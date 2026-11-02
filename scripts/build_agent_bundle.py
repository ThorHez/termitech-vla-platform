"""Build the authenticated-download Agent bundle from this checkout and uv.lock."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE_MEMBER = "vla_agent/agent-release.json"


def stamp_release(wheel: Path, stage: Path, root: Path | None = None):
    """Stamp a temporary wheel, preserving the checkout and its uncommitted work."""
    root = ROOT if root is None else root
    with zipfile.ZipFile(wheel) as archive:
        entries = [(item, archive.read(item)) for item in archive.infolist()]
    records = [item.filename for item, _ in entries if item.filename.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("Agent wheel 缺少唯一的 RECORD")
    record = records[0]
    hashes = {"wheel:" + item.filename: hashlib.sha256(content).hexdigest()
              for item, content in entries if item.filename not in {record, RELEASE_MEMBER}}
    hashes.update({"bundle:" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in stage.iterdir() if path.is_file() and path != wheel
                   and path.name not in {"release.json", "manifest.json"}})
    hashes.update({"project:" + name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("pyproject.toml", "uv.lock")})
    release = {"schema_version": 1,
               "version": tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"],
               "build_id": hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    content = (json.dumps(release, sort_keys=True) + "\n").encode()
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
    rows = list(csv.reader(io.StringIO(next(data for item, data in entries if item.filename == record).decode())))
    rows = [row for row in rows if row[0] != RELEASE_MEMBER]
    rows.append([RELEASE_MEMBER, "sha256=" + digest, str(len(content))])
    updated_record = io.StringIO(newline="")
    csv.writer(updated_record, lineterminator="\n").writerows(rows)
    temporary = wheel.with_suffix(".whl.stamping")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item, data in entries:
            if item.filename != RELEASE_MEMBER:
                archive.writestr(item, updated_record.getvalue().encode() if item.filename == record else data)
        archive.writestr(RELEASE_MEMBER, content)
    os.replace(temporary, wheel)
    (stage / "release.json").write_bytes(content)
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/agent-dist")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("需要 uv 来构建 Agent wheel 和导出锁定依赖")
    with tempfile.TemporaryDirectory(prefix="vla-agent-build-") as temporary:
        stage = Path(temporary)
        subprocess.run([uv, "build", "--wheel", "--out-dir", str(stage)], cwd=ROOT, check=True)
        subprocess.run([uv, "export", "--locked", "--no-dev", "--no-emit-project", "--no-header",
                        "--format", "requirements-txt", "--output-file", str(stage / "requirements.txt"),
                        "--quiet"], cwd=ROOT, check=True)
        for name in ("install-agent.sh", "install_agent.py", "agent_ownership.py"):
            shutil.copyfile(ROOT / "deploy" / name, stage / name)
        shutil.copyfile(ROOT / "apps/control_plane/vla_control/agent_bootstrap.sh", stage / "agent-bootstrap.sh")
        wheels = list(stage.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Agent 构建需要且只能生成一个 wheel")
        release = stamp_release(wheels[0], stage)
        manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(stage.iterdir())}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_archive = destination / "vla-agent-bundle.tar.gz.building"
        with tarfile.open(temporary_archive, "w:gz") as archive:
            for path in sorted(stage.iterdir()):
                archive.add(path, arcname=path.name)
        output = destination / "vla-agent-bundle.tar.gz"
        os.replace(temporary_archive, output)
        checksum = hashlib.sha256(output.read_bytes()).hexdigest()
        (destination / "vla-agent-bundle.sha256").write_text(checksum + "  " + output.name + "\n")
        print(f"Agent 安装包：{output}\n版本：{release['version']}\n构建：{release['build_id']}\nSHA-256：{checksum}")


if __name__ == "__main__":
    main()
