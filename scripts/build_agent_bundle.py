"""Build the authenticated-download Agent bundle from this checkout and uv.lock."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]


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
        subprocess.run([uv, "export", "--locked", "--no-dev", "--no-emit-project",
                        "--format", "requirements-txt", "--output-file", str(stage / "requirements.txt"),
                        "--quiet"], cwd=ROOT, check=True)
        for name in ("install-agent.sh", "install_agent.py", "agent_ownership.py"):
            shutil.copyfile(ROOT / "deploy" / name, stage / name)
        shutil.copyfile(ROOT / "apps/control_plane/vla_control/agent_bootstrap.sh", stage / "agent-bootstrap.sh")
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
        print(f"Agent 安装包：{output}\nSHA-256：{checksum}")


if __name__ == "__main__":
    main()
