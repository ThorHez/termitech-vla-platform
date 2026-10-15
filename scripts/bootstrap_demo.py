"""Prepare an explicit, isolated CPU smoke fixture; never touch real training data."""

import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare_demo(base: Path | None = None, port: int = 8000) -> dict:
    base = (base or ROOT / ".local").resolve()
    base.mkdir(parents=True, exist_ok=True)
    credentials_file = base / "dev-credentials.json"
    if credentials_file.exists():
        credentials = json.loads(credentials_file.read_text())
    else:
        credentials = {"admin_token": secrets.token_urlsafe(32), "enrollment_token": secrets.token_urlsafe(32)}
        credentials_file.write_text(json.dumps(credentials, indent=2))
        credentials_file.chmod(0o600)
    code = base / "demo-code"
    code.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "templates" / "smoke_training.py", code / "smoke_training.py")
    (code / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    if not (code / ".git").exists():
        subprocess.run(["git", "init", "-b", "main", str(code)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(code), "add", "."], check=True, capture_output=True)
    changed = subprocess.run(["git", "-C", str(code), "diff", "--cached", "--quiet"]).returncode
    if changed:
        subprocess.run(["git", "-C", str(code), "-c", "user.name=VLA local smoke", "-c",
                        "user.email=smoke@localhost", "-c", "commit.gpgsign=false", "commit", "-m",
                        "Record platform smoke workload"], check=True, capture_output=True)
    paths = {kind: base / "demo-assets" / kind for kind in ("dataset", "model", "norm")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.txt").write_text("平台链路测试占位资产；不是机器人数据或 VLA 权重。\n")
    output = base / "outputs"
    output.mkdir(exist_ok=True)
    config = {"control_plane_url": f"http://127.0.0.1:{port}",
              "enrollment_token": credentials["enrollment_token"], "name": "本机 · 链路测试节点",
              "roots": [str(base)], "output_root": str(output), "state_dir": str(base / "agent-state"),
              "poll_interval_seconds": 0.5, "heartbeat_interval_seconds": 2,
              "profiles": [{"id": "local-smoke", "name": "CPU 链路测试环境", "code_dir": str(code),
                            "python": sys.executable, "adapter": "smoke", "recipe": "smoke",
                            "execution_mode": "native", "metadata": {"simulation": True}}]}
    config_file = base / "agent.json"
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    config_file.chmod(0o600)
    env = {**os.environ, "VLA_ADMIN_TOKEN": credentials["admin_token"],
           "VLA_ENROLLMENT_TOKEN": credentials["enrollment_token"],
           "VLA_DATABASE_URL": f"sqlite:///{base / 'control.db'}",
           "VLA_SCHEDULER_INTERVAL_SECONDS": "0.5", "VLA_HOST": "127.0.0.1", "VLA_PORT": str(port),
           "VLA_WEB_DIST": str(ROOT / "apps" / "web" / "dist")}
    return {"base": base, "credentials": credentials, "config": config,
            "config_file": config_file, "env": env, "assets": paths}


if __name__ == "__main__":
    fixture = prepare_demo()
    print(f"已准备本地测试环境：{fixture['base']}")
    print(f"管理员访问令牌保存在 {fixture['base'] / 'dev-credentials.json'}")
