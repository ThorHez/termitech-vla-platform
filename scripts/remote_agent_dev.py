"""Run the local control plane and an SSH reverse tunnel for a remote Agent."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("端口必须是整数") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须在 1–65535 之间")
    return port


def load_credentials(path: Path) -> dict[str, str]:
    """Reuse development credentials, creating them with private permissions."""
    if path.is_symlink():
        raise RuntimeError(f"凭据文件不能是符号链接：{path}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                {"admin_token": secrets.token_urlsafe(32),
                 "enrollment_token": secrets.token_urlsafe(32)},
                stream, indent=2,
            )
            stream.write("\n")
    path.chmod(0o600)
    try:
        credentials = json.loads(path.read_text())
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError(f"凭据文件不是有效 JSON：{path}") from exc
    required = ("admin_token", "enrollment_token")
    if not isinstance(credentials, dict) or any(
        not isinstance(credentials.get(key), str) or not credentials[key].strip()
        for key in required
    ):
        raise RuntimeError(f"凭据文件缺少 admin_token 或 enrollment_token：{path}")
    if credentials["admin_token"] == credentials["enrollment_token"]:
        raise RuntimeError("管理员令牌和节点登记令牌必须不同")
    return {key: credentials[key] for key in required}


def require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"无法占用本机 127.0.0.1:{port}；端口可能已被使用。"
                "请自行停止已有服务或使用 --local-port，脚本不会结束已有进程。"
            ) from exc


def private_log(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "wb", buffering=0)


def stop_children(children: list[subprocess.Popen]) -> None:
    """Signal only the process groups created by this invocation."""
    for child in reversed(children):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for child in reversed(children):
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def wait_for_api(child: subprocess.Popen, url: str, stop: threading.Event) -> bool:
    deadline = time.monotonic() + 20
    while not stop.is_set() and time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("本机中台启动失败，请查看 control-plane.log")
        try:
            with urlopen(url + "/healthz", timeout=1) as response:
                healthy = response.status == 200 and json.load(response).get("status") == "ok"
            if healthy and child.poll() is None:
                return True
        except (URLError, HTTPError, OSError, ValueError):
            pass
        stop.wait(0.1)
    if stop.is_set():
        return False
    raise RuntimeError("本机中台启动超时，请查看 control-plane.log")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", default="root@115.190.212.169")
    parser.add_argument("--ssh-port", type=port_number, default=22)
    parser.add_argument("--local-port", type=port_number, default=8000)
    parser.add_argument("--remote-port", type=port_number, default=18000)
    args = parser.parse_args()
    if not args.ssh_target or args.ssh_target.startswith("-") or any(
        character.isspace() for character in args.ssh_target
    ):
        parser.error("--ssh-target 必须是 SSH 主机名或 user@host")

    stop = threading.Event()
    children: list[subprocess.Popen] = []
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda *_: stop.set())

    try:
        require_free_port(args.local_port)
        web_dist = ROOT / "apps" / "web" / "dist"
        if not (web_dist / "index.html").is_file():
            raise RuntimeError("缺少网页构建产物；请先在 apps/web 执行 pnpm build")
        ssh = shutil.which("ssh")
        if not ssh:
            raise RuntimeError("找不到 ssh，请先安装 OpenSSH 客户端")
        LOCAL.mkdir(parents=True, exist_ok=True)
        credentials_path = LOCAL / "dev-credentials.json"
        credentials = load_credentials(credentials_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
        log_dir = LOCAL / "remote-agent-dev" / run_id
        log_dir.mkdir(parents=True, mode=0o700)
        local_url = f"http://127.0.0.1:{args.local_port}"
        remote_url = f"http://127.0.0.1:{args.remote_port}"
        api_env = {
            **os.environ,
            "VLA_ADMIN_TOKEN": credentials["admin_token"],
            "VLA_ENROLLMENT_TOKEN": credentials["enrollment_token"],
            "VLA_DATABASE_URL": f"sqlite:///{LOCAL / 'control.db'}",
            "VLA_SCHEDULER_ENABLED": "true",
            "VLA_SCHEDULER_INTERVAL_SECONDS": "0.5",
            "VLA_HOST": "127.0.0.1",
            "VLA_PORT": str(args.local_port),
            "VLA_WEB_DIST": str(web_dist),
            "PYTHONUNBUFFERED": "1",
        }
        print(f"本次日志目录：{log_dir}", flush=True)
        with ExitStack() as stack:
            api_log = stack.enter_context(private_log(log_dir / "control-plane.log"))
            ssh_log = stack.enter_context(private_log(log_dir / "ssh-tunnel.log"))
            # Cleanup runs before the log streams close, including startup failures.
            stack.callback(stop_children, children)
            if stop.is_set():
                return 0
            api = subprocess.Popen(
                [sys.executable, "-m", "vla_control.main"],
                cwd=ROOT, env=api_env, stdin=subprocess.DEVNULL,
                stdout=api_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            children.append(api)
            if not wait_for_api(api, local_url, stop):
                return 0
            forward = f"127.0.0.1:{args.remote_port}:127.0.0.1:{args.local_port}"
            tunnel = subprocess.Popen(
                [ssh, "-N", "-T", "-p", str(args.ssh_port),
                 "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                 "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                 "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1",
                 "-o", "ControlMaster=no", "-o", "ControlPath=none",
                 "-R", forward, args.ssh_target],
                cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=ssh_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            children.append(tunnel)
            print(
                f"本机中台已就绪：{local_url}\n"
                f"SSH 隧道正在连接 {args.ssh_target}；远端 Agent 应使用 {remote_url}。\n"
                f"登录令牌请从 {credentials_path} 的 admin_token 字段读取。\n"
                "连接是否完成以远端健康检查和 Agent 心跳为准；本脚本不启动本机 Agent。\n"
                "保持本进程运行；Ctrl+C 停止本次启动的中台和隧道。",
                flush=True,
            )
            while not stop.wait(0.25):
                if api.poll() is not None:
                    raise RuntimeError("本机中台已退出，请查看 control-plane.log")
                if tunnel.poll() is not None:
                    raise RuntimeError(
                        "SSH 隧道已退出，请查看 ssh-tunnel.log。需要已可用的非交互 SSH 认证；"
                        "远端端口冲突也会导致退出。"
                    )
        print("已停止本次启动的中台和隧道；远端 Agent 服务未被修改。", flush=True)
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"启动或运行失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
