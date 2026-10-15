"""Start the local API and smoke Agent. Training is CPU-only and clearly labeled."""

import argparse
import json
import signal
import subprocess
import sys
import time
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from bootstrap_demo import ROOT, prepare_demo


def request(url, token=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if body is None else json.dumps(body).encode()
    with urlopen(Request(url, data=data, headers=headers), timeout=5) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    fixture = prepare_demo(port=args.port)
    api = f"http://127.0.0.1:{args.port}"
    children = []
    try:
        children.append(subprocess.Popen([sys.executable, "-m", "vla_control.main"], env=fixture["env"], cwd=ROOT))
        for _ in range(100):
            if children[0].poll() is not None:
                raise RuntimeError("管理服务启动失败，请查看日志")
            try:
                request(api + "/healthz")
                break
            except (URLError, HTTPError):
                time.sleep(0.1)
        else:
            raise RuntimeError("管理服务启动超时")
        children.append(subprocess.Popen([sys.executable, "-m", "vla_agent.main", "--config",
                                         str(fixture["config_file"])], cwd=ROOT))
        token = fixture["credentials"]["admin_token"]
        for _ in range(100):
            nodes = request(api + "/api/v1/nodes", token)
            node = next((n for n in nodes if n["name"] == fixture["config"]["name"]), None)
            if node:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Agent 登记超时")
        existing = request(api + "/api/v1/assets", token)
        for kind, path in fixture["assets"].items():
            if not any(a["node_id"] == node["id"] and a["path"] == str(path) for a in existing):
                request(api + "/api/v1/assets", token, {"name": "链路测试 / " + kind,
                        "kind": kind.upper() if kind != "dataset" else "DATASET", "node_id": node["id"],
                        "path": str(path), "version": "demo-v1", "metadata": {"simulation": True}})
        print(f"\n平台已启动：{api}\n这是本地 CPU 链路测试环境。", flush=True)
        print(f"登录令牌：{token}\nCtrl+C 停止本地管理服务和 Agent。", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
