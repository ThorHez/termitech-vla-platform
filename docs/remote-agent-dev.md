# 本机中台与远端 Agent 联调

`scripts/remote_agent_dev.py` 在本机运行 API、内嵌 Scheduler 和已构建的网页，通过 SSH 反向端口转发供远端 Agent 连接。它复用 `.local/control.db` 和 `.local/dev-credentials.json`，不创建演示资产，不调用 `prepare_demo()`，也不启动本机 Agent。

## 启动

先安装依赖并构建网页：

```bash
uv sync --locked
cd apps/web
pnpm install --frozen-lockfile
pnpm build
cd ../..
```

确认本机到目标服务器已有可用的非交互 SSH 认证和已确认的主机密钥，然后运行：

```bash
uv run python scripts/remote_agent_dev.py
```

默认连接目标为 `root@115.190.212.169`，本机管理入口为 `http://127.0.0.1:8000`，远端 Agent 的 `control_plane_url` 应设置为 `http://127.0.0.1:18000`。远端 Agent 服务的安装、配置和启动单独完成，本脚本不远程安装或修改服务。

管理员登录使用 `.local/dev-credentials.json` 中的 `admin_token`。Agent 初次登记使用同一文件的 `enrollment_token`。脚本不打印这些值，不把它们加入命令行参数；中台凭据通过子进程环境传入。凭据不存在时生成两个不同的随机值，并以 `0600` 权限保存；已有凭据和数据库原样复用。

可覆盖连接目标和端口：

```bash
uv run python scripts/remote_agent_dev.py \
  --ssh-target root@115.190.212.169 \
  --ssh-port 22 \
  --local-port 8000 \
  --remote-port 18000
```

SSH 使用 `-N -T -R 127.0.0.1:18000:127.0.0.1:8000`，设置 `BatchMode=yes`、`ExitOnForwardFailure=yes`、连接超时和保活，并通过 `ControlMaster=no`、`ControlPath=none` 创建本次独立连接。脚本不会提示输入密码，不修改 SSH 配置或密钥，也不会复用、关闭已有的 SSH 主连接。

## 验证与日志

终端输出本次日志目录 `.local/remote-agent-dev/<时间戳-PID>/`，其中 `control-plane.log` 保存中台输出，`ssh-tunnel.log` 保存 SSH 诊断。每次启动使用独立日志文件，权限为 `0600`。

本机健康检查通过后才启动隧道。SSH 进程创建成功不等于连接已建立，远端可用性应在服务器上检查：

```bash
curl --fail http://127.0.0.1:18000/healthz
```

返回 `{"status":"ok"}` 后，观察网页机器列表中的 Agent 心跳及资源数据。复用数据库时可能仍显示此前的本机演示节点，它不会由此脚本启动，最终会因心跳过期显示离线。

## 停止与故障行为

保持脚本在前台运行。`Ctrl+C` 或 `SIGTERM` 会结束本次启动的 API 和 SSH 隧道；远端 Agent 的 systemd 服务仍常驻，连接恢复后继续上报。联调期间本机休眠、网络中断或进程退出会让中台不可达，此方式用于开发验证，正式使用应部署长期在线中台。

本机端口已占用时脚本拒绝启动，不结束已有服务。远端转发端口冲突时 SSH 退出，脚本随即清理自己的 API。API 或隧道异常退出也会结束另一进程，错误原因保存在日志中。清理范围限定于本次创建的子进程，不接触其他中台、SSH 或远端训练进程。
