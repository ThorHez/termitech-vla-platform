# 本机中台与远端 Agent 联调

`scripts/remote_agent_dev.py` 运行本机 API、内嵌 Scheduler 和已构建网页，复用 `.local/control.db` 与 `.local/dev-credentials.json`，不创建演示资产或启动本机 Agent。SSH 接入与隧道由中台管理，脚本不再绑定固定服务器或启动独立 SSH 命令。

## 启动

```bash
uv sync --locked
cd apps/web
pnpm install --frozen-lockfile
pnpm build
cd ../..
uv run python scripts/build_agent_bundle.py
uv run python -m vla_control.users create-admin \
  --database-url sqlite:///./.local/control.db --username admin
uv run python scripts/remote_agent_dev.py
```

默认管理入口 `http://127.0.0.1:8000`，可用 `--local-port` 修改。使用独立管理员账号登录；账号已存在时跳过创建步骤。旧 `admin_token` 不再用于网页登录。仅本机 HTTP 联调显式允许非 Secure Cookie，生产入口须使用 HTTPS。若服务已迁移至另一台机器，不要再用迁移前本机数据库启动第二个中台。

## 从页面接入

1. 打开机器资源页，点击“接入机器”。
2. 填裸金属 IP、SSH 端口、用户名、密码和 Agent 安装目录，测试 SSH。
3. 若探测到同目录已有 Agent，使用它的状态目录、输出目录及训练环境，保留原机器身份。
4. 配置实际训练 Python、代码目录和适配器，提交远程安装；配方在创建训练时选择。
5. 中台自动上传安装包与私有配置、安装/更新并启动 Agent，页面显示阶段日志、节点上线及环境探测结果。

本机入口属于 loopback 时，中台通过该机器的 SSH 密码认证创建独立反向隧道。远端端口由 SSH 服务自动分配，只监听远端 `127.0.0.1`；实际端口写入 Agent 配置，不再复用固定 18000。转发目标由 `VLA_TUNNEL_TARGET_HOST` / `VLA_TUNNEL_TARGET_PORT` 指定，启动脚本自动配置为当前本机端口。

同一中台可管理多台服务器，无需手工启动多个隧道命令。断线后中台自动重连并复用原端口；中台重启后读取持久化部署记录恢复隧道。应用应运行一个 API worker 来持有隧道与部署后台线程。

## 持久化与日志

SSH 密码在数据库中加密存储；应用密钥、专属 SSH 主机指纹文件保存在 `.local/remote-management/`。应用不读取或改写用户 `~/.ssh`，密码不进入进程参数、明文配置、API 响应或部署日志。首次连接记录主机指纹，后续变更拒绝连接。

部署记录和安全阶段日志在页面可查询，关闭弹窗不会中止安装。启动器日志在 `.local/remote-agent-dev/<时间戳-PID>/control-plane.log`，权限 0600。部署成功必须由实际 Agent 新心跳及对应 onboarding ID 确认，并通过训练 Python 环境探测；不以 SSH 命令返回零就认定接入成功。

## 停止与生产部署

Ctrl+C 关闭本机中台及其隧道，远端 Agent 的 systemd 服务继续运行。本机休眠或离线时远端无法上报，恢复后中台重连。原有训练进程不会因页面关闭或隧道中断被终止。

长期中台设置 `VLA_AGENT_CONTROL_PLANE_URL` 为裸金属可访问的 HTTPS 基础 URL，Agent 即可直连，不需要 SSH 转发。特殊机器可在高级设置覆盖入口。数据库及 `VLA_REMOTE_STATE_DIR` 必须一起持久化；Compose 已挂载专属 `remote_state` 卷。
