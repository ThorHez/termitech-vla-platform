# 将本地中台迁移至 Linux

> 2026-10-22 安全暂停：公网中台已停止并取消开机自启，后续监听地址已改为 `127.0.0.1`。下文公网启动与验收记录仅描述此前迁移过程，不是当前启用指令。仓库已提供 [HTTPS 配置](https.md)及[用户账号配置](users-and-permissions.md)能力，尚未据此重新部署公网入口；未经确认不要启动服务。

中台包含网页、API、内嵌调度器，以及 SSH 隧道管理器。迁移需要同时保留 SQLite 数据库、管理员/登记令牌和 `remote-management`，其中 `credentials.key` 用于解密数据库里的 SSH 密码。只复制代码或数据库会丢失连接恢复能力。

训练项目、训练解释器、数据集、模型、输出和 Agent 的本地状态仍保存在对应训练机器。中台不依赖 `third_party/openpi-umi` 训练仓库，也不需要在控制服务环境安装训练依赖。

后续 HTTPS 升级使用 Caddy 占用公网 80/443，中台移到 `127.0.0.1:8000`。合并 HTTPS 环境补丁时保留目标端最新数据库、用户与 Agent 凭据和 `remote-management`；同步将 `VLA_TUNNEL_TARGET_PORT` 及旧的内部默认入口从 80 改为 8000。三台 Agent 原有的反向隧道监听端口 35071、36569、44003 不变，不需重装节点。证书申请、续期与离线配置准备见 [HTTPS 说明](https.md)。

## 准备发布包

在当前工作区执行：

```sh
uv run python scripts/deploy_control_plane.py package --build \
  --output .local/migration/release-20261022.tar.gz
```

该命令构建网页和 Agent 包，再按显式目录清单打包当前工作区，包含尚未提交的修改和新增代码。不会只取 Git HEAD，也不会携带 `.venv`、`node_modules`、`.local`、`.env` 或训练仓库。发布包内含源码、网页产物、Agent 包、同一版本 wheel、锁定依赖和逐文件 SHA-256 清单；终端仅输出包路径、校验和、大小等非敏感信息。已完成构建时可省略 `--build`。

## 目标布局与独立环境

本次目标是 `root@115.190.212.169`，路径 `/data4/termitech-vla-platform`：

```text
/data4/termitech-vla-platform/
  apps/ packages/ deploy/     # 当前工作区发布包中的源码
  scripts/ docs/ artifacts/   # 网页、wheel、锁定依赖和 Agent 包
  .venv/                      # 中台专用 Python 3.11+，不能复用训练环境
  .local/
    control.db
    dev-credentials.json
    control.env               # 0600，systemd EnvironmentFile
    remote-management/        # 0700，包含密钥、主机指纹和部署日志
    backups/
```

先检查端口、磁盘、现有服务及正在运行的训练。使用现有可用 Python 3.11+ 创建新的隔离环境，再安装发布包内锁定依赖及 wheel，例如：

```sh
/path/to/python3.11 -m venv /data4/termitech-vla-platform/.venv
/data4/termitech-vla-platform/.venv/bin/python -m pip install \
  --require-hashes --only-binary=:all: \
  -r /data4/termitech-vla-platform/artifacts/requirements.txt
/data4/termitech-vla-platform/.venv/bin/python -m pip install \
  --no-deps /data4/termitech-vla-platform/artifacts/termitech_vla_platform-0.1.0-py3-none-any.whl
```

不要更改系统 Python、已有 Agent 环境或训练虚拟环境。首次迁移直接在该目录保存发布源码，后续升级前保留完整旧版本备份。

## 一致性切换

1. 在旧中台仍运行时完成源码包上传、校验、独立运行环境安装及服务文件准备，不启动新中台。
2. 确认没有进行中的部署、配方回写、预检或训练启动事务。记录节点、资产、配方、训练记录和连接数量，以及现有训练进程。
3. 停止本地中台及其管理的 SSH 隧道，保留远端 Agent 和训练服务。
4. 生成最终一致性备份：

   ```sh
   uv run python scripts/deploy_control_plane.py snapshot --controller-stopped \
     --output .local/migration/state-20261022.tar.gz
   ```

   此命令使用 SQLite Backup API，检查备份完整性，并将数据库、令牌及整个 SSH 元数据目录放入私有归档。`--controller-stopped` 是调用者对中台已停止的明确确认，脚本本身不会停止服务。切换期间禁止启动两个使用同一节点凭据的调度中台。

5. 私密传输归档并校验 SHA-256。解压到独立 `.local/`，将归档中的 `credentials.json` 保存为 `dev-credentials.json`，目录 0700、文件 0600。保留本地原件和目标切换前备份。历史中引用的本地 CPU 演示资源保留原路径作为历史记录，不改写为训练机器上未经验证的新路径。
6. 用迁移后的令牌生成服务配置：

   ```sh
   python scripts/deploy_control_plane.py render \
     --credentials /data4/termitech-vla-platform/.local/dev-credentials.json \
     --release-dir /data4/termitech-vla-platform \
     --state-dir /data4/termitech-vla-platform/.local \
     --python /data4/termitech-vla-platform/.venv/bin/python \
     --port 80 --output-dir /data4/termitech-vla-platform/.local/service-files
   ```

   将生成的 `control.env` 安装至 `.local/control.env`（0600），`termitech-vla-control.service` 安装至 `/etc/systemd/system/`，运行 `systemd-analyze verify` 后再 `daemon-reload`、`enable --now termitech-vla-control.service`。服务使用 root，以读取本次迁移的私有凭据并保留原有 SSH 管理权限；后台为单进程单 worker，内嵌调度器开启。

7. 验证网页、`/healthz`、原管理员登录、元数据数量及源路径保持一致，等待所有原有节点收到新心跳。再核对训练进程、日志及 Agent 运行用户，确认没有重启训练。

网页入口为 `http://115.190.212.169`（端口需确认空闲且网络允许访问）。Agent 回连入口继续使用 `http://127.0.0.1:80` 和原有 SSH 反向隧道，中台把隧道转发目标指向自身 80 端口。目标主机同时承担 T001host Agent 时，也必须验证中台能通过已保存的连接登录该主机自身并恢复其反向隧道。保留管理员令牌，不在命令行、部署日志或文档中打印。

## 回滚

启动目标前发生错误，可保留旧中台继续运行。

新中台接受过心跳、Agent 事件或命令确认后，数据库就已经产生新状态，即使没有新增训练、资产或配方，也不能直接从迁移前数据库恢复。回滚应先停止目标中台并确认其隧道退出，用 SQLite Backup API 保存目标端最新数据库，同时保存当前令牌和 `remote-management`，校验后私密同步回本机，再启动旧中台。这样已经确认的 Agent 事件不会因恢复旧数据库而丢失。全过程保持只有一个中台在管理这些节点。

## 2026-10-22 迁移验收历史（随后已安全停用）

停止前的入口为 `http://115.190.212.169`（80 端口）。外部网络对 8000/8080 的 HTTP 请求超时，80 端口已通过公网访问验证。生产服务使用独立的 Python 3.13.9 正式版，目录为 `/data4/termitech-vla-platform/.venv`，管理员令牌保持原值。

systemd 单元是 `termitech-vla-control.service`，迁移验收时曾启用开机自启，随后按安全暂停要求已停止并禁用；内嵌调度器及 SSH 隧道管理器都由这一个进程运行。原本机中台已停止。原有三台 Agent 的节点身份、配置及反向端口保持不变：T001host 35071、bms-xwld-0001 36569、T003host 44003。

业务数据位于 `/data4/termitech-vla-platform/.local/control.db`，密钥、主机指纹和完整部署日志位于同目录的 `remote-management/`。一致性快照和源站历史归档均保留在私有 `.local` 目录，源站原件也保留。迁移验收确认 26 项资产、14 次训练、8199 条事件、1 个保存配方及 2 个配方修订的数量一致。

```sh
systemctl status termitech-vla-control.service
journalctl -u termitech-vla-control.service -n 100 --no-pager
```

后续备份使用本文件配套脚本的 `snapshot` 子命令。它会显式关闭 SQLite 连接，把输出数据库设为独立 DELETE 日志模式；校验清单和归档不包含临时 WAL/SHM 文件。
