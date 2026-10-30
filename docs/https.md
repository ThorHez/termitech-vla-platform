# HTTPS

网页和管理 API 支持 HTTPS，可由 Caddy 终止 TLS，也可由中台直接加载 PEM 证书。用户会话保留 `Secure`、`HttpOnly` 和 `SameSite=Strict` Cookie。网页与 API 使用同一来源，不需要将凭据发送到另一域名。

配置生成与服务启用是两个独立步骤：渲染器不会安装服务、申请公网证书、修改防火墙或启动中台。`115.190.212.169` 已于 2026-10-23 按用户授权启用 HTTPS，实际服务状态应从目标机核实；不能从配置文件存在推断服务已经启动或停止。

## 本机测试证书

```sh
.venv/bin/python scripts/generate_local_tls.py --output-dir .local/tls
```

生成本地 CA `root-ca.pem`、CA 私钥 `root-ca-key.pem`、服务证书链 `fullchain.pem` 和服务私钥 `server-key.pem`。证书仅包含 `localhost`、`127.0.0.1`、`::1`，默认有效期 30 天；私钥权限为 0600，目录为 0700。重复运行会校验并复用完整、匹配且尚未到期的文件；文件不完整或即将到期时，改用新的空目录生成。

本地 CA **没有自动加入系统或浏览器信任库**。浏览器首次打开本地 HTTPS 页面会提示证书不受信任；这不等同于公开 CA 签发的证书。只在明确决定信任该本机 CA 后，手动导入 `root-ca.pem`。不要共享私钥，也不要将本地 CA 用于公网服务。

命令行验收可以只为该次请求指定 CA，无需修改系统信任：

```sh
curl --cacert .local/tls/root-ca.pem https://127.0.0.1:8443/healthz
```

该命令要求已启动本地 HTTPS 预览；生成证书本身不启动服务。现有本地预览使用独立测试数据库，账户与原业务机器记录的来源不因 HTTPS 改变。

## 生成反向代理配置

`scripts/configure_https.py` 为 Caddy **2.11.7** 生成配置，支持三个模式：

| 模式 | 证书来源 | 适用场景 |
| --- | --- | --- |
| `acme` | Caddy 运行时向 Let's Encrypt 申请并自动续期 | 公网 IP 或域名 |
| `provided` | 指定已有 PEM 证书链和私钥 | 已有正式证书或受管内部 CA |
| `internal` | Caddy 运行时创建自己的内部 CA | 隔离测试，客户端需单独信任 CA |

例如为现有公网 IP 准备配置，但保持服务停止：

```sh
.venv/bin/python scripts/configure_https.py \
  --mode acme \
  --site https://115.190.212.169 \
  --upstream http://127.0.0.1:8000 \
  --output-dir .local/https-public-ip
```

输出目录包含：

- `Caddyfile`：HTTPS 入口、HTTP 跳转、流式响应和代理头配置。
- `control-https.env`：中台环境变量的**补丁**，需要合并进原私有 `control.env`；不可覆盖数据库、登记凭据、SSH 状态目录等其他配置。补丁中的空值也要合并，用于清除原生 TLS 的证书与私钥配置，避免代理上游继续监听 TLS。
- `termitech-vla-https.service`：独立 `caddy` 用户运行的 systemd 服务，默认二进制 `/usr/bin/caddy`、配置 `/etc/termitech-vla/https/Caddyfile`。
- `https-manifest.json`：入口、上游、模式和预期 Caddy 版本等非凭据部署信息。

输出文件默认 0600，目录 0700；脚本拒绝覆盖已有输出。安装配置时需显式授予 `caddy` 用户读取 Caddyfile 的权限。服务的证书及 ACME 账户状态保存到 `/var/lib/termitech-vla-caddy`；保留该目录并让服务用户独占读取。新机器需先安装匹配版本的 Caddy 并创建服务用户，脚本不会下载二进制或创建账户。

Compose 部署将上游设为 `--upstream http://api:8000`。此时清单标记 `deployment_kind=compose`，输出的环境补丁和 systemd 单元仅含说明注释，**不要安装或合并这两个文件**。使用 `deploy/compose.https.yaml` 覆盖文件配置容器网络与可信代理，在私有 Compose `.env` 中设置 `VLA_PUBLIC_URL` 和 `VLA_HTTPS_CONFIG_DIR`，将生成的 Caddyfile 挂载给代理；Compose 负责保持 API 容器的监听设置及代理地址一致。

已有证书使用 `provided` 模式，路径为**服务器上的最终路径**：

```sh
.venv/bin/python scripts/configure_https.py \
  --mode provided \
  --site https://115.190.212.169 \
  --upstream http://127.0.0.1:8000 \
  --cert-file /etc/termitech-vla/https/fullchain.pem \
  --key-file /etc/termitech-vla/https/server-key.pem \
  --output-dir .local/https-provided
```

渲染器不会读取或复制证书私钥；部署时验证证书链、有效期、密钥匹配及 SAN 包含入口 IP/域名。证书与私钥必须放在 `caddy` 用户可读的位置；生成的服务设置 `ProtectHome=true`，不要放在 `/root` 或用户家目录下。私钥用 0600 归属 `caddy`，或使用适当的专用组及 0640。

本地代理预览可以结合前述证书，设置 `--site https://127.0.0.1:8443 --bind 127.0.0.1 --http-port 8000 --https-port 8443 --upstream http://127.0.0.1:8001`。`--site` 是浏览器最终入口，其端口必须与 `--https-port` 一致；HTTPS 标准端口为 443 时可省略入口 URL 中的端口。

## 公网 IP 证书与续期

Let's Encrypt 已支持公网 IP 证书，使用 `shortlived` 配置档，证书有效期约 160 小时。渲染器对 IP 入口显式选择此配置档；**持续自动续期是上线条件**。公网入口不要求先购买域名。[Let's Encrypt 官方说明](https://letsencrypt.org/2026/01/15/6day-and-ip-general-availability/)、[Caddy ACME issuer 配置](https://caddyserver.com/docs/caddyfile/directives/tls#issuers)。

未来启用时，公网 TCP 80 和 443 需能到达 Caddy，用于 HTTP 跳转和 ACME 验证；云安全组、主机防火墙与端口占用都要与此一致。外部只公开代理端口，中台 `127.0.0.1:8000` 保持回环监听。可先用 `--acme-staging` 测试申请链路，测试 CA 的证书不被浏览器信任；完成后用新的输出目录生成正式配置。`--email` 可登记证书账户联系地址。

ACME 模式由持续运行的 Caddy 负责续期，保留其状态目录，不要只复制一份短期证书后停止客户端。检查服务日志中的申请、续期失败以及客户端实际收到的到期时间。`provided` 模式的续期由证书管理方负责；更新 PEM 文件后重启代理加载新证书。当前生成的 systemd 服务关闭 Caddy 管理 API，不提供 `systemctl reload`；重启前执行配置校验并安排短暂入口中断。

IP 客户端通常不发送 TLS SNI；公网 IP 经 NAT 映射到私网网卡时，按本地连接地址选择证书可能找不到公网 IP 证书。渲染器仅对字面 IPv4/IPv6 入口设置全局 `default_sni` 为入口 IP，确保无 SNI 的连接也选择该证书；域名入口仍要求 SNI 与 Host 一致，未知 Host 仍被拒绝。[Caddy default_sni 官方说明](https://caddyserver.com/docs/caddyfile/options#default-sni)。

## 中台配置与 Agent 连通性

当前固定 Caddy 2.11.7 所带 Go/x/net 版本存在 HTTP/2 拒绝服务公告，渲染器暂时仅启用 `h1`（HTTPS 上的 HTTP/1.1），保留流式响应。升级到经过审计的修复版二进制并复验前，不应直接重新开放 HTTP/2/3。此项缓解不等于所有 Go 组件漏洞均已修复。[Go 公告示例](https://pkg.go.dev/vuln/GO-2026-6603)。

生成配置显式限制请求头为 16 KiB、请求体为 10 MB。读取请求头最长 15 秒，读取请求体最长 60 秒，正文停顿期限为 15 秒并要求平均每秒至少 1024 字节；这适合管理 API 的 JSON 请求，训练数据仍保留在执行机器本地。代理连接上游最长等待 3 秒，等待首个上游响应头最长 90 秒，留出已有 SSH 探测和数据库忙等的余量。该响应头期限不会截断已经开始的日志流；没有配置统一的响应总时长。

这些期限和大小上限是资源耗尽防护的一部分，不是全局请求限速或 DDoS 容量承诺。登录的账号/IP 失败次数限制也不能替代边缘限流、连接准入、服务资源配额、审计保留策略和云端流量清洗。部署时先在隔离环境验证正常 SSH 检查、目录联想与日志下载，再检查目标服务实际加载的配置。[Caddy 请求期限](https://caddyserver.com/docs/caddyfile/options#timeouts)、[上游响应头期限](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#the-http-transport)、[OWASP DoS 指引](https://cheatsheetseries.owasp.org/cheatsheets/Denial_of_Service_Cheat_Sheet.html)。

2026-10-23 的生产修复另通过 systemd `security-audit-runtime.conf` drop-in 对控制台和 Caddy 指定进程局部的 `SSL_CERT_FILE=/etc/termitech-vla/https/trusted-ca-20261019.pem` 与空的 `SSL_CERT_DIR=/etc/termitech-vla/https/empty-ca`，避免重新合并旧系统 CA；信任包来自锁定的 certifi 2026.10.19，并已验证 ACME 目录的正常 HTTPS 访问。该文件不是渲染器自动生成的内容，后续更新 certifi 时需同步替换、记录摘要并复验这份局部副本，不能只升级 Python 包。不要替换共享 Conda 或全局 CA 来代替进程隔离；保留原服务 drop-in 及证书续期状态。完整续期周期仍需持续观察。

反向代理后的中台配置如下，另保留现有数据库及 Agent 凭据：

```text
VLA_HOST=127.0.0.1
VLA_PORT=8000
VLA_PUBLIC_URL=https://115.190.212.169
VLA_REQUIRE_HTTPS=true
VLA_SESSION_COOKIE_SECURE=true
VLA_FORWARDED_ALLOW_IPS=127.0.0.1,::1
VLA_AGENT_CONTROL_PLANE_URL=http://127.0.0.1:8000
VLA_TUNNEL_TARGET_HOST=127.0.0.1
VLA_TUNNEL_TARGET_PORT=8000
```

`VLA_PUBLIC_URL` 只接受 HTTPS 来源（协议、主机及可选端口），不能包含用户名、路径、查询串或片段。管理页面及 API 校验 Host；明文页面导航跳转至配置的 HTTPS 入口，明文管理 API 在处理凭据和正文前拒绝请求。只有明确配置的代理来源可以影响协议和客户端地址识别；不可将可信代理设置成 `*` 或所有地址网段。代理与中台不在同一主机时，应据实际地址调整可信代理，并保持上游网络受限。

需要同时保留本机和局域网 HTTPS 入口时，可设置 `VLA_PUBLIC_URL_ALIASES=https://192.168.1.20:8443,https://localhost:8443`。别名必须逐个列出完整 HTTPS 来源，不能包含空项或通配符，并且必须保留主 `VLA_PUBLIC_URL`；每个主机与端口组合单独匹配。代理还需显式监听、路由这些地址，证书 SAN 也必须覆盖对应 IP 或域名。别名不会开放跨来源请求：登录和写操作的 Origin 仍须与当前访问入口一致，Cookie 仍限定在当前主机；各入口需要分别登录。明文页面导航仍跳转至主入口。

`/healthz` 与 `/agent/v1/` 保留内部 HTTP，Agent 协议仍使用独立节点凭据。现有节点通过 SSH 加密隧道回连，不需要重装或修改身份。中台由原 80 端口迁到 8000 时，将隧道的**中台端转发目标**改为 8000；远端机器上原有的回环监听端口不变。此前三个节点使用的 35071、36569、44003 不能被统一改成 8000。原先的 `VLA_AGENT_CONTROL_PLANE_URL` 若指定 `http://127.0.0.1:80`，同步调整为新的内部回环入口，避免新配置沿用旧默认值。

所有 Agent 均经内部 SSH 隧道访问时，生成配置加 `--internal-agents-only`：公网 HTTP 和 HTTPS 入口对 `/agent`、`/agent/*`、`/docs`、`/redoc` 及 `/openapi.json` 返回 404，内部回环接口与节点身份不变。该选项默认关闭，以兼容直接连接 HTTPS 的 Agent；启用前确认没有节点依赖公网代理访问 Agent 协议。

反向代理方案下不要再设置 `VLA_TLS_CERT_FILE`、`VLA_TLS_KEY_FILE`；中台与本机代理之间使用回环 HTTP。如选择由中台直接提供 TLS，则两项需同时设置为可读、匹配的 PEM 文件绝对路径，`VLA_PORT` 改为 TLS 监听端口，并保留 `VLA_REQUIRE_HTTPS=true`、`VLA_SESSION_COOKIE_SECURE=true`。原生 TLS 不负责申请或续期证书，也不会额外创建 HTTP 跳转监听器。现有 SSH HTTP 隧道依赖明文回环上游，生产迁移优先采用反向代理方案。

## 安装与验收边界

生产仍停止时可以完成配置准备：备份目标端**最新**数据库、私有环境文件和 `remote-management`；更新应用及首次管理员账号；合并环境补丁；安装 Caddyfile、服务文件和所需证书；检查端口及权限。执行 `caddy adapt --config /etc/termitech-vla/https/Caddyfile --adapter caddyfile --validate` 与 `systemd-analyze verify` 校验。仅配置校验不证明证书申请成功或公网链路可用。

由 HTTP 切换为 HTTPS 时，先停止中台，再使用新版本命令撤销目标数据库中的旧用户会话，随后才启动 HTTPS 服务。旧 HTTP Cookie 可能未带 Secure；仅修改配置不能使已发出的 Cookie 自动获得该属性。

```sh
.venv/bin/python -m vla_control.users revoke-sessions --all \
  --database-url sqlite:////data4/termitech-vla-platform/.local/control.db
```

此操作保留账号、密码及 Agent 凭据，所有网页登录需要重新登录。使用正在升级的目标数据库，不要对旧副本执行后误认为生产会话已撤销。

实际启动、开端口及申请公网证书属于后续重新上线操作；本次实现不执行。上线验收时使用普通证书校验访问 `/healthz` 和登录页，确认 HTTP 跳转至唯一 HTTPS 入口、会话 Cookie 带 Secure、未登录和越权 API 被拒绝、真实机器重新收到心跳、元数据和训练进程保持一致。不要通过关闭 TLS 校验来判断验收通过。

公网中台已接收新事件后，如需回滚，使用目标端最新业务状态，不可恢复迁移前旧数据库。完整数据边界和回滚步骤见[迁移说明](control-plane-migration.md)，账号配置见[用户与权限](users-and-permissions.md)。HTTPS 提供传输保护，不替代后端权限、系统维护或访问范围限制。

## 2026-10-22 本地验收

本机预览已运行于 `https://127.0.0.1:8443`，原 `http://127.0.0.1:8000` 对正常页面导航跳转至该入口；管理 API 明文请求被拒绝。内部 API 和 CPU 演示 Agent 使用回环 8001 端口，唯一管理员账号及测试资产保留，原业务数据和远端连接未切换。

全量 Python 回归 1562 项通过，3 项可选 Caddy 配置测试在该轮跳过，随后显式指定已校验的 Caddy 2.11.7，补跑这 3 项及同组部署测试均通过；前端 36 项通过。真实 TLS 集成测试验证 IP 证书链、原生 TLS、HTTPS 代理、转发头防伪造、Secure Cookie、CSRF 与退出。独立 Chrome 使用仅限本地证书公钥的进程级固定配置，验证了 TLS 1.3 页面登录、刷新、退出和演示节点显示；未改变系统信任库。这些结果不代表公网证书已签发或 Compose 已实际运行。

有 Caddy 的环境可运行真实代理测试，不会自动下载二进制：

```sh
VLA_TEST_CADDY_BINARY=/path/to/caddy .venv/bin/python -m pytest \
  tests/deployment/test_https_deployment.py tests/integration/test_https.py
```
