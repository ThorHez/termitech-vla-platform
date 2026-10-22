# server-169 Agent 部署与联调记录

验收日期：2026-10-15（Asia/Shanghai）。

## 部署结果

| 项目 | 当前配置 |
| --- | --- |
| 服务器 | `115.190.212.169`，主机名 `T001host` |
| 中台节点名称 | `server-169 · A100 裸金属` |
| 节点 ID | `d9ced156-ff4d-42ef-bdb1-49f64b5ad9d0` |
| GPU | 8 × NVIDIA A100-SXM4-80GB，NVML 上报可用 |
| Agent 根目录 | `/data2/termitech-vla-platform` |
| Agent 环境 | 独立 Python 3.11 venv：`/data2/termitech-vla-platform/.venv` |
| 服务账户 | `vla-agent`，无交互登录 shell |
| systemd 单元 | `termitech-vla-agent.service`，已启用开机自启 |
| 服务状态 | 验收结束时 `active/running`，MainPID `2274332` |
| Agent 配置 | `/data2/termitech-vla-platform/config/agent.json`，权限 `0640` |
| 持久状态 | `/data2/termitech-vla-platform/state` |
| 任务输出 | `/data2/termitech-vla-platform/runs/<attempt_id>` |
| 测试代码 | `/data2/termitech-vla-platform/smoke-code`，独立 Git 仓库 |
| 部署目录占用 | 验收结束时约 93 MiB |
| 数据盘可用空间 | 本次心跳约 41.85 GiB；以后以当前机器页面为准 |

Agent 是宿主机上的原生 Python 服务，训练由独立 Runner 执行。服务使用 `KillMode=process`，本轮已验证重启 Agent 不结束正在运行的训练进程。

安装包版本为 `termitech-vla-platform 0.1.0`。wheel SHA-256：

`bf107ae0690c27ecf2e22127003aeac7bbf28cdb84ec33c3ed7b343a117e5e1a`

安装时使用从 `uv.lock` 导出的带哈希依赖列表；安装后 `pip check` 通过。包、依赖列表和校验清单位于服务器的 `install/` 目录。原有 openpi-umi 虚拟环境未被用于安装平台依赖。

## 本次中台连接

本次按用户选择使用本机开发中台，而非远端 Docker Compose 中台：

- 本机 API、网页和内嵌 Scheduler：`http://127.0.0.1:8000`。
- 本机数据库：工作区 `.local/control.db`（SQLite）。
- SSH 反向转发：服务器 `127.0.0.1:18000` → 本机 `127.0.0.1:8000`。
- Agent 的 `control_plane_url`：`http://127.0.0.1:18000`。
- 中台入口：[机器详情](http://127.0.0.1:8000/#/nodes/d9ced156-ff4d-42ef-bdb1-49f64b5ad9d0)。

中台和隧道由本机 `scripts/remote_agent_dev.py` 管理；该脚本保持运行时远端可连接。本机休眠、隧道退出或中台停止后，远端 Agent 仍常驻，但暂时无法上报。本轮未开放公网管理端口，未修改既有 SSH 配置或服务器防火墙。

重新连接：

`uv run python scripts/remote_agent_dev.py`

管理员令牌从本机 `.local/dev-credentials.json` 的 `admin_token` 字段读取。Agent 登记成功后，远端配置中的 enrollment token 已移除；后续通过持久状态中的节点凭据重新连接。不要删除 `state/` 来重启 Agent。

## 真实机器验收

验收通过 6 个检查阶段，共创建 5 个隔离测试任务：

| 用例 | 证据与结果 | Run ID |
| --- | --- | --- |
| GPU 上报 | 8 个唯一 UUID、型号、显存、利用率和进程列表；NVML 可用 | 无任务 |
| 成功与幂等 | 30 步完成，重复提交返回同一任务；30 个指标事件、真实日志、代码快照和本机产物 | `81b811e3-5db7-4944-ae1e-80169874da59` |
| 取消 | 最终 CANCELLED，按 PID + 创建时间确认原训练进程退出 | `ec2ce46c-e7f0-4e2a-8d8e-565d88a32f10` |
| Agent 重启 | Agent PID 改变；训练 PID `2274263` 及创建时间不变；同一 Attempt 完成 300 步，只有一个 STARTED 事件 | `fae7766d-2a45-4459-a87e-aa0934509b68` |
| 代码漂移 | 预检后修改独立测试代码，启动前以 `CODE_CHANGED_SINCE_PREFLIGHT` 拒绝；没有 STARTED；恢复原字节后预检通过 | `01404190-87f6-46cf-95c1-f7db33d85c0a` |
| 双 GPU 预约 | 同一 Attempt 原子预约两张真实 GPU，本机复检与锁通过；任务结束后预约释放 | `df8b3bd2-d318-4df4-b91c-3f2c2b8d7a88` |

双 GPU 用例仍执行 CPU 合成数据负载，仅验证资源预约和释放；本轮没有进行 GPU 数值计算或 VLA 模型训练，也没有验证 GPU 训练进程的归属识别。

测试使用的实际代码 commit：

`0cd01ada96ed4b1016dfabe3fb5b3570de916d76`

这个 commit 属于隔离的 smoke 仓库，不是 openpi-umi。产物均带 `simulation: true`，不是 VLA 权重。页面中的代码漂移任务 FAILED 属于预期验收结果。

详细原始报告保存在本工作区 `.local/remote-deploy/acceptance.json`；本次验收脚本为 `.local/remote-deploy/verify_agent.py`，两者不纳入源码发布。验收结束时：

- Agent 在线，活动测试 Attempt 为 0，待上传事件为 0。
- GPU 预约已释放，`nvidia-smi` 未列出计算进程。
- smoke 仓库恢复原始字节，Git 状态干净。
- 原训练库 `/data2/hzl_workspace_for_pi/openpi-umi` 仍位于 `6c798798c9d2ae39db005d351aa570a9fb46341e`，保留部署前的六个已修改文件；本次未改写该仓库、数据或权重。

## 运维与后续接入

在服务器上查看状态和日志：

`systemctl status termitech-vla-agent.service`

`journalctl -u termitech-vla-agent.service -f`

只重启 Agent：

`systemctl restart termitech-vla-agent.service`

停止 Agent 不会主动结束独立 Runner。需要结束训练时，先在中台取消对应任务并确认终态。重启期间 journal 中出现既有训练进程仍存活的记录符合当前 `KillMode=process` 配置；本轮已确认这些进程随后正常完成。

后续已通过网页生成配置与下载安装包更新本节点，保留节点 ID。现在登记了 `remote-smoke` 和 `openpi-umi-native` 两个环境；后者解释器为 `/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python`，两个环境实际探测均通过。真实 SFT 配方仍保持未验收，需要选择数据、模型和归一化统计后完成小步数训练验收；现有 OpenPI 代码的未提交修改保留。详细说明见 [界面接入记录](../node-onboarding.md)。

本页前面的服务 PID、wheel 校验值及 Run ID 是首次部署时的证据。更新后服务 PID 和安装包校验值已变化，当前以服务状态与 `install/onboarding-20261015/manifest.json` 为准。更新后的六项真实机器验收再次通过；最新报告为本工作区 `.local/remote-deploy/acceptance.json`，初始报告保存在 `.local/remote-deploy/acceptance-initial-20261015.json`。

允许登记资产的受管根目录为：

- `/data2/hzl_workspace_for_pi`
- `/data2/hzl_workspace_for_pi_mem`
- `/data2/RLinf`
- `/data2/termitech-vla-platform/smoke-code`
- `/data2/termitech-vla-platform/runs`

资产路径还需对服务账户可读；代码、环境或数据目录的现有权限未被批量更改。长期使用时，将中台部署到稳定在线的服务器并通过 HTTPS 接入，迁移时保留数据库、节点身份和 Agent 状态。


## 中台 SSH 自动安装与重启恢复验收（2026-10-15）

本轮以管理员 API 提交 `115.190.212.169:22` 的 root 密码认证，指定 `/data2/termitech-vla-platform`。中台实际完成 SSH 与权限探测、读取已有配置、上传当前安装包并核对 SHA-256、安装依赖、更新并启动 systemd Agent。密码未写入命令行或明文配置，部署 API 不返回密码或登记令牌。

- 部署 ID：`ce9c98fb-3c61-441a-ac9b-7f83d2c6fcd5`，最终 `SUCCEEDED / ONLINE`。
- 原节点 ID `d9ced156-ff4d-42ef-bdb1-49f64b5ad9d0` 保留，8 张 A100 遥测正常。
- 原 `remote-smoke` 与 `openpi-umi-native` 均携带本次 onboarding 标识并返回 READY；SFT recipe 仍未执行真实训练验收。
- 安装包本次实际上传 SHA-256：`510e2246d0205802bad876917575b306aea1c82c0535be8a1e578eaf9e5f762d`。后续控制面小修已重新构建供未来部署的包，本记录对应实际上传版本。
- 2026-10-15 18:51 观察服务 `active/running`，账户 `vla-agent`，当时 MainPID `2373842`。

本轮不再由启动脚本创建外部 SSH 命令。中台自动分配远端 loopback 端口，本次为 `37745`，只作为观察证据，不是固定配置。随后实际结束本机中台并重新启动，凭据自动解密，原端口隧道恢复 HEALTHY，18:51:11 UTC+8 收到比重启前更新的 Agent 心跳。无需人工建立隧道或重新输入 SSH 密码。

中台连接密钥和主机记录目录实际权限为 0700，两个文件为 0600。需随数据库保存 `.local/remote-management`，生产 Compose 使用专属持久卷。详细脱敏报告在当前工作区 `.local/ssh-onboarding-acceptance.json`，包含 SSH 检查、部署状态历史和重启恢复结果。

OpenPI checkout 仍为 `6c798798c9d2ae39db005d351aa570a9fb46341e`；检查时有六个已跟踪文件修改及三个未跟踪文件，本轮仅只读查看，未清理或覆盖。安装耗时主要来自目标机器首次下载新增依赖，本轮未运行新的 VLA/GPU 计算训练。


## 卸载以重新测试首次创建（2026-10-15 19:29 UTC+8）

按用户明确要求，使用新增管理员卸载 API 实际执行了 Agent 卸载，任务 ID `eeeb529f-8958-47bc-92dc-394b0e67ace1`，最终 `UNINSTALLED`。执行前中台无活动训练/部署，远端再次检查活动/UNKNOWN Attempt 及待上报事件后才停止服务。

- 标准 `termitech-vla-agent.service` 已 stop/disable 并移除，`LoadState=not-found`。
- 原 `.venv/config/home/tmp/state` 及 unit 归档至 `/data2/termitech-vla-platform.uninstalled.20261015T112915Z.5107746c9012`，目录 0700；原配置与身份状态不再存在。
- 原 `runs`、`smoke-code`、UMI 训练代码与 `/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python` 保留。
- 原节点标记 REMOVED，历史训练和资产记录保留；关联 SSH 凭据清除，旧通道和自动重连停用。
- 使用同一只读系统探测脚本验证原路径和 `/data2/termitech-vla-agent_test`，均不再发现标准 Agent 服务或旧配置，安装条件通过。

本次为用户准备首次创建测试，当时未再次安装 Agent。脱敏验收记录在 `.local/uninstall-169-acceptance.json`。本节描述 19:29 的卸载状态；后续重新接入结果见下节。

## 清华镜像安装与重新接入（2026-10-15 20:30 UTC+8）

用户在 `/data2/termitech-vla-agent` 重新创建 Agent，官方 PyPI 下载较慢，首轮作业在 900 秒超时；增加细化日志后的下一轮也长时间停留依赖下载。按用户要求，将 Agent 安装器默认索引切至清华官方推荐的 `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`，保留 HTTPS、锁定版本、`--require-hashes` 和二进制包限制。仅修改 Agent 安装子进程环境，不写全局或训练环境 pip 配置；移除继承的额外索引、TLS 豁免和安装重定向等 PIP_* 设置。

在 169 实测下载锁定的 idna wheel 至内存，约 0.52 秒完成，SHA-256 与 uv.lock 一致。随后严格核对当前安装器父子进程及目录，仅停止仍使用旧源的 Agent pip 子进程 `2442489`，由原安装器记录失败并退出，再通过中台重试同一目录。

- 新部署：`01f75086-4a9a-414f-a507-7762247c5d97`，20:29:35 开始、20:30:02 完成，约 26.9 秒，最终 `SUCCEEDED / ONLINE`。
- 依赖下载及安装耗时 12.5 秒，安装日志明确显示清华镜像 URL；manifest 与依赖哈希校验通过。
- 实际上传安装包 SHA-256：`ac3058898a15f2148e878b982193899393df3543032adabee89d50b790462c16`。
- 新节点：`8879a636-0d8f-4d25-8556-b6fab2430758`；`T001host` ONLINE，8 张 GPU 上报，`openpi-native-1` 的训练 Python 检查 READY。
- 实测 systemd：loaded / active / running，enabled，观察时 MainPID `2450528`。
- 本轮没有活动训练任务，未执行新的 VLA/GPU 计算训练。旧节点和历史关系保留。

安装器专项 27 项通过；中台实际下载的新安装包与本地 SHA-256 一致。证据：`.local/tsinghua-agent-deployment-acceptance.json`、`.local/tsinghua-mirror-169-check.json`；网页截图 `.local/ui-tsinghua-agent-online.jpg`。

## 路径联想 Agent 更新（2026-10-15 20:48 UTC+8）

用户随后再次测试卸载及创建，当前在线身份为 `0fcb9ef5-8fc7-4feb-9603-44fc05dae5cc`，20:33 创建任务 `4a7ab734-6d00-40ba-98d8-b90c5e1acba3` 已成功。此次先检查没有活动训练或部署，再使用该节点原配置和路径更新 Agent，增加 PATH_SUGGEST；未恢复已被用户删除的旧身份。

- 更新部署 `32118143-9ad8-4c85-9d9a-e0811605af89`，20:47:44 至 20:48:00，SUCCEEDED / ONLINE；节点 ID、Agent state、训练环境与受管目录保留。
- 实际上传包 SHA-256：`61123ff63c6be81fe11ff025b9932ac344d633f98df62d703a62af87154a4911`。
- 从中台及浏览器查询 `/data2/hz`，真实返回 `hzl_workspace_for_pi/` 与 `hzl_workspace_for_pi_mem/`；点击目录下钻、键盘选择及离线机器切换均通过。
- 仅列文件和目录名称、类型；不读取文件内容，不登记新资产，不启动训练。路径真实落点、扫描数量/时长、结果数量及请求超时均受限制。

完整 Python 回归 392 项及前端构建/格式检查通过。证据 `.local/path-suggestion-169-acceptance.json`，截图 `.local/ui-path-suggestions-169.jpg`。


## 配方联想 Agent 更新（2026-10-15 21:08 UTC+8）

在确认没有活动训练或部署后，沿用现有 SSH 管理连接，原地更新当前节点 `0fcb9ef5-8fc7-4feb-9603-44fc05dae5cc`。本机中台同步加载配方查询接口，保留节点身份、Agent 状态目录、训练 Python 和代码目录。

- 部署 `aae9ade3-aab7-4c4b-9d38-b608431317c4`，21:08:24 至 21:08:39，SUCCEEDED / ONLINE；systemd 服务实际返回 active，环境检查 READY。
- 实际上传包 SHA-256：`ed7c74b928559c208c3e569a8132173d5ca8019aab3de8e5ea51ee77644e2e64`。
- 真实代码 `/data2/hzl_workspace_for_pi/openpi-umi/src/openpi/training/config.py`：前缀 `pi05_umi` 返回 43 个名称；`pi05_umi_32d_80k_95_real_umi_batch_72_v4` 返回 16 个；不存在的前缀返回空列表。
- 浏览器已验证前缀筛选、上下键加 Enter 填入且不提交表单，以及点击 `pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta` 填入完整名称。页面显示代码路径与待验收状态。
- 当前环境没有已验收配方名单，因此发现项均为待验收；没有修改验收凭据或启动训练。动态 RoboArena 生成器不能静态列出，部分名称不符合平台既有命名规则，页面分别显示限制提示。

最终相关 Agent 与独立 HTTP/Agent/Runner 测试 48 项通过；此前完整回归 496 项通过，前端构建与格式检查通过。证据 `.local/recipe-suggestion-169-acceptance.json`，截图 `.local/ui-recipe-suggestions-169.jpg`。


## 配方详情与自定义库更新（2026-10-16 10:35 UTC+8）

确认没有活动训练或部署后，原地更新当前节点 `0fcb9ef5-8fc7-4feb-9603-44fc05dae5cc`。本机中台新增配方表并保留现有记录、SSH 凭据和隧道，Agent 的身份、环境及路径保持不变。

- 部署 `da5c8453-e646-4b36-8f18-d37eeef05944`，10:35:31 至 10:35:46，SUCCEEDED / ONLINE；远端 systemd 实际为 active。
- 上传包 SHA-256：`6048b029b06b539d218188dd3f246e96a70cb363b67e91f1dfdae45013c9b69b`。
- 真实 `pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta` 返回 74 个详情字段，源码行 4430–4484；静态步数 60000、batch 72、峰值学习率 0.00003。
- 在真实页面另存“Elec Meta 小批量（演示）”v1：batch=36、峰值学习率=0.00002、num_workers=4；随后 v2 将 num_workers 改为2。新修订均保持待验收。
- API 核实 v1 定义未变；刷新页面仍能选中 v2。下一步真实显示配方默认值 steps=60000、batch=36、learning_rate=0.00002；没有提交预检或启动训练。
- 原源码 SHA-256 前后均为 `be8f124d6f6a0d168e9cf7581d9c7ac089af09524ee45265605977b18c74d0f5`，没有回写 config.py。该示例为多数据集配方，页面明确显示当前首版可查看另存、训练仍需多数据集资产绑定适配。

完整回归 630 项、前端构建和格式检查通过。证据 `.local/recipe-library-169-acceptance.json`，截图 `.local/ui-recipe-library-save.jpg` 和 `.local/ui-recipe-library-saved-v2.jpg`。不代表真实 VLA/GPU 训练验收。


## 配方关键词包含搜索更新（2026-10-16 10:49 UTC+8）

原地更新现有节点 `0fcb9ef5-8fc7-4feb-9603-44fc05dae5cc`，修复仅前缀匹配造成 `elec` 无结果的问题。现为不区分大小写的包含匹配、开头匹配优先。部署前检查没有活动训练或部署，原节点身份及环境保留。

- 部署 `12d3e881-daf1-4b13-b372-7169f55fe8e9`，10:49:11 至 10:49:27，SUCCEEDED。
- 上传包 SHA-256：`544ed33416bee1860eb734e25dc22971da943c6910a5ce943fe2a16b5ba6cbcd`。
- 实际 API 查询 `elec`、`ELEC`、`meta` 均返回指定 `…v4_elec_meta` 配方，各 1 条；不依赖浏览器缓存全量配方后过滤。
- Agent 相关 74、中台搜索 72、真实 HTTP/Agent/Runner 集成 9 项通过；前端构建及格式检查通过。

证据 `.local/recipe-keyword-169-acceptance.json`，页面截图 `.local/ui-recipe-keyword-elec.jpg`。路径补全仍保持目录前缀规则，配方验收规则保持独立。

前端同时区分搜索词与完整选择：输入 `elec` 只显示联想，不将关键词作为 `config_name` 请求详情；选中结果或输入确切完整名称后才加载详情。实际页面已验证上下键加 Enter 可填入完整配方并读取详情，未启动训练。
