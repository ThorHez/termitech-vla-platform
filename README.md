# Termitech VLA Platform

裸金属本地数据与模型的 VLA 训练管理平台。中央控制面保存资源和训练元数据；每台裸金属运行 Agent，使用本机训练环境和本地文件。训练数据保持本地；真机部署支持通过已接入机器的 SSH 连接，使用 rsync 同步完整 checkpoint、归一化参数与可移植推理配置。

## 当前版本

这是首个可运行的开发版本，包含：

- 中文网页：工作台、机器资源、本机资产、模板、训练提交和训练详情。
- FastAPI 管理服务和数据库调度器；SQLite 用于本地开发，PostgreSQL 用于中央部署。
- Agent 主动连接、节点心跳、NVML GPU 指标、CPU/内存/磁盘监控。
- 机器资源占用历史：CPU、内存、产出磁盘和逐卡 GPU 趋势，支持最近 1 小时到 30 天，参见[资源历史说明](docs/resource-history.md)。
- 同机资产检查、异步预检、整组 GPU 预约、幂等提交、独立 Runner、取消、日志和训练指标。
- 训练多指标展示：自动识别已上报的数值字段，可多选查看独立曲线、最近值和采样步数，参见[训练指标说明](docs/training-metrics.md)。
- 每次实际执行前采集 Git 提交、分支、本地修改摘要和真实模块路径；与预检版本比较。
- 本地输入与输出关联。训练产物可以追溯到来源任务及代码快照。
- 训练过程中的检查点自动登记：显示已保存步数、保存时间和可用状态；保留策略清理的历史检查点标记为已清理。
- 本地路径联想：登记资产时按所选机器的实际目录补全路径，支持点击目录深入及键盘选择；仅查询受管目录，不读取文件内容。
- 训练配方联想：新建训练时按所选机器和环境读取 OpenPI 配方注册代码，输入关键词搜索并选择 `config_name`（不区分大小写，名称开头匹配优先）；Agent 在预检时自动验证所选配方与本机资产，无需手工维护配方验收名单。
- 自定义配方：读取配方详情与 Python 定义，编辑常用训练参数及数据、模型中的静态值，查看差异并另存；Agent 将新配方写入其来源配置文件后自动创建仅包含该文件的 Git commit，记录新 SHA，训练冻结对应修订，参见[配方库说明](docs/recipe-library.md)。
- 显式 CPU 链路测试模板，以及由 Agent 自动预检后使用的 OpenPI SFT 适配器。
- SSH 机器接入：填写 IP、SSH 用户名/密码和 Agent 安装目录，由中台测试连接、上传安装包、安装并启动 Agent，确认真实上线与训练环境检查。
- 安装诊断：细化环境、依赖、配置与服务阶段，逐行回传安装输出和静默等待提示，展示耗时并可下载脱敏日志；中台与目标机保留安装日志。
- 机器管理：管理员可重命名机器，删除中台连接，或远程卸载 Agent 并重置身份；保留训练代码、数据、模型、输出和历史记录，活动任务期间禁止删除。

权限支持独立用户、管理员/操作员/只读三种角色、限时会话、账号禁用和操作审计，参见[用户与权限](docs/users-and-permissions.md)。当前用户共享一个工作空间，项目级隔离、checkpoint 恢复和 Docker 训练执行尚未实现。已实现的 Runner 使用登记的本机 Python/uv/venv 环境。完整目标参见 [详细设计](docs/VLA训练平台详细设计V1.0.md)，当前接口参见 [实现契约](docs/implementation-contract.md)。

HTTPS 支持 Caddy 自动申请/续期公网 IP 或域名证书、已有 PEM 证书和本机测试证书；管理入口校验 Host 与可信代理，参见 [HTTPS 配置](docs/https.md)。生成配置不会启动服务；此前暂停的公网中台仍保持停用。

## 仓库结构

```text
apps/
  web/                         React + TypeScript 控制台
  control_plane/vla_control/    API、SQLAlchemy 模型、调度与事件处理
  agent/vla_agent/              节点 Agent、独立 Runner、状态、采集
packages/
  protocol/vla_protocol/        协议版本和规范化摘要，无 ORM 依赖
  adapters/vla_adapters/        轻量适配器、OpenPI 入口 wrapper
templates/                     可审阅的 CPU 测试训练脚本
scripts/                       本地启动与隔离测试环境准备
deploy/                        中央服务 Compose、Agent 配置与 systemd
tests/                         API、Agent、适配器与跨进程集成测试
docs/                          设计和接口约定
third_party/openpi-umi/         独立 Git 子模块，UMI/SFT、ReCap、MEM 与推理源码
third_party/arx5-sdk/           固定基础版本与实机补丁的 ARX5 SDK 子模块
third_party/umi-arx-deploy/     固定基础版本与实机补丁的 UMI 部署、eval 子模块
third_party/patches/            ARX/UMI 实机源码补丁，随平台仓库保存
third_party/robot-sources.lock.json  两个机器人工程的来源、版本与文件校验清单
```

OpenPI 训练源码通过 Git 子模块关联到 `git@github.com:ThorHez/openpi-umi.git`，固定版本由平台仓库中的 gitlink 记录。首次检出后执行：

```bash
git submodule update --init third_party/openpi-umi
```

ARX SDK 与实机 eval 工程也已纳入 `third_party`。它们保留 `server-lenovo-111` 当前使用的源码修订；上游基础提交与实机补丁分别记录，避免遗漏未推送或未提交的代码。首次检出后执行：

```bash
git submodule update --init third_party/arx5-sdk third_party/umi-arx-deploy
python scripts/robot_sources.py apply
python scripts/robot_sources.py check
```

这些命令只恢复和校验源码，不安装 SDK、不启动 CAN、机械臂、相机或 eval。两个工程仍使用各自的运行环境，不加入平台 Python 依赖或 Agent 安装包。来源、补丁维护及入口说明见[机器人第三方源码](docs/robot-sources.md)。

训练源码仍保留独立提交历史，在子模块目录内开发和提交；发布时先推送 OpenPI 提交，再提交平台的子模块版本指针。统一开发分支为 `codex/unified-training`，从 `wbcd` 合入 MEM，保留各训练阶段的独立入口。平台 Python 环境不安装 JAX/PyTorch；训练运行在 profile 指定的解释器中。服务器上的已有代码目录不会随本地子模块更新，部署新的训练版本时应登记独立目录并重新预检。

训练配方按方式维护在 `src/openpi/training/recipes/sft.py`、`recap.py`、`mem.py`、`rlt.py` 和 `uot.py`。`config.py` 保留共享配置类型、数据处理器和 `get_config()` / `cli()` 兼容入口，聚合各文件中已有的命名配方；本次拆分不改变配方参数。RLT 暂无已实现配方，文件仅保留空注册列表，不代表训练模板已启用。配方联想仍从 `config.py` 的注册关系出发，详情定位到实际配方文件；另存回写到同一文件并单独提交，旧单文件训练项目继续兼容。

## 本地启动

需要 Python 3.11+、uv、Git、Node.js 22+ 和 pnpm。

```bash
uv sync --locked
cd apps/web
pnpm install --frozen-lockfile
pnpm build
cd ../..
uv run python scripts/dev.py
```

打开 <http://127.0.0.1:8000>，使用本机演示账号登录，账号信息保存在 `.local/dev-credentials.json`（权限 0600），不在终端打印密码。该脚本创建明确标记的本地 CPU 测试节点、独立 Git 测试代码仓库、占位资产，并同时运行 API 和 Agent。

在网页选择「平台链路测试」，GPU 数设为 0，预检后提交。可以观察真实子进程的状态、指标、日志和 Git 快照；输出带 `simulation: true`，不是 VLA 权重。所有开发状态位于被 Git 忽略的 `.local/`，重复启动复用节点与记录。

前端开发时可额外运行 `cd apps/web && pnpm dev`，Vite 将 API 请求代理到 `127.0.0.1:8000`。

Ctrl+C 停止开发 API 和 Agent。独立 Runner 按设计可以继续运行；退出开发环境前，先在页面取消仍在运行的测试任务。

## 中央部署

```bash
cp deploy/.env.example deploy/.env
# 将 deploy/.env 中的数据库密码、Agent 登记令牌替换为不同的随机凭据。
docker compose --env-file deploy/.env -f deploy/compose.yaml up --build -d
docker compose --env-file deploy/.env -f deploy/compose.yaml exec api \
  python -m vla_control.users create-admin --username admin
```

Compose 包含 PostgreSQL、API/Web 和独立 Scheduler，默认仅将端口发布到宿主机 `127.0.0.1:8000`。团队访问时按 [HTTPS 配置](docs/https.md)设置反向代理及中台公开来源。Agent 的节点令牌和用户会话互相隔离；登记令牌由管理员保管。首次管理员由上述交互命令创建，不开放注册。

初始版本自动创建数据库表，尚不支持跨版本数据库迁移。正式接入团队数据前，应建立数据库备份与迁移流程。PostgreSQL 与 Docker 部署需在实际目标环境验收，本地 SQLite 的通过结果不代替该验证。

## 接入裸金属

在网页“机器资源 → 接入机器”填写远端 IP、SSH 端口、用户名、密码和 Agent 安装目录，测试连接后配置训练 Python 绝对路径、代码目录及适配器；接入机器不需要填写训练配方。点击安装后，中台上传安装包与配置，远程安装并启动 systemd Agent，展示进度及真实上线结果，无需人工下载、上传或执行命令。详见 [机器与训练环境接入说明](docs/node-onboarding.md)。首次运行前由管理员执行 `uv run python scripts/build_agent_bundle.py` 生成安装包，Docker 部署会在镜像构建时生成。

管理员可以从机器卡片或详情页点击“重命名”，修改平台中的显示名称，支持在线、离线或正在训练的机器。保存后机器列表、资产及历史训练页面同步使用新名称，机器 ID 与既有记录关联保留。自定义名称会持久保存，Agent 后续心跳、重启或更新不会覆盖；名称去除首尾空白后须为 1 至 200 个字符。

如果先使用本机中台连接远端 Agent，参见 [SSH 隧道联调说明](docs/remote-agent-dev.md)。该模式保持管理端口在 loopback 上，不需要开放公网管理端口。

远端需要 Linux/systemd、Git、SSH 密码认证，以及 root 或 sudo 安装权限。安装优先发现 Conda，并在 Agent 安装目录下创建专用 Python 3.11 Conda 环境；只有没有 Conda 时，才使用默认 Python 3.11+ 创建 venv。SSH 无需预先激活 Conda，检测会显示实际解释器路径和版本。训练环境、代码、数据和模型需已在目标机器准备好。中台不向 base 或训练环境安装 Agent 依赖。已有 Agent 更新会保留状态目录及节点身份，并拒绝更新仍有活动任务的机器。

Agent 服务及训练进程使用 SSH 登录用户运行，普通用户的 sudo 仅用于安装和服务管理；使用 root 接入时，Agent 使用 root。安装前会使用该账户验证训练 Python 和代码目录的实际访问能力。

Agent 依赖默认从清华 PyPI 镜像下载，源地址会显示在安装日志中；继续校验锁定版本与 SHA-256，仅对 Agent 的独立环境生效。详见[清华 PyPI 使用说明](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。

SSH 密码在中台加密保存，用于部署和隧道重连；应用密钥与主机指纹保存在 `VLA_REMOTE_STATE_DIR`，默认数据库同目录的 `remote-management`。该目录需与数据库一起持久化备份，密钥丢失后需重新输入 SSH 密码。本机联调默认自动维护每台机器独立的反向隧道；配置可达的 HTTPS 中台入口时，Agent 直接连接。

机器上线后，在网页登记本机 DATASET、MODEL 和 NORM 的绝对路径；创建训练时选择配方（代码中的 `config_name`），完成训练预检再提交。每次任务冻结所选配方，并在任务详情及 Agent 代码快照中记录；启动时再次比对配方、代码和环境。手工部署仍可参考 `deploy/agent.example.json` 与 systemd 模板，但不是网页接入流程。

登记 DATASET 时可填写机器人类型、单臂/双臂、EEF/Joint 动作表示、每路视觉输入的名称与真实字段名、任务描述；高级字段包含动作/状态维度、采样率、动作与旋转编码、实际数据字段清单。已有数据集可点击“补录/编辑数据特征”，不会修改本机文件、路径或资产版本，也可暂不填写以保留旧流程。任务描述用于说明资产，不会自动替换训练 prompt。

创建训练时，平台将登记特征与当前训练环境中处理器的静态 Repack 输入契约匹配：完整字段与视角唯一匹配、阶段和模型变换兼容、参数无需补填时，自动应用该处理器并收起高级配置；多候选、动态处理代码或信息不足时展示原因并保留手动选择。平台不按处理器名称猜选，也不把机器人型号当作动作编码证明；相机数量相同但字段名不同不能视为兼容。归一化参数仍逐数据集绑定，混采权重仍由本次训练决定。

登记描述随预检和训练冻结保存，后续修改不会改写旧任务。使用登记描述的训练需要新版 Agent：本机预检核对真实 `meta/info.json` 的相机字段、声明字段、可选动作维度和 FPS，并回传一致描述，旧 Agent 忽略描述时拒绝启动。`task_index` 派生的 `task` 需要实际 `meta/tasks.jsonl` 证据；仅有多段机器人状态字段时不能核验统一 `state_dim`，可先留空。静态匹配和元信息校验不证明机器人硬件、动作单位或训练效果。

Agent 本地状态必须持久保存。状态数据库损坏、Runner 丢失或节点失联时，平台不会直接认定 GPU 已释放；需要对账。服务账号应具备读取受管输入、写入输出和使用 GPU 的权限。当前 native 模式依赖受信 recipe，不提供文件系统级沙箱。

一个 SFT profile 的示例：

```json
{
  "id": "umi-sft",
  "name": "UMI SFT 本机环境",
  "code_dir": "/data/code/openpi-umi",
  "python": "/data/code/openpi-umi/.venv/bin/python",
  "adapter": "openpi-sft",
  "recipe": null,
  "execution_mode": "native"
}
```

接入时 `recipe` 可以省略或为 null。创建训练后，Agent 使用所选训练解释器解析配方，检查数据集、归一化统计、模型检查点、实际入口、输出空间和 GPU 配置，并将结果保存到本次预检。无需将名称加入 `metadata.validated_recipes`，自定义修订也无需手工写入 `metadata.validated_recipe_revisions`；旧字段保留历史兼容信息，不再作为启动条件。

环境探测 READY 只说明解释器与代码环境可用；训练预检 READY 表示本次配方配置和本机资产检查通过，尚未执行梯度更新或保存训练 checkpoint。预检不占用 GPU，也不等于训练效果验收。提交和实际启动仍核对本次自动预检证据、代码与运行环境，发生变化时需要重新预检。

网页“新建训练 → 资产与参数”需要选择训练脚本入口。项目基地址固定为所选环境的 `code_dir`（例如 `/data2/hzl_workspace_for_pi/openpi-umi`），入口填写相对路径，如 `scripts/train_multi_dataset.py`。路径联想由该机器 Agent 提供，只浏览当前项目的目录和 `.py` 文件，可点击目录逐层展开，也可上下键选择、Enter 填入；不支持通过绝对路径、`..` 或逃逸软链接选择项目外文件。入口写入本次预检和执行快照，启动时核对文件 SHA-256，并执行核对过的源码。脚本需要提供 `main(config)` 与 `init_wandb` 接口；多数据集配方不能选择已知单数据集 `scripts/train.py`。旧 API 省略 `entrypoint` 时保留自动选择兼容。

数据必须是本地 LeRobot 数据目录，模型为包含 `params` 的 checkpoint 或 params 目录，NORM 指向包含 `norm_stats.json` 的目录或文件。单数据集配方显式绑定这些输入和输出目录，保留 recipe 的模型结构。多数据集逐行绑定 DATASET、各自 NORM、数据处理器及混采权重；处理器可由登记特征自动匹配、在高级配置中选择，或显式沿用配方数据项。训练按冻结的完整绑定列表构建子数据集与权重，逐项检查本机资产；state padding 等整体配置仍来自所选配方。多 GPU batch 必须整除卡数。

数据资产登记原始状态/动作字段的语义标签，包括独立的 EEF/Joint、绝对/增量及真实形状，无法确认的内容保留未知。新建训练第一页的每个数据集卡片可推荐处理器、选择现有处理器或配置字段组合；状态拼接、动作布局、1–3 路视觉映射和归一化掩码保存在本次训练绑定中，由统一 `ComposableDataConfig` 构建流水线。预检核验源标签、实际元信息、模型/阶段与归一化维度，冻结本次方案及实现源码摘要。历史资产方案可手动导入，旧配方与 MEM/Hybrid 专用处理器继续兼容，参见[数据处理组件组合](docs/composed-processing.md)。

登记或编辑数据集时，在线 Agent 可从实际 LeRobot 元信息读取字段、形状、采样率和任务文本，推断可确定的特征并填入表单默认值。界面显示推断依据，保留手工及已登记配置，切换目录后拒绝旧结果；读取不会直接保存资产，也不会按动作维数猜测 EEF/Joint、单位或坐标系。

当前适配器针对 `scripts/train.py`（单数据集）、`scripts/train_multi_dataset.py`（多数据集）、`get_config()` 和 dataclass 配置接口。Agent 预检会检查所用入口与数据协议；各分支 recipe 的训练效果仍需真实训练验收，SFT 入口存在不代表任意 MEM/ReCap recipe 可直接使用。

## 测试

```bash
uv run pytest
cd apps/web && pnpm build
```

集成测试启动独立管理服务和 Agent，通过真实 HTTP 完成 CPU 训练、取消和代码变更拦截。测试只在临时目录中创建 Git 仓库，不访问真实训练服务器。没有 GPU 的开发机通过 NVML 不可用状态明确展示监控缺失，不将其当成零占用。

验证记录和剩余工作见 [开发状态](docs/development-status.md)。

## 技术依据

OpenPI 适配参考仓库的 [训练入口](https://github.com/ThorHez/openpi-umi/blob/main/scripts/train.py) 和 [配置定义](https://github.com/ThorHez/openpi-umi/blob/main/src/openpi/training/config.py)。实际任务使用的版本始终以 Agent 采集的完整 SHA 为准。

实现使用 [SQLAlchemy 事务](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)、[Paramiko SSH](https://docs.paramiko.org/en/stable/api/client.html)、[NVML](https://docs.nvidia.com/deploy/nvml-api/latest/) 和 [Vite](https://vite.dev/guide/) 官方接口。
