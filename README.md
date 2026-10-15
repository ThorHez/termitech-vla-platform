# Termitech VLA Platform

裸金属本地数据与模型的 VLA 训练管理平台。中央控制面保存资源和训练元数据；每台裸金属运行 Agent，使用本机训练环境和本地文件。数据和权重不跨机器同步。

## 当前版本

这是首个可运行的开发版本，包含：

- 中文网页：工作台、机器资源、本机资产、模板、训练提交和训练详情。
- FastAPI 管理服务和数据库调度器；SQLite 用于本地开发，PostgreSQL 用于中央部署。
- Agent 主动连接、节点心跳、NVML GPU 指标、CPU/内存/磁盘监控。
- 同机资产检查、异步预检、整组 GPU 预约、幂等提交、独立 Runner、取消、日志和训练指标。
- 每次实际执行前采集 Git 提交、分支、本地修改摘要和真实模块路径；与预检版本比较。
- 本地输入与输出关联。训练产物可以追溯到来源任务及代码快照。
- 显式 CPU 链路测试模板，以及管理员配置并验证后使用的 OpenPI SFT 适配器。

MEM 两阶段、ReCap 派生链路和 RLT 仍标记为待接入。当前权限为单项目管理员令牌模式，多用户/项目授权、checkpoint 恢复和 Docker 训练执行尚未实现。已实现的 Runner 使用登记的本机 Python/uv/venv 环境。完整目标参见 [详细设计](docs/VLA训练平台详细设计V1.0.md)，当前接口参见 [实现契约](docs/implementation-contract.md)。

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
```

openpi-umi 继续维护在独立代码库中，平台不拉取、切换或改写它。平台 Python 环境不安装 JAX/PyTorch；训练运行在 profile 指定的解释器中。

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

打开 <http://127.0.0.1:8000>，使用终端打印的管理员令牌登录。令牌也保存在 `.local/dev-credentials.json`（权限 0600）。该脚本创建明确标记的本地 CPU 测试节点、独立 Git 测试代码仓库、占位资产，并同时运行 API 和 Agent。

在网页选择「平台链路测试」，GPU 数设为 0，预检后提交。可以观察真实子进程的状态、指标、日志和 Git 快照；输出带 `simulation: true`，不是 VLA 权重。所有开发状态位于被 Git 忽略的 `.local/`，重复启动复用节点与记录。

前端开发时可额外运行 `cd apps/web && pnpm dev`，Vite 将 API 请求代理到 `127.0.0.1:8000`。

Ctrl+C 停止开发 API 和 Agent。独立 Runner 按设计可以继续运行；退出开发环境前，先在页面取消仍在运行的测试任务。

## 中央部署

```bash
cp deploy/.env.example deploy/.env
# 将 deploy/.env 中三个值替换为各不相同的随机凭据。
docker compose --env-file deploy/.env -f deploy/compose.yaml up --build -d
```

Compose 包含 PostgreSQL、API/Web 和独立 Scheduler，默认仅将端口发布到宿主机 `127.0.0.1:8000`。团队访问时由现有 HTTPS 反向代理提供入口。Agent 的节点令牌和管理员令牌互相隔离；登记令牌由管理员保管。

初始版本自动创建数据库表，尚不支持跨版本数据库迁移。正式接入团队数据前，应建立数据库备份与迁移流程。PostgreSQL 与 Docker 部署需在实际目标环境验收，本地 SQLite 的通过结果不代替该验证。

## 接入裸金属

1. 在裸金属安装本平台 Python 包，独立于训练虚拟环境。
2. 复制 `deploy/agent.example.json`，设置中央 URL、登记凭据、受管根目录、输出目录和 Agent 状态目录。
3. 在 `profiles` 中登记实际代码目录、Python 解释器、recipe 和 adapter。
4. 使用 `vla-agent --config /etc/vla/agent.json` 启动；稳定运行后使用附带的 systemd 模板。
5. 在网页登记本机 DATASET、MODEL 和 NORM 的绝对路径，完成预检再提交。

Agent 本地状态必须持久保存。状态数据库损坏、Runner 丢失或节点失联时，平台不会直接认定 GPU 已释放；需要对账。服务账号应具备读取受管输入、写入输出和使用 GPU 的权限。当前 native 模式依赖受信 recipe，不提供文件系统级沙箱。

一个 SFT profile 的示例：

```json
{
  "id": "umi-sft",
  "name": "UMI SFT 已验收环境",
  "code_dir": "/data/code/openpi-umi",
  "python": "/data/code/openpi-umi/.venv/bin/python",
  "adapter": "openpi-sft",
  "recipe": "your_verified_recipe",
  "execution_mode": "native",
  "metadata": {"validated": true}
}
```

只有在对应 recipe 和资产协议已完成本机小数据训练验收后，管理员才应设置 `validated: true`。数据必须是本地 LeRobot 数据目录，模型为包含 `params` 的 checkpoint 或 params 目录，NORM 指向包含 `norm_stats.json` 的目录或文件。wrapper 显式覆盖这三个输入和输出目录，保留 recipe 的模型结构与数据变换。多 GPU batch 必须整除卡数。

当前适配器针对 `scripts/train.py`、`get_config()` 和 dataclass 配置接口。各分支 recipe 的动作维度、history、相机和归一化协议仍需对应算法工程验收；SFT 入口存在不代表任意 MEM/ReCap recipe 可直接使用。

## 测试

```bash
uv run pytest
cd apps/web && pnpm build
```

集成测试启动独立管理服务和 Agent，通过真实 HTTP 完成 CPU 训练、取消和代码变更拦截。测试只在临时目录中创建 Git 仓库，不访问真实训练服务器。没有 GPU 的开发机通过 NVML 不可用状态明确展示监控缺失，不将其当成零占用。

验证记录和剩余工作见 [开发状态](docs/development-status.md)。

## 技术依据

OpenPI 适配参考仓库的 [训练入口](https://github.com/ThorHez/openpi-umi/blob/main/scripts/train.py) 和 [配置定义](https://github.com/ThorHez/openpi-umi/blob/main/src/openpi/training/config.py)。实际任务使用的版本始终以 Agent 采集的完整 SHA 为准。

实现使用 [SQLAlchemy 事务](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)、[NVML](https://docs.nvidia.com/deploy/nvml-api/latest/) 和 [Vite](https://vite.dev/guide/) 官方接口。
