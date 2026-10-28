# ARX5 与 UMI 实机源码

平台的 `third_party` 现在包含三个相互独立的 Git 子模块：

| 路径 | 用途 | 固定公开基线 |
| --- | --- | --- |
| `third_party/openpi-umi` | 模型训练与策略推理 | 沿用已有 gitlink，本次不修改 |
| `third_party/arx5-sdk` | ARX5 硬件接口、控制器和 ZMQ 服务 | `a8890c9bae94464abd1cb7c5e4da7c4a62104a3a` |
| `third_party/umi-arx-deploy` | 相机、观测处理与实机部署、eval 客户端 | `31c9323298cdad3595fc1d1bb92c4be3e757e4b2` |

主要入口为：

- `third_party/arx5-sdk/python/communication/zmq_server.py`
- `third_party/umi-arx-deploy/scripts/eval_arx5_pi_hzl.py`
- `third_party/umi-arx-deploy/scripts/eval_arx5_pi_hzl_show.py`
- `third_party/umi-arx-deploy/scripts/eval_arx5_pi_hzl_v2.py`

UMI 部署工程的公开上游为 `https://github.com/ThorHez/umi-arx.git`；子模块使用本地名称 `umi-arx-deploy`，对应用户指定的部署工作区。延迟与 CAN 诊断源码同时保留，包括 `inspect_robot_can_feedback.py`、`measure_robot_obs_latency.py` 和 `tools/robot_latency/zmq_server.py`；具体路径以锁定清单为准，不能仅凭文件名认定应作为正式启动入口。

本次只集成源码与版本证据，不启动、停止或重启 SDK、CAN、相机、机械臂或 eval；也不自动安装两个工程的依赖。网页中的 SDK/eval 启停、参数编辑和日志管理需另行接入。

## 首次检出与恢复实机修改

在平台仓库根目录执行：

```sh
git submodule update --init third_party/arx5-sdk third_party/umi-arx-deploy
python scripts/robot_sources.py apply
python scripts/robot_sources.py check
```

两个新子模块固定在各自的公开基线。`apply` 只恢复本仓库保存的实机源码补丁，不获取网络数据、不切换分支、不执行 `reset`，也不触碰已有的 `openpi-umi`。Git 子模块初始化单独执行，因此检查命令不会隐式克隆或改写其他检出目录。

也可分别处理一个工程：

```sh
python scripts/robot_sources.py apply --project arx5-sdk
python scripts/robot_sources.py check --project umi-arx-deploy
```

`check` 只读取文件与 Git 状态。输出 `ready` 表示补丁、HEAD 和全部清单文件的内容、大小、可执行标志已核对；输出 `patch_required` 时退出码为 1，说明仍需显式应用补丁。任何版本、补丁摘要或已有源码不匹配也会返回非零状态，并说明原因。

`apply` 会先预检所有选中工程，要求 HEAD 等于固定基线，并验证补丁 SHA-256 和 `git apply --check`。已经完整匹配清单的工作区可重复运行，不会重复打补丁。存在其他已跟踪源码修改且不匹配清单时，命令拒绝覆盖；现有未跟踪文件与补丁新增路径冲突时，Git 的预检也会拒绝操作。命令不会删除工作区文件或执行硬件代码。

## 为什么公开基线之外还保存补丁

来源与逐文件记录保存在 `third_party/robot-sources.lock.json`；补丁保存在 `third_party/patches/`。

ARX SDK 的来源是 `server-lenovo-111:/home/lenovo/kian/arx5-sdk`，当时位于 `main`，来源 HEAD 为 `11a8b73efcc78ee27fc210762b9b4c71e8e91227`，并有额外的工作区修改。该来源 HEAD 是未发布的本地提交，不能只把它写成要求其他人从公开上游下载的 gitlink。本集成固定其可获取的公开父提交 `a8890c9...`，把该本地提交及后续源码变化保存为 `arx5-sdk-deployment.patch`。

UMI 的来源是 `server-lenovo-111:/home/lenovo/mason/code/umi-arx-deploy`，当时位于 `main`，来源 HEAD 为 `9f91e02d19ec73fb94009878ec41b981c2993d52`。该 HEAD 是未发布的本地 merge 提交，本集成固定其可获取的公开父提交 `31c9323...`，用 `umi-arx-deploy-deployment.patch` 保存公开基线到该实机工作区的源码差异，以及选入的未跟踪 eval、CAN/延迟诊断、测试、工具和文档，使部署入口不因只克隆公开上游而丢失。

补丁应用后两个子模块显示本地修改是预期状态：平台 gitlink 固定公开基线，父仓库跟踪的补丁和文件哈希清单固定实机修订。发布或搬迁时应同时保留 `.gitmodules`、两个 gitlink、补丁、清单与恢复脚本；仅记录 dirty 子模块状态无法在另一台机器上复现。

## 文件与运行环境边界

清单只描述本次选定的源码快照。ARX SDK 不叠加个人编辑器设置 `.vscode/settings.json` 的本地修改；UMI 排除运行配置 `scripts/epoch=0100-train_loss=0.012.yaml`、运行日志 `shellgame_grasp_h16_trace.log`、本机安装记录 `tools/robot_latency/installation.json` 和无现有源码引用的历史备份 `tools/robot_latency/zmq_server.before.py`。远端未跟踪的环境、缓存、采集数据、模型、日志和其他运行目录不作为源码导入。

UMI 的固定上游提交自身含有部分历史编译文件及共享库，SDK 自身含有机械臂模型与网格资源。这些保留在原工程版本中，不代表已经为当前系统重新编译、验证二进制兼容性或连接硬件。平台 Python 环境仍不安装 JAX/PyTorch、ARX SDK 或 UMI 的依赖；实机执行应使用分别核验过的工程目录与解释器。

这两个工程不会自动加入控制面发布包或 Agent wheel，也不随本次源码集成安装到已有机器。控制面与 Agent 仍通过原先的显式源码清单打包，Docker 构建上下文排除第三方工程。

ARX SDK 保留其原有 MIT `LICENSE`。当前 UMI 固定版本没有顶层 LICENSE 文件，本集成记录该现状，不为其新增或推断授权条款。

## 更新快照

后续升级时，先确认实际使用的上游、提交和工作区差异，保留当前平台与子模块中的其他工作。更新公开基线或实机补丁后，应重新生成补丁摘要和完整文件清单，并从干净基线检出验证 `apply` 和 `check`。不要通过重置已有实机目录来对齐清单，也不要仅为消除 dirty 标记而丢弃来源修订。

源码哈希与脚本语法检查只证明版本和文件完整性，不证明 SDK 可用、相机正常、模型完成推理或机械臂动作正确。
