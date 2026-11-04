# 训练共享剩余显存

“新建训练”的资源参数中可选择“GPU 使用模式”。默认“独占使用”仍要求整卡空闲。选择“共享剩余显存”后，必须勾选具体 GPU 并填写每卡显存预算；支持 OpenPI SFT 和 ReCap GPU 训练，不用于 CPU 归一化统计。

例如一张 80 GiB 显卡被外部推理进程使用 24 GiB，仍可选择该卡并设置 50 GiB 训练预算。界面、预检、调度和训练启动前均核对实际剩余显存，要求足以容纳预算并另留 1 GiB。这里使用 NVML 实际 `memory_free_mb`，不把总显存减已用显存的估算当作真实可用容量。

共享允许与已识别的外部计算或图形进程共存；未知进程、遥测缺失、其他用户的预约、平台训练分配及活动推理继续拦截。每张 GPU 仍只分配给一个平台任务，Agent 的文件锁和正式环境资源保护继续生效。现有进程不会被停止或释放。

训练启动时，Agent 在导入 JAX 前强制使用按需增长的 BFC 分配池，按预算设置 `XLA_CLIENT_MEM_FRACTION`，清理继承的旧 fraction、其他分配器和统一内存覆盖。多卡共用一个比例，因此采用 `budget / max(total_memory)` 并向下取整，确保任一显卡的 BFC 池不超过预算；不同容量的卡实际池大小可能更小。这是显存池配置，不是进程全部 CUDA 内存的硬件隔离。通信库和 CUDA 上下文等可能另行分配，外部进程也可能继续增长；训练本身超过预算时仍需减少 batch size 或使用更省显存的训练方式。

共享策略作为 `training_gpu_policy: {"mode":"shared","memory_mb":51200}` 冻结在本次预检、运行计划和启动命令中。修改模式、卡号或预算都需要重新预检确认。确认页和训练详情显示模式及预算，草稿、个人模板和重新训练保留该设置。Agent 须声明 `TRAINING_SHARED_GPU_V1`；旧 Agent 会明确提示升级，不会忽略预算后直接启动。

JAX 分配池语义依据 [JAX 显存说明](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) 及 OpenPI 锁定的 JAX 0.5.3 对应 [XLA 配置](https://github.com/openxla/xla/blob/df971129bd82e381954da0185b534220e21798a4/xla/python/xla_client.py) 和 [BFC 实现](https://github.com/openxla/xla/blob/df971129bd82e381954da0185b534220e21798a4/xla/pjrt/gpu/gpu_helpers.cc)。
