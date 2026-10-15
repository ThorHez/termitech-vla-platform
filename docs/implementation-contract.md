# 首个可运行版本：实现契约

设计文档定义完整目标，本契约固定当前并行实现的接口。首个版本完成单项目管理员模式下的资源、资产、预检、单阶段执行、取消、监控及代码追溯。完整多角色权限、多阶段算法和 checkpoint 恢复待后续验收。

## 代码布局

`apps/control_plane/vla_control`、`apps/agent/vla_agent`、`packages/protocol/vla_protocol`、`packages/adapters/vla_adapters` 为一个 Python distribution 的四个包。前端为 `apps/web` React + TypeScript + Vite。

## HTTP

用户接口统一 `/api/v1`，需要 `Authorization: Bearer <VLA_ADMIN_TOKEN>`。Agent 接口统一 `/agent/v1`，使用各节点独立 token。登记时使用 `VLA_ENROLLMENT_TOKEN`；用户/Agent 凭据不能混用。列表直接返回 JSON 数组。时间为 UTC ISO 8601。错误 `{error: {code, message, details}}`。

- GET `/api/v1/overview`: `{nodes_online,nodes_total,gpus_total,gpus_available,runs_active,runs_failed,assets_total}`。
- GET `/api/v1/nodes`: 节点数组；GET `/nodes/{id}` 返回节点及关联 assets/profiles/runs。
- GET `/api/v1/assets?node_id=&kind=`；POST `/assets`: `{name,kind,node_id,path,version,metadata}`，kind DATASET/MODEL/NORM。初始 REGISTERED，Agent 预检后验证；逻辑归档 POST `/assets/{id}/archive`。
- GET `/api/v1/profiles?node_id=`：由 Agent 配置并登记，管理员不通过浏览器提交任意命令。
- GET `/api/v1/templates`：模板数组，`id,name,algorithm,status,description,parameters,required_inputs`。status AVAILABLE 或 PLANNED；只有 AVAILABLE 可提交。
- POST `/api/v1/run-previews`: `{name,node_id,profile_id,template_id,inputs:{dataset:id,model:id,norm:id},parameters:{steps,batch_size,learning_rate},gpu_count}`。返回 `{id,status:"PENDING"}`；GET `/run-previews/{id}` 返回 `{id,status,checks,code_snapshot,error,request}`。Agent 异步预检，成功 READY，失败 FAILED；有效期 15 分钟。
- POST `/api/v1/runs`: `{preflight_id}` + `Idempotency-Key`；服务器冻结预检请求与代码基线。
- GET `/api/v1/runs`；GET `/runs/{id}`：Run 含 `id,name,node_id,profile_id,template_id,state,created_at,inputs,parameters,gpu_count,code_expectation,attempts,outputs,queue_reason`。
- POST `/api/v1/runs/{id}/cancel`；GET `/runs/{id}/events?after_seq=0`: 事件数组；GET `/runs/{id}/logs?after_seq=0`: `{items:[{seq,text,timestamp}],next_seq}`。
- GET `/api/v1/alerts`：当前异常数组。
- GET `/healthz` 无鉴权。

节点字段：`id,name,hostname,status,last_seen,roots,output_root,gpus,metrics`；profile 字段：`id,node_id,name,code_dir,python,adapter,recipe,execution_mode,docker_image,metadata`。GPU 字段 `uuid,name,memory_total_mb,memory_used_mb,utilization_pct,processes:[{pid,memory_mb}],reserved_by`；未知值 null。Agent 上报不含 reserved_by，由控制面结合预约填充。

## Agent 协议

- POST `/agent/v1/register` enrollment bearer，body `{name,hostname,roots,output_root,profiles:[{id,name,code_dir,python,adapter,recipe,execution_mode,docker_image,metadata}]}`；返回 `{node_id,token}`。首次登记后持久化凭据，重启复用，避免重复登记。
- POST `/agent/v1/heartbeat` node bearer，body `{node_id,gpus,metrics,active_attempts:[{attempt_id,pid,state}]}`，返回 `{ok:true}`。
- GET `/agent/v1/commands?node_id=` 返回待确认命令数组；命令在 APPLIED/REJECTED 前可重复返回，Agent 必须幂等。
- POST `/agent/v1/commands/{id}/ack` body `{node_id,status:"APPLIED"|"REJECTED",result:{...}}`。
- POST `/agent/v1/events/batch` body `{node_id,events:[{attempt_id,seq,type,timestamp,data}]}`；返回 `{accepted:n}`。每个 Attempt seq 单调，唯一约束去重。Agent 只有成功上传后删除本地待发送事件。

命令 `{id,type,payload}`：

- PREFLIGHT payload `{preview_id,profile_id,template_id,inputs:{role:{id,path,kind}},parameters,gpu_count}`。ack result `{code_snapshot,checks:[{name,ok,message}],error?:{code,message}}`；控制面存 code_snapshot，绝不接受浏览器传 SHA。
- START payload `{run_id,attempt_id,execution_token,profile_id,template_id,inputs:{role:{id,path,kind}},parameters,gpu_uuids,code_expectation}`；Runner 负责独立进程、持久化事件和退出确认。ack 只表示 Runner 已接受，不代表训练开始或成功。
- CANCEL payload `{attempt_id,execution_token}`；以 Runner 的终止事件为准释放 GPU。

事件类型：`STARTED` data `{pid,code_snapshot,output_dir}`；`METRIC` `{step,total_steps,loss,...}`；`LOG` `{text}`；`SUCCEEDED` `{exit_code:0,outputs:[{kind,name,path,version,metadata}]}`；`FAILED` `{exit_code,error:{code,message}}`；`CANCELLED` `{exit_code}`。允许 `RESOURCE` 保存进程指标。只有终态事件释放 GPU。Agent 断网时状态 UNKNOWN，预约保留。

## 状态和调度

Run 状态 QUEUED → STARTING → RUNNING → SUCCEEDED/FAILED；取消 CANCELLING → CANCELLED；节点失联时活动 Run 为 UNKNOWN（保存原状态）。单节点锁加 `(node_id,gpu_uuid)` 唯一预约，整组原子申请。无可用 GPU 则排队。显式开发 smoke 模板允许 gpu_count=0；真正 openpi 模板至少一张。Agent 必须复检本机路径、代码快照和 GPU 外部占用。API 进程内调度支持开发，独立 `vla-scheduler` 支持部署，同一个数据库锁保证一致性。

## 共享 Python API

`vla_protocol` 提供 `PROTOCOL_VERSION = 1` 和 `canonical_digest(value)`；无需依赖 ORM。Agent 与 API 可使用普通 dict 避免共享持久化模型。

`vla_adapters` 由根任务实现：`get_templates() -> list[dict]`；`build_launch(profile,template_id,inputs,parameters,output_dir,gpu_uuids) -> dict`，返回 `{argv,cwd,env}`。只接受可信配置中的 recipe；调用本包受控 wrapper，不接受 shell。`smoke` 为显式测试模板，记录为测试成果；真实 SFT 入口为 `openpi-sft`，需经过管理员配置 `adapter:"openpi-sft"` 的 profile，调用已安装的 openpi 与训练脚本。MEM/ReCap/RLT 返回 PLANNED，不可误报已验收。

代码快照由 Agent 实现：repo_url,code_dir,branch,commit_sha,commit_message,committed_at,dirty,changes_digest,untracked_digest,module_origin,captured_at,complete。无有效 Git 仓库拒绝真实训练；smoke fixture 也建独立 Git 仓库保证验证链真实。真实 openpi 的 module_origin 在 profile.python 下探测；启动前比较快照身份与变化摘要。

## 当前交付诚实边界

本地 smoke 是控制链路测试，不是 VLA 算法或 GPU 训练验收。前端默认真实空状态，通过显式 demo 开关才可看演示数据，必须标示演示。生产 HTTPS、真实机器环境、模板数据协议和恢复验收按主设计继续执行。
