# 首个可运行版本：实现契约

设计文档定义完整目标，本契约固定当前实现的接口。当前版本完成共享工作空间中按角色授权的资源、资产、预检、单阶段执行、取消、监控及代码追溯。ReCap 两个训练阶段分别创建独立作业；项目隔离、跨阶段自动编排和 checkpoint 恢复待后续验收。

## 代码布局

`apps/control_plane/vla_control`、`apps/agent/vla_agent`、`packages/protocol/vla_protocol`、`packages/adapters/vla_adapters` 为一个 Python distribution 的四个包。前端为 `apps/web` React + TypeScript + Vite。

## HTTP

用户接口统一 `/api/v1`，使用 HttpOnly Cookie 会话，写请求需要 `X-CSRF-Token`；后端按管理员、操作员、只读角色逐接口授权，未配置权限的接口默认拒绝。Agent 接口统一 `/agent/v1`，使用各节点独立 token。登记时使用 `VLA_ENROLLMENT_TOKEN`；用户/Agent 凭据不能混用。列表直接返回 JSON 数组。时间为 UTC ISO 8601。错误 `{error: {code, message, details}}`。首次管理员通过服务器交互命令创建，旧共享管理员令牌默认停用，详见[用户与权限](users-and-permissions.md)。

- POST `/auth/login`：`{username,password}`，返回 `{user,permissions,csrf_token,expires_at}` 并设置 Cookie；GET `/auth/me` 恢复当前会话；POST `/auth/logout` 撤销当前会话。
- POST `/auth/password`：`{current_password,new_password}`，撤销该用户旧会话并轮换当前 Cookie 与 CSRF；初始/重置密码必须先更换才能访问业务接口。
- 管理员 GET/POST `/users`，PATCH `/users/{user_id}` 修改角色、显示名称及启用状态，POST `/users/{user_id}/reset-password` 重置；禁止禁用或降级末位启用管理员。修改角色、禁用和重置会撤销旧会话。
- 管理员 GET `/audit?limit=100`（最多 500）读取最近审计；不保存请求正文、密码、查询串或凭据。

- GET `/api/v1/overview`: `{nodes_online,nodes_total,gpus_total,gpus_available,runs_active,runs_failed,assets_total}`。
- GET `/api/v1/nodes`: 节点数组；GET `/nodes/{id}` 返回节点及关联 assets/profiles/runs。
- GET `/api/v1/assets?node_id=&kind=`；POST `/assets`: `{name,kind,node_id,path,version,metadata}`，kind DATASET/MODEL/NORM。初始 REGISTERED，Agent 预检后验证；逻辑归档 POST `/assets/{id}/archive`。
- POST `/api/v1/nodes/{id}/path-suggestions`：`{prefix:"",limit:30}`，limit 最多 50，返回 202 `{id,node_id,status:PENDING|READY|FAILED,prefix,parent,entries:[{name,path,kind:directory|file}],truncated,error}`；GET `/nodes/{id}/path-suggestions/{request_id}` 轮询相同结构。查询仅限在线、心跳新鲜且不在部署维护中的机器，使用节点 Agent 的受管根目录。空前缀和根前缀可列可访问 roots，目录路径带尾 `/`，目录内按 basename 前缀匹配；不读文件内容、不递归扫描。15 秒超时，10 秒内相同查询复用，节点 pending 上限 20；过期 PATH_SUGGEST 命令被清理为 REJECTED。旧 Agent 返回明确升级提示，管理员仍可手输路径登记。
- GET `/api/v1/profiles?node_id=`：由 Agent 配置并登记，管理员不通过浏览器提交任意命令。
- 配方详情与自定义库：POST/GET `/nodes/{id}/recipe-details` 异步读取有界静态配置；GET/POST `/recipes`、POST `/recipes/{id}/revisions`、GET `/recipe-revisions/{id}` 保存不可变修订与原始详情，完整字段及行为见[配方库契约](recipe-library.md)。来源不接受浏览器指定，支持受约束数值及已读取的数据、模型、优化器和学习率静态字段，不接收可执行表达式。
- 配方回写：POST `/recipe-revisions/{id}/writeback` 接收 `{config_name}`；GET `/recipe-writebacks/{id}` 查询持久任务和提交 SHA。同一修订与名称重复请求复用任务。Agent 只在空闲代码目录将独立配置写入 `config.py`，自动创建仅包含本次文件变更的 Git commit，保留无关暂存及未暂存修改。成功后新增绑定新配置名、源码和 SHA 的不可变修订；已提交但详情读取失败时明确保留提交凭据。待确认写入任务不按只读查询规则过期，期间暂停同目录预检、提交及调度。
- POST `/api/v1/nodes/{id}/recipe-suggestions`：`{profile_id,prefix:"",limit:30}`，limit 最多 50；返回 202 `{id,node_id,profile_id,prefix,status:PENDING|READY|FAILED,entries:[{name,validated,source:code|registered|both}],truncated,source,warnings,error}`，GET 同路径 `/{request_id}` 轮询。仅查询所选在线机器上的活动训练环境，15 秒超时，短时缓存绑定环境配置；晚到 ACK 不恢复过期查询。代码来源是环境代码目录内的 OpenPI 配置注册列表，结果与管理员已登记名称合并。保留字段名 `prefix` 兼容旧调用，其值作为关键词做不区分大小写的包含匹配；名称开头匹配排在其他包含匹配之前，再按名称排序，最后截断。缓存按匹配语义版本隔离，避免复用旧前缀搜索的空结果。`validated` 由中台现有配方规则计算，发现名称不改变验收凭据。旧 Agent 可继续手输配方。
- GET `/api/v1/templates`：模板数组，`id,name,algorithm,status,description,parameters,required_inputs`，ReCap 另含 `runtime_adapter,stages`（阶段参数、默认入口及数据要求）。status AVAILABLE 或 PLANNED；只有 AVAILABLE 可提交。
- POST `/api/v1/run-previews`: `{name,node_id,profile_id,template_id,recipe?,training_stage?,stage_config?,inputs:{dataset:id,model:id,norm:id},dataset_bindings?,parameters:{steps,batch_size,learning_rate},gpu_count}`。配方为 ASCII 命名配置，匹配 `[A-Za-z0-9_-]{1,255}`；省略时仅可使用旧环境默认配方，smoke 固定为 `smoke`。服务端解析并冻结 `request.recipe`。返回 `{id,status:"PENDING"}`；GET `/run-previews/{id}` 返回 `{id,status,checks,code_snapshot,error,request}`。Agent 异步预检，成功 READY，失败 FAILED；有效期 15 分钟。
- POST `/api/v1/runs`: `{preflight_id}` + `Idempotency-Key`；服务器冻结预检请求与代码基线。
- GET `/api/v1/runs`；GET `/runs/{id}`：Run 含 `id,name,node_id,profile_id,template_id,recipe,state,created_at,inputs,parameters,gpu_count,code_expectation,attempts,outputs,queue_reason`。recipe 来自冻结的 `code_expectation.recipe`；历史无记录返回 null，不推测当前环境默认值，无需新增数据库列。
- POST `/api/v1/runs/{id}/cancel`；GET `/runs/{id}/events?after_seq=0`: 事件数组；GET `/runs/{id}/logs?after_seq=0`: `{items:[{seq,text,timestamp}],next_seq}`。
- GET `/api/v1/alerts`：当前异常数组。
- POST `/api/v1/ssh-connections/test`：`{host,port:22,username,password,install_dir}`，管理员密码 SSH 检查。返回 `{connection_id,host,port,username,fingerprint,hostname,checks,can_install,existing_agent?,expires_at}`；existing_agent 只包含白名单本地配置和节点 ID，无节点 token。密码加密持久化，不在任何响应返回。
- POST `/api/v1/remote-deployments`：`{connection_id,install_dir,agent:{name,control_plane_url?,roots,output_root,state_dir,profiles}}` + `Idempotency-Key`，202 返回异步部署记录。连接需测试且未过期，目录变化必须重新测试。
- GET `/api/v1/remote-deployments`：最近部署数组；GET `/{id}`：`{id,connection_id,host,port,username,install_dir,onboarding_id,node_id,state,stage,logs,history,agent,connection_mode,tunnel_state,error,created_at,updated_at,started_at,finished_at}`。agent 不含登记令牌，logs 含 `{timestamp,stage,message}`。
- GET `/api/v1/remote-deployments/{id}/logs/download`：管理员 Bearer 鉴权，返回 no-store 的 JSONL 下载附件。完整日志保存在远程管理状态目录的 `deployment-logs/<id>.jsonl`，API 内嵌 logs 仅保留最近 300 条；旧记录没有文件时导出现存阶段日志。URL 不携带令牌。
- POST `/api/v1/remote-deployments/{id}/retry` + `Idempotency-Key`：失败或中断的部署重新检查并安装，返回新的异步部署记录。原记录保留。
- GET `/api/v1/nodes/{id}/management`：返回 SSH 管理来源、安装路径及 `can_disconnect/can_uninstall` 和不可操作原因；不返回 SSH 密码。
- POST `/api/v1/nodes/{id}/disconnect` + `Idempotency-Key`：删除中台连接、关闭隧道并停用恢复、清除关联 SSH 凭据。保留远端程序，节点标为 DISCONNECTED；历史记录可查询。
- POST `/api/v1/nodes/{id}/uninstall` + `Idempotency-Key`：202 创建远程卸载任务，使用既有部署轮询接口。`operation=uninstall`，成功终态 UNINSTALLED，附 `operation_result.backup_dir/removed_paths/service_removed`；节点标为 REMOVED。远程失败不提前删除连接或凭据。
- GET `/api/v1/agent-config-defaults`：管理员读取平台自动提供的连接入口，返回 `{control_plane_url:string|null,source:"configured"|"request"|"unavailable",message:string}`；不包含登记令牌，响应 no-store。
- POST `/api/v1/agent-configs`：管理员生成接入配置，必填 profiles 的训练 Python 绝对路径、代码目录及 adapter；recipe 可省略/null/空白，规范化为 null。旧配置的非空 recipe 作为默认配方保留。`control_plane_url` 可省略或传 null，服务端采用默认入口，显式 URL 作为高级覆盖仍兼容。返回 `{id,filename,config}`，config 含登记令牌及稳定的 onboarding_id，SFT 的 validated 初始为 false，不创建节点。响应 no-store。
- GET `/api/v1/agent-package`：管理员下载当前 Agent 安装包；缺少构建产物返回 503 AGENT_PACKAGE_UNAVAILABLE，响应 no-store。
- GET `/healthz` 无鉴权。

Agent 连接入口优先使用中台环境变量 `VLA_AGENT_CONTROL_PLANE_URL`。未配置时，仅采用公开 HTTPS 请求入口；本机 HTTP、localhost、loopback、未指定地址和私网 IP 不自动推导。解析不手工信任 `X-Forwarded-*` 头。SSH 部署遇到 loopback 入口时，由中台管理独立反向隧道，远端端口自动分配后写入配置，监听远端 `127.0.0.1`；回连目标由 `VLA_TUNNEL_TARGET_HOST` / `VLA_TUNNEL_TARGET_PORT` 配置。HTTPS 入口采用直连，不创建隧道。默认地址无效返回 `503 AGENT_DEFAULT_URL_INVALID`，不回显原始环境值；既没有显式地址也无可用默认值则返回 `422 AGENT_CONTROL_PLANE_URL_REQUIRED`。普通向导自动使用默认入口，特殊网络覆盖放在高级连接设置中。原 `/agent-configs` 和 `/agent-package` 保留兼容手工接入，网页主流程使用远程部署接口。

部署状态 `QUEUED → RUNNING → WAITING_AGENT → SUCCEEDED/FAILED`，中台重启将未完成作业标记 `INTERRUPTED`，不自动重跑安装。SUCCEEDED 需要实际节点新心跳、匹配新 onboarding ID 的全部 profiles 和训练 Python READY；recipe 是否训练验收独立表示。成功后更新同一节点的名称与 roots 等元数据，保留节点身份与资产关联。关闭页面不停止部署。重复幂等键不重复安装，并发同目标安装拒绝；更新成功后旧部署隧道标记 SUPERSEDED 并释放。

安装细阶段为 `INSTALL_CHECKING / INSTALL_ENVIRONMENT / INSTALL_DEPENDENCIES / INSTALL_PACKAGE / INSTALL_VERIFY / INSTALL_CONFIG / INSTALL_SERVICE`，主进度均归属安装步骤；旧 INSTALLING 继续兼容。SSH 双流增量组行后实时回传，事件白名单解析、输出限长和脱敏，不接受远程输出任意改变部署状态。安装器输出 `VLA_INSTALL_EVENT ` + JSON `{phase,message,level,elapsed_seconds?}`，无输出时定期说明仍在等待。裸金属独立安装日志目录 0700、文件 0600，安装失败后保留；中台完整文件与数据库最近日志均使用脱敏文本。启动确认仍以真实服务、Agent 心跳与环境检查为准。

删除连接与卸载增加终态 DISCONNECTED / UNINSTALLED，任务含 `operation=install|disconnect|uninstall`。这些历史记录不可使用旧凭据重试，旧通道不再自动恢复。曾卸载节点不会通过旧部署记录的 prior_node 重新绑定，新创建使用新身份。数据库保留 Node/Run/Profile/Asset 关联，默认 GET `/nodes` 排除 DISCONNECTED/REMOVED；`include_inactive=true` 可读历史节点。旧节点 Agent 请求会被拒绝，Scheduler 不将其重新改成 OFFLINE 或再次调度。

部署期间锁定已知节点的训练调度窗口，新的预检/提交返回 `409 NODE_MAINTENANCE`，排队任务保持排队并显示同名原因。已有活动训练或未确认命令时拒绝部署，安装器在目标机再次检查本地任务。

SSH 凭据使用 Fernet 加密存储，0600 加密密钥与专属 known_hosts 在 `VLA_REMOTE_STATE_DIR`（默认数据库同目录的 remote-management）。首次连接记录指纹，已知指纹变化拒绝连接。API 响应、日志、进程参数均不含 SSH 密码或登记令牌；上传私有配置放在远端临时私有目录，安装后清理。隧道断线自动重连，中台重启恢复已有有效部署的通道。当前仅支持单 API worker 持有部署线程和隧道。

节点字段：`id,name,hostname,status,last_seen,roots,output_root,gpus,metrics`；profile 字段：`id,node_id,name,code_dir,python,adapter,recipe,execution_mode,docker_image,metadata`。GPU 字段 `uuid,name,memory_total_mb,memory_used_mb,utilization_pct,processes:[{pid,memory_mb}],reserved_by`；未知值 null。Agent 上报不含 reserved_by，由控制面结合预约填充。

`metrics.runtime_checks` 是数组，每项包含 `profile_id,status:CHECKING|READY|FAILED,checked_at,checks:[{name,ok,message}]`，可附 `python_version,executable,prefix,module_origin,error`。检查结果独立于 profile.metadata，不改变运行环境摘要，也不自动标记 recipe 为已验收。

## Agent 协议

- POST `/agent/v1/register` enrollment bearer，body `{name,hostname,roots,output_root,profiles:[{id,name,code_dir,python,adapter,recipe,execution_mode,docker_image,metadata}]}`；返回 `{node_id,token}`。首次登记后持久化凭据，重启复用，避免重复登记。
- POST `/agent/v1/heartbeat` node bearer，body `{node_id,gpus,metrics,active_attempts:[{attempt_id,pid,state}]}`，返回 `{ok:true}`。
- GET `/agent/v1/commands?node_id=` 返回待确认命令数组；命令在 APPLIED/REJECTED 前可重复返回，Agent 必须幂等。
- POST `/agent/v1/commands/{id}/ack` body `{node_id,status:"APPLIED"|"REJECTED",result:{...}}`。
- POST `/agent/v1/events/batch` body `{node_id,events:[{attempt_id,seq,type,timestamp,data}]}`；返回 `{accepted:n}`。每个 Attempt seq 单调，唯一约束去重。Agent 只有成功上传后删除本地待发送事件。

命令 `{id,type,payload}`：

- PREFLIGHT payload `{preview_id,profile_id,template_id,recipe,training_stage?,stage_config?,inputs:{role:{id,path,kind}},parameters,gpu_count}`。ack result `{code_snapshot,checks:[{name,ok,message}],error?:{code,message}}`；`code_snapshot.recipe` 必须与请求一致，缺失返回 AGENT_RECIPE_UNSUPPORTED 要求升级 Agent，不同返回 RECIPE_MISMATCH。控制面存 code_snapshot，绝不接受浏览器传 SHA。
- START payload `{run_id,attempt_id,execution_token,profile_id,template_id,recipe?,inputs:{role:{id,path,kind}},parameters,gpu_uuids,code_expectation}`；新任务 recipe 来自冻结快照，历史无记录则不发送此字段以兼容旧任务。Runner 负责独立进程、持久化事件和退出确认。ack 只表示 Runner 已接受，不代表训练开始或成功。
- CANCEL payload `{attempt_id,execution_token}`；以 Runner 的终止事件为准释放 GPU。
- PATH_SUGGEST payload `{prefix,limit,expires_at}`：Agent 返回 `{prefix,parent,entries:[{name,path,kind}],truncated}`，只读取本机普通文件和目录的名称、类型。检查父路径与每个匹配项的真实落点，跳过越界或失效符号链接；拒绝 `..`、控制字符、非绝对前缀及过期命令。扫描数量和耗时设上限，截断时 `truncated=true`。控制面再次校验节点归属、前缀与返回路径，联想结果不代表资产已登记或通过训练预检。
- RECIPE_SUGGEST payload `{profile_id,prefix,limit,expires_at}`：返回 `{profile_id,prefix,entries:[{name}],truncated,source,warnings}`。Agent 在 profile 的本机受管代码目录内静态解析 OpenPI `_CONFIGS` 注册列表，不导入或执行训练代码；源文件数量、大小及结果条数受限，拒绝越界符号链接、非法名称及过期请求。无法解析的动态生成片段返回固定警告。中台校验环境归属、请求快照、名称关键词匹配及安全结果结构，验收状态由中台独立计算。
- RECIPE_DETAIL payload `{profile_id,recipe,expires_at}` 返回选定配方源码片段、字段及默认值、文件摘要与 Git/本地 tracked 变化指纹；不执行或导入训练代码。保存修订必须使用近期成功且环境未变化的详情。
- RECIPE_WRITE payload `{operation_id,profile_id,definition,config_name,expires_at}` 将冻结修订中的静态修改物化为独立代码配方，并返回提交 SHA、变更路径和新来源详情。源码文件本身已有改动、来源漂移或存在活动训练时拒绝写入。Agent 用持久操作记录恢复源码、Git 引用和索引更新；重复执行复用同一提交，不自动推送。START 注册与回写共享代码目录锁。
- 自定义训练使用 `recipe_revision_id`，控制面解析并冻结 `recipe_definition`；PREFLIGHT/START 同时携带定义与目标 `node_id`，预检和实际执行快照必须回传完全一致的定义。`runtime_digest` 绑定修订摘要，源码指纹不一致时拒绝执行。共享 `validate_overrides/validate_definition/resolve_definition/apply_overrides` 限定修改项，Agent 自动预检绑定确切摘要，不继承代码配方或旧修订的检查结果。旧任务不携带此扩展字段。

事件类型：`STARTED` data `{pid,code_snapshot,output_dir}`；`METRIC` `{step,total_steps,loss,...}`；`LOG` `{text}`；`ARTIFACT` `{outputs:[{kind,name,path,version,metadata}],removed_paths:[path]}`；`SUCCEEDED` `{exit_code:0,outputs:[{kind,name,path,version,metadata}]}`；`FAILED` `{exit_code,error:{code,message}}`；`CANCELLED` `{exit_code}`。允许 `RESOURCE` 保存进程指标。只有终态事件释放 GPU。Agent 断网时状态 UNKNOWN，预约保留。

Agent 在训练过程中后台只读发现已提交的 OpenPI/Orbax 检查点：要求数字步数目录、有效的 `_CHECKPOINT_METADATA.commit_timestamp_nsecs`、params 元数据和实际数据分片，跳过临时目录。通过事务 outbox 上报，重启、重复投递和最终成功上报按尝试、类型、路径与版本去重。增量产出绑定执行节点和当前尝试的输出子目录，不改变训练状态或 GPU 预约。已确认被保留策略清理的路径标为 `UNAVAILABLE`，保留来源记录；重新出现可恢复 `READY`。页面在运行中自动刷新产出，显示步数、保存时间及可用状态；保存完成不代表整个训练结束或模型已评估。

## 状态和调度

Run 状态 QUEUED → STARTING → RUNNING → SUCCEEDED/FAILED；取消 CANCELLING → CANCELLED；节点失联时活动 Run 为 UNKNOWN（保存原状态）。单节点锁加 `(node_id,gpu_uuid)` 唯一预约，整组原子申请。无可用 GPU 则排队。显式开发 smoke 模板允许 gpu_count=0；真正 openpi 模板至少一张。Agent 必须复检本机路径、代码快照和 GPU 外部占用。API 进程内调度支持开发，独立 `vla-scheduler` 支持部署，同一个数据库锁保证一致性。

## 共享 Python API

`vla_protocol` 提供 `PROTOCOL_VERSION = 1` 和 `canonical_digest(value)`；无需依赖 ORM。Agent 与 API 可使用普通 dict 避免共享持久化模型。

`vla_adapters` 提供 `get_templates() -> list[dict]`；`resolve_recipe(profile,requested=None,require_validated=True) -> str`；`build_launch(profile,template_id,inputs,parameters,output_dir,gpu_uuids,recipe=None,recipe_definition=None,entrypoint=None,dataset_bindings=None,training_stage=None,stage_config=None) -> dict`，返回 `{argv,cwd,env}`。控制面与 Agent 共用配方解析规则；`require_validated` 参数保留调用兼容，手工 `metadata.validated_recipes`、`validated_recipe_revisions` 和 `validated` 不再作为启动门槛。新接入环境可没有默认配方，SFT 预检必须明确配方，Agent 负责自动检查并返回绑定本次请求的证据，提交时再次校验。调用本包受控 wrapper，不接受 shell。`smoke` 为显式测试模板，记录为测试成果；真实 SFT 入口为 `openpi-sft`，调用已安装的 openpi 与训练脚本。ReCap 返回 AVAILABLE 并提供 value/offline_rl 阶段配置，复用 openpi-sft 本机运行环境；MEM/RLT 仍返回 PLANNED。AVAILABLE 只表示开放配置与预检入口，不表示真实算法训练已验收。

SFT 与 ReCap `code_snapshot.automatic_preflight` 固定 `{schema_version:1,scope:"configuration_and_local_assets",plan,plan_digest}`。`plan` 包含实际入口、全部数据集和归一化参数、模型检查点、输入、参数与 GPU 数量；Agent 使用指定训练解释器解析配置，检查受管路径、数据协议、模型与输出空间，不初始化 GPU 或启动训练。逐项检查包含 `recipe_configuration/data_contract/norm_contract/model_checkpoint/training_entrypoint/output_storage/gpu_capacity`。提交验证完整证据与请求绑定，实际启动再次生成并比较计划；旧 Agent 或旧预检缺少证据时需要升级并重新预检。此证据不表示梯度更新、checkpoint 保存或训练效果已经验收。

代码快照由 Agent 实现：repo_url,code_dir,branch,commit_sha,commit_message,committed_at,dirty,changes_digest,untracked_digest,module_origin,captured_at,complete,runtime_digest,recipe。运行环境摘要同时绑定原始 profile 配置与本次配方，不能通过覆盖 profile.recipe 绕过环境漂移检测；旧命令未显式指定 recipe 时保留旧摘要格式。无有效 Git 仓库拒绝真实训练；smoke fixture 也建独立 Git 仓库保证验证链真实。真实 openpi 的 module_origin 在 profile.python 下探测；启动前比较快照身份与变化摘要。

## 当前交付诚实边界

本地 smoke 是控制链路测试，不是 VLA 算法或 GPU 训练验收。前端默认真实空状态，通过显式 demo 开关才可看演示数据，必须标示演示。生产 HTTPS、真实机器环境、模板数据协议和恢复验收按主设计继续执行。


训练入口配置：`PreviewInput.entrypoint` 接受所选环境项目内的相对 `.py` 文件路径。网页必填；旧 API 的省略/null 保持自动选择兼容。控制面转发到 PREFLIGHT 与 START，不新增 Run 数据库列，而是冻结在 `code_expectation.requested_entrypoint`；显式入口还记录 `requested_entrypoint_sha256`。Agent 解析真实路径并拒绝逃逸软链接，预检/启动/结束快照比较路径及内容摘要；自动计划同时记录所选相对路径、实际绝对路径与源码摘要。wrapper 编译执行核对过的源码，避免旧 `.pyc` 导致执行代码与预检不同。

项目路径联想复用 `POST /nodes/{node_id}/path-suggestions`，增加可选 `profile_id`；此时控制面从 Profile 固定 `base_path=code_dir`，请求前缀限制在项目下，空前缀打开项目根。命令使用 `python_only:true`，Agent 独立核对当前 Profile 和目录。响应必须回显 `profile_id/base_path`，只返回项目内目录及 `.py` 文件；与通用资产路径缓存隔离，Profile 或基目录变化后旧结果失效。


多数据集配置：网页创建 SFT 或 ReCap 任务使用 `dataset_bindings:[{dataset_asset_id,norm_asset_id,recipe_index,weight}]` 和 `inputs:{model:id}`；1–64 组，权重为正有限数值且总和有限，每行用 recipe_index（原配方数据处理项的零起始索引）或 processor:{name,parameters} 指定处理方式，二者必选其一；独立处理器不受当前配方数据项列表限制。中台展开为 `dataset_0/norm_0` 等角色，校验每个资产的同机归属、种类与状态，全部保存为 RunInput 关系，支持占用/归档保护。该列表与单组 `inputs.dataset/norm` 不可混用。仅单数据集 SFT 省略/null 保留旧 API 的 dataset/norm 角色兼容；多数据集配方及 ReCap 必须显式逐组绑定，缺少时返回 DATASET_BINDINGS_REQUIRED，不隐含使用原配方目录。

数据集、归一化、初始模型与混采权重由本次训练的资产绑定唯一决定。配方中的 repo_id、assets、norm 路径、weight_loader 和 data.weights 仅显示源码参考，禁止另存或回写这些字段及其容器；历史修订仍可读取，但含此类覆盖时需先保存干净版本。网页不从源码路径或权重自动选择任务资产；新行从未选择资产、等权重开始，用户显式确认。显式绑定完整替换训练数据列表，允许增删、重排和复用原配方的数据处理项；不隐含加入原配方的其他数据。旧行沿用 recipe_index 对应的图像、状态、动作处理；独立 processor 行从所选训练环境导出的具体 DataConfigFactory 类及其参数构造，分别覆盖 repo_id、归一化资产及混采权重；state padding 与合并归一化策略保留。单数据集配方仅接受一行、索引 0。绑定列表冻结在 `code_snapshot.dataset_bindings` 和 `automatic_preflight.plan.dataset_bindings`，通过 PREFLIGHT/START 传递；页面改动后重新预检，控制面、Agent 和 wrapper 均核对行数、顺序、路径、权重和配对。生效计划固定 input_authority:registered_assets、每组 dataset_asset_id/norm_asset_id/recipe_index 以及 model.asset_id；提交与启动均验证资产身份。旧 Agent 或旧预检缺少该证明时拒绝新提交，要求升级后重新预检；已有运行的冻结历史保留。


独立数据处理器目录：POST `/api/v1/nodes/{id}/data-processors` 接收 `{profile_id}`，GET 同路径 `/{query_id}` 轮询。Agent 在后台使用登记的训练解释器执行 CPU/offline 目录探针，读取本项目配置模块导出的具体 dataclass DataConfigFactory 子类，不局限所选配方。响应 catalog.entries 提供类名、说明、可编辑的有界静态参数、复杂参数限制及来源；发现不代表数据兼容或已通过训练验收。资源字段、嵌套 Multi 工厂、合成 Fake 及未支持的 RLDS 加载器不可作为独立数据处理器。查询按机器、Profile 和代码来源隔离，拒绝过期或漂移结果。

新行使用 `processor:{name,parameters}`，name 是本机配置模块导出的简单类名，parameters 为有界的 JSON 静态值，可包括 `base_config.prompt_from_task` 等已声明处理参数。禁止可执行表达式、任意导入、未声明参数及资产路径覆盖。处理器、参数和资源绑定共同进入冻结请求、运行环境摘要、预检计划及实际启动检查；处理器类的真实身份与来源另行记录。用户切换类后使用新类默认值，不继承原配方或旧类的隐藏参数。旧 recipe_index 行及历史任务格式保留。单数据集配方仍只接受一行；多数据集配方保留其 state padding 和合并归一化策略。

训练缓存预检：CPU wrapper 在导入 OpenPI/datasets 前应用项目内已核对的 `src/openpi/shared/runtime_paths.py` 缓存默认值。`training_cache` 为 SFT 与 ReCap 必需检查：已有缓存锁以不截断方式验证写入访问，再对独有临时锁测试 FileLock；对每个实际数据集按 LeRobot 相同 data_dir 构造 `datasets.load_dataset_builder("parquet",...)`。不调用 download_and_prepare、as_dataset 或训练数据加载器。缓存路径、每个 builder 的目录及初始化源码摘要进入冻结计划，启动前重做；缓存路径均限制在 Agent 受管目录。权限失败使用 TRAINING_CACHE_UNWRITABLE，无法确认的配置使用 TRAINING_CACHE_UNAVAILABLE。此探针会操作缓存锁、创建并清理自己的小锁文件，不生成 Arrow 数据或使用 GPU；不修改缓存所有者或访问权限。旧的缺少缓存检查的 SFT 或 ReCap 预检不可提交，需重新预检。


## ReCap 分阶段训练契约

`template_id:recap` 必须选择 `training_stage:value|offline_rl`，`stage_config` 仅包含该阶段支持的字段；省略/null 规范化为 `{}`，保留配方值，不由 API 补入阶段默认值。其他模板拒绝非 null 的阶段字段。阶段不修改已登记 profile，使用 `runtime_adapter:openpi-sft` 匹配本机环境；每次作业仅执行所选训练阶段，不自动串联标注或后续训练。

| 阶段 | 输入协议与模型 | 独立阶段参数 | 默认项目内入口 |
| --- | --- | --- | --- |
| `value` 价值函数训练 | 已含有限 `value_target` 的数据、初始权重及逐组归一化统计；配方模型必须为 `Pi0ValueConfig`，数据映射保留 `value_target` | `val_ratio` 数值 0–0.9；`eval_interval` 整数 1–1000000；`eval_batches` 整数 1–10000 | 单数据集 `scripts/train_value.py`；多数据集 `scripts/train_value_multi_dataset.py` |
| `offline_rl` 离线强化学习／ACP | 已完成离线优势标注的派生数据、初始策略权重及逐组归一化统计；必须为读取 `is_positive` 的 ACP 策略配方，拒绝价值模型配方 | `acp_dropout` 数值 0–1，覆盖每个数据工厂的优势条件丢弃概率 | 单数据集 `scripts/train.py`；多数据集 `scripts/train_multi_dataset.py` |

两阶段均要求显式 `dataset_bindings` 与 `inputs:{model:id}`，逐行绑定同机 DATASET/NORM 资产、原配方处理项和混采权重，不能省略绑定而隐含使用配方中的其他数据。公共训练步数、batch、学习率和 GPU 数量仍按模板规则校验；阶段参数拒绝未知字段、布尔伪数值、非有限数值及越界值。

训练入口保持项目内相对 `.py` 路径、真实路径边界、`main(config)`／`init_wandb` 接口及源码摘要校验。ReCap 另外拒绝价值阶段使用已知策略入口、ACP 使用已知价值入口、多数据集价值阶段使用单数据集价值入口，以及 `train_advantage.py`、`train_multi_dataset_hybrid.py`。手选其他入口仍须通过实际配方、接口和阶段数据协议检查，文件存在本身不构成阶段匹配证明。

Agent 在 CPU 自动预检中核对模型类型、权重加载器和每组实际 parquet 的映射列、监督字段及有效标签；每组最多抽取 16 个均匀分布文件的前 64 行监督列，记录抽查范围，不声称遍历全部数据。价值阶段还检查入口实际使用的 `~/.cache/jax` 可写性。输出 `recap_stage` 检查和 `plan.recap_stage_contract:{stage,model_type,loader_type,datasets}`。`training_stage/stage_config` 同时冻结在请求、代码快照、runtime digest 和自动计划，经 PREFLIGHT/START 传递；提交与启动复核，阶段、参数或数据证据漂移时拒绝或取消，旧 Agent 缺少阶段证据不能返回 READY。无新增 Run 数据库列；任务详情从冻结快照恢复阶段配置。

ACP 仅消费已登记的标注派生数据，不会自动运行 `lerobot_value_infer.py`，也不会向原始数据写入目标列或优势标签。目标列准备、价值模型推理与优势标注、派生版本发布及血缘仍是独立接入边界。配置入口可用、CPU 预检通过或测试 fixture 成功均不等于真实 ReCap 训练验收；真实价值模型与 ACP 策略的梯度、指标、checkpoint、取消及恢复边界需分别记录。
