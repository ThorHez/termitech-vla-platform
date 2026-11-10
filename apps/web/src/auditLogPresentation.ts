export interface AuditEntry {
  id: string;
  created_at: string;
  actor_id: string | null;
  actor_username: string | null;
  action: string;
  method: string;
  path: string;
  status: number;
  outcome?: AuditOutcome;
  details?: Record<string, unknown> | null;
  remote_addr?: string | null;
}

export interface AuditEventsPage {
  items: AuditEntry[];
  next_cursor: string | null;
  has_more: boolean;
}

export type AuditOutcome = 'succeeded' | 'accepted' | 'denied' | 'failed';
export interface AuditFilters {
  actor_id: string;
  action: string;
  outcome: '' | AuditOutcome;
  from_time: string;
  to_time: string;
  q: string;
}
export const emptyAuditFilters = (): AuditFilters => ({
  actor_id: '',
  action: '',
  outcome: '',
  from_time: '',
  to_time: '',
  q: '',
});

export const auditOutcomeLabels: Record<AuditOutcome, string> = {
  succeeded: '成功',
  accepted: '已受理',
  denied: '拒绝',
  failed: '失败',
};
export function auditOutcome(status: number): AuditOutcome {
  return status === 202
    ? 'accepted'
    : status === 401 || status === 403
      ? 'denied'
      : status >= 400
        ? 'failed'
        : 'succeeded';
}

const safeText = (value: unknown, limit = 240) =>
  typeof value === 'string' && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value)
    ? value
        .trim()
        .replace(/\bBearer\s+[^\s,;]+/gi, '[已隐藏]')
        .replace(
          /\b(?:password|token|csrf(?:_token)?|secret|authorization)\s*[:=]\s*[^\s,;]+/gi,
          '[已隐藏]',
        )
        .slice(0, limit)
    : '';

const detailLabels: Record<string, string> = {
  category: '分类',
  target_type: '对象类型',
  target_id: '对象 ID',
  target_name: '对象名称',
  asset_kind: '资产类型',
  node_id: '机器 ID',
  count: '总数量',
  related_ids: '关联任务',
  error_code: '错误码',
  code: '错误码',
  request_id: '请求标识',
  duration_ms: '耗时',
  created_count: '新增数量',
  skipped_count: '跳过数量',
  username: '账号',
  attempted_username: '尝试账号',
  role: '角色',
  enabled: '账号启用',
};
const detailKeys = ['operation_label', ...Object.keys(detailLabels)];

/** Never retain arbitrary nested request data in audit UI state. */
export function safeAuditDetails(value: unknown): Record<string, string | number | boolean> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const result: Record<string, string | number | boolean> = {};
  for (const key of detailKeys) {
    const item = (value as Record<string, unknown>)[key];
    if (
      key === 'duration_ms' ||
      key === 'created_count' ||
      key === 'skipped_count' ||
      key === 'count'
    ) {
      if (
        typeof item === 'number' &&
        Number.isFinite(item) &&
        item >= 0 &&
        (key === 'duration_ms' || Number.isSafeInteger(item))
      )
        result[key] = item;
    } else if (key === 'enabled') {
      if (typeof item === 'boolean') result[key] = item;
    } else {
      const text = safeText(item, key === 'target_name' ? 500 : 200);
      if (text) result[key] = text;
    }
  }
  return result;
}

export function safeAuditPath(path: unknown) {
  return safeText(path, 2048).split(/[?#]/, 1)[0];
}

export function sanitizeAuditEntry(entry: AuditEntry): AuditEntry {
  return {
    id: safeText(entry.id, 100),
    created_at: safeText(entry.created_at, 60),
    actor_id: safeText(entry.actor_id, 100) || null,
    actor_username: safeText(entry.actor_username, 100) || null,
    action: safeText(entry.action, 120),
    method: safeText(entry.method, 16).toUpperCase(),
    path: safeAuditPath(entry.path),
    status: entry.status,
    outcome: auditOutcome(entry.status),
    details: safeAuditDetails(entry.details),
    remote_addr: safeText(entry.remote_addr, 100) || null,
  };
}

export const auditActionGroups: { label: string; actions: Record<string, string> }[] = [
  {
    label: '资产与数据集',
    actions: {
      'assets.register': '登记资产',
      'assets.archive': '删除资产登记',
      'assets.update_profile': '修改数据集特征',
      'assets.bind_norm': '修改归一化绑定',
      'assets.validate': '发起数据集校验',
      'assets.cancel_validation': '取消数据集校验',
      'datasets.scan': '扫描数据集目录',
      'datasets.register_batch': '批量登记数据集',
    },
  },
  {
    label: '训练',
    actions: {
      'runs.preflight': '发起训练预检',
      'runs.create': '创建训练任务',
      'runs.cancel': '取消训练任务',
      'runs.delete': '删除训练记录',
    },
  },
  {
    label: '资产同步',
    actions: {
      'transfers.create': '创建资产同步任务',
      'transfers.create_batch': '批量创建资产同步',
      'transfers.retry': '重试资产同步',
      'transfers.cancel': '取消资产同步',
      'transfers.delete': '删除资产同步记录',
      'checkpoints.transfer': '创建模型部署同步',
      'checkpoints.retry': '重试模型部署同步',
      'checkpoints.cancel': '取消模型部署同步',
    },
  },
  {
    label: '推理部署',
    actions: {
      'deployments.create': '创建推理部署',
      'deployments.refresh': '刷新推理部署状态',
      'deployments.start': '启动推理服务',
      'deployments.stop': '停止推理服务',
      'deployments.test': '测试推理服务',
      'deployments.delete_record': '删除部署记录',
    },
  },
  {
    label: '登录与用户',
    actions: {
      'auth.login': '登录',
      'auth.login_failed': '登录失败',
      'auth.logout': '退出登录',
      'auth.password': '修改本人密码',
      'auth.password_failed': '修改本人密码失败',
      'auth.reauth': '确认敏感操作身份',
      'auth.reauth_failed': '敏感操作身份确认失败',
      'auth.sessions_revoked': '撤销登录会话',
      'users.create': '创建用户',
      'users.update': '修改用户信息或权限',
      'users.reset_password': '重置用户密码',
    },
  },
  {
    label: '个人配置与公共配方',
    actions: {
      'drafts.save': '保存个人训练草稿',
      'drafts.clear': '清空个人训练草稿',
      'training_templates.create': '保存个人训练模板',
      'training_templates.update': '修改个人训练模板',
      'training_templates.sharing': '更改训练模板共享范围',
      'training_templates.delete': '删除个人训练模板',
      'dataset_templates.save': '保存个人格式模板',
      'dataset_templates.update': '修改个人格式模板',
      'dataset_templates.sharing': '更改数据集格式模板共享范围',
      'dataset_templates.delete': '删除个人格式模板',
      'recipes.create': '登记公共配方',
      'recipes.revise': '修订公共配方',
      'recipes.writeback': '回写机器配方代码',
    },
  },
  {
    label: '机器与资源',
    actions: {
      'machines.rename': '修改机器名称',
      'machines.onboard': '创建机器接入任务',
      'machines.retry_onboarding': '重试机器接入任务',
      'machines.agent_config': '生成 Agent 接入配置',
      'machines.ssh_test': '测试 SSH 连接',
      'machines.ssh_connect': '建立 SSH 连接',
      'machines.ssh_retest': '重新测试 SSH 连接',
      'machines.storage': '修改机器存储目录',
      'machines.rebind_profile': '切换训练环境',
      'machines.add_profile': '添加训练环境',
      'machines.disconnect': '断开机器',
      'machines.uninstall': '卸载机器 Agent',
      'machines.upgrade': '升级机器 Agent',
      'machines.download_agent': '下载 Agent 安装包',
      'machines.download_logs': '下载机器接入日志',
      'resources.reserve': '预留 GPU',
      'resources.release': '释放 GPU 预留',
      'terminal.create_session': '申请机器终端会话',
      'terminal.open': '打开机器终端',
      'terminal.open_failed': '机器终端连接失败',
      'terminal.close': '关闭机器终端',
    },
  },
  {
    label: '机器查询',
    actions: {
      'queries.paths': '查询机器路径',
      'queries.dataset_fields': '读取数据集字段',
      'queries.recipes': '查询机器配方',
      'queries.recipe_details': '读取机器配方详情',
      'queries.processors': '查询数据处理器',
      'queries.inference_configs': '查询推理配置',
      'queries.ssh_paths': '查询 SSH 接入路径',
    },
  },
];

const actionLabels: Record<string, string> = {
  ...Object.assign({}, ...auditActionGroups.map((group) => group.actions)),
  'auth.login': '登录',
  'auth.logout': '退出登录',
  'auth.password': '修改密码',
  'auth.reauth': '验证当前密码',
  'auth.sessions_revoked': '撤销账号会话',
  'auth.login_failed': '登录失败',
  'users.create': '创建用户',
  'users.update': '修改用户',
  'users.reset_password': '重置用户密码',
  'user.create': '创建用户',
  'user.update': '修改用户',
  'user.reset_password': '重置用户密码',
  'asset.register': '登记资产',
  'asset.archive': '删除资产登记',
  'run.create': '创建训练任务',
  'run.cancel': '取消训练',
};
export function auditOperationLabel(entry: AuditEntry) {
  const label = safeAuditDetails(entry.details).operation_label;
  if (typeof label === 'string') return label;
  if (actionLabels[entry.action]) return actionLabels[entry.action];
  const path = safeAuditPath(entry.path).replace(/^\/api\/v1/, '');
  if (entry.method === 'POST' && path === '/assets') return '登记资产';
  if (/^\/assets\/[^/]+\/archive$/.test(path)) return '删除资产登记';
  if (/^\/assets\/[^/]+\/dataset-profile$/.test(path)) return '修改数据集特征';
  if (/^\/assets\/[^/]+\/norm-binding$/.test(path)) return '修改归一化绑定';
  if (/^\/assets\/[^/]+\/validations/.test(path)) return '数据源校验';
  if (path === '/run-previews') return '提交训练预检';
  if (path === '/runs' && entry.method === 'POST') return '创建训练任务';
  if (/^\/runs\/[^/]+\/cancel$/.test(path)) return '取消训练';
  if (/^\/runs\/[^/]+$/.test(path) && entry.method === 'DELETE') return '删除训练记录';
  if (path === '/users' && entry.method === 'POST') return '创建用户';
  if (/^\/users\/[^/]+\/reset-password$/.test(path)) return '重置用户密码';
  if (/^\/users\/[^/]+$/.test(path) && entry.method === 'PATCH') return '修改用户';
  if (/^\/nodes\/[^/]+\/upgrade$/.test(path)) return '升级 Agent';
  if (/^\/nodes\/[^/]+\/uninstall$/.test(path)) return '卸载 Agent';
  if (/^\/nodes\/[^/]+\/disconnect$/.test(path)) return '断开机器';
  if (/suggestions|recipe-details|data-processors|dataset-inspections/.test(path))
    return '查询机器信息';
  return (
    (
      {
        POST: '提交请求',
        PUT: '更新请求',
        PATCH: '修改请求',
        DELETE: '删除请求',
        GET: '读取请求',
      } as Record<string, string>
    )[entry.method] || '接口操作'
  );
}

const targetLabels: Record<string, string> = {
  account: '账号',
  asset: '资产',
  dataset: '数据集',
  user: '用户',
  node: '机器',
  run: '训练任务',
  recipe: '配方',
  deployment: '部署',
  transfer: '同步任务',
  session: '会话',
  dataset_scan: '数据集扫描',
  dataset_template: '个人格式模板',
  query: '机器查询',
  asset_transfer: '资产同步任务',
  asset_transfer_batch: '批量资产同步',
  run_preview: '训练预检',
  policy_deployment: '推理部署',
  deployment_record: '部署记录',
  checkpoint_transfer: '模型部署同步',
  training_draft: '训练草稿',
  training_template: '个人训练模板',
  recipe_writeback: '配方代码回写',
  agent_config: 'Agent 接入配置',
  ssh_connection: 'SSH 连接',
  terminal: '机器终端',
  remote_deployment: '机器接入任务',
  agent_package: 'Agent 安装包',
  resource: '业务对象',
};
const categoryLabels: Record<string, string> = {
  auth: '登录与账号',
  users: '用户管理',
  assets: '资产',
  training: '训练',
  transfers: '资产同步',
  deployments: '推理部署',
  resources: '资源',
  machines: '机器',
  queries: '机器查询',
  templates: '个人模板',
  drafts: '训练草稿',
  recipes: '公共配方',
  api: '接口操作',
};
const assetKindLabels: Record<string, string> = {
  DATASET: '数据集',
  NORM: '归一化统计（norm_stats）',
  MODEL: '模型',
};
export function auditTarget(entry: AuditEntry) {
  const details = safeAuditDetails(entry.details);
  const kind = typeof details.target_type === 'string' ? details.target_type : '';
  const id = typeof details.target_id === 'string' ? details.target_id : '';
  const name = typeof details.target_name === 'string' ? details.target_name : '';
  return { label: name || targetLabels[kind] || kind || '—', id };
}

export function auditDetailRows(entry: AuditEntry) {
  return Object.entries(safeAuditDetails(entry.details)).flatMap(([key, value]) => {
    if (!detailLabels[key]) return [];
    const identifiers =
      key === 'category'
        ? categoryLabels
        : key === 'target_type'
          ? targetLabels
          : key === 'asset_kind'
            ? assetKindLabels
            : undefined;
    const translated = identifiers?.[String(value)];
    return [
      {
        key,
        label: detailLabels[key],
        value:
          translated ||
          (key === 'duration_ms'
            ? `${value} ms`
            : typeof value === 'boolean'
              ? value
                ? '是'
                : '否'
              : String(value)),
        identifier: translated ? String(value) : undefined,
      },
    ];
  });
}

export function auditTime(value: string) {
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? '未知时间'
    : timestamp.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
}

export function submitAuditFilters(draft: AuditFilters): AuditFilters {
  const result = {
    ...draft,
    actor_id: draft.actor_id.trim(),
    action: draft.action.trim(),
    q: draft.q.trim(),
  };
  if (
    result.actor_id &&
    result.actor_id !== 'unattributed' &&
    !/^[a-f\d]{8}-(?:[a-f\d]{4}-){3}[a-f\d]{12}$/i.test(result.actor_id)
  )
    throw new Error('请选择有效的用户。');
  if (result.q.length > 100) throw new Error('关键词最多 100 个字符。');
  if (result.action.length > 120) throw new Error('操作标识最多 120 个字符。');
  for (const field of ['from_time', 'to_time'] as const) {
    if (!result[field]) continue;
    const time = new Date(result[field]);
    if (Number.isNaN(time.getTime())) throw new Error('请填写有效的起止时间。');
    result[field] = time.toISOString();
  }
  if (result.from_time && result.to_time && result.from_time > result.to_time)
    throw new Error('开始时间不能晚于结束时间。');
  return result;
}

export function auditEventsPath(filters: AuditFilters, cursor: string | null = null) {
  const params = new URLSearchParams({ limit: '50' });
  for (const [key, value] of Object.entries(filters)) if (value) params.set(key, value);
  if (cursor) params.set('cursor', cursor);
  return `/audit/events?${params}`;
}
