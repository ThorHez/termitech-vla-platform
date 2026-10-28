import type { Asset, Node, Run } from './types';
import type { PolicyWarning } from './policyWarningInfo';
import type { CheckpointTransfer } from './checkpointTransfer';

export type InferenceBackend = 'auto' | 'jax' | 'pytorch';
export type ResolvedInferenceBackend = Exclude<InferenceBackend, 'auto'>;

export const inferenceBackendLabels: Record<InferenceBackend, string> = {
  auto: '自动识别',
  jax: 'JAX',
  pytorch: 'PyTorch',
};

export function inferenceBackendError(backend: InferenceBackend, node?: Node) {
  if (backend === 'auto') return undefined;
  if (
    !Array.isArray(node?.metrics?.capabilities) ||
    !node.metrics.capabilities.includes('POLICY_DEPLOY_BACKEND_V1')
  )
    return '当前 Agent 尚不支持指定推理框架，请先更新 Agent。权重同步可继续进行。';
  return undefined;
}

export function resolvedInferenceBackendLabel(backend?: ResolvedInferenceBackend | null) {
  return backend ? inferenceBackendLabels[backend] : '尚未识别，等待 Agent 上报';
}

export interface PolicyDeployment {
  id: string;
  record_id?: string | null;
  name: string;
  node_id: string;
  profile_id: string;
  checkpoint_path: string;
  config_name: string;
  inference_backend?: InferenceBackend;
  resolved_backend?: ResolvedInferenceBackend | null;
  source_run_id?: string | null;
  transfer_id?: string | null;
  port: number;
  gpu_uuid?: string | null;
  gpu_mode?: 'exclusive' | 'shared';
  gpu_memory_mb?: number | null;
  default_prompt?: string | null;
  state: string;
  reported_state?: string;
  connection_state?: string;
  endpoint?: string | null;
  pid?: number | null;
  logs: string;
  checks: { name: string; ok: boolean; message?: string }[];
  error?: { code: string; message: string } | null;
  warnings?: PolicyWarning[];
  pending_command?: string | null;
  test_result?: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface DeploymentRecordSummary {
  id: string;
  name: string;
  node_id: string | null;
  profile_id: string | null;
  source_run_id: string | null;
  state: string;
  created_at: string;
  updated_at: string;
  deployment_id: string | null;
  selected_transfer_id: string | null;
  transfer_ids: string[];
}

export interface DeploymentRecordDetail extends DeploymentRecordSummary {
  deployment: PolicyDeployment | null;
  transfers: CheckpointTransfer[];
}

export const deploymentLabels: Record<string, string> = {
  PENDING: '等待 Agent',
  STARTING: '模型加载中',
  RUNNING: '服务就绪',
  STOPPING: '正在停止',
  STOPPED: '已停止',
  FAILED: '部署失败',
  UNKNOWN: '状态待确认',
};

export const deploymentActive = (state: string) =>
  ['PENDING', 'STARTING', 'RUNNING', 'STOPPING', 'UNKNOWN'].includes(state);

export const deploymentStartable = (state: string) => ['STOPPED', 'FAILED'].includes(state);

export function deploymentStartBlockedReason(
  item: PolicyDeployment,
  node: Node | undefined,
  writable: boolean,
  busy = false,
) {
  if (!deploymentStartable(item.state)) return '当前服务状态不能启动，请先确认服务已停止。';
  if (!writable) return '当前账号没有启动服务的权限。';
  if (busy || item.pending_command) return '已有部署操作正在处理，请等待状态回执后再启动。';
  if (!node) return '正在读取目标机器状态，确认在线后可启动服务。';
  if (node.status !== 'ONLINE') return '目标机器当前离线，恢复在线后可启动服务。';
  const backendError = inferenceBackendError(item.inference_backend || 'auto', node);
  if (backendError) return backendError;
  return undefined;
}

function deploymentRevision(value: string) {
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds)) return undefined;
  const fraction = /\.(\d+)(?:Z|[+-]\d\d:\d\d)$/.exec(value)?.[1] || '';
  return [milliseconds, Number((fraction + '000000').slice(3, 6))];
}

export function deploymentDetailReceipt(
  id: string,
  polled?: PolicyDeployment,
  startReceipt?: PolicyDeployment,
  initial?: PolicyDeployment,
) {
  const receipt = startReceipt?.id === id ? startReceipt : undefined;
  let current = polled?.id === id ? polled : undefined;
  const embedded = initial?.id === id ? initial : undefined;
  if (embedded && current) {
    const embeddedTime = deploymentRevision(embedded.updated_at);
    const polledTime = deploymentRevision(current.updated_at);
    if (
      embeddedTime &&
      (!polledTime ||
        embeddedTime[0] > polledTime[0] ||
        (embeddedTime[0] === polledTime[0] && embeddedTime[1] > polledTime[1]))
    )
      current = embedded;
  } else if (!current) current = embedded;
  if (!current) return receipt || (initial?.id === id ? initial : undefined);
  if (!receipt) return current;
  const pollTime = deploymentRevision(current.updated_at);
  const receiptTime = deploymentRevision(receipt.updated_at);
  if (pollTime && receiptTime) {
    if (
      pollTime[0] > receiptTime[0] ||
      (pollTime[0] === receiptTime[0] && pollTime[1] > receiptTime[1])
    )
      return current;
    if (
      pollTime[0] === receiptTime[0] &&
      pollTime[1] === receiptTime[1] &&
      current.state === receipt.state &&
      current.pending_command === receipt.pending_command
    )
      return current;
  }
  return receipt;
}

export function deployableOutputs(run?: Run): Asset[] {
  return [...(run?.outputs || [])]
    .filter(
      (asset) =>
        asset.kind === 'MODEL' &&
        asset.state !== 'UNAVAILABLE' &&
        asset.metadata?.availability !== 'removed' &&
        asset.node_id === run?.node_id,
    )
    .sort((a, b) => Number(b.metadata?.step || 0) - Number(a.metadata?.step || 0));
}

export function sourceDeploymentBlocker(run?: Run): string | undefined {
  if (!run) return undefined;
  if (run.template_id !== 'openpi-sft')
    return '当前支持 OpenPI 动作策略部署，请选择 OpenPI 训练任务。';
  if (
    run.training_stage === 'value' ||
    run.code_expectation?.training_stage === 'value' ||
    run.code_expectation?.automatic_preflight?.plan.training_stage === 'value'
  )
    return '此任务训练的是价值模型，请选择动作策略模型。';
  if ((run.code_expectation?.automatic_preflight?.plan.datasets?.length || 0) > 1)
    return '此任务包含多组数据与归一化参数，尚需配置推理数据绑定后才能部署。';
  if (['QUEUED', 'STARTING', 'RUNNING', 'CANCELLING', 'UNKNOWN'].includes(run.state))
    return '来源训练尚未结束，请等待训练完成后部署权重。';
  if (!trainingInferenceConfig(run))
    return '此训练任务缺少已确认的生成配置，无法自动生成推理配置。请使用保存了生成配置的训练任务，或改选本地权重并选择已有配置。';
  return undefined;
}

export function trainingInferenceConfig(run?: Run) {
  const snapshot = run?.code_expectation;
  const plan = snapshot?.automatic_preflight?.plan;
  const generated = plan?.generated_config;
  const recipe = snapshot?.recipe ?? plan?.recipe;
  if (
    generated?.schema_version !== 1 ||
    generated.language !== 'python' ||
    generated.filename !== 'training_config.py' ||
    typeof generated.source !== 'string' ||
    !generated.source.trim() ||
    !/^[a-f0-9]{64}$/.test(generated.sha256) ||
    snapshot?.confirmed_config_sha256 !== generated.sha256 ||
    (snapshot.recipe != null && plan?.recipe != null && snapshot.recipe !== plan.recipe) ||
    typeof recipe !== 'string' ||
    !/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}$/.test(recipe)
  )
    return undefined;
  return {
    recipe,
    sha256: generated.sha256,
    autoConfig: snapshot.auto_config,
  };
}

export function deploymentClientExample(port: number) {
  return `import json\nimport numpy as np\nfrom openpi_client import websocket_client_policy\n\ndef arrays(value):\n    if isinstance(value, dict):\n        if set(value) == {"__array__", "dtype"}:\n            return np.asarray(value["__array__"], dtype=value["dtype"])\n        return {k: arrays(v) for k, v in value.items()}\n    if isinstance(value, list):\n        return np.asarray(value)\n    return value\n\n# 在部署机器上运行；字段需与模型的观测输入一致。\n# RGB 图像可写为 {"__array__": [[[0, 0, 0]]], "dtype": "uint8"}。\n# 组合式处理配置可直接使用原始字段，由推理适配转换图像类型。\nwith open("observation.json") as f:\n    observation = arrays(json.load(f))\npolicy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=${port})\nresult = policy.infer(observation)\nprint(result["actions"].shape)\nprint(result["actions"])`;
}
