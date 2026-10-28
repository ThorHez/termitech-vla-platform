import type { Node, Run } from './types';
import type { PolicyWarning } from './policyWarningInfo';

export function hasTransferCapability(node: Node | undefined, capability: string) {
  return (
    Array.isArray(node?.metrics?.capabilities) && node.metrics.capabilities.includes(capability)
  );
}

export function checkpointSyncAllowed(input: {
  sourceRun?: Pick<Run, 'node_id'>;
  sourceNode?: Node;
  targetNode?: Node;
  profileSelected: boolean;
  directory: string;
  checkpoint: string;
  blocked?: string;
  syncing: boolean;
  busy: boolean;
}) {
  return (
    !!input.sourceRun &&
    !!input.targetNode &&
    input.sourceRun.node_id !== input.targetNode.id &&
    input.sourceNode?.status === 'ONLINE' &&
    input.targetNode.status === 'ONLINE' &&
    hasTransferCapability(input.sourceNode, 'POLICY_EXPORT') &&
    hasTransferCapability(input.targetNode, 'POLICY_DEPLOY_PORTABLE_V1') &&
    input.profileSelected &&
    input.directory.trim().startsWith('/') &&
    input.checkpoint.trim().startsWith('/') &&
    !input.blocked &&
    !input.syncing &&
    !input.busy
  );
}

export interface CheckpointTransfer {
  id: string;
  record_id?: string | null;
  name: string;
  source_node_id: string;
  source_run_id: string;
  source_checkpoint_path: string;
  target_node_id: string;
  target_profile_id: string;
  target_directory: string;
  target_path: string | null;
  state: string;
  bytes_total: number;
  bytes_transferred: number;
  files_total: number;
  files_transferred: number | null;
  speed_bytes_per_second: number;
  progress_pct: number;
  attempt: number;
  logs: string;
  error?: { code: string; message: string } | null;
  warnings?: PolicyWarning[];
  manifest_sha256?: string | null;
  portable_digest?: string | null;
  created_at: string;
  updated_at: string;
  finished_at?: string | null;
}

export const transferLabels: Record<string, string> = {
  QUEUED: '等待同步',
  PREPARING: '检查机器与目录',
  EXPORTING: '准备推理配置',
  TRANSFERRING: '正在同步权重',
  VERIFYING: '校验文件完整性',
  SUCCEEDED: '同步校验完成',
  FAILED: '同步失败',
  CANCELLED: '已取消同步',
  INTERRUPTED: '同步已中断',
  CANCELLING: '正在取消',
};

export const transferTerminal = (state: string) =>
  ['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(state);

export function normalizeTransferPath(path: string) {
  const value = path.trim().replace(/\/+/g, '/');
  return value.replace(/\/+$/, '') || (value.startsWith('/') ? '/' : '');
}

export function transferForId(id: string, ...receipts: (CheckpointTransfer | undefined)[]) {
  return id ? receipts.find((receipt) => receipt?.id === id) : undefined;
}

export function transferBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  const index = Math.min(3, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** index).toLocaleString('zh-CN', { maximumFractionDigits: 1 })} ${['B', 'KiB', 'MiB', 'GiB'][index]}`;
}

export function transferElapsed(createdAt: string, finishedAt?: string | null, now = Date.now()) {
  const start = Date.parse(createdAt);
  const end = finishedAt ? Date.parse(finishedAt) : now;
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return undefined;
  const seconds = Math.floor((end - start) / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours ? `${hours}小时` : ''}${hours || minutes ? `${minutes}分` : ''}${seconds % 60}秒`;
}

export function transferMatches(
  transfer: CheckpointTransfer | undefined,
  binding: {
    runId: string;
    checkpoint: string;
    nodeId: string;
    profileId: string;
    directory: string;
  },
) {
  return (
    !!transfer &&
    transfer.source_run_id === binding.runId &&
    normalizeTransferPath(transfer.source_checkpoint_path) ===
      normalizeTransferPath(binding.checkpoint) &&
    transfer.target_node_id === binding.nodeId &&
    transfer.target_profile_id === binding.profileId &&
    normalizeTransferPath(transfer.target_directory) === normalizeTransferPath(binding.directory)
  );
}
