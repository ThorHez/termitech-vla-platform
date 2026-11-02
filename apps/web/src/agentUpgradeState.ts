import type { RemoteDeployment } from './types';

export function agentReleaseLabel(
  version?: string | null,
  buildId?: string | null,
  fallback = '未上报',
) {
  return `${version || fallback}${buildId ? `（构建 ${buildId.slice(0, 12)}）` : ''}`;
}

export const nodeOperationComplete = (job: RemoteDeployment) =>
  ['SUCCEEDED', 'UNINSTALLED', 'DISCONNECTED', 'FAILED', 'INTERRUPTED'].includes(job.state);

export const nodeOperationSucceeded = (job: RemoteDeployment) =>
  ['SUCCEEDED', 'UNINSTALLED', 'DISCONNECTED'].includes(job.state);

export const nodeOperationFailed = (job: RemoteDeployment) =>
  ['FAILED', 'INTERRUPTED'].includes(job.state);

export function nodeOperationLabel(job: RemoteDeployment) {
  if (job.operation === 'upgrade')
    return job.state === 'SUCCEEDED'
      ? 'Agent 升级完成'
      : nodeOperationFailed(job)
        ? 'Agent 升级未完成'
        : '正在升级 Agent';
  if (job.state === 'DISCONNECTED') return '连接已断开';
  if (job.state === 'UNINSTALLED') return 'Agent 已卸载';
  return nodeOperationFailed(job) ? '卸载未完成' : '正在远程卸载 Agent';
}

// Ignore another machine's receipt rather than resuming the wrong operation.
export function latestNodeUpgrade(job: RemoteDeployment | null | undefined, nodeId: string) {
  return job?.node_id === nodeId && job.operation === 'upgrade' ? job : undefined;
}
