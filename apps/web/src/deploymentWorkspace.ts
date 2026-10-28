import type { DeploymentRecordSummary } from './policyDeployment';
import { deploymentLabels } from './policyDeployment';

export interface DeploymentWorkspaceEntry {
  key: string;
  record: DeploymentRecordSummary;
}

export const deploymentRecordLabels: Record<string, string> = {
  ...deploymentLabels,
  DRAFT: '待配置',
  PREPARING: '准备中',
  READY_TO_DEPLOY: '待配置启动',
  CANCELLED: '同步已取消',
  INTERRUPTED: '同步已中断',
};

export type DeploymentWorkspaceRoute =
  | { kind: 'list' }
  | { kind: 'deployment'; id: string }
  | { kind: 'preparation'; id: string }
  | { kind: 'not-found' };

export function buildDeploymentWorkspaceEntries(
  records: DeploymentRecordSummary[],
): DeploymentWorkspaceEntry[] {
  const seen = new Set<string>();
  return records
    .filter((record) => {
      if (seen.has(record.id)) return false;
      seen.add(record.id);
      return true;
    })
    .map((record, index) => ({
      key: `record:${record.id}`,
      record,
      index,
      timestamp: Number.isFinite(Date.parse(record.created_at))
        ? Date.parse(record.created_at)
        : -Infinity,
    }))
    .sort((left, right) =>
      right.timestamp === left.timestamp
        ? left.index - right.index
        : right.timestamp > left.timestamp
          ? 1
          : -1,
    )
    .map(({ key, record }) => ({ key, record }));
}

function decodeId(segment: string): string | undefined {
  try {
    const id = decodeURIComponent(segment);
    return /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(id) ? id : undefined;
  } catch {
    return undefined;
  }
}

export function parseDeploymentWorkspaceRoute(path: string): DeploymentWorkspaceRoute {
  if (path === '/deployments') return { kind: 'list' };
  if (typeof path !== 'string') return { kind: 'not-found' };
  const preparation = /^\/deployments\/preparations\/([^/]+)$/.exec(path);
  if (preparation) {
    const id = decodeId(preparation[1]);
    return id ? { kind: 'preparation', id } : { kind: 'not-found' };
  }
  const deployment = /^\/deployments\/([^/]+)$/.exec(path);
  if (deployment) {
    const id = decodeId(deployment[1]);
    if (id && id !== 'preparations') return { kind: 'deployment', id };
  }
  return { kind: 'not-found' };
}

export function deploymentRecordPath(id: string): string {
  return `/deployments/${encodeURIComponent(id)}`;
}

export function deploymentPreparationPath(id: string): string {
  return `/deployments/preparations/${encodeURIComponent(id)}`;
}
