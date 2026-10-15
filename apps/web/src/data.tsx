import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';
import { useApi } from './api';
import type { Alert, Asset, Gpu, Node, Overview, Profile, Run, Template } from './types';

function useResources(enabled: boolean) {
  const overview = useApi<Overview>(enabled ? '/overview' : null);
  const nodes = useApi<Node[]>(enabled ? '/nodes' : null);
  const assets = useApi<Asset[]>(enabled ? '/assets' : null);
  const profiles = useApi<Profile[]>(enabled ? '/profiles' : null, 15000);
  const templates = useApi<Template[]>(enabled ? '/templates' : null, 30000);
  const runs = useApi<Run[]>(enabled ? '/runs' : null);
  const alerts = useApi<Alert[]>(enabled ? '/alerts' : null);
  const refresh = () => {
    overview.refresh();
    nodes.refresh();
    assets.refresh();
    profiles.refresh();
    templates.refresh();
    runs.refresh();
    alerts.refresh();
  };
  return { enabled, overview, nodes, assets, profiles, templates, runs, alerts, refresh };
}
const DataContext = createContext<ReturnType<typeof useResources> | null>(null);
export function DataProvider({ enabled, children }: { enabled: boolean; children: ReactNode }) {
  const resources = useResources(enabled);
  return <DataContext.Provider value={resources}>{children}</DataContext.Provider>;
}
export function useData() {
  const context = useContext(DataContext);
  if (!context) throw new Error('DataProvider missing');
  return context;
}
export const assetKinds: Record<string, string> = {
  DATASET: '数据集',
  MODEL: '模型',
  NORM: '归一化参数',
};
export const inputRoles: Record<string, string> = {
  dataset: '数据集',
  model: '基础模型',
  norm: '归一化参数',
};
export const activeStates = new Set(['QUEUED', 'STARTING', 'RUNNING', 'CANCELLING', 'UNKNOWN']);
export function gpuIdle(gpu: Gpu) {
  return (
    !gpu.reserved_by &&
    Array.isArray(gpu.processes) &&
    gpu.processes.length === 0 &&
    gpu.utilization_pct != null &&
    gpu.memory_used_mb != null &&
    gpu.memory_total_mb != null &&
    gpu.utilization_pct <= 5 &&
    gpu.memory_used_mb <= Math.max(128, gpu.memory_total_mb * 0.02)
  );
}
export function gpuAverage(gpus?: Gpu[]) {
  if (!gpus?.length || gpus.some((gpu) => gpu.utilization_pct == null)) return undefined;
  return gpus.reduce((sum, gpu) => sum + (gpu.utilization_pct as number), 0) / gpus.length;
}
export function requiredInputs(template?: Template) {
  return Array.isArray(template?.required_inputs)
    ? template.required_inputs
    : Object.keys(template?.required_inputs || {});
}
