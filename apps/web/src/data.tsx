import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { useApi, usePageVisible } from './api';
import { deduplicateDiskAlerts } from './workspaceAlertData';
import type { Alert, Asset, Gpu, Node, Overview, Profile, RunSummary, Template } from './types';
import type { MemberSummary } from './resourcePermissions';

type DataResource =
  | 'members'
  | 'overview'
  | 'nodes'
  | 'assets'
  | 'profiles'
  | 'templates'
  | 'runs'
  | 'alerts';
type DataRequests = Readonly<Partial<Record<DataResource, number>>>;

const pageRequests: Record<string, DataRequests> = {
  '/overview': { overview: 15000, nodes: 15000, runs: 15000, alerts: 10000 },
  '/alerts': { alerts: 10000, nodes: 0 },
  '/nodes': { nodes: 15000 },
  '/assets': { assets: 30000, nodes: 15000, profiles: 0, runs: 0, members: 0 },
  '/templates': { templates: 30000 },
  '/runs': { nodes: 0, templates: 0, members: 0 },
  '/deployments': { nodes: 0, members: 0 },
};

function requestsForRoute(route: string): DataRequests {
  const [path, query] = route.split('?', 2);
  if (path.startsWith('/nodes/')) return { templates: 0, alerts: 10000 };
  if (path.startsWith('/runs/sync/')) return { nodes: 0, members: 0 };
  if (path.startsWith('/runs/')) return { nodes: 0, templates: 0, members: 0 };
  if (path.startsWith('/deployments/')) return { nodes: 0, profiles: 0, members: 0 };
  if (path === '/scenes' && new URLSearchParams(query).get('scene')) return { assets: 0, runs: 0 };
  return pageRequests[path] ?? {};
}

function useResource<T>(endpoint: string, interval?: number) {
  const active = interval !== undefined;
  const result = useApi<T>(active ? endpoint : null, interval ?? 0);
  return {
    ...result,
    loading: active && (result.loading || (result.data === undefined && !result.error)),
  };
}

function useResources(enabled: boolean, route: string) {
  const visible = usePageVisible();
  const pollingEnabled = enabled && visible;
  const [extraRequests, setExtraRequests] = useState<DataRequests[]>([]);
  const request = useCallback((scope: DataRequests) => {
    const entry = { ...scope };
    setExtraRequests((current) => [...current, entry]);
    return () => setExtraRequests((current) => current.filter((value) => value !== entry));
  }, []);
  const scopes = [requestsForRoute(route), ...extraRequests];
  const intervalFor = (resource: DataResource) => {
    if (!pollingEnabled) return undefined;
    const intervals = scopes.flatMap((scope) => scope[resource] ?? []);
    const repeating = intervals.filter((value) => value > 0);
    return repeating.length ? Math.min(...repeating) : intervals[0];
  };
  const members = useResource<MemberSummary[]>('/members', intervalFor('members'));
  const overview = useResource<Overview>('/overview', intervalFor('overview'));
  const nodes = useResource<Node[]>('/nodes', intervalFor('nodes'));
  const assets = useResource<Asset[]>('/assets?view=summary', intervalFor('assets'));
  const profiles = useResource<Profile[]>('/profiles', intervalFor('profiles'));
  const templates = useResource<Template[]>('/templates', intervalFor('templates'));
  const runs = useResource<RunSummary[]>('/runs?view=summary', intervalFor('runs'));
  const alerts = useResource<Alert[]>('/alerts', intervalFor('alerts'));
  const currentAlerts = useMemo(
    () => (alerts.data ? deduplicateDiskAlerts(alerts.data) : undefined),
    [alerts.data],
  );
  const refresh = () => {
    members.refresh();
    overview.refresh();
    nodes.refresh();
    assets.refresh();
    profiles.refresh();
    templates.refresh();
    runs.refresh();
    alerts.refresh();
  };
  return {
    enabled,
    pollingEnabled,
    members,
    overview,
    nodes,
    assets,
    profiles,
    templates,
    runs,
    alerts: { ...alerts, data: currentAlerts },
    refresh,
    request,
  };
}
const DataContext = createContext<ReturnType<typeof useResources> | null>(null);
export function DataProvider({
  enabled,
  route = '/overview',
  children,
}: {
  enabled: boolean;
  route?: string;
  children: ReactNode;
}) {
  const resources = useResources(enabled, route);
  return <DataContext.Provider value={resources}>{children}</DataContext.Provider>;
}
export function useData() {
  const context = useContext(DataContext);
  if (!context) throw new Error('DataProvider missing');
  return context;
}
export function useMembers() {
  return useContext(DataContext)?.members;
}
export function useDataRequests(requests?: DataRequests) {
  const request = useContext(DataContext)?.request;
  useEffect(() => (requests ? request?.(requests) : undefined), [request, requests]);
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
    !gpu.user_reservation &&
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
export function requiredInputs(template?: Pick<Template, 'required_inputs'>) {
  return Array.isArray(template?.required_inputs)
    ? template.required_inputs
    : Object.keys(template?.required_inputs || {});
}
