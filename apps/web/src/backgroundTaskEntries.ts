import type { AssetTransfer } from './assetTransfer';
import { assetTransferLabels } from './assetTransfer';
import type { Node, Run } from './types';
import { transferBytes } from './checkpointTransfer';

export type BackgroundTask =
  | { key: string; kind: 'training'; item: Run }
  | { key: string; kind: 'data-sync'; item: AssetTransfer };

export type BackgroundTaskType = 'ALL' | BackgroundTask['kind'];
export type BackgroundTaskStatus = 'ALL' | 'ACTIVE' | 'SUCCEEDED' | 'FAILED' | 'CANCELLED';

const activeTraining = new Set(['QUEUED', 'STARTING', 'RUNNING', 'CANCELLING', 'UNKNOWN']);
const activeSync = new Set([
  'QUEUED',
  'PREPARING',
  'SCANNING',
  'TRANSFERRING',
  'VERIFYING',
  'CANCELLING',
]);

export function backgroundTaskPath(task: BackgroundTask) {
  return task.kind === 'training'
    ? `/runs/${encodeURIComponent(task.item.id)}`
    : `/runs/sync/${encodeURIComponent(task.item.id)}`;
}

export function buildBackgroundTasks(runs: Run[] = [], transfers: AssetTransfer[] = []) {
  const items: BackgroundTask[] = [
    ...runs.map((item): BackgroundTask => ({ key: `training:${item.id}`, kind: 'training', item })),
    ...transfers.map(
      (item): BackgroundTask => ({ key: `data-sync:${item.id}`, kind: 'data-sync', item }),
    ),
  ];
  const seen = new Set<string>();
  return items
    .filter((task) => {
      if (seen.has(task.key)) return false;
      seen.add(task.key);
      return true;
    })
    .map((task, index) => ({ task, index, timestamp: Date.parse(task.item.created_at) }))
    .sort((left, right) => {
      const a = Number.isFinite(left.timestamp) ? left.timestamp : -Infinity;
      const b = Number.isFinite(right.timestamp) ? right.timestamp : -Infinity;
      return a === b ? left.index - right.index : b > a ? 1 : -1;
    })
    .map(({ task }) => task);
}

export function backgroundTaskActive(task: BackgroundTask) {
  return (task.kind === 'training' ? activeTraining : activeSync).has(task.item.state);
}

export function backgroundTaskActiveCounts(runs: Run[] = [], transfers: AssetTransfer[] = []) {
  const counts = { training: 0, sync: 0, total: 0 };
  for (const task of buildBackgroundTasks(runs, transfers)) {
    if (!backgroundTaskActive(task)) continue;
    if (task.kind === 'training') counts.training += 1;
    else counts.sync += 1;
    counts.total += 1;
  }
  return counts;
}

export function filterBackgroundTasks(
  tasks: BackgroundTask[],
  {
    type = 'ALL',
    status = 'ALL',
    query = '',
    nodes = [],
  }: {
    type?: BackgroundTaskType;
    status?: BackgroundTaskStatus;
    query?: string;
    nodes?: Node[];
  } = {},
) {
  const search = query.trim().toLocaleLowerCase();
  const machine = (id: string) => {
    const node = nodes.find((item) => item.id === id);
    return `${id} ${node?.name || ''} ${node?.hostname || ''}`;
  };
  return tasks.filter((task) => {
    if (type !== 'ALL' && task.kind !== type) return false;
    if (status === 'ACTIVE' && !backgroundTaskActive(task)) return false;
    if (status === 'FAILED' && !['FAILED', 'INTERRUPTED'].includes(task.item.state)) return false;
    if (['SUCCEEDED', 'CANCELLED'].includes(status) && task.item.state !== status) return false;
    if (!search) return true;
    const item = task.item;
    const content =
      task.kind === 'training'
        ? `${item.name} ${item.id} ${machine(task.item.node_id)} 模型训练`
        : `${item.name} ${item.id} ${machine(task.item.source_node_id)} ${machine(task.item.target_node_id)} ${task.item.source_path} ${task.item.target_path || task.item.target_directory} 数据同步`;
    return content.toLocaleLowerCase().includes(search);
  });
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

export interface BackgroundTaskProgress {
  percent: number | null;
  preparing: boolean;
  label: string;
  detail?: string;
}

export function backgroundTaskProgress(task: BackgroundTask): BackgroundTaskProgress {
  if (task.kind === 'training') {
    const attempt = task.item.attempts?.at(-1);
    const step = attempt?.latest_metrics?.step;
    const total = attempt?.latest_metrics?.total_steps ?? task.item.parameters?.steps;
    const known = finite(step) && step >= 0 && finite(total) && total > 0;
    return {
      percent:
        task.item.state === 'SUCCEEDED' ? 100 : known ? Math.min(99, (step / total) * 100) : null,
      preparing: false,
      label: known
        ? `${step.toLocaleString('zh-CN')} / ${total.toLocaleString('zh-CN')} 步`
        : '暂无步数上报',
    };
  }
  const item = task.item;
  const preparing = ['QUEUED', 'PREPARING', 'SCANNING'].includes(item.state);
  if (preparing)
    return {
      percent: null,
      preparing: true,
      label: assetTransferLabels[item.state],
      detail: '文件总量待统计',
    };
  const measured = finite(item.progress_pct)
    ? item.progress_pct
    : finite(item.bytes_total) &&
        item.bytes_total > 0 &&
        finite(item.bytes_transferred) &&
        item.bytes_transferred >= 0
      ? (item.bytes_transferred / item.bytes_total) * 100
      : null;
  const percent =
    item.state === 'SUCCEEDED'
      ? 100
      : measured === null
        ? null
        : Math.max(0, Math.min(99, measured));
  const amount =
    finite(item.bytes_total) && item.bytes_total > 0
      ? `${transferBytes(item.bytes_transferred)} / ${transferBytes(item.bytes_total)}`
      : '文件总量待统计';
  const speed =
    backgroundTaskActive(task) &&
    finite(item.speed_bytes_per_second) &&
    item.speed_bytes_per_second > 0
      ? ` · ${transferBytes(item.speed_bytes_per_second)}/s`
      : '';
  return {
    percent,
    preparing: false,
    label: percent === null ? '进度待更新' : `${percent.toFixed(percent === 100 ? 0 : 1)}%`,
    detail: amount + speed,
  };
}
