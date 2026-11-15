import { ApiError, api } from './api';
import { buildBackgroundTasks, filterBackgroundTasks } from './backgroundTaskEntries';
import type { AssetTransfer } from './assetTransfer';
import type { Node, RunSummary } from './types';
import type {
  BackgroundTask,
  BackgroundTaskStatus,
  BackgroundTaskType,
} from './backgroundTaskEntries';

export const BACKGROUND_TASK_PAGE_SIZE = 20;

export interface BackgroundTaskPage {
  items: BackgroundTask[];
  total: number;
  total_all: number;
  page: number;
  page_size: number;
  total_pages: number;
  history_limited?: boolean;
  node_id?: string;
}

export interface BackgroundTaskQuery {
  page: number;
  type: BackgroundTaskType;
  status: BackgroundTaskStatus;
  query: string;
  nodeId?: string;
}

export const initialBackgroundTaskQuery: BackgroundTaskQuery = {
  page: 1,
  type: 'ALL',
  status: 'ALL',
  query: '',
};

export function backgroundTaskQueryFromSearch(search: string): BackgroundTaskQuery {
  const params = new URLSearchParams(search);
  const nodeId = params.get('node') || undefined;
  const type = params.get('type');
  const status = params.get('status');
  const page = Number(params.get('page'));
  return {
    page: Number.isSafeInteger(page) && page > 0 ? page : 1,
    type: nodeId || type === 'training' ? 'training' : type === 'data-sync' ? 'data-sync' : 'ALL',
    status:
      status === 'ACTIVE' || status === 'SUCCEEDED' || status === 'FAILED' || status === 'CANCELLED'
        ? status
        : 'ALL',
    query: (params.get('q') || '').slice(0, 200),
    nodeId,
  };
}

export function backgroundTaskListPath(value: BackgroundTaskQuery) {
  const query = new URLSearchParams();
  if (value.nodeId) query.set('node', value.nodeId);
  if (value.type !== 'ALL') query.set('type', value.type);
  if (value.status !== 'ALL') query.set('status', value.status);
  if (value.query.trim()) query.set('q', value.query.trim());
  if (value.page > 1) query.set('page', String(value.page));
  return query.size ? `/runs?${query}` : '/runs';
}

export function backgroundTaskPagePath(value: BackgroundTaskQuery) {
  const query = new URLSearchParams({
    view: 'summary',
    page: String(value.page),
    page_size: String(BACKGROUND_TASK_PAGE_SIZE),
    type: value.type,
    status: value.status,
    q: value.query.trim(),
  });
  if (value.nodeId) query.set('node_id', value.nodeId);
  return `/background-tasks?${query}`;
}

type QueryAction =
  | { kind: 'page'; page: number }
  | { kind: 'type'; value: BackgroundTaskType }
  | { kind: 'status'; value: BackgroundTaskStatus }
  | { kind: 'query'; value: string }
  | { kind: 'receipt'; request: BackgroundTaskQuery; page: number };

export function backgroundTaskQueryReducer(
  state: BackgroundTaskQuery,
  action: QueryAction,
): BackgroundTaskQuery {
  if (action.kind === 'receipt') {
    if (backgroundTaskPagePath(state) !== backgroundTaskPagePath(action.request)) return state;
    return action.page === state.page ? state : { ...state, page: action.page };
  }
  if (action.kind === 'page')
    return Number.isSafeInteger(action.page) && action.page > 0
      ? { ...state, page: action.page }
      : state;
  return { ...state, [action.kind]: action.value, page: 1 };
}

export function backgroundTaskPageRange(
  value: Pick<BackgroundTaskPage, 'page' | 'page_size' | 'total'>,
) {
  if (!value.total) return { start: 0, end: 0 };
  return {
    start: (value.page - 1) * value.page_size + 1,
    end: Math.min(value.page * value.page_size, value.total),
  };
}
export function legacyBackgroundTaskPage(
  request: BackgroundTaskQuery,
  runs: RunSummary[],
  transfers: AssetTransfer[],
  nodes: Node[] = [],
): BackgroundTaskPage {
  const all = request.nodeId
    ? buildBackgroundTasks(runs.filter((run) => run.node_id === request.nodeId))
    : buildBackgroundTasks(runs, transfers);
  const filtered = filterBackgroundTasks(all, {
    type: request.type,
    status: request.status,
    query: request.query,
    nodes,
  });
  const totalPages = Math.max(1, Math.ceil(filtered.length / BACKGROUND_TASK_PAGE_SIZE));
  const page = Math.max(1, Math.min(request.page, totalPages));
  const start = (page - 1) * BACKGROUND_TASK_PAGE_SIZE;
  return {
    items: filtered.slice(start, start + BACKGROUND_TASK_PAGE_SIZE),
    total: filtered.length,
    total_all: all.length,
    page,
    page_size: BACKGROUND_TASK_PAGE_SIZE,
    total_pages: totalPages,
    history_limited: true,
    ...(request.nodeId ? { node_id: request.nodeId } : {}),
  };
}

export async function fetchBackgroundTaskPage(
  request: BackgroundTaskQuery,
  nodes: Node[] = [],
  signal?: AbortSignal,
): Promise<BackgroundTaskPage> {
  try {
    const page = await api<BackgroundTaskPage>(backgroundTaskPagePath(request), { signal });
    if (
      !request.nodeId ||
      (page.node_id === request.nodeId &&
        Array.isArray(page.items) &&
        page.items.every(
          (task) => task.kind === 'training' && task.item.node_id === request.nodeId,
        ))
    )
      return page;
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) throw error;
  }
  if (request.nodeId) {
    const runs = await api<RunSummary[]>('/runs?view=summary', { signal });
    return legacyBackgroundTaskPage(request, runs, [], nodes);
  }
  const [runs, transfers] = await Promise.all([
    api<RunSummary[]>('/runs?view=summary', { signal }),
    api<AssetTransfer[]>('/asset-transfers?limit=200', { signal }),
  ]);
  return legacyBackgroundTaskPage(request, runs, transfers, nodes);
}
