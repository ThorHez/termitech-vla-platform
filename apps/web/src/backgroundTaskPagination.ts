import { ApiError, api } from './api';
import { buildBackgroundTasks, filterBackgroundTasks } from './backgroundTaskEntries';
import type { AssetTransfer } from './assetTransfer';
import type { Node, Run } from './types';
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
}

export interface BackgroundTaskQuery {
  page: number;
  type: BackgroundTaskType;
  status: BackgroundTaskStatus;
  query: string;
}

export const initialBackgroundTaskQuery: BackgroundTaskQuery = {
  page: 1,
  type: 'ALL',
  status: 'ALL',
  query: '',
};

export function backgroundTaskPagePath(value: BackgroundTaskQuery) {
  const query = new URLSearchParams({
    page: String(value.page),
    page_size: String(BACKGROUND_TASK_PAGE_SIZE),
    type: value.type,
    status: value.status,
    q: value.query.trim(),
  });
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
  runs: Run[],
  transfers: AssetTransfer[],
  nodes: Node[] = [],
): BackgroundTaskPage {
  const all = buildBackgroundTasks(runs, transfers);
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
  };
}

export async function fetchBackgroundTaskPage(
  request: BackgroundTaskQuery,
  nodes: Node[] = [],
  signal?: AbortSignal,
): Promise<BackgroundTaskPage> {
  try {
    return await api<BackgroundTaskPage>(backgroundTaskPagePath(request), { signal });
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) throw error;
  }
  const [runs, transfers] = await Promise.all([
    api<Run[]>('/runs', { signal }),
    api<AssetTransfer[]>('/asset-transfers?limit=200', { signal }),
  ]);
  return legacyBackgroundTaskPage(request, runs, transfers, nodes);
}
