import { useCallback, useEffect, useRef, useState } from 'react';
import { appendTrainingMetricPage } from './trainingMetricFeed';
import type { TrainingMetricFeed, TrainingMetricPage } from './trainingMetricFeed';
import type { RunEvent } from './types';

const LEGACY_TOKEN_KEY = 'termitech.admin-token';
const SESSION_EXPIRED = 'termitech:session-expired';
let csrfToken: string | undefined;
let operationUser: string | undefined;
let sessionGeneration = 0;
let reauthenticate: ((signal?: AbortSignal | null) => Promise<void>) | undefined;

// Human sessions live in HttpOnly cookies. Only the CSRF value is kept in memory.
export function setSessionCsrf(token?: string, userId?: string) {
  if (csrfToken !== token || operationUser !== userId) sessionGeneration += 1;
  csrfToken = token;
  operationUser = userId;
}

/** Bind multi-step work to the account that started it, including after a wait. */
export function createOperationClient() {
  const generation = sessionGeneration;
  const requireCurrent = () => {
    if (generation !== sessionGeneration)
      throw new ApiError('SESSION_CHANGED', '登录状态已改变，请重新执行此操作。', 401);
  };
  return {
    requireCurrent,
    async api<T>(path: string, options: RequestInit = {}) {
      requireCurrent();
      const result = await api<T>(path, options);
      requireCurrent();
      return result;
    },
    post<T>(path: string, body: unknown = {}, options: RequestInit = {}) {
      return this.api<T>(path, { ...options, method: 'POST', body: JSON.stringify(body) });
    },
  };
}
export function discardLegacyToken() {
  try {
    window.sessionStorage.removeItem(LEGACY_TOKEN_KEY);
  } catch {
    // Storage can be disabled; authentication does not depend on browser storage.
  }
}
export function subscribeSessionExpiry(listener: () => void) {
  window.addEventListener(SESSION_EXPIRED, listener);
  return () => window.removeEventListener(SESSION_EXPIRED, listener);
}
export function notifySessionExpired() {
  setSessionCsrf();
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(SESSION_EXPIRED));
}
export function registerReauthentication(handler: (signal?: AbortSignal | null) => Promise<void>) {
  reauthenticate = handler;
  return () => {
    if (reauthenticate === handler) reauthenticate = undefined;
  };
}

export async function authenticatedFetch(path: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers);
  headers.set('Accept', 'application/json');
  headers.delete('Authorization');
  const method = (options.method || 'GET').toUpperCase();
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method) && csrfToken) {
    headers.set('X-CSRF-Token', csrfToken);
    if (operationUser) headers.set('X-Operation-User', operationUser);
  }
  if (options.body) headers.set('Content-Type', 'application/json');
  const generation = sessionGeneration;
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      credentials: 'same-origin',
      headers,
      signal: options.signal ?? AbortSignal.timeout(20000),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new ApiError('NETWORK_ERROR', '无法连接管理服务，请检查服务是否启动及网络连接。', 0);
  }
  if (
    response.status === 401 &&
    path !== '/auth/login' &&
    path !== '/auth/me' &&
    generation === sessionGeneration
  ) {
    notifySessionExpired();
  }
  return response;
}

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  options.signal?.throwIfAborted();
  const generation = sessionGeneration;
  // The server rejects the first sensitive request before applying any change.
  // Retry it once after password confirmation, only in the same browser session.
  for (let attempt = 0; ; attempt += 1) {
    const response = await authenticatedFetch(path, options);
    const body = await response.text();
    let data: unknown;
    try {
      data = body ? JSON.parse(body) : null;
    } catch {
      throw new ApiError(
        'INVALID_RESPONSE',
        '服务返回了无效响应，请检查 API 代理配置。',
        response.status,
      );
    }
    if (response.ok) {
      options.signal?.throwIfAborted();
      if (generation !== sessionGeneration)
        throw new ApiError('SESSION_CHANGED', '登录状态已改变，请重新执行此操作。', 401);
      return data as T;
    }
    const detail = data as { error?: { code?: string; message?: string }; detail?: string } | null;
    const code = detail?.error?.code || `HTTP_${response.status}`;
    const handler = reauthenticate;
    if (
      attempt === 0 &&
      response.status === 403 &&
      code === 'REAUTH_REQUIRED' &&
      !path.startsWith('/auth/') &&
      !['GET', 'HEAD', 'OPTIONS'].includes((options.method || 'GET').toUpperCase()) &&
      handler &&
      csrfToken &&
      generation === sessionGeneration
    ) {
      options.signal?.throwIfAborted();
      await handler(options.signal);
      options.signal?.throwIfAborted();
      if (generation !== sessionGeneration || !csrfToken) {
        throw new ApiError('SESSION_CHANGED', '登录状态已改变，请重新执行此操作。', 401);
      }
      continue;
    }
    throw new ApiError(
      code,
      detail?.error?.message ||
        detail?.detail ||
        (response.status === 403
          ? '当前账号没有执行此操作的权限。'
          : response.status === 401
            ? '登录已失效，请重新登录。'
            : `请求失败（${response.status}）`),
      response.status,
    );
  }
}

export const post = <T>(path: string, body: unknown = {}, headers?: Record<string, string>) =>
  api<T>(path, { method: 'POST', body: JSON.stringify(body), headers });

export function useApi<T>(path: string | null, interval = 8000) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<Date>();
  const [revision, setRevision] = useState(0);
  const lastPath = useRef(path);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    if (!path) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    if (lastPath.current !== path) {
      setData(undefined);
      lastPath.current = path;
    }
    setLoading(true);
    const fetchData = async () => {
      try {
        const result = await api<T>(path, {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
        });
        if (!cancelled) {
          setData(result);
          setError(undefined);
          setUpdatedAt(new Date());
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        if (!cancelled) {
          setLoading(false);
          if (interval > 0) timer = setTimeout(fetchData, interval);
        }
      }
    };
    void fetchData();
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [path, interval, revision]);
  return { data, error, loading, updatedAt, refresh };
}

export function navigate(path: string) {
  window.location.hash = path;
}

// Retain a bounded local tail while advancing the server cursor. Re-fetching the
// first page would freeze live logs as soon as an execution emits 1,000 events.
export function useRunFeed<T extends { seq: number }>(
  path: string,
  kind: 'events' | 'logs',
  live = true,
) {
  const [data, setData] = useState<T[]>([]);
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const [complete, setComplete] = useState(false);
  const [revision, setRevision] = useState(0);
  const liveRef = useRef(live);
  const previousLive = useRef(live);
  const wake = useRef<(() => void) | undefined>(undefined);
  liveRef.current = live;
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    let active = true;
    let cursor = 0;
    let rows: T[] = [];
    let inFlight = false;
    let requested = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setData([]);
    setLoading(true);
    setComplete(false);
    const poll = async () => {
      if (!active || inFlight) return;
      inFlight = true;
      requested = false;
      let fullPage = false;
      let failed = false;
      try {
        const result = await api<T[] | { items: T[]; next_seq: number }>(
          `${path}?after_seq=${cursor}&limit=1000`,
          { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]) },
        );
        if (!active) return;
        const batch = kind === 'logs' ? (result as { items: T[] }).items : (result as T[]);
        fullPage = batch.length === 1000;
        if (batch.length) {
          cursor = Math.max(cursor, ...batch.map((item) => item.seq));
          rows = [...rows, ...batch].slice(-5000);
          setData(rows);
        }
        if (active) {
          setError(undefined);
          setComplete(!liveRef.current && !fullPage && !requested);
        }
      } catch (err) {
        failed = true;
        if (active) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setComplete(false);
        }
      } finally {
        inFlight = false;
        if (active) {
          setLoading(false);
          if (requested || fullPage || failed || liveRef.current)
            timer = setTimeout(poll, requested ? 0 : fullPage ? 50 : 3000);
        }
      }
    };
    wake.current = () => {
      requested = true;
      setComplete(false);
      if (!inFlight) {
        clearTimeout(timer);
        void poll();
      }
    };
    void poll();
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
      wake.current = undefined;
    };
  }, [path, kind, revision]);
  useEffect(() => {
    if (previousLive.current === live) return;
    previousLive.current = live;
    // A final read after the terminal transition also covers an older request
    // that started before the server committed the last logs and terminal event.
    wake.current?.();
  }, [live]);
  return { data, error, loading, complete, refresh };
}
export function useRunMetrics(runId: string, attemptId: string | undefined, live = true) {
  const path = attemptId
    ? `/runs/${encodeURIComponent(runId)}/metrics?attempt_id=${encodeURIComponent(attemptId)}`
    : null;
  const [state, setState] = useState<{
    path: string | null;
    data: RunEvent[];
    loading: boolean;
    complete: boolean;
    totalCount: number;
    error?: Error;
  }>({ path: null, data: [], loading: false, complete: false, totalCount: 0 });
  const [revision, setRevision] = useState(0);
  const liveRef = useRef(live);
  const previousLive = useRef(live);
  const wake = useRef<(() => void) | undefined>(undefined);
  liveRef.current = live;
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    if (!path || !attemptId) return;
    let active = true;
    let feed: TrainingMetricFeed = { items: [], cursor: 0, totalCount: 0 };
    let recoveringHistory = false;
    let inFlight = false;
    let requested = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setState({ path, data: [], loading: true, complete: false, totalCount: 0 });
    const poll = async () => {
      if (!active || inFlight) return;
      inFlight = true;
      requested = false;
      let more = false;
      let failed = false;
      try {
        const page = await api<TrainingMetricPage>(`${path}&after_seq=${feed.cursor}&limit=1000`, {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
        });
        if (!active) return;
        feed = appendTrainingMetricPage(feed, page, attemptId);
        more = page.has_more;
        if (!more && feed.items.length !== feed.totalCount) {
          // A late event can be inserted before the acknowledged cursor. A
          // fresh metric-only scan repairs that gap without touching log state.
          if (recoveringHistory) throw new Error('指标历史正在变化，请重新读取完整记录。');
          recoveringHistory = true;
          feed = { items: [], cursor: 0, totalCount: page.total_count };
          more = true;
          setState((current) => ({ ...current, complete: false, totalCount: page.total_count }));
        } else {
          if (!more) recoveringHistory = false;
          setState({
            path,
            data: feed.items,
            loading: false,
            complete: !more && !requested,
            totalCount: feed.totalCount,
          });
        }
      } catch (error) {
        failed = true;
        if (recoveringHistory) {
          feed = { items: [], cursor: 0, totalCount: feed.totalCount };
          recoveringHistory = false;
        }
        if (active)
          setState((current) => ({
            ...current,
            loading: false,
            complete: false,
            error: error instanceof Error ? error : new Error(String(error)),
          }));
      } finally {
        inFlight = false;
        if (active && (requested || more || failed || liveRef.current)) {
          timer = setTimeout(poll, requested ? 0 : more ? 50 : 3000);
        }
      }
    };
    wake.current = () => {
      requested = true;
      setState((current) => ({ ...current, complete: false }));
      if (!inFlight) {
        clearTimeout(timer);
        void poll();
      }
    };
    void poll();
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
      wake.current = undefined;
    };
  }, [path, attemptId, revision]);
  useEffect(() => {
    if (previousLive.current === live) return;
    previousLive.current = live;
    wake.current?.();
  }, [live]);
  // A route/attempt change must never render the previous execution's data,
  // including the render before this effect clears its state.
  return path && state.path === path
    ? { ...state, refresh }
    : {
        data: [] as RunEvent[],
        loading: Boolean(path),
        complete: !path,
        totalCount: 0,
        error: undefined,
        refresh,
      };
}

export function useRoute() {
  const [route, setRoute] = useState(window.location.hash.slice(1) || '/overview');
  useEffect(() => {
    const handler = () => setRoute(window.location.hash.slice(1) || '/overview');
    window.addEventListener('hashchange', handler);
    return () => window.removeEventListener('hashchange', handler);
  }, []);
  return route;
}

export function date(value?: string | null, time = true) {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    ...(time ? ({ hour: '2-digit', minute: '2-digit' } as const) : {}),
  });
}
export function displayNumber(value?: number | null, suffix = '') {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${Math.round(value * 10) / 10}${suffix}`
    : '—';
}
