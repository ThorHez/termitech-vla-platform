import { useCallback, useEffect, useRef, useState } from 'react';

const LEGACY_TOKEN_KEY = 'termitech.admin-token';
const SESSION_EXPIRED = 'termitech:session-expired';
let csrfToken: string | undefined;
let sessionGeneration = 0;
let reauthenticate: ((signal?: AbortSignal | null) => Promise<void>) | undefined;

// Human sessions live in HttpOnly cookies. Only the CSRF value is kept in memory.
export function setSessionCsrf(token?: string) {
  if (csrfToken !== token) sessionGeneration += 1;
  csrfToken = token;
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
    if (response.ok) return data as T;
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
export function useRunFeed<T extends { seq: number }>(path: string, kind: 'events' | 'logs') {
  const [data, setData] = useState<T[]>([]);
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    let active = true;
    let cursor = 0;
    let rows: T[] = [];
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setData([]);
    setLoading(true);
    const poll = async () => {
      let nextDelay = 3000;
      try {
        const result = await api<T[] | { items: T[]; next_seq: number }>(
          `${path}?after_seq=${cursor}&limit=1000`,
          { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]) },
        );
        const batch = kind === 'logs' ? (result as { items: T[] }).items : (result as T[]);
        if (batch.length) {
          cursor = Math.max(cursor, ...batch.map((item) => item.seq));
          rows = [...rows, ...batch].slice(-5000);
        }
        if (active) {
          setData(rows);
          setError(undefined);
        }
        if (batch.length === 1000) nextDelay = 50;
      } catch (err) {
        if (active) setError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        if (active) {
          setLoading(false);
          timer = setTimeout(poll, nextDelay);
        }
      }
    };
    void poll();
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [path, kind, revision]);
  return { data, error, loading, refresh };
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
