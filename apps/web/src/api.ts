import { useCallback, useEffect, useRef, useState } from 'react';

const TOKEN_KEY = 'termitech.admin-token';
export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY) || '';
}
export function saveToken(token: string) {
  sessionStorage.setItem(TOKEN_KEY, token.trim());
}
export function clearToken() {
  sessionStorage.removeItem(TOKEN_KEY);
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
  const headers = new Headers(options.headers);
  headers.set('Accept', 'application/json');
  headers.set('Authorization', `Bearer ${getToken()}`);
  if (options.body) headers.set('Content-Type', 'application/json');
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      headers,
      signal: options.signal ?? AbortSignal.timeout(20000),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new ApiError('NETWORK_ERROR', '无法连接管理服务，请检查服务是否启动及网络连接。', 0);
  }
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
  if (!response.ok) {
    const detail = data as { error?: { code?: string; message?: string }; detail?: string } | null;
    throw new ApiError(
      detail?.error?.code || `HTTP_${response.status}`,
      detail?.error?.message || detail?.detail || `请求失败（${response.status}）`,
      response.status,
    );
  }
  return data as T;
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
