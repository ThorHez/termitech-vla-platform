import { useCallback, useEffect, useRef, useState } from 'react';
import { backgroundTaskPagePath, fetchBackgroundTaskPage } from './backgroundTaskPagination';
import type { BackgroundTaskPage, BackgroundTaskQuery } from './backgroundTaskPagination';
import type { Node } from './types';

export function useBackgroundTaskPage(
  request: BackgroundTaskQuery,
  enabled: boolean,
  nodes: Node[],
) {
  const [data, setData] = useState<BackgroundTaskPage>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(enabled);
  const [revision, setRevision] = useState(0);
  const nodesRef = useRef(nodes);
  const requestRef = useRef(request);
  nodesRef.current = nodes;
  requestRef.current = request;
  const path = enabled ? backgroundTaskPagePath(request) : null;
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    if (!path) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setLoading(true);
    const fetchPage = async () => {
      try {
        const page = await fetchBackgroundTaskPage(
          requestRef.current,
          nodesRef.current,
          AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
        );
        if (!cancelled) {
          setData(page);
          setError(undefined);
        }
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause : new Error(String(cause)));
      } finally {
        if (!cancelled) {
          setLoading(false);
          timer = setTimeout(fetchPage, 3000);
        }
      }
    };
    void fetchPage();
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [path, revision]);
  return { data, error, loading, refresh };
}
