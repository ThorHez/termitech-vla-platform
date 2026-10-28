import { useEffect, useState } from 'react';
import { api } from './api';
import {
  canonicalDatasetInspectionPath,
  inspectionMatches,
  type DatasetInspectionDTO,
  type DatasetInspectionQuery,
} from './datasetInspectionHelpers';
import type { Asset, Node } from './types';

const pause = (signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, 500);
    signal.addEventListener('abort', abort, { once: true });
  });

/** Source metadata stays local to this training binding; registration is unchanged. */
export function useDatasetImageMetadata(node?: Node, dataset?: Asset, enabled = true) {
  const path = canonicalDatasetInspectionPath(dataset?.path || '');
  const scope = `${node?.id || ''}\0${dataset?.id || ''}\0${path}`;
  const ready = Boolean(
    enabled && node?.status === 'ONLINE' && dataset?.node_id === node?.id && path,
  );
  const [revision, setRevision] = useState(0);
  const [view, setView] = useState<{
    scope: string;
    revision: number;
    loading: boolean;
    inspection?: DatasetInspectionDTO;
    error?: string;
  }>();
  useEffect(() => {
    if (!ready || !node) return;
    const controller = new AbortController();
    const { signal } = controller;
    setView({ scope, revision, loading: true });
    const timer = setTimeout(() => {
      void (async () => {
        try {
          let result = await api<DatasetInspectionQuery>(
            `/nodes/${encodeURIComponent(node.id)}/dataset-inspections`,
            {
              method: 'POST',
              body: JSON.stringify({ path }),
              signal: AbortSignal.any([signal, AbortSignal.timeout(20000)]),
            },
          );
          if (signal.aborted) return;
          if (!inspectionMatches(result, node.id, path))
            throw new Error('图像元信息与当前机器或数据集路径不一致，请重新读取。');
          const queryId = result.id;
          const deadline = Date.now() + 65000;
          while (result.status === 'PENDING') {
            if (Date.now() > deadline) throw new Error('读取图像元信息超时，请重试。');
            await pause(signal);
            if (signal.aborted) return;
            result = await api<DatasetInspectionQuery>(
              `/nodes/${encodeURIComponent(node.id)}/dataset-inspections/${encodeURIComponent(queryId)}`,
              { signal: AbortSignal.any([signal, AbortSignal.timeout(20000)]) },
            );
            if (signal.aborted) return;
            if (!inspectionMatches(result, node.id, path, queryId))
              throw new Error('图像元信息已失效，请按当前数据集重新读取。');
          }
          if (result.status !== 'READY' || !result.inspection)
            throw new Error(result.error?.message || '未能读取图像字段形状，请手动核对或重试。');
          setView({ scope, revision, loading: false, inspection: result.inspection });
        } catch (error) {
          if (!signal.aborted)
            setView({ scope, revision, loading: false, error: error instanceof Error ? error.message : String(error) });
        }
      })();
    }, 350);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [scope, ready, revision]);
  const current = view?.scope === scope && view.revision === revision ? view : undefined;
  return {
    scope,
    loading: ready && (!current || current.loading),
    inspection: enabled ? current?.inspection : undefined,
    error: enabled ? current?.error : undefined,
    refresh: () => setRevision((value) => value + 1),
  };
}
