import { useCallback, useEffect, useRef, useState } from 'react';
import { createOperationClient } from './api';
import { useAuth } from './auth';
import type { Asset, DatasetBinding, Node } from './types';

export type DatasetSampleCount = {
  status: 'loading' | 'ready' | 'unavailable';
  count?: number;
  message?: string;
  checkedAt?: string;
  source?: 'validation' | 'metadata';
};

const validCount = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
const pathKey = (path: string) => path.replace(/\/+$/u, '').replace(/\/\.\//gu, '/');
const unavailable = (message: string): DatasetSampleCount => ({ status: 'unavailable', message });

/** Account-local live metadata; changing weights never reads a dataset again. */
export function useDatasetSampleCounts(
  bindings: readonly DatasetBinding[],
  assets: readonly Asset[],
  node?: Node,
): { counts: Record<string, DatasetSampleCount>; refresh: () => void } {
  const { user, loading, expired, expiresAt } = useAuth();
  const authenticated = Boolean(user?.id && !loading && !expired);
  const owner = JSON.stringify([user?.id, authenticated, expiresAt]);
  const ids = [
    ...new Set(bindings.map((binding) => binding.dataset_asset_id).filter(Boolean)),
  ].sort();
  const selected = ids.map((id) =>
    assets.find((asset) => asset.id === id && asset.kind === 'DATASET'),
  );
  const identities = ids.map((id, index) => {
    const asset = selected[index];
    return JSON.stringify([
      owner,
      node?.id,
      node?.status,
      id,
      asset?.node_id,
      asset?.path,
      asset?.version,
      asset?.status,
      asset?.state,
      asset?.validation?.status,
      asset?.validation?.timestamp_contract?.status,
      asset?.validation?.timestamp_contract?.checked_rows,
      asset?.validation?.checked_at,
      asset?.validation?.completed_at,
      asset?.validation?.source_fingerprint,
    ]);
  });
  const scope = JSON.stringify(identities);
  const cache = useRef({
    owner,
    values: new Map<string, DatasetSampleCount>(),
    metadataOnly: new Set<string>(),
  });
  if (cache.current.owner !== owner)
    cache.current = { owner, values: new Map(), metadataOnly: new Set() };
  const [revision, setRevision] = useState({ scope: '', sequence: 0 });
  const forceRead = revision.scope === scope;
  const key = JSON.stringify([scope, revision.sequence]);
  const active = useRef(key);
  active.current = key;
  const [view, setView] = useState<{ key: string; counts: Record<string, DatasetSampleCount> }>();
  const initial: Record<string, DatasetSampleCount> = Object.fromEntries(
    ids.map((id, index) => {
      const asset = selected[index];
      let count: DatasetSampleCount;
      if (!authenticated) count = unavailable('请登录后读取数据集样本数。');
      else if (!asset || !node || asset.node_id !== node.id || !asset.path)
        count = unavailable('当前机器的数据集资产或路径不可用。');
      else if (['ARCHIVED', 'UNAVAILABLE'].includes(asset.status || asset.state || ''))
        count = unavailable('数据集已归档或不可用。');
      else if (cache.current.values.has(identities[index]))
        count = cache.current.values.get(identities[index])!;
      else if (
        !forceRead &&
        !cache.current.metadataOnly.has(identities[index]) &&
        asset.validation?.status === 'SUCCEEDED' &&
        asset.validation.timestamp_contract?.status === 'verified' &&
        validCount(asset.validation.timestamp_contract?.checked_rows)
      )
        count = {
          status: 'ready',
          count: asset.validation.timestamp_contract.checked_rows,
          checkedAt: asset.validation.checked_at || asset.validation.completed_at || undefined,
          source: 'validation',
        };
      else if (node.status !== 'ONLINE') count = unavailable('机器离线，恢复连接后可读取样本数。');
      else count = { status: 'loading' };
      return [id, count];
    }),
  );
  const refresh = useCallback(() => {
    if (cache.current.owner === owner)
      for (const identity of identities) {
        cache.current.values.delete(identity);
        // A failed explicit reread must not revive older validation proof when
        // the user subsequently adds another dataset to this selection.
        cache.current.metadataOnly.add(identity);
      }
    setRevision((previous) => ({ scope, sequence: previous.sequence + 1 }));
  }, [scope]);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    setView({ key, counts: initial });
    const pending = selected.filter((asset): asset is Asset =>
      Boolean(asset && initial[asset.id]?.status === 'loading'),
    );
    if (!pending.length) return () => controller.abort();
    const operation = createOperationClient();
    const identityById = new Map(ids.map((id, index) => [id, identities[index]]));
    let next = 0;
    const publish = (id: string, value: DatasetSampleCount) => {
      if (signal.aborted || active.current !== key || cache.current.owner !== owner) return;
      const identity = identityById.get(id)!;
      if (value.status === 'ready') cache.current.values.set(identity, value);
      else cache.current.values.delete(identity);
      setView((previous) =>
        previous?.key === key && active.current === key && !signal.aborted
          ? { key, counts: { ...previous.counts, [id]: value } }
          : previous,
      );
    };
    const worker = async () => {
      while (!signal.aborted && active.current === key && next < pending.length) {
        const asset = pending[next++];
        try {
          const result = await operation.api<{
            asset_id: string;
            node_id: string;
            checked_at?: string;
            dataset?: { status?: string; path?: string; total_frames?: unknown };
          }>(`/assets/${encodeURIComponent(asset.id)}/statistics`, {
            signal: AbortSignal.any([signal, AbortSignal.timeout(20000)]),
          });
          if (signal.aborted || active.current !== key) return;
          const dataset = result?.dataset;
          const expectedPath = pathKey(asset.path);
          const actualPath = typeof dataset?.path === 'string' ? pathKey(dataset.path) : '';
          if (
            result?.asset_id !== asset.id ||
            result.node_id !== asset.node_id ||
            dataset?.status !== 'READY' ||
            !expectedPath ||
            ![expectedPath, `${expectedPath}/meta/info.json`].includes(actualPath) ||
            !validCount(dataset.total_frames)
          )
            throw new Error('数据集样本数或来源与当前选择不一致，请重新读取。');
          publish(asset.id, {
            status: 'ready',
            count: dataset.total_frames,
            source: 'metadata',
            checkedAt: typeof result.checked_at === 'string' ? result.checked_at : undefined,
          });
        } catch (error) {
          publish(
            asset.id,
            unavailable(error instanceof Error ? error.message : '读取样本数失败，请重试。'),
          );
        }
      }
    };
    void Promise.all(Array.from({ length: Math.min(2, pending.length) }, () => worker()));
    return () => controller.abort();
  }, [scope, revision.sequence]);

  return { counts: view?.key === key ? view.counts : initial, refresh };
}
