import { useCallback, useEffect, useRef, useState } from 'react';
import { LoaderCircle, RefreshCw, XCircle } from 'lucide-react';
import { createOperationClient, useApi } from './api';
import { useAuth } from './auth';
import {
  DATASET_BASE_VALIDATION_REQUIRED,
  currentDatasetValidation,
  datasetValidationBlockedReason,
  datasetValidationPresentation,
  datasetValidationProfiles,
  readDatasetValidation,
  requestDatasetValidation,
} from './datasetValidationHelpers';
import type { Asset, DatasetSourceValidation, Node, Profile } from './types';

interface ValidationEnvironment {
  node?: Node;
  profiles: Profile[];
  profileId: string;
}

interface ValidationRowState {
  loading: boolean;
  ready: boolean;
  pending: boolean;
  succeeded: boolean;
}

type SubmitValidation = () => Promise<void>;

export function PreflightDatasetValidation({
  asset,
  node,
  profiles,
  profileId,
  previouslyPassed = false,
  batchSubmitting = false,
  onStateChange,
  onSubmitChange,
}: ValidationEnvironment & {
  asset: Asset;
  previouslyPassed?: boolean;
  batchSubmitting?: boolean;
  onStateChange?: (assetId: string, state: ValidationRowState) => void;
  onSubmitChange?: (assetId: string, submit?: SubmitValidation) => void;
}) {
  const { can } = useAuth();
  const permitted = can('datasets.validate');
  const unavailable = !['REGISTERED', 'READY'].includes(asset.status || asset.state || '');
  const [receipt, setReceipt] = useState<DatasetSourceValidation>();
  const [polling, setPolling] = useState(false);
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState<string>();
  const [client] = useState(createOperationClient);
  const submitting = useRef<AbortController>(undefined);
  const detail = useApi<DatasetSourceValidation | null>(
    permitted && !unavailable ? `/assets/${encodeURIComponent(asset.id)}/validation` : null,
    polling ? 3000 : 0,
  );
  const report = currentDatasetValidation(undefined, detail.data, receipt);
  const waiting = report?.status === 'PENDING';
  const current =
    receipt &&
    report?.scope === 'dataset_source' &&
    report?.profile_id === profileId &&
    (report === receipt || (receipt.id && report.id === receipt.id))
      ? report
      : undefined;
  const succeeded = current?.status === 'SUCCEEDED';
  const compatible = datasetValidationProfiles(profiles, asset.node_id);
  const blocker = datasetValidationBlockedReason(asset, report, node, profileId, compatible, busy);
  const ready =
    permitted &&
    !unavailable &&
    !succeeded &&
    !blocker &&
    detail.data !== undefined &&
    !detail.error;
  const loading = permitted && !unavailable && detail.data === undefined && !detail.error;
  const blockerMessage =
    blocker || (detail.data === undefined && !detail.error ? '正在读取校验状态…' : undefined);

  useEffect(() => () => submitting.current?.abort(), []);
  useEffect(() => setPolling(waiting), [waiting]);
  useEffect(() => {
    onStateChange?.(asset.id, {
      loading,
      ready,
      pending: permitted && !unavailable && (busy || waiting),
      succeeded: permitted && !unavailable && succeeded,
    });
  }, [asset.id, onStateChange, loading, ready, busy, waiting, permitted, unavailable, succeeded]);
  useEffect(() => {
    if (!receipt && report?.status === 'PENDING' && report.profile_id === profileId) {
      setReceipt(report);
    }
  }, [receipt, report, profileId]);
  const refresh = detail.refresh;
  const submit = useCallback(async () => {
    if (!ready || submitting.current) return;
    const controller = new AbortController();
    submitting.current = controller;
    setBusy(true);
    setSubmitError(undefined);
    try {
      client.requireCurrent();
      const latest = await readDatasetValidation(
        asset.id,
        AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
      );
      client.requireCurrent();
      controller.signal.throwIfAborted();
      if (latest?.status === 'PENDING') {
        setReceipt(latest);
        refresh();
        return;
      }
      const next = await requestDatasetValidation(
        asset.id,
        profileId,
        AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
      );
      client.requireCurrent();
      controller.signal.throwIfAborted();
      setReceipt(next);
      refresh();
    } catch (cause) {
      if (!controller.signal.aborted)
        setSubmitError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      submitting.current = undefined;
      if (!controller.signal.aborted) setBusy(false);
    }
  }, [ready, asset.id, profileId, client, refresh]);
  useEffect(() => {
    onSubmitChange?.(asset.id, ready ? submit : undefined);
    return () => onSubmitChange?.(asset.id);
  }, [asset.id, onSubmitChange, ready, submit]);

  if (unavailable) return <p>该数据集已归档或不可用。</p>;
  if (!permitted) return <p>当前账号没有提交数据集校验的权限，请联系操作员或管理员。</p>;
  if (succeeded) return <p role="status">基础校验已通过。</p>;

  return (
    <div>
      {previouslyPassed && !receipt && !busy && !waiting && <p>当前有该环境的通过记录。</p>}
      <button
        type="button"
        className="button small outline"
        disabled={!ready || batchSubmitting}
        onClick={() => void submit()}
      >
        {busy || waiting ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}
        {busy ? '提交中…' : waiting ? '校验中…' : '刷新数据集校验'}
      </button>
      {waiting && (
        <p role="status">
          {report.profile_id === profileId
            ? '已提交后台校验，等待 Agent 返回结果。'
            : '该数据集正在其他训练环境校验，请等待完成后再刷新。'}
        </p>
      )}
      {!busy && !waiting && blockerMessage && <p>{blockerMessage}</p>}
      {receipt?.status === 'PENDING' &&
        receipt.profile_id === profileId &&
        report &&
        !current &&
        !waiting && <p role="status">本次校验结果已被新的校验记录替换，请重新刷新校验。</p>}
      {current && current.status !== 'PENDING' && (
        <p role="status">
          {current.error?.message || datasetValidationPresentation(current).label}
        </p>
      )}
      {submitError && <p role="status">提交校验失败：{submitError}</p>}
      {detail.error && (
        <p role="status">
          校验状态读取失败：{detail.error.message}{' '}
          <button type="button" className="button small outline" onClick={detail.refresh}>
            重新读取校验状态
          </button>
        </p>
      )}
    </div>
  );
}

export function PreflightDatasetValidationBatch({
  affected,
  additional,
  unresolved,
  fallback,
  node,
  profiles,
  profileId,
  changed,
  onStatusChange,
}: ValidationEnvironment & {
  affected: Asset[];
  additional: Asset[];
  unresolved: number;
  fallback?: string;
  changed: () => void;
  onStatusChange: (pending: boolean, refreshed: boolean) => void;
}) {
  const { can } = useAuth();
  const [affectedIds] = useState(() => new Set(affected.map((asset) => asset.id)));
  const [allRequested, setAllRequested] = useState(false);
  const [batchSubmitting, setBatchSubmitting] = useState(false);
  const [states, setStates] = useState<Record<string, ValidationRowState>>({});
  const submissions = useRef(new Map<string, SubmitValidation>());
  const batch = useRef<AbortController>(undefined);
  const stateChanged = useCallback((id: string, next: ValidationRowState) => {
    setStates((current) => {
      const prev = current[id];
      if (
        prev?.loading === next.loading &&
        prev.ready === next.ready &&
        prev.pending === next.pending &&
        prev.succeeded === next.succeeded
      )
        return current;
      return { ...current, [id]: next };
    });
  }, []);
  const submitChanged = useCallback((id: string, submit?: SubmitValidation) => {
    if (submit) submissions.current.set(id, submit);
    else submissions.current.delete(id);
  }, []);

  const assets = [...affected, ...additional];
  const required = assets.filter((asset) => affectedIds.has(asset.id));
  const optional = assets.filter((asset) => !affectedIds.has(asset.id));
  const targets = allRequested ? assets : required;
  const pending = batchSubmitting || targets.some((asset) => states[asset.id]?.pending);
  const succeeded = targets.filter((asset) => states[asset.id]?.succeeded).length;
  const refreshed = targets.length > 0 && unresolved === 0 && succeeded === targets.length;
  const canRefresh = (items: Asset[]) =>
    !batchSubmitting &&
    items.every((asset) => states[asset.id] && !states[asset.id].loading) &&
    items.some((asset) => states[asset.id].ready);
  useEffect(() => onStatusChange(pending, refreshed), [onStatusChange, pending, refreshed]);
  useEffect(
    () => () => {
      batch.current?.abort();
      onStatusChange(false, false);
    },
    [onStatusChange],
  );
  const hadPending = useRef(false);
  useEffect(() => {
    if (hadPending.current && !pending) changed();
    hadPending.current = pending;
  }, [changed, pending]);

  async function refreshAssets(items: Asset[], all = false) {
    if (batch.current) return;
    const controller = new AbortController();
    batch.current = controller;
    if (all) setAllRequested(true);
    setBatchSubmitting(true);
    try {
      // Bound request fan-out while letting individual failures remain retryable.
      const ready = items.filter((asset) => submissions.current.has(asset.id));
      for (let index = 0; index < ready.length && !controller.signal.aborted; index += 4) {
        await Promise.allSettled(
          ready.slice(index, index + 4).map((asset) => submissions.current.get(asset.id)?.()),
        );
      }
    } finally {
      batch.current = undefined;
      if (!controller.signal.aborted) setBatchSubmitting(false);
    }
  }

  const renderRows = (items: Asset[], previouslyPassed = false) => (
    <ul className="preflight-validation-list">
      {items.map((asset) => (
        <li key={asset.id}>
          <strong>{asset.name}</strong>
          <PreflightDatasetValidation
            asset={asset}
            node={node}
            profiles={profiles}
            profileId={profileId}
            previouslyPassed={previouslyPassed}
            batchSubmitting={batchSubmitting}
            onStateChange={stateChanged}
            onSubmitChange={submitChanged}
          />
        </li>
      ))}
    </ul>
  );

  return (
    <div
      className={
        refreshed ? 'preflight-validation-batch' : 'preflight-validation-batch error-notice'
      }
    >
      {!refreshed && <XCircle size={18} aria-hidden="true" />}
      <div>
        {!refreshed && (
          <>
            <strong role="alert" title={DATASET_BASE_VALIDATION_REQUIRED}>
              数据集基础校验缺失或已失效
            </strong>
            {unresolved > 0 && <p>{fallback || '部分数据集无法定位，请在资产详情中核对校验。'}</p>}
            {targets.length > 0 && (
              <p role="status">
                本次已完成 {succeeded} / {targets.length} 个数据集的基础校验。
              </p>
            )}
          </>
        )}
        {can('datasets.validate') && (
          <div className="preflight-validation-actions">
            {!refreshed && targets.length > 1 && (
              <button
                type="button"
                className="button small outline"
                disabled={!canRefresh(targets)}
                onClick={() => void refreshAssets(targets)}
              >
                {batchSubmitting ? (
                  <LoaderCircle size={14} className="spin" />
                ) : (
                  <RefreshCw size={14} />
                )}
                刷新待校验数据集
              </button>
            )}
            {!allRequested && optional.length > 0 && (
              <button
                type="button"
                className="button small outline"
                disabled={!canRefresh(assets)}
                onClick={() => void refreshAssets(assets, true)}
              >
                刷新本次所有数据集
              </button>
            )}
          </div>
        )}
        {renderRows(required)}
        {optional.length > 0 && (
          <details className="preflight-validation-optional" open={allRequested || undefined}>
            <summary>同一训练环境已有通过记录（{optional.length} 个）</summary>
            <p>
              旧版本可能只报告第一个缺失项。如需重新检查本次绑定的每个数据集，可点击「刷新本次所有数据集」。
            </p>
            {renderRows(optional, true)}
          </details>
        )}
      </div>
    </div>
  );
}
