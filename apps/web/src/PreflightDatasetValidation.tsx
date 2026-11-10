import { useEffect, useRef, useState } from 'react';
import { LoaderCircle, RefreshCw } from 'lucide-react';
import { useApi } from './api';
import { useAuth } from './auth';
import {
  currentDatasetValidation,
  datasetValidationBlockedReason,
  datasetValidationPresentation,
  datasetValidationProfiles,
  readDatasetValidation,
  requestDatasetValidation,
} from './datasetValidationHelpers';
import type { Asset, DatasetSourceValidation, Node, Profile } from './types';

export function PreflightDatasetValidation({
  asset,
  node,
  profiles,
  profileId,
  changed,
  onPendingChange,
}: {
  asset: Asset;
  node?: Node;
  profiles: Profile[];
  profileId: string;
  changed: () => void;
  onPendingChange: (pending: boolean) => void;
}) {
  const { can, user } = useAuth();
  const permitted =
    can('assets.write') ||
    (can('datasets.validate') && !!user?.id && asset.registered_by_user_id === user.id);
  const unavailable = ['ARCHIVED', 'UNAVAILABLE'].includes(asset.status || asset.state || '');
  const [receipt, setReceipt] = useState<DatasetSourceValidation>();
  const [polling, setPolling] = useState(false);
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState<string>();
  const submitting = useRef(false);
  const mounted = useRef(true);
  const notified = useRef<string | undefined>(undefined);
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
  const ready = !blocker && detail.data !== undefined && !detail.error;
  const blockerMessage =
    blocker || (detail.data === undefined && !detail.error ? '正在读取校验状态…' : undefined);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      onPendingChange(false);
    };
  }, [onPendingChange]);
  useEffect(() => {
    onPendingChange(busy || waiting);
    setPolling(waiting);
  }, [busy, waiting, onPendingChange]);
  useEffect(() => {
    if (!receipt && report?.status === 'PENDING' && report.profile_id === profileId) {
      setReceipt(report);
    }
  }, [receipt, report, profileId]);
  useEffect(() => {
    if (!current || current.status === 'PENDING') return;
    const resultKey = JSON.stringify([current.id, current.requested_at, current.status]);
    if (notified.current === resultKey) return;
    notified.current = resultKey;
    changed();
  }, [current, changed]);

  if (unavailable) return <p>该数据集已归档或不可用。</p>;
  if (!permitted) return <p>请联系数据集登记人或管理员刷新校验。</p>;

  async function submit() {
    if (!ready || submitting.current) return;
    submitting.current = true;
    setBusy(true);
    setSubmitError(undefined);
    try {
      const latest = await readDatasetValidation(asset.id, AbortSignal.timeout(20000));
      if (!mounted.current) return;
      if (latest?.status === 'PENDING') {
        setReceipt(latest);
        detail.refresh();
        return;
      }
      const next = await requestDatasetValidation(asset.id, profileId);
      if (mounted.current) {
        setReceipt(next);
        detail.refresh();
      }
    } catch (cause) {
      if (mounted.current) setSubmitError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      submitting.current = false;
      if (mounted.current) setBusy(false);
    }
  }

  return (
    <div>
      {succeeded ? (
        <p role="status">基础校验已通过，请点击「重新预检」。</p>
      ) : (
        <button
          type="button"
          className="button small outline"
          disabled={!ready}
          onClick={() => void submit()}
        >
          {busy || waiting ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}
          {busy ? '提交中…' : waiting ? '校验中…' : '刷新数据集校验'}
        </button>
      )}
      {waiting && (
        <p role="status">
          {report.profile_id === profileId
            ? '已提交后台校验，等待 Agent 返回结果。'
            : '该数据集正在其他训练环境校验，请等待完成后再刷新。'}
        </p>
      )}
      {!busy && !waiting && !succeeded && blockerMessage && <p>{blockerMessage}</p>}
      {receipt?.status === 'PENDING' &&
        receipt.profile_id === profileId &&
        report &&
        !current &&
        !waiting && <p role="status">本次校验结果已被新的校验记录替换，请重新刷新校验。</p>}
      {current && current.status !== 'PENDING' && !succeeded && (
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
