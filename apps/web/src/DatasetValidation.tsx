import { useEffect, useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, LoaderCircle, RefreshCw } from 'lucide-react';
import { date, useApi } from './api';
import { useAuth } from './auth';
import { Badge, ErrorNotice, Field } from './components';
import {
  currentDatasetValidation,
  datasetValidationBlockedReason,
  datasetValidationPresentation,
  datasetValidationProfiles,
  requestDatasetValidation,
} from './datasetValidationHelpers';
import type { Asset, DatasetSourceValidation, Node, Profile } from './types';
import './dataset-validation.css';

export function DatasetValidationBadge({
  validation,
}: {
  validation?: DatasetSourceValidation | null;
}) {
  const presentation = datasetValidationPresentation(validation);
  return (
    <span className="dataset-validation-badge" aria-label={`数据集基础校验：${presentation.label}`}>
      <Badge {...presentation} />
    </span>
  );
}

export function DatasetValidation({
  asset,
  node,
  profiles: suppliedProfiles,
  changed,
}: {
  asset: Asset;
  node?: Node;
  profiles?: Profile[];
  changed?: () => void;
}) {
  const { can } = useAuth();
  const [receipt, setReceipt] = useState<DatasetSourceValidation>();
  const [polling, setPolling] = useState(asset.validation?.status === 'PENDING');
  const detail = useApi<DatasetSourceValidation | null>(
    `/assets/${encodeURIComponent(asset.id)}/validation`,
    polling ? 3000 : 0,
  );
  const profilesQuery = useApi<Profile[]>(
    suppliedProfiles ? null : `/profiles?node_id=${encodeURIComponent(asset.node_id)}`,
    0,
  );
  const profiles = datasetValidationProfiles(
    suppliedProfiles || profilesQuery.data || [],
    asset.node_id,
  );
  const report = currentDatasetValidation(asset.validation, detail.data, receipt);
  const [profileId, setProfileId] = useState(asset.validation?.profile_id || '');
  const selected = profiles.some((profile) => profile.id === profileId)
    ? profileId
    : profiles.some((profile) => profile.id === report?.profile_id)
      ? report!.profile_id!
      : profiles.length === 1
        ? profiles[0].id
        : '';
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const pending = useRef(false);
  const published = useRef(JSON.stringify(asset.validation || null));
  const writable = can('assets.write') && (asset.status || asset.state) !== 'ARCHIVED';
  const blocker = datasetValidationBlockedReason(asset, report, node, selected, profiles, busy);
  useEffect(() => setPolling(report?.status === 'PENDING'), [report?.status]);
  useEffect(() => {
    if (detail.data === undefined) return;
    const value = JSON.stringify(detail.data);
    if (value !== published.current) {
      published.current = value;
      changed?.();
    }
  }, [detail.data, changed]);
  const completed = report?.completed_at || report?.checked_at;
  return (
    <section className="asset-detail-section dataset-validation" aria-label="数据集基础校验">
      <div className="dataset-validation-heading">
        <h4>数据集基础校验</h4>
        <DatasetValidationBadge validation={report} />
      </div>
      <p className="dataset-validation-scope">
        检查数据源元信息、文件结构、时间戳与有限原始样本。相机映射、状态拼接等训练组合配置仍需在训练向导中校验。
      </p>
      {report?.requested_at && <p>最近提交：{date(report.requested_at)}</p>}
      {completed && <p>最近完成：{date(completed)}</p>}
      {report?.profile_id && (
        <p>
          校验环境：
          {profiles.find((profile) => profile.id === report.profile_id)?.name || report.profile_id}
        </p>
      )}
      {report?.status === 'PENDING' && (
        <p role="status">已提交后台校验，等待 Agent 返回结果；可关闭此窗口。</p>
      )}
      {report?.status === 'STALE' && (
        <p className="dataset-validation-warning">数据或配置已变化，请重新校验后使用最新结果。</p>
      )}
      {report?.reason && <p className="dataset-validation-warning">{report.reason}</p>}
      {report?.error && (
        <p className="dataset-validation-warning" role="status">
          {report.error.code} · {report.error.message}
        </p>
      )}
      {Array.isArray(report?.checks) && report.checks.length > 0 && (
        <details
          className="dataset-validation-checks"
          open={report.status === 'FAILED' || report.status === 'BLOCKED'}
        >
          <summary>最近检查结果（{report.checks.length} 项）</summary>
          <ul>
            {report.checks.map((check, index) => (
              <li key={`${check.name}:${index}`} className={check.ok ? '' : 'failed'}>
                {check.ok ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                <span>{check.message || check.name}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      <ErrorNotice error={detail.error} retry={detail.refresh} title="校验报告读取失败" />
      {writable && (
        <div className="dataset-validation-actions">
          <Field label="校验使用的训练环境">
            <select
              aria-label="校验使用的训练环境"
              value={selected}
              disabled={busy || report?.status === 'PENDING'}
              onChange={(event) => setProfileId(event.target.value)}
            >
              <option value="">{profiles.length ? '请选择训练环境' : '暂无可用环境'}</option>
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </select>
          </Field>
          <button
            type="button"
            className="button outline"
            disabled={!!blocker}
            title={blocker}
            onClick={async () => {
              if (pending.current || blocker || !can('assets.write')) return;
              pending.current = true;
              setBusy(true);
              setError(undefined);
              try {
                setReceipt(await requestDatasetValidation(asset.id, selected));
                detail.refresh();
                changed?.();
              } catch (cause) {
                setError(cause instanceof Error ? cause : new Error(String(cause)));
              } finally {
                pending.current = false;
                setBusy(false);
              }
            }}
          >
            {busy ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}
            {busy ? '提交中…' : report ? '重新校验' : '开始基础校验'}
          </button>
          {blocker && <p className="field-hint">{blocker}</p>}
          <ErrorNotice error={error || profilesQuery.error} title="无法提交校验" />
        </div>
      )}
    </section>
  );
}
