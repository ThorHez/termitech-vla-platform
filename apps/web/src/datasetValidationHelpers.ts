import { api, post } from './api';
import type { Asset, DatasetSourceValidation, Node, Profile } from './types';

export function datasetValidationPresentation(report?: DatasetSourceValidation | null) {
  if (!report) return { state: 'REGISTERED', label: '尚未校验' };
  if (report.scope !== 'dataset_source') return { state: 'UNKNOWN', label: '校验范围待确认' };
  switch (report.status) {
    case 'PENDING':
      return { state: 'STARTING', label: '待校验 / 后台校验中' };
    case 'SUCCEEDED':
      return { state: 'VALID', label: '基础校验通过' };
    case 'FAILED':
      return { state: 'FAILED', label: '基础校验失败' };
    case 'STALE':
      return { state: 'UNKNOWN', label: '需重验' };
    case 'BLOCKED':
      return { state: 'UNKNOWN', label: '暂无法校验' };
    default:
      return { state: 'UNKNOWN', label: '校验状态待确认' };
  }
}

export function datasetValidationProfiles(profiles: Profile[], nodeId: string) {
  return profiles.filter(
    (profile) =>
      profile.node_id === nodeId &&
      profile.adapter === 'openpi-sft' &&
      profile.active !== false &&
      (!profile.execution_mode || profile.execution_mode === 'native'),
  );
}

function reportTime(report: DatasetSourceValidation) {
  return Math.max(
    0,
    ...[report.requested_at, report.completed_at, report.checked_at]
      .map((value) => (value ? Date.parse(value) : 0))
      .filter(Number.isFinite),
  );
}

export function currentDatasetValidation(
  initial?: DatasetSourceValidation | null,
  polled?: DatasetSourceValidation | null,
  submitted?: DatasetSourceValidation,
) {
  const candidates = [...(polled === null ? [] : [initial]), polled, submitted].filter(
    (report): report is DatasetSourceValidation => !!report,
  );
  return candidates.reduce<DatasetSourceValidation | undefined>((latest, candidate) => {
    if (!latest || reportTime(candidate) > reportTime(latest)) return candidate;
    if (reportTime(candidate) < reportTime(latest)) return latest;
    if (latest.id === candidate.id && latest.status === 'STALE') return latest;
    if (latest.id === candidate.id && latest.status !== 'PENDING' && candidate.status === 'PENDING')
      return latest;
    return candidate;
  }, undefined);
}

export function datasetValidationBlockedReason(
  asset: Pick<Asset, 'state' | 'status'>,
  report: DatasetSourceValidation | undefined | null,
  node: Node | undefined,
  profileId: string,
  profiles: Profile[],
  busy = false,
) {
  if ((asset.status || asset.state) === 'ARCHIVED') return '已归档资产保留历史报告。';
  if (busy || report?.status === 'PENDING') return '已有后台校验任务，请等待结果。';
  if (!node) return '正在读取资产所在机器。';
  if (node.status !== 'ONLINE') return '机器离线，恢复在线后可重新校验。';
  if (!profiles.length) return '请先为此机器配置原生 OpenPI 训练环境。';
  if (!profileId || !profiles.some((profile) => profile.id === profileId))
    return '请选择本次校验使用的训练环境。';
  return undefined;
}

export function readDatasetValidation(assetId: string, signal?: AbortSignal) {
  return api<DatasetSourceValidation | null>(`/assets/${encodeURIComponent(assetId)}/validation`, {
    signal,
  });
}

export function requestDatasetValidation(assetId: string, profileId?: string) {
  return post<DatasetSourceValidation>(
    `/assets/${encodeURIComponent(assetId)}/validations`,
    profileId ? { profile_id: profileId } : {},
  );
}
