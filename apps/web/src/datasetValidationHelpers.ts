import { api } from './api';
import type {
  Asset,
  DatasetSourceValidation,
  Node,
  PreflightCheck,
  Preview,
  Profile,
} from './types';

export const DATASET_BASE_VALIDATION_REQUIRED = 'DATASET_BASE_VALIDATION_REQUIRED';

function validationPath(path: string) {
  const parts = path.split('/');
  if (!path.startsWith('/') || parts.includes('..') || /[\r\n\0]/u.test(path)) return undefined;
  return `/${parts.filter((part) => part && part !== '.').join('/')}`;
}

export function isPreflightDatasetValidationCheck(check: PreflightCheck) {
  return (
    !check.ok &&
    (check.code === DATASET_BASE_VALIDATION_REQUIRED ||
      check.message?.startsWith(`${DATASET_BASE_VALIDATION_REQUIRED}:`))
  );
}

export function preflightValidationAssets(preview: Preview, assets: readonly Asset[]) {
  const request = preview.request;
  if (
    preview.status !== 'FAILED' ||
    preview.error?.code !== DATASET_BASE_VALIDATION_REQUIRED ||
    !request?.profile_id
  )
    return undefined;
  const registered = new Map(
    assets
      .filter((asset) => asset.kind === 'DATASET' && asset.node_id === request.node_id)
      .map((asset) => [asset.id, asset]),
  );
  const boundIds = new Set(
    Object.entries(request.inputs || {}).flatMap(([role, id]) =>
      /^dataset(?:_\d+)?$/u.test(role) ? [id] : [],
    ),
  );
  const bound = [...boundIds].flatMap((id) => {
    const asset = registered.get(id);
    return asset ? [asset] : [];
  });
  const reportedIds = new Set(
    preview.checks?.flatMap((check) =>
      isPreflightDatasetValidationCheck(check) && check.dataset_asset_id
        ? [check.dataset_asset_id]
        : [],
    ),
  );
  if (reportedIds.size) {
    const affected = bound.filter((asset) => reportedIds.has(asset.id));
    return {
      affected,
      additional: [],
      unresolved: reportedIds.size - affected.length,
      legacy: false,
    };
  }

  // Older Agents identify only the first failure by its resolved path.
  const message = preview.error.message?.replace(/^DATASET_BASE_VALIDATION_REQUIRED:\s*/u, '');
  const match = message?.match(
    /^数据集 (\/[^\r\n]*) (?:的基础校验证据缺失或已失效|缺少匹配的基础校验结果|的 ReCap 监督基础校验证据缺失或已失效)(?:[，；。]|$)/u,
  );
  const path = match && validationPath(match[1]);
  const candidates = path ? bound.filter((asset) => validationPath(asset.path) === path) : [];
  const reported = candidates.length === 1 ? candidates[0] : undefined;
  const affected = bound.filter(
    (asset) =>
      asset === reported ||
      asset.validation?.status !== 'SUCCEEDED' ||
      asset.validation.profile_id !== request.profile_id,
  );
  return {
    affected,
    additional: bound.filter((asset) => !affected.includes(asset)),
    unresolved: Number(!reported) + boundIds.size - bound.length,
    legacy: true,
  };
}

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
    case 'CANCELLED':
      return { state: 'CANCELLED', label: '基础校验已取消' };
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

export function requestDatasetValidation(
  assetId: string,
  profileId?: string,
  signal?: AbortSignal,
) {
  return api<DatasetSourceValidation>(`/assets/${encodeURIComponent(assetId)}/validations`, {
    method: 'POST',
    body: JSON.stringify(profileId ? { profile_id: profileId } : {}),
    signal,
  });
}
