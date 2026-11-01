import type { Asset } from './types';

export const DATASET_NORM_PATH = '__path__';

export function datasetNormPathError(path: string): string | undefined {
  const value = path.trim();
  if (!value) return '请填写 norm_stats 的完整路径。';
  if (!value.startsWith('/')) return 'norm_stats 需要填写机器上的绝对路径。';
  if (value.length > 4096 || /[\\\u0000-\u001f\u007f-\u009f]/.test(path))
    return 'norm_stats 路径无效或超过 4096 个字符。';
  if (value.split('/').includes('..')) return 'norm_stats 路径不能包含 ..。';
  if (value.split('/').every((part) => !part || part === '.'))
    return '请选择 norm_stats 目录或 norm_stats.json 文件，不能使用根目录。';
}

export function datasetNormBindingPayload(
  normId: string,
  path = '',
): { norm_asset_id: string | null } | { norm_path: string } {
  return normId === DATASET_NORM_PATH
    ? { norm_path: path.trim() }
    : { norm_asset_id: normId || null };
}

export function datasetNormInputError(
  nodeId: string,
  normId: string,
  path: string,
  assets: Asset[],
): string | undefined {
  if (normId === DATASET_NORM_PATH)
    return !nodeId ? '请先选择数据集所在机器。' : datasetNormPathError(path);
  return datasetNormSelectionError(nodeId, normId, assets);
}

const unavailable = (asset: Asset) =>
  ['ARCHIVED', 'UNAVAILABLE'].includes(asset.status || asset.state || '');

export function datasetNormCandidates(nodeId: string, assets: Asset[]) {
  return assets.filter(
    (asset) => asset.node_id === nodeId && asset.kind === 'NORM' && !unavailable(asset),
  );
}

export function datasetNormSelectionError(
  nodeId: string,
  normId: string | null | undefined,
  assets: Asset[],
): string | undefined {
  if (!normId) return undefined;
  const norm = assets.find((item) => item.id === normId);
  if (!norm) return '已绑定的 norm_stats 资产不存在或尚未读取，请重新选择。';
  if (norm.kind !== 'NORM') return 'norm_stats 必须选择归一化参数资产。';
  if (norm.node_id !== nodeId) return 'norm_stats 必须与数据集位于同一台机器。';
  if (unavailable(norm)) return '已绑定的 norm_stats 已归档或不可用，请重新绑定。';
}

export function defaultDatasetNormSync(asset: Asset, assets: Asset[]): boolean {
  return (
    asset.kind === 'DATASET' &&
    Boolean(asset.norm_asset_id) &&
    !datasetNormSelectionError(asset.node_id, asset.norm_asset_id, assets)
  );
}

export function datasetNormTransferError(
  asset: Asset,
  assets: Asset[],
  include: boolean,
): string | undefined {
  if (!include) return undefined;
  if (asset.kind !== 'DATASET') return '只有数据集可以一并同步绑定的 norm_stats。';
  if (!asset.norm_asset_id) return '此数据集尚未绑定 norm_stats，请先绑定后再选择一并同步。';
  return datasetNormSelectionError(asset.node_id, asset.norm_asset_id, assets);
}

export function boundDatasetNorm(asset: Asset, assets: Asset[]) {
  return asset.norm_asset_id ? assets.find((item) => item.id === asset.norm_asset_id) : undefined;
}
