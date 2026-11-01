import {
  assetTargetDirectoryError,
  assetTargetNameError,
  defaultAssetTargetName,
} from './assetTransfer';
import type { AssetTransfer } from './assetTransfer';
import { defaultDatasetNormSync, datasetNormTransferError } from './datasetNormAssets';
import type { Asset, Node } from './types';

export const BATCH_ASSET_SYNC_LIMIT = 20;

export interface BatchAssetSyncRow {
  assetId: string;
  targetName: string;
  includeNormStats: boolean;
  excludeTrainState: boolean;
  expectedNormAssetId?: string;
}

export interface BatchAssetSyncResult {
  idempotency_key: string;
  transfers: AssetTransfer[];
}

export function batchSyncSourceError(asset: Asset | undefined, nodes: Node[]): string | undefined {
  if (!asset) return '源资产已不存在。';
  if (['ARCHIVED', 'UNAVAILABLE'].includes(asset.status || asset.state || ''))
    return '已归档或不可用资产不能发起同步。';
  if (!['DATASET', 'MODEL', 'NORM'].includes(asset.kind)) return '当前资产类型不支持同步。';
  if (assetTargetDirectoryError(asset.path) || !asset.path.trim().replace(/\/+$/, ''))
    return '源资产需要有效的具体目录或文件路径。';
  if (nodes.find((node) => node.id === asset.node_id)?.status !== 'ONLINE')
    return '源机器离线，请先恢复机器连接。';
}

export function reconcileBatchSyncSelection(ids: string[], assets: Asset[], nodes: Node[]) {
  return [...new Set(ids)].filter(
    (id) =>
      !batchSyncSourceError(
        assets.find((asset) => asset.id === id),
        nodes,
      ),
  );
}

export function toggleBatchSyncVisible(
  ids: string[],
  visibleEligibleIds: string[],
  checked: boolean,
) {
  const visible = new Set(visibleEligibleIds);
  return checked
    ? [...new Set([...ids, ...visibleEligibleIds])]
    : ids.filter((id) => !visible.has(id));
}

export function batchSyncRows(
  assets: Asset[],
  availableAssets: Asset[] = assets,
): BatchAssetSyncRow[] {
  return assets.map((asset) => {
    const includeNormStats = defaultDatasetNormSync(asset, availableAssets);
    return {
      assetId: asset.id,
      targetName: defaultAssetTargetName(asset),
      includeNormStats,
      excludeTrainState: false,
      ...(includeNormStats ? { expectedNormAssetId: asset.norm_asset_id! } : {}),
    };
  });
}

export function batchSyncRequest(
  rows: BatchAssetSyncRow[],
  targetNodeId: string,
  directory: string,
  availableAssets: Asset[] = [],
) {
  return {
    target_node_id: targetNodeId,
    target_directory: directory.trim().replace(/\/+$/, '') || '/',
    entries: rows.map((row) => ({
      source_asset_id: row.assetId,
      target_name: row.targetName.trim(),
      include_norm_stats: row.includeNormStats,
      ...(row.includeNormStats && row.expectedNormAssetId
        ? { expected_norm_asset_id: row.expectedNormAssetId }
        : {}),
      ...(row.excludeTrainState &&
      availableAssets.find((asset) => asset.id === row.assetId)?.kind === 'MODEL'
        ? { exclude_train_state: true }
        : {}),
    })),
  };
}

export function batchSyncError({
  rows,
  assets,
  nodes,
  targetNodeId,
  directory,
}: {
  rows: BatchAssetSyncRow[];
  assets: Asset[];
  nodes: Node[];
  targetNodeId: string;
  directory: string;
}): string | undefined {
  if (!rows.length) return '请先选择要同步的资产。';
  if (rows.length > BATCH_ASSET_SYNC_LIMIT)
    return `一次最多同步 ${BATCH_ASSET_SYNC_LIMIT} 项资产，请减少选择后分批操作。`;
  if (new Set(rows.map((row) => row.assetId)).size !== rows.length)
    return '选择中存在重复源资产，请重新选择。';
  const names = new Set<string>();
  for (const row of rows) {
    const asset = assets.find((item) => item.id === row.assetId);
    const sourceError = batchSyncSourceError(asset, nodes);
    if (sourceError) return `${asset?.name || row.assetId}：${sourceError}`;
    if (row.excludeTrainState && asset!.kind !== 'MODEL')
      return `${asset!.name}：只有模型资产可以排除 train_state/。`;
    const nameError = assetTargetNameError(row.targetName);
    if (nameError) return `${asset!.name}：${nameError}`;
    const name = row.targetName.trim();
    if (names.has(name)) return `目标子目录“${name}”重名，请为每项资产填写不同名称。`;
    names.add(name);
    const normError = datasetNormTransferError(asset!, assets, row.includeNormStats);
    if (normError) return `${asset!.name}：${normError}`;
    if (
      row.includeNormStats &&
      (!row.expectedNormAssetId || row.expectedNormAssetId !== asset!.norm_asset_id)
    )
      return `${asset!.name}：绑定的 norm_stats 已变化，请重新确认是否一并同步。`;
  }
  const target = nodes.find((node) => node.id === targetNodeId);
  if (!target) return '请选择目标机器。';
  if (target.status !== 'ONLINE') return '目标机器离线，请先恢复机器连接。';
  if (
    rows.some((row) => assets.find((asset) => asset.id === row.assetId)?.node_id === targetNodeId)
  )
    return '目标机器不能与任何所选资产的源机器相同。';
  const directoryError = assetTargetDirectoryError(directory);
  if (directoryError) return directoryError;
  if (
    new TextEncoder().encode(
      JSON.stringify({
        ...batchSyncRequest(rows, targetNodeId, directory, assets),
        idempotency_key: '0'.repeat(36),
      }),
    ).byteLength >
    128 * 1024
  )
    return '批量同步请求超过 128 KiB，请缩短名称或减少选择。';
}

export function batchSyncIdentity(
  previous: { fingerprint: string; key: string } | undefined,
  request: unknown,
  generateKey: () => string,
) {
  const fingerprint = JSON.stringify(request);
  return previous?.fingerprint === fingerprint ? previous : { fingerprint, key: generateKey() };
}

export function batchSyncReceiptMatches(
  result: BatchAssetSyncResult,
  key: string,
  request: ReturnType<typeof batchSyncRequest>,
) {
  if (
    !result ||
    result.idempotency_key !== key ||
    !Array.isArray(result.transfers) ||
    result.transfers.length !== request.entries.length
  )
    return false;
  const ids = new Set<string>();
  const sources = new Set<string>();
  return result.transfers.every((transfer) => {
    const entry = request.entries.find((item) => item.source_asset_id === transfer.source_asset_id);
    if (
      !entry ||
      !transfer.id ||
      ids.has(transfer.id) ||
      sources.has(transfer.source_asset_id) ||
      transfer.target_node_id !== request.target_node_id ||
      typeof transfer.target_directory !== 'string' ||
      canonicalDirectory(transfer.target_directory) !==
        canonicalDirectory(request.target_directory) ||
      Boolean(transfer.include_norm_stats) !== entry.include_norm_stats ||
      Boolean(transfer.exclude_train_state) !== Boolean(entry.exclude_train_state) ||
      transfer.target_name !== entry.target_name
    )
      return false;
    ids.add(transfer.id);
    sources.add(transfer.source_asset_id);
    return true;
  });
}

function canonicalDirectory(value: string) {
  return '/' + value.trim().split('/').filter(Boolean).join('/');
}
