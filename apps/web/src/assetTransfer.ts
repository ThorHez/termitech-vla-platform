import type { Asset, Node } from './types';

export interface AssetNormTransfer {
  source_asset_id: string;
  name: string;
  source_path: string;
  target_name?: string;
  target_path?: string | null;
  asset_path?: string | null;
  target_asset_id?: string | null;
  state: string;
  bytes_total?: number;
  bytes_transferred?: number;
  files_total?: number;
  files_transferred?: number | null;
  progress_pct?: number;
  error?: { code: string; message: string } | null;
}

export interface AssetTransfer {
  owner_user_id?: string | null;
  execution_user_id?: string | null;
  id: string;
  name: string;
  source_asset_id: string;
  source_kind: string;
  source_node_id: string;
  source_path: string;
  target_node_id: string;
  target_directory: string;
  target_name: string;
  target_path?: string | null;
  asset_path?: string | null;
  target_asset_id?: string | null;
  include_norm_stats?: boolean;
  exclude_train_state?: boolean;
  norm_sync?: AssetNormTransfer | null;
  state: string;
  bytes_total: number;
  bytes_transferred: number;
  files_total: number;
  files_transferred?: number | null;
  speed_bytes_per_second?: number;
  progress_pct?: number;
  attempt?: number;
  logs?: string;
  error?: { code: string; message: string } | null;
  created_at: string;
  updated_at: string;
  finished_at?: string | null;
}

export const assetTransferLabels: Record<string, string> = {
  QUEUED: '等待同步',
  PREPARING: '检查机器与目录',
  SCANNING: '统计源文件',
  TRANSFERRING: '正在同步文件',
  VERIFYING: '校验文件完整性',
  SUCCEEDED: '同步完成，已登记',
  FAILED: '同步失败',
  CANCELLED: '已取消同步',
  INTERRUPTED: '同步已中断',
  CANCELLING: '正在停止同步',
};

export const assetTransferTerminal = (state: string) =>
  ['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(state);

export const assetTransferTaskPath = (id: string) => `/runs/sync/${encodeURIComponent(id)}`;

export function assetTransferCreateRequest(
  asset: Asset,
  targetNodeId: string,
  directory: string,
  targetName: string,
  includeNormStats = false,
  excludeTrainState = false,
) {
  return {
    source_asset_id: asset.id,
    target_node_id: targetNodeId,
    target_directory: directory.trim().replace(/\/+$/, '') || '/',
    target_name: targetName.trim(),
    include_norm_stats: includeNormStats,
    ...(includeNormStats ? { expected_norm_asset_id: asset.norm_asset_id } : {}),
    ...(asset.kind === 'MODEL' && excludeTrainState ? { exclude_train_state: true } : {}),
  };
}

export function assetTargetNameError(value: string): string | undefined {
  const name = value.trim();
  if (!name) return '请填写目标子目录名。';
  if (name.length > 160) return '目标子目录名最多 160 个字符。';
  if (new TextEncoder().encode(name).length > 255) return '目标子目录名过长，请缩短名称。';
  if (name === '.' || name === '..' || /[/\\\x00-\x1f\x7f-\x9f]/.test(name)) {
    return '目标子目录名不能包含路径分隔符、控制字符，也不能是 . 或 ..。';
  }
}

export function assetTargetDirectoryError(value: string): string | undefined {
  const directory = value.trim();
  if (!directory.startsWith('/')) return '请填写目标机器上的绝对父目录。';
  if (directory.length > 4096) return '目标父目录过长，请缩短路径。';
  if (
    /[\\\x00-\x1f\x7f-\x9f]/.test(directory) ||
    directory.split('/').some((part) => part === '.' || part === '..')
  ) {
    return '目标父目录不能包含控制字符或 .、.. 路径段。';
  }
}

export function assetTargetPath(directory: string, name: string): string | undefined {
  if (assetTargetDirectoryError(directory) || assetTargetNameError(name)) return undefined;
  return `${directory.trim().replace(/\/+$/, '')}/${name.trim()}`;
}

export function defaultAssetTargetName(asset: Pick<Asset, 'name' | 'path'>): string {
  const basename = asset.path.replace(/\/+$/, '').split('/').at(-1) || '';
  const fromFile = /\.(json|npz|pt|pth|bin|safetensors|ckpt)$/i.test(basename);
  for (const candidate of [fromFile ? asset.name : basename, asset.name, 'synced-asset']) {
    if (!assetTargetNameError(candidate)) return candidate.trim();
  }
  return 'synced-asset';
}

export function assetSyncBlocker(input: {
  asset: Asset;
  sourceNode?: Node;
  targetNode?: Node;
  directory: string;
  targetName: string;
}): string | undefined {
  if ((input.asset.status || input.asset.state) === 'ARCHIVED') return '已归档资产不能发起同步。';
  if ((input.asset.status || input.asset.state) === 'UNAVAILABLE')
    return '源资产当前不可用，请先确认源文件仍然存在。';
  if (!['DATASET', 'NORM', 'MODEL'].includes(input.asset.kind)) return '当前资产类型暂不支持同步。';
  if (input.asset.path.trim().replace(/\/+$/, '') === '')
    return '不能同步机器根目录，请登记具体的数据目录或文件。';
  if (input.sourceNode?.id !== input.asset.node_id || input.sourceNode.status !== 'ONLINE') {
    return '源机器离线，请先恢复机器连接。';
  }
  if (!input.targetNode) return '请选择目标机器。';
  if (input.targetNode.id === input.asset.node_id) return '请选择另一台机器作为同步目标。';
  if (input.targetNode.status !== 'ONLINE') return '目标机器离线，请先恢复机器连接。';
  return assetTargetDirectoryError(input.directory) || assetTargetNameError(input.targetName);
}
