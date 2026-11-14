import type { User } from './auth';

export const assetDeletionsPath = '/assets/deletions';

export interface AssetPurgeInspection {
  id: string;
  status: 'PENDING' | 'RUNNING' | 'READY' | 'MISSING' | 'FAILED';
  requested_at: string;
  error_code: string | null;
  error_message: string | null;
}

export interface AssetFileDeletion {
  id: string;
  status: 'PENDING' | 'RUNNING' | 'DELETED' | 'MISSING' | 'FAILED';
  requested_at: string;
  finished_at: string | null;
  error_code: string | null;
  error_message: string | null;
}

export interface ArchivedAsset {
  id: string;
  name: string;
  kind: string;
  node_id: string;
  node_name: string | null;
  path: string;
  archived_at: string | null;
  archived_by: string | null;
  inspection: AssetPurgeInspection | null;
  file_deletion: AssetFileDeletion | null;
  requires_new_inspection?: boolean;
  can_purge: boolean;
  purge_block_reason?: string | null;
}

export function canManageAssetDeletions(
  user: Pick<User, 'role'> | undefined,
  can: (permission: string) => boolean,
) {
  return user?.role === 'admin' && can('assets.purge');
}

export function canOpenAssetPurge(asset: ArchivedAsset) {
  const status = asset.file_deletion?.status;
  return asset.can_purge === true && Boolean(asset.path) && (!status || status === 'FAILED');
}

export function canConfirmAssetPurge(
  asset: ArchivedAsset,
  inspection: AssetPurgeInspection | null,
) {
  if (!canOpenAssetPurge(asset) || !inspection?.id || asset.inspection?.id !== inspection.id)
    return false;
  const deletion = asset.file_deletion;
  if (deletion && !asset.requires_new_inspection) return true;
  if (deletion && (!inspection.requested_at || inspection.requested_at <= deletion.requested_at))
    return false;
  return inspection.status === 'READY' || inspection.status === 'MISSING';
}

export function assetPurgeConfirmation(
  asset: ArchivedAsset,
  inspection: AssetPurgeInspection | null,
  typed: { path: string; inspectionId?: string },
) {
  if (
    !inspection ||
    !canConfirmAssetPurge(asset, inspection) ||
    typed.path !== asset.path ||
    typed.inspectionId !== inspection.id
  )
    return;
  return { confirm_path: typed.path, inspection_id: inspection.id };
}
