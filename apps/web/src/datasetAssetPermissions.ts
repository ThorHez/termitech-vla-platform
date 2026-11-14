import type { User } from './auth';
import type { Asset } from './types';

type DatasetAsset = Pick<Asset, 'kind' | 'status' | 'state' | 'registered_by_user_id'>;
type PermissionCheck = (permission: string) => boolean;

export function canEditDatasetAsset(
  user: Pick<User, 'id' | 'role'> | undefined,
  asset: DatasetAsset,
  can: PermissionCheck,
) {
  return (
    asset.kind === 'DATASET' &&
    asset.status !== 'ARCHIVED' &&
    asset.state !== 'ARCHIVED' &&
    (can('assets.write') ||
      (user?.role === 'operator' &&
        Boolean(user.id) &&
        asset.registered_by_user_id === user.id &&
        can('datasets.write')))
  );
}

export function canBindDatasetNorm(
  user: Pick<User, 'id' | 'role'> | undefined,
  asset: DatasetAsset,
  can: PermissionCheck,
) {
  return (asset.status || asset.state) !== 'UNAVAILABLE' && canEditDatasetAsset(user, asset, can);
}
