import { createOperationClient } from './api';
import type { User } from './auth';
import type { Asset } from './types';

export function canArchiveAsset(
  user: Pick<User, 'id' | 'role'> | undefined,
  asset: Pick<Asset, 'kind' | 'state' | 'status' | 'registered_by_user_id'>,
  can: (permission: string) => boolean,
) {
  if (!user?.id || asset.status === 'ARCHIVED' || asset.state === 'ARCHIVED') return false;
  return (
    can('assets.write') ||
    (user.role === 'operator' &&
      can('assets.archive') &&
      ['DATASET', 'NORM', 'MODEL'].includes(asset.kind) &&
      asset.registered_by_user_id === user.id)
  );
}

/** A confirmation and its completion callback belong to the opening session. */
export function createAssetArchiveOperation() {
  const client = createOperationClient();
  return async (
    assetId: string,
    {
      isActive,
      signal,
      onArchived,
    }: { isActive: () => boolean; signal: AbortSignal; onArchived: () => void },
  ) => {
    const requireActive = () => {
      client.requireCurrent();
      signal.throwIfAborted();
      if (!isActive()) throw new DOMException('删除窗口已失效。', 'AbortError');
    };
    requireActive();
    await client.post(`/assets/${encodeURIComponent(assetId)}/archive`, {}, { signal });
    requireActive();
    onArchived();
  };
}
