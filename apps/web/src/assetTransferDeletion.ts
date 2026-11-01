import { api } from './api';
import { assetTransferTerminal } from './assetTransfer';
import type { AssetTransfer } from './assetTransfer';

export function canDeleteAssetTransferRecord(item: Pick<AssetTransfer, 'state'>) {
  return assetTransferTerminal(item.state);
}

export function deleteAssetTransferRecord(id: string) {
  return api<void>(`/asset-transfers/${encodeURIComponent(id)}`, { method: 'DELETE' });
}
