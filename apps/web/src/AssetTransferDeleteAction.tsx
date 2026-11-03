import { useState } from 'react';
import { createPortal } from 'react-dom';
import { LoaderCircle, Trash2 } from 'lucide-react';
import { useAuth } from './auth';
import { canManageResource } from './resourcePermissions';
import { ErrorNotice, Modal } from './components';
import { canDeleteAssetTransferRecord, deleteAssetTransferRecord } from './assetTransferDeletion';
import type { AssetTransfer } from './assetTransfer';

export function AssetTransferDeleteAction({
  item,
  compact = false,
  disabled = false,
  onDeleted,
  onBusyChange,
}: {
  item: AssetTransfer;
  compact?: boolean;
  disabled?: boolean;
  onDeleted?: () => void;
  onBusyChange?: (busy: boolean) => void;
}) {
  const { can, user } = useAuth();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  if (
    !can('assets.transfer') ||
    !canManageResource(user, item.owner_user_id) ||
    !canDeleteAssetTransferRecord(item)
  )
    return null;
  const close = () => {
    if (!busy) setConfirm(false);
  };
  const remove = async () => {
    if (
      busy ||
      disabled ||
      !can('assets.transfer') ||
      !canManageResource(user, item.owner_user_id) ||
      !canDeleteAssetTransferRecord(item)
    )
      return;
    setBusy(true);
    onBusyChange?.(true);
    setError(undefined);
    try {
      await deleteAssetTransferRecord(item.id);
      setConfirm(false);
      onDeleted?.();
    } catch (cause) {
      setError(cause as Error);
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  };
  return (
    <>
      <button
        type="button"
        className={compact ? 'icon-button danger-text' : 'button outline danger-text'}
        aria-label={`删除同步记录 ${item.name}`}
        title="删除同步记录"
        disabled={busy || disabled}
        onClick={() => {
          setError(undefined);
          setConfirm(true);
        }}
      >
        <Trash2 size={compact ? 16 : 14} />
        {!compact && '删除同步记录'}
      </button>
      {confirm &&
        createPortal(
          <Modal
            title="删除数据同步记录？"
            close={close}
            footer={
              <>
                <button type="button" className="button outline" disabled={busy} onClick={close}>
                  取消
                </button>
                <button
                  type="button"
                  className="button danger"
                  disabled={busy || disabled}
                  onClick={remove}
                >
                  {busy && <LoaderCircle size={16} className="spin" />}
                  {busy ? '正在删除…' : '确认删除同步记录'}
                </button>
              </>
            }
          >
            <p>
              删除「{item.name}
              」后，此同步记录及其日志将不再显示。源资产、已登记的目标资产以及机器上的文件和部分同步文件继续保留。
            </p>
            <ErrorNotice error={error} />
          </Modal>,
          document.body,
        )}
    </>
  );
}
