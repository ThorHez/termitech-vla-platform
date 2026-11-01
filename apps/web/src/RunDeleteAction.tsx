import { useState } from 'react';
import { createPortal } from 'react-dom';
import { LoaderCircle, Trash2 } from 'lucide-react';
import { useAuth } from './auth';
import { ErrorNotice, Modal } from './components';
import { canDeleteRunRecord, deleteRunRecord } from './runDeletion';
import type { Run } from './types';

export function RunDeleteAction({
  run,
  compact = false,
  onDeleted,
}: {
  run: Run;
  compact?: boolean;
  onDeleted?: () => void;
}) {
  const { can } = useAuth();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  if (!can('training.write') || !canDeleteRunRecord(run)) return null;
  const close = () => {
    if (!busy) setConfirm(false);
  };
  const remove = async () => {
    if (busy || !can('training.write') || !canDeleteRunRecord(run)) return;
    setBusy(true);
    setError(undefined);
    try {
      await deleteRunRecord(run.id);
      setConfirm(false);
      onDeleted?.();
    } catch (cause) {
      setError(cause as Error);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <button
        type="button"
        className={compact ? 'icon-button danger-text' : 'button outline danger-text'}
        aria-label={`删除训练记录 ${run.name}`}
        title="删除训练记录"
        disabled={busy}
        onClick={() => {
          setError(undefined);
          setConfirm(true);
        }}
      >
        <Trash2 size={compact ? 16 : 14} />
        {!compact && '删除记录'}
      </button>
      {confirm &&
        createPortal(
          <Modal
            title="删除训练任务记录？"
            close={close}
            footer={
              <>
                <button type="button" className="button outline" disabled={busy} onClick={close}>
                  取消
                </button>
                <button type="button" className="button danger" disabled={busy} onClick={remove}>
                  {busy && <LoaderCircle size={16} className="spin" />}
                  {busy ? '正在删除…' : '确认删除记录'}
                </button>
              </>
            }
          >
            <p>
              删除「{run.name}
              」后，该任务及其日志、训练指标将不再显示。已登记的模型、检查点和机器文件继续保留。
            </p>
            <p className="field-hint">有关联部署记录时，需先删除部署记录。</p>
            <ErrorNotice error={error} />
          </Modal>,
          document.body,
        )}
    </>
  );
}
