import { useState } from 'react';
import { createPortal } from 'react-dom';
import { LoaderCircle, Trash2 } from 'lucide-react';
import { useAuth } from './auth';
import { ErrorNotice, Modal } from './components';
import { canDeleteDeploymentRecord, deleteDeploymentRecord } from './deploymentRecordDeletion';
import type { CheckpointTransfer } from './checkpointTransfer';
import type { DeploymentRecordSummary, PolicyDeployment } from './policyDeployment';

export function DeploymentRecordDeleteAction({
  record,
  deployment,
  transfers,
  compact = false,
  disabled = false,
  onDeleted,
}: {
  record: DeploymentRecordSummary;
  deployment?: PolicyDeployment | null;
  transfers?: CheckpointTransfer[];
  compact?: boolean;
  disabled?: boolean;
  onDeleted?: () => void;
}) {
  const { can } = useAuth();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  if (!can('training.write') || !canDeleteDeploymentRecord(record, deployment, transfers))
    return null;
  const close = () => {
    if (!busy) setConfirm(false);
  };
  const remove = async () => {
    if (
      busy ||
      disabled ||
      !can('training.write') ||
      !canDeleteDeploymentRecord(record, deployment, transfers)
    )
      return;
    setBusy(true);
    setError(undefined);
    try {
      await deleteDeploymentRecord(record.id);
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
        aria-label={`删除部署记录 ${record.name}`}
        title="删除部署记录"
        disabled={busy || disabled}
        onClick={() => {
          setError(undefined);
          setConfirm(true);
        }}
      >
        <Trash2 size={compact ? 16 : 14} />
        {!compact && '删除部署记录'}
      </button>
      {confirm &&
        createPortal(
          <Modal
            title="删除部署记录"
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
                  {busy ? '正在删除…' : '确认删除部署记录'}
                </button>
              </>
            }
          >
            <p>
              删除「{record.name}
              」后，此部署记录及相关日志将不再显示。机器上的模型、权重文件和已登记资产继续保留。
            </p>
            <ErrorNotice error={error} />
          </Modal>,
          document.body,
        )}
    </>
  );
}
