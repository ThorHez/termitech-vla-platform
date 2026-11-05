import { useEffect, useRef, useState } from 'react';
import { LoaderCircle } from 'lucide-react';
import { useAuth } from './auth';
import { canArchiveAsset, createAssetArchiveOperation } from './assetArchive';
import { ErrorNotice, Modal } from './components';
import type { Asset } from './types';

interface Props {
  asset: Asset;
  ownerUserId: string;
  close: () => void;
  onArchived: () => void;
}

export function AssetArchiveDialog(props: Props) {
  const { user, can } = useAuth();
  if (user?.id !== props.ownerUserId || !canArchiveAsset(user, props.asset, can)) return null;
  return <AssetArchiveDialogContent key={`${props.ownerUserId}:${props.asset.id}`} {...props} />;
}

function AssetArchiveDialogContent({ asset, ownerUserId, close, onArchived }: Props) {
  const { user, can } = useAuth();
  const archive = useRef(createAssetArchiveOperation());
  const controller = useRef<AbortController | undefined>(undefined);
  const mounted = useRef(true);
  const pending = useRef(false);
  const allowed = useRef(true);
  allowed.current = user?.id === ownerUserId && canArchiveAsset(user, asset, can);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const isActive = () => mounted.current && allowed.current;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.current?.abort();
    };
  }, []);
  const dismiss = () => {
    if (!pending.current) close();
  };
  const confirm = async () => {
    if (pending.current || !isActive()) return;
    const abort = new AbortController();
    controller.current = abort;
    pending.current = true;
    setBusy(true);
    setError(undefined);
    try {
      await archive.current(asset.id, {
        isActive,
        signal: AbortSignal.any([abort.signal, AbortSignal.timeout(20000)]),
        onArchived,
      });
    } catch (cause) {
      if (isActive() && !abort.signal.aborted) setError(cause as Error);
    } finally {
      pending.current = false;
      if (isActive()) setBusy(false);
    }
  };
  return (
    <Modal
      title="删除资产登记"
      close={dismiss}
      footer={
        <>
          <button className="button outline" disabled={busy} onClick={dismiss}>
            取消
          </button>
          <button className="button danger" disabled={busy} onClick={() => void confirm()}>
            {busy && <LoaderCircle size={16} className="spin" />}确认删除登记
          </button>
        </>
      }
    >
      <p>
        删除「{asset.name}
        」的资产登记，将其从新训练的可选资产中移除。历史训练关联和本机文件会继续保留。
      </p>
      <ErrorNotice error={error} />
    </Modal>
  );
}
