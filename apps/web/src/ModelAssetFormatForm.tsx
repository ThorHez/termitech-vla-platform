import { useEffect, useId, useRef, useState } from 'react';
import { LoaderCircle, Save } from 'lucide-react';
import { createOperationClient } from './api';
import type { User } from './auth';
import { ErrorNotice, Field, Modal } from './components';
import { modelAssetFormat, modelFormatLabels } from './modelAssetFormat';
import type { Asset } from './types';

type DeclaredFormat = keyof typeof modelFormatLabels;
export type ModelFormatChoice = DeclaredFormat | 'other' | '';

const modelFormats: DeclaredFormat[] = ['openpi-orbax', 'openpi-pytorch'];

export function modelAssetFormatLabel(asset: Asset) {
  const format = modelAssetFormat(asset);
  return format ? modelFormatLabels[format] : undefined;
}

export function canEditModelAssetFormat(
  user: Pick<User, 'id' | 'role'> | undefined,
  asset: Asset,
  can: (permission: string) => boolean,
) {
  const metadata = asset.metadata ?? {};
  return Boolean(
    user?.id &&
      asset.kind === 'MODEL' &&
      ['READY', 'REGISTERED'].includes(asset.status || asset.state || '') &&
      asset.origin_attempt_id == null &&
      metadata.format !== 'openpi-transferred' &&
      !Object.hasOwn(metadata, 'transfer_id') &&
      !Object.hasOwn(metadata, 'asset_sync') &&
      can('assets.register') &&
      (user.role === 'admin' || asset.registered_by_user_id === user.id),
  );
}

export function ModelAssetFormatField({
  value,
  onChange,
  disabled,
}: {
  value: ModelFormatChoice;
  onChange: (value: ModelFormatChoice) => void;
  disabled?: boolean;
}) {
  return (
    <Field label="模型格式" required>
      <select
        aria-label="模型格式"
        required
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value as ModelFormatChoice)}
      >
        <option value="" disabled>
          选择模型格式
        </option>
        {modelFormats.map((format) => (
          <option key={format} value={format}>
            {modelFormatLabels[format]}
          </option>
        ))}
        <option value="other">其他模型格式</option>
      </select>
    </Field>
  );
}

export function ModelAssetFormatDialog({
  asset,
  close,
  saved,
}: {
  asset: Asset;
  close: () => void;
  saved: () => void;
}) {
  const initialFormat = modelAssetFormat(asset) ?? 'other';
  const [format, setFormat] = useState<ModelFormatChoice>(initialFormat);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [client] = useState(createOperationClient);
  const request = useRef<AbortController>(undefined);
  const formId = useId();
  useEffect(() => () => request.current?.abort(), []);
  return (
    <Modal
      title="修改模型格式"
      subtitle={asset.name}
      close={() => {
        if (!request.current) close();
      }}
      footer={
        <>
          <button type="button" className="button outline" disabled={busy} onClick={close}>
            取消
          </button>
          <button
            type="submit"
            form={formId}
            className="button primary"
            disabled={busy || !format || format === initialFormat}
          >
            {busy ? <LoaderCircle size={15} className="spin" /> : <Save size={15} />}
            保存格式
          </button>
        </>
      }
    >
      <form
        id={formId}
        onSubmit={async (event) => {
          event.preventDefault();
          if (request.current || !format || format === initialFormat) return;
          const controller = new AbortController();
          request.current = controller;
          setBusy(true);
          setError(undefined);
          try {
            await client.api<Asset>(`/assets/${encodeURIComponent(asset.id)}/model-format`, {
              method: 'PATCH',
              signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
              body: JSON.stringify({ format: format === 'other' ? null : format }),
            });
            if (!controller.signal.aborted) saved();
          } catch (err) {
            if (!controller.signal.aborted)
              setError(err instanceof Error ? err : new Error(String(err)));
          } finally {
            request.current = undefined;
            if (!controller.signal.aborted) setBusy(false);
          }
        }}
      >
        <ErrorNotice error={error} title="无法保存模型格式" />
        <ModelAssetFormatField value={format} onChange={setFormat} disabled={busy} />
      </form>
    </Modal>
  );
}
