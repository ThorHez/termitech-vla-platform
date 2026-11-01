import { useId, useRef, useState } from 'react';
import { Link2, LoaderCircle } from 'lucide-react';
import { api } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Field, Modal } from './components';
import { useData } from './data';
import { PathAutocomplete } from './PathAutocomplete';
import {
  DATASET_NORM_PATH,
  boundDatasetNorm,
  datasetNormBindingPayload,
  datasetNormCandidates,
  datasetNormInputError,
  datasetNormSelectionError,
} from './datasetNormAssets';
import type { Asset, Node } from './types';
import './dataset-norm-binding.css';

export function DatasetNormSummary({ asset, assets }: { asset: Asset; assets: Asset[] }) {
  const norm = boundDatasetNorm(asset, assets);
  const error = datasetNormSelectionError(asset.node_id, asset.norm_asset_id, assets);
  return (
    <span
      className={`dataset-norm-summary ${error ? 'unavailable' : ''}`}
      title={norm?.path || error}
    >
      <Link2 size={11} aria-hidden="true" />
      norm_stats：{asset.norm_asset_id ? norm?.name || asset.norm_asset_id : '未绑定（可选）'}
      {error && ' · 不可用'}
    </span>
  );
}

export function DatasetNormField({
  nodeId,
  assets,
  value,
  onChange,
  pathValue = '',
  onPathChange,
  node,
  disabled = false,
}: {
  nodeId: string;
  assets: Asset[];
  value: string;
  onChange: (value: string) => void;
  pathValue?: string;
  onPathChange?: (value: string) => void;
  node?: Node;
  disabled?: boolean;
}) {
  const candidates = datasetNormCandidates(nodeId, assets);
  const byPath = Boolean(onPathChange) && value === DATASET_NORM_PATH;
  const selected = byPath ? undefined : assets.find((asset) => asset.id === value);
  const error = byPath
    ? datasetNormInputError(nodeId, value, pathValue, assets)
    : datasetNormSelectionError(nodeId, value, assets);
  return (
    <div className="dataset-norm-field">
      <Field
        label="绑定 norm_stats（可选）"
        hint={
          onPathChange
            ? '选择同一台机器上的归一化参数，也可以直接填写路径。'
            : '仅选择同一台机器上的归一化参数资产；可稍后绑定或解除。'
        }
      >
        <select
          aria-label="绑定 norm_stats"
          value={value}
          disabled={disabled || !nodeId}
          aria-invalid={Boolean(error)}
          onChange={(event) => {
            const next = event.target.value;
            if (next !== DATASET_NORM_PATH) onPathChange?.('');
            onChange(next);
          }}
        >
          <option value="">不绑定</option>
          {onPathChange && <option value={DATASET_NORM_PATH}>直接填写路径（自动登记）</option>}
          {value && !byPath && !candidates.some((asset) => asset.id === value) && (
            <option value={value} disabled>
              {selected?.name || value} · 不可用
            </option>
          )}
          {candidates.map((asset) => (
            <option key={asset.id} value={asset.id}>
              {asset.name} · {asset.version}
            </option>
          ))}
        </select>
      </Field>
      {byPath && (
        <div className="dataset-norm-path-entry">
          <PathAutocomplete
            node={node?.id === nodeId ? node : undefined}
            value={pathValue}
            onChange={onPathChange!}
            label="norm_stats 绝对路径"
            placeholder="例如 /data/norm_stats 或 /data/norm_stats.json"
            disabled={disabled || !nodeId}
            hint="支持目录或 norm_stats.json 文件。保存时自动登记并绑定，已有登记会复用；文件内容在训练预检中验证。"
          />
        </div>
      )}
      {selected && <code className="dataset-norm-path">{selected.path}</code>}
      {error && (
        <p className="dataset-norm-error" role="alert">
          {error}
        </p>
      )}
      {nodeId && !candidates.length && !value && (
        <p className="field-hint">
          {onPathChange
            ? '此机器还没有可选的 norm_stats，可选择“直接填写路径”自动登记。'
            : '此机器还没有可用的 norm_stats 资产，可先登记归一化参数后再绑定。'}
        </p>
      )}
    </div>
  );
}

export function DatasetNormBindingDialog({ asset, close }: { asset: Asset; close: () => void }) {
  const { assets, nodes } = useData();
  const { can } = useAuth();
  const [value, setValue] = useState(asset.norm_asset_id || '');
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const pending = useRef(false);
  const formId = useId();
  const blocker = datasetNormInputError(asset.node_id, value, path, assets.data || []);
  const writable =
    can('assets.write') &&
    asset.kind === 'DATASET' &&
    !['ARCHIVED', 'UNAVAILABLE'].includes(asset.status || asset.state || '');
  if (!writable)
    return (
      <Modal title="无法修改绑定" close={close}>
        只有具备资产编辑权限的账号可以修改可用数据集的 norm_stats 绑定。
      </Modal>
    );
  return (
    <Modal
      title="绑定 norm_stats"
      subtitle="为数据集保存可选的归一化参数关系。"
      close={() => {
        if (!pending.current) close();
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
            disabled={busy || Boolean(blocker)}
          >
            {busy ? <LoaderCircle size={15} className="spin" /> : <Link2 size={15} />}
            {value === DATASET_NORM_PATH ? '自动登记并绑定' : value ? '保存绑定' : '保存为不绑定'}
          </button>
        </>
      }
    >
      <form
        id={formId}
        className="dataset-norm-form"
        onSubmit={async (event) => {
          event.preventDefault();
          if (pending.current || blocker || !writable) return;
          pending.current = true;
          setBusy(true);
          setError(undefined);
          try {
            await api(`/assets/${encodeURIComponent(asset.id)}/norm-binding`, {
              method: 'PUT',
              body: JSON.stringify(datasetNormBindingPayload(value, path)),
            });
            assets.refresh();
            close();
          } catch (err) {
            setError(err instanceof Error ? err : new Error(String(err)));
          } finally {
            pending.current = false;
            setBusy(false);
          }
        }}
      >
        <div className="dataset-norm-source">
          <strong>{asset.name}</strong>
          <code>{asset.path}</code>
        </div>
        <ErrorNotice error={error || assets.error} title="无法保存绑定" />
        <DatasetNormField
          nodeId={asset.node_id}
          assets={assets.data || []}
          value={value}
          onChange={setValue}
          pathValue={path}
          onPathChange={setPath}
          node={nodes.data?.find((node) => node.id === asset.node_id)}
          disabled={busy}
        />
        <p className="field-hint">
          保存后可在同步时选择一并传输；不改变已经创建的训练任务及其归一化参数。
        </p>
      </form>
    </Modal>
  );
}
