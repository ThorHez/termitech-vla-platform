import { useEffect, useId, useRef, useState } from 'react';
import {
  ArrowRight,
  CheckCircle2,
  FolderSync,
  LoaderCircle,
  RefreshCw,
  Square,
} from 'lucide-react';
import { date, navigate, post, useApi } from './api';
import { useAuth } from './auth';
import { canManageResource, ResourceOwner } from './resourcePermissions';
import {
  Back,
  Badge,
  Empty,
  ErrorNotice,
  Field,
  Loading,
  Modal,
  PageHeading,
  Panel,
  ProgressBar,
} from './components';
import { assetKinds, useData } from './data';
import { nodeAddressText } from './nodeAddress';
import { AssetSyncDirectory } from './AssetSyncDirectory';
import {
  boundDatasetNorm,
  defaultDatasetNormSync,
  datasetNormSelectionError,
  datasetNormTransferError,
} from './datasetNormAssets';
import { transferBytes, transferElapsed } from './checkpointTransfer';
import {
  assetSyncBlocker,
  assetSyncDirectory,
  assetTransferCreateRequest,
  assetTargetNameError,
  assetTargetPath,
  assetTransferLabels,
  assetTransferTaskPath,
  assetTransferTerminal,
  defaultAssetTargetName,
} from './assetTransfer';
import type { AssetSyncDirectoryDraft, AssetTransfer } from './assetTransfer';
import type { Asset } from './types';
import './asset-transfers.css';
import { AssetTransferDeleteAction } from './AssetTransferDeleteAction';

export function AssetSyncDialog({
  asset,
  close,
  created,
}: {
  asset: Asset;
  close: () => void;
  created: (transfer: AssetTransfer) => void;
}) {
  const { can } = useAuth();
  const { nodes, assets } = useData();
  const [targetId, setTargetId] = useState('');
  const [directoryDraft, setDirectoryDraft] = useState<AssetSyncDirectoryDraft>({ parent: '' });
  const [targetName, setTargetName] = useState(() => defaultAssetTargetName(asset));
  const [includeNormStats, setIncludeNormStats] = useState(() =>
    defaultDatasetNormSync(asset, assets.data || []),
  );
  const [excludeTrainState, setExcludeTrainState] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const pending = useRef(false);
  const lastRequest = useRef<{ signature: string; key: string } | undefined>(undefined);
  const formId = useId();
  const sourceNode = nodes.data?.find((node) => node.id === asset.node_id);
  const targetNode = nodes.data?.find((node) => node.id === targetId);
  const targets = (nodes.data || []).filter((node) => node.id !== asset.node_id);
  const { directory, error: directoryError } = assetSyncDirectory(directoryDraft);
  const blocker =
    datasetNormTransferError(asset, assets.data || [], includeNormStats) ||
    directoryError ||
    assetSyncBlocker({ asset, sourceNode, targetNode, directory, targetName });
  const preview = assetTargetPath(directory, targetName);
  if (!can('assets.transfer')) {
    return (
      <Modal title="权限不足" close={close}>
        同步资产需要操作员或管理员权限。
      </Modal>
    );
  }
  return (
    <Modal
      title="同步到其他机器"
      subtitle="源机器直传目标机器；控制台负责调度、校验和进度。"
      close={() => {
        if (!pending.current) close();
      }}
      className="asset-sync-modal"
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
            {busy ? <LoaderCircle size={15} className="spin" /> : <FolderSync size={15} />}
            {busy ? '正在创建同步任务…' : '开始同步'}
          </button>
        </>
      }
    >
      <form
        id={formId}
        className="asset-sync-form"
        onSubmit={async (event) => {
          event.preventDefault();
          if (pending.current || blocker || !can('assets.transfer')) return;
          pending.current = true;
          setBusy(true);
          setError(undefined);
          const request = assetTransferCreateRequest(
            asset,
            targetId,
            directory,
            targetName,
            includeNormStats,
            excludeTrainState,
          );
          const signature = JSON.stringify(request);
          if (lastRequest.current?.signature !== signature) {
            lastRequest.current = { signature, key: crypto.randomUUID() };
          }
          try {
            const transfer = await post<AssetTransfer>('/asset-transfers', {
              ...request,
              idempotency_key: lastRequest.current.key,
            });
            created(transfer);
          } catch (err) {
            setError(err instanceof Error ? err : new Error(String(err)));
          } finally {
            pending.current = false;
            setBusy(false);
          }
        }}
      >
        <div className="asset-sync-source">
          <strong>{asset.name}</strong>
          <span>
            {assetKinds[asset.kind] || asset.kind} · {asset.version}
          </span>
          <span>
            {sourceNode?.name || asset.node_id}
            {sourceNode && nodeAddressText(sourceNode) ? ` · ${nodeAddressText(sourceNode)}` : ''}
          </span>
          <code>{asset.path}</code>
        </div>
        <ErrorNotice error={error} title="无法开始同步" />
        <AssetSyncNormOption
          asset={asset}
          assets={assets.data || []}
          checked={includeNormStats}
          onChange={setIncludeNormStats}
          disabled={busy}
        />
        <AssetSyncModelOption
          asset={asset}
          checked={excludeTrainState}
          onChange={setExcludeTrainState}
          disabled={busy}
        />
        <Field label="目标机器" required>
          <select
            required
            aria-label="目标机器"
            value={targetId}
            disabled={busy}
            onChange={(event) => {
              setTargetId(event.target.value);
              setDirectoryDraft({ parent: '' });
              setError(undefined);
            }}
          >
            <option value="">选择另一台在线机器</option>
            {targets.map((node) => (
              <option key={node.id} value={node.id} disabled={node.status !== 'ONLINE'}>
                {node.name}
                {nodeAddressText(node) ? ` · ${nodeAddressText(node)}` : ''}
                {node.status === 'ONLINE' ? '' : ' · 离线'}
              </option>
            ))}
          </select>
        </Field>
        {!targets.some((node) => node.status === 'ONLINE') && (
          <p className="asset-sync-notice">
            暂无其他在线机器，请先在机器资源中接入或恢复目标机器。
          </p>
        )}
        <AssetSyncDirectory
          key={targetId}
          node={targetNode}
          draft={directoryDraft}
          onChange={setDirectoryDraft}
          busy={busy}
        />
        <Field label="目标子目录名" required hint="仅填写一层目录名，不包含路径分隔符。">
          <input
            required
            aria-label="目标子目录名"
            maxLength={160}
            value={targetName}
            disabled={busy}
            aria-invalid={Boolean(targetName && assetTargetNameError(targetName))}
            onChange={(event) => setTargetName(event.target.value)}
          />
        </Field>
        <div className="asset-sync-destination" aria-live="polite">
          <span>目标目录</span>
          <code>{preview || '选择父目录并填写子目录名后显示'}</code>
        </div>
        <p className="asset-sync-notice">
          目标机器需能访问源机器的
          SSH。提交后可离开页面，在“后台任务”中查看进度、取消或重试。目标已有目录不会被覆盖；重试只继续本任务保留的部分文件。完成文件校验后会自动登记目标资产，并保留数据特征。训练仍需预检。
        </p>
        {blocker && (
          <p className="field-hint" role="status">
            {blocker}
          </p>
        )}
      </form>
    </Modal>
  );
}

export function AssetSyncNormOption({
  asset,
  assets,
  checked,
  onChange,
  disabled = false,
}: {
  asset: Asset;
  assets: Asset[];
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  if (asset.kind !== 'DATASET') return null;
  const norm = boundDatasetNorm(asset, assets);
  const bindingError = datasetNormSelectionError(asset.node_id, asset.norm_asset_id, assets);
  const unavailable = !asset.norm_asset_id || Boolean(bindingError);
  return (
    <div className="asset-sync-norm-option">
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled || (unavailable && !checked)}
          onChange={(event) => onChange(event.target.checked)}
        />
        一并同步已绑定的 norm_stats
      </label>
      {asset.norm_asset_id ? (
        <div className="asset-sync-norm-description">
          <strong>{norm?.name || asset.norm_asset_id}</strong>
          {norm && <code>{norm.path}</code>}
          {bindingError && (
            <p className="dataset-norm-error" role="status">
              {bindingError} 可仅同步数据集，或返回资产列表修改绑定。
            </p>
          )}
        </div>
      ) : (
        <p className="asset-sync-notice">
          此数据集尚未绑定 norm_stats。可在资产列表中先绑定，或仅同步数据集。
        </p>
      )}
      {checked && !unavailable && (
        <p className="asset-sync-notice">
          归一化参数保存到独立的 norm-stats 目录；两份文件均校验完成后，自动绑定到目标数据集。
        </p>
      )}
    </div>
  );
}

export function AssetSyncModelOption({
  asset,
  checked,
  onChange,
  disabled = false,
}: {
  asset: Asset;
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  if (asset.kind !== 'MODEL') return null;
  return (
    <div className="asset-sync-norm-option">
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        不同步 train_state/
      </label>
      <p className="asset-sync-notice">
        仅跳过所选模型目录根部的 train_state/，params
        与其他文件保留。需要恢复完整训练状态时，请保留该目录。
      </p>
    </div>
  );
}

export function AssetNormTransferInfo({ item }: { item: AssetTransfer }) {
  if (!item.include_norm_stats) return null;
  const norm = item.norm_sync;
  const fullyBound =
    item.state === 'SUCCEEDED' && Boolean(item.target_asset_id && norm?.target_asset_id);
  const parentStopped = ['FAILED', 'CANCELLED', 'INTERRUPTED'].includes(item.state);
  const notStarted = !norm || norm.state === 'QUEUED';
  const activeNorm = [
    'QUEUED',
    'PREPARING',
    'SCANNING',
    'TRANSFERRING',
    'VERIFYING',
    'CANCELLING',
  ].includes(norm?.state || 'QUEUED');
  const phase = fullyBound
    ? '已同步并绑定'
    : norm?.state === 'SUCCEEDED'
      ? parentStopped
        ? '文件已校验，整体任务未完成'
        : '文件已校验，等待整体登记'
      : parentStopped && activeNorm
        ? notStarted
          ? '尚未执行'
          : '已随整体任务停止'
        : assetTransferLabels[norm?.state || 'QUEUED'] || norm?.state;
  return (
    <section className="asset-transfer-norm" aria-label="附带 norm_stats 同步">
      <div className="asset-transfer-norm-heading">
        <strong>附带 norm_stats</strong>
        <Badge
          state={
            fullyBound
              ? 'READY'
              : parentStopped && activeNorm
                ? 'CANCELLED'
                : norm?.state || 'QUEUED'
          }
          label={phase}
        />
      </div>
      <span>{norm?.name || '已绑定的归一化参数'}</span>
      <dl className="asset-transfer-paths">
        <div>
          <dt>源参数</dt>
          <dd>
            <code>{norm?.source_path || '等待来源确认'}</code>
          </dd>
        </div>
        <div>
          <dt>目标参数</dt>
          <dd>
            <code>
              {norm?.asset_path ||
                norm?.target_path ||
                (norm?.target_name
                  ? assetTargetPath(item.target_directory, norm.target_name)
                  : undefined) ||
                '准备后显示'}
            </code>
          </dd>
        </div>
      </dl>
      {Number(norm?.bytes_total) > 0 && (
        <p className="asset-sync-notice">
          {transferBytes(norm?.bytes_transferred || 0)} / {transferBytes(norm!.bytes_total!)}
          {Number.isFinite(norm?.progress_pct)
            ? ` · ${fullyBound ? 100 : Math.min(99, norm!.progress_pct!).toFixed(1)}%`
            : ''}
        </p>
      )}
      <ErrorNotice
        error={norm?.error ? new Error(`${norm.error.code} · ${norm.error.message}`) : undefined}
        title="norm_stats 同步未完成"
      />
      {!fullyBound && (
        <p className="asset-sync-notice">
          数据集与归一化参数均校验完成后，才会登记目标资产并建立绑定。
        </p>
      )}
    </section>
  );
}

export function AssetSyncBackgroundNotice({ transfer }: { transfer: AssetTransfer }) {
  return (
    <div className="asset-sync-background-notice" role="status">
      <CheckCircle2 size={18} aria-hidden="true" />
      <div>
        <strong>数据同步已加入后台任务</strong>
        <span>
          {transfer.name} · 源机器直传目标机器，可继续操作或关闭页面，在“后台任务”中查看进度。
        </span>
      </div>
      <a className="button outline small" href={`#${assetTransferTaskPath(transfer.id)}`}>
        查看任务
        <ArrowRight size={13} />
      </a>
    </div>
  );
}

export function AssetTransferTaskDetail({ id }: { id: string }) {
  const detail = useApi<AssetTransfer>(`/asset-transfers/${encodeURIComponent(id)}`, 2000);
  const item = detail.data?.id === id ? detail.data : undefined;
  const { assets } = useData();
  const refreshed = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (
      item?.state === 'SUCCEEDED' &&
      item.target_asset_id &&
      refreshed.current !== item.target_asset_id
    ) {
      refreshed.current = item.target_asset_id;
      assets.refresh();
    }
  }, [item?.state, item?.target_asset_id, assets.refresh]);
  return (
    <>
      <Back to="/runs" label="返回训练任务" />
      <PageHeading
        eyebrow="BACKGROUND TASK"
        title={item?.name || '数据同步任务'}
        description="数据同步 · 后台执行，可离开页面后再回来查看。"
      />
      <ErrorNotice error={detail.error} retry={detail.refresh} title="同步任务读取失败" />
      {item ? (
        <Panel title="同步进度" subtitle="每 2 秒自动更新；准备、传输、校验与资产登记分别展示。">
          <AssetTransferCard
            item={item}
            changed={detail.refresh}
            deleted={() => navigate('/runs')}
            viewTarget={(nodeId) => navigate(`/assets?node=${encodeURIComponent(nodeId)}`)}
          />
        </Panel>
      ) : detail.loading ? (
        <Loading label="正在读取数据同步任务…" />
      ) : null}
    </>
  );
}

export function AssetTransferCard({
  item,
  changed,
  viewTarget,
  viewProgress,
  deleted,
}: {
  item: AssetTransfer;
  changed: () => void;
  viewTarget?: (nodeId: string) => void;
  viewProgress?: (transfer: AssetTransfer) => void;
  deleted?: () => void;
}) {
  const { nodes, refresh } = useData();
  const { can, user } = useAuth();
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<Error>();
  const [now, setNow] = useState(() => Date.now());
  const writable = can('assets.transfer') && canManageResource(user, item.owner_user_id);
  const terminal = assetTransferTerminal(item.state);
  const preparing = ['QUEUED', 'PREPARING', 'SCANNING'].includes(item.state);
  useEffect(() => {
    if (terminal) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [terminal]);
  const elapsed = transferElapsed(item.created_at, item.finished_at, now);
  const source = nodes.data?.find((node) => node.id === item.source_node_id);
  const target = nodes.data?.find((node) => node.id === item.target_node_id);
  const progress = Number.isFinite(item.progress_pct)
    ? Math.max(0, Math.min(99, item.progress_pct!))
    : 0;
  const registered = item.state === 'SUCCEEDED' && Boolean(item.target_asset_id);
  const perform = async (operation: 'cancel' | 'retry') => {
    if (busy || deleting || !writable) return;
    setBusy(true);
    setError(undefined);
    try {
      await post(`/asset-transfers/${encodeURIComponent(item.id)}/${operation}`);
      changed();
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };
  return (
    <article className="asset-transfer-card" aria-label={`数据同步 ${item.name}`}>
      <ResourceOwner ownerUserId={item.owner_user_id} />
      <div className="asset-transfer-heading">
        <div>
          <strong>
            <FolderSync size={15} />
            {item.name}
          </strong>
          <span>
            {assetKinds[item.source_kind] || item.source_kind} ·{' '}
            {source?.name || item.source_node_id}
            <ArrowRight size={12} />
            {target?.name || item.target_node_id}
          </span>
        </div>
        <Badge
          state={item.state === 'SUCCEEDED' ? 'READY' : item.state}
          label={
            item.state === 'SUCCEEDED' && !registered
              ? '文件同步完成'
              : assetTransferLabels[item.state] || item.state
          }
        />
      </div>
      {item.source_kind === 'MODEL' && (
        <section className="asset-transfer-model-scope" aria-label="模型同步范围">
          <div className="asset-transfer-model-scope-heading">
            <h4>模型同步范围</h4>
            <span
              className={`asset-transfer-policy ${item.exclude_train_state ? 'excludes-state' : ''}`}
            >
              {item.exclude_train_state ? '排除 train_state/' : '完整复制'}
            </span>
          </div>
          <dl>
            <div>
              <dt>模型根部 train_state/</dt>
              <dd>{item.exclude_train_state ? '不参与同步' : '参与同步（如存在）'}</dd>
            </div>
            <div>
              <dt>params 与其他文件</dt>
              <dd>正常同步（如存在）</dd>
            </div>
          </dl>
          <p>
            本任务配置已固定，重试沿用。排除选项只影响模型根部 train_state/，嵌套同名目录仍会同步。
          </p>
          {item.exclude_train_state && (
            <p>该目录若保存了优化器等训练状态，本副本不会包含这些状态。</p>
          )}
        </section>
      )}
      {preparing ? (
        <div className="asset-transfer-progress" role="status">
          <div
            className="progress teal asset-transfer-indeterminate"
            role="progressbar"
            aria-label={assetTransferLabels[item.state]}
          >
            <span />
          </div>
          <span>准备中</span>
        </div>
      ) : (
        <div className="asset-transfer-progress" aria-live="polite">
          <ProgressBar value={item.state === 'SUCCEEDED' ? 100 : progress} />
          <span>{item.state === 'SUCCEEDED' ? '100' : progress.toFixed(1)}%</span>
        </div>
      )}
      <div className="asset-transfer-metrics">
        {preparing ? (
          <span>文件总大小与数量待统计</span>
        ) : (
          <>
            <span>
              {transferBytes(item.bytes_transferred || 0)} / {transferBytes(item.bytes_total || 0)}
            </span>
            <span>
              {item.files_transferred ?? '—'} / {item.files_total || 0} 个文件
            </span>
          </>
        )}
        {!terminal && !preparing && Number(item.speed_bytes_per_second) > 0 && (
          <span>{transferBytes(item.speed_bytes_per_second!)}/s</span>
        )}
        <span>
          第 {item.attempt || 1} 次 · 创建于 {date(item.created_at)}
        </span>
        {elapsed && <span>任务耗时 {elapsed}</span>}
      </div>
      <dl className="asset-transfer-paths">
        <div>
          <dt>源资产</dt>
          <dd>
            <code>{item.source_path}</code>
          </dd>
        </div>
        <div>
          <dt>{registered ? '目标资产' : '目标目录'}</dt>
          <dd>
            <code>
              {item.asset_path ||
                item.target_path ||
                assetTargetPath(item.target_directory, item.target_name) ||
                '待确认'}
            </code>
          </dd>
        </div>
      </dl>
      <AssetNormTransferInfo item={item} />
      {item.state === 'CANCELLING' && (
        <p className="asset-sync-notice" role="status">
          正在停止传输，停止确认前保留当前任务。
        </p>
      )}
      {item.state === 'CANCELLED' && (
        <p className="asset-sync-notice" role="status">
          同步已停止，保留记录与本任务部分文件供重试。
        </p>
      )}
      <ErrorNotice
        error={
          error ||
          (item.error ? new Error(`${item.error.code} · ${item.error.message}`) : undefined)
        }
        title="同步未完成"
      />
      {registered && (
        <p className="asset-transfer-verified">
          <CheckCircle2 size={14} />
          {item.include_norm_stats && item.norm_sync?.target_asset_id
            ? '数据集与 norm_stats 均已校验并登记，已绑定目标数据集；训练使用前仍需预检。'
            : '文件完整性校验通过，目标资产已登记；训练使用前仍需预检。'}
        </p>
      )}
      <div className="asset-transfer-actions">
        {viewProgress && (
          <button type="button" className="button outline small" onClick={() => viewProgress(item)}>
            {terminal ? '查看详情' : '查看进度'}
            <ArrowRight size={13} />
          </button>
        )}
        {writable && ['FAILED', 'CANCELLED', 'INTERRUPTED'].includes(item.state) && (
          <button
            type="button"
            className="button outline small"
            disabled={busy || deleting}
            onClick={() => void perform('retry')}
          >
            <RefreshCw size={13} />
            重试同步
          </button>
        )}
        {writable && !terminal && (
          <button
            type="button"
            className="button ghost small"
            disabled={busy || deleting || item.state === 'CANCELLING'}
            onClick={() => void perform('cancel')}
          >
            <Square size={12} />
            取消同步
          </button>
        )}
        {registered && viewTarget && (
          <button
            type="button"
            className="button primary small"
            onClick={() => viewTarget(item.target_node_id)}
          >
            查看目标资产
            <ArrowRight size={13} />
          </button>
        )}
        <AssetTransferDeleteAction
          item={item}
          disabled={busy}
          onBusyChange={setDeleting}
          onDeleted={() => {
            changed();
            refresh();
            deleted?.();
          }}
        />
        {busy && <LoaderCircle size={14} className="spin" />}
      </div>
      {item.logs && (
        <details className="asset-transfer-logs">
          <summary>同步日志</summary>
          <pre>{item.logs}</pre>
        </details>
      )}
    </article>
  );
}

export function AssetTransferProgressDialog({
  initialTransfer,
  close,
  changed,
  viewTarget,
}: {
  initialTransfer: AssetTransfer;
  close: () => void;
  changed: () => void;
  viewTarget?: (nodeId: string) => void;
}) {
  const detail = useApi<AssetTransfer>(
    `/asset-transfers/${encodeURIComponent(initialTransfer.id)}`,
    2000,
  );
  const item = detail.data?.id === initialTransfer.id ? detail.data : initialTransfer;
  return (
    <Modal
      title="数据同步进度"
      subtitle="同步阶段、已传大小与速度持续更新。"
      close={close}
      className="asset-transfer-progress-modal"
      footer={
        <button type="button" className="button outline" onClick={close}>
          关闭
        </button>
      }
    >
      <ErrorNotice
        error={detail.error}
        retry={detail.refresh}
        title="进度更新失败，以下保留最近状态"
      />
      <AssetTransferCard
        item={item}
        changed={() => {
          detail.refresh();
          changed();
        }}
        viewTarget={viewTarget}
        deleted={close}
      />
      <p className="asset-sync-notice" role="status">
        每 2 秒自动更新。关闭窗口不会取消同步，可在“训练任务”中重新查看进度。
      </p>
    </Modal>
  );
}

export function AssetTransfers({
  revision,
  viewTarget,
  viewProgress,
}: {
  revision: number;
  viewTarget: (nodeId: string) => void;
  viewProgress?: (transfer: AssetTransfer) => void;
}) {
  const { enabled, assets, nodes } = useData();
  const [sourceFilter, setSourceFilter] = useState('');
  const list = useApi<AssetTransfer[]>(
    enabled
      ? `/asset-transfers?limit=50&revision=${revision}${sourceFilter ? `&source_asset_id=${encodeURIComponent(sourceFilter)}` : ''}`
      : null,
    3000,
  );
  const seenSuccess = useRef(new Set<string>());
  useEffect(() => setSourceFilter(''), [revision]);
  useEffect(() => {
    let changed = false;
    for (const item of list.data || []) {
      if (item.state === 'SUCCEEDED' && item.target_asset_id && !seenSuccess.current.has(item.id)) {
        seenSuccess.current.add(item.id);
        changed = true;
      }
    }
    if (changed) assets.refresh();
  }, [list.data, assets.refresh]);
  if (!list.error && !list.data?.length && !sourceFilter) return null;
  const items = (list.data || []).filter(
    (item) => !sourceFilter || item.source_asset_id === sourceFilter,
  );
  const sourceChoices = new Map(
    (assets.data || []).map((asset) => [
      asset.id,
      {
        id: asset.id,
        name: asset.name,
        nodeId: asset.node_id,
      },
    ]),
  );
  for (const item of list.data || []) {
    if (!sourceChoices.has(item.source_asset_id))
      sourceChoices.set(item.source_asset_id, {
        id: item.source_asset_id,
        name: item.name,
        nodeId: item.source_node_id,
      });
  }
  return (
    <Panel
      title="数据同步"
      subtitle="新建任务由源机器直传目标机器；控制台负责调度、校验和进度。"
      className="asset-transfers-panel"
    >
      <ErrorNotice error={list.error} retry={list.refresh} title="同步任务读取失败" />
      <div className="asset-transfer-filter">
        <label>
          源资产
          <select
            aria-label="筛选数据同步的源资产"
            value={sourceFilter}
            onChange={(event) => setSourceFilter(event.target.value)}
          >
            <option value="">全部源资产</option>
            {[...sourceChoices.values()].map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} ·{' '}
                {nodes.data?.find((node) => node.id === item.nodeId)?.name || item.nodeId}
              </option>
            ))}
          </select>
        </label>
        <span>最近 {list.data?.length || 0} 条任务</span>
      </div>
      {list.loading && !list.data ? (
        <Loading label="正在读取同步任务…" />
      ) : items.length ? (
        items.map((item) => (
          <AssetTransferCard
            key={item.id}
            item={item}
            changed={list.refresh}
            viewTarget={viewTarget}
            viewProgress={viewProgress}
          />
        ))
      ) : (
        !list.error && (
          <Empty
            icon={FolderSync}
            title="没有符合条件的同步任务"
            description="选择其他源资产查看同步记录。"
          />
        )
      )}
    </Panel>
  );
}
