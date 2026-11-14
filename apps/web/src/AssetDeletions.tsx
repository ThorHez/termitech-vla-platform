import { useCallback, useEffect, useRef, useState } from 'react';
import { Archive, LoaderCircle, RefreshCw, RotateCcw, Server, Trash2 } from 'lucide-react';
import { createOperationClient, date, useApi } from './api';
import { useAuth } from './auth';
import {
  assetDeletionsPath,
  assetPurgeConfirmation,
  canConfirmAssetPurge,
  canManageAssetDeletions,
  canOpenAssetPurge,
} from './assetPurge';
import type { ArchivedAsset, AssetFileDeletion, AssetPurgeInspection } from './assetPurge';
import { Back, Badge, Empty, ErrorNotice, Field, Loading, Modal, PageHeading } from './components';
import { assetKinds, useData } from './data';
import type { Asset } from './types';
import './asset-deletions.css';

const fileStates = {
  PENDING: ['QUEUED', '等待删除'],
  RUNNING: ['PENDING', '正在删除'],
  DELETED: ['SUCCEEDED', '文件已删除'],
  MISSING: ['MISSING', '文件已不存在'],
  FAILED: ['FAILED', '删除失败'],
};

const inspectionStates = {
  PENDING: '等待文件检查',
  RUNNING: '正在检查文件',
  READY: '已核对文件路径',
  MISSING: '原路径已无文件，此操作将记录文件不存在。',
  FAILED: '文件检查失败',
};

export function AssetDeletionsPage() {
  const { user, can } = useAuth();
  if (!user || !canManageAssetDeletions(user, can))
    return (
      <>
        <Back to="/assets" label="返回数据与模型" />
        <PageHeading title="权限不足" description="已删除资产由平台管理员管理。" />
      </>
    );
  return <AssetDeletionsContent key={user.id} />;
}

function AssetDeletionsContent() {
  const { assets, overview } = useData();
  const ledger = useApi<ArchivedAsset[]>(assetDeletionsPath, 4000);
  const [selection, setSelection] = useState<
    {
      id: string;
      path: string;
    } & ({ action: 'purge' } | { action: 'restore'; archivedAt: string | null })
  >();
  const [submitted, setSubmitted] = useState<{
    assetId: string;
    receipt: AssetFileDeletion;
    previous?: ArchivedAsset[];
  }>();
  const [restored, setRestored] = useState<{ asset: Asset; previous?: ArchivedAsset[] }>();
  const latestRows = useRef(ledger.data);
  latestRows.current = ledger.data;
  const rows =
    restored && restored.previous === ledger.data
      ? ledger.data?.filter((asset) => asset.id !== restored.asset.id)
      : ledger.data;
  const selected = rows?.find(
    (asset) => asset.id === selection?.id && asset.path === selection?.path,
  );
  const onSubmitted = (assetId: string, receipt: AssetFileDeletion) => {
    setSubmitted({ assetId, receipt, previous: latestRows.current });
    setSelection(undefined);
    ledger.refresh();
  };
  const onRestored = (asset: Asset) => {
    setRestored({ asset, previous: latestRows.current });
    setSelection(undefined);
    assets.refresh();
    overview.refresh();
    ledger.refresh();
  };

  return (
    <>
      <Back to="/assets" label="返回数据与模型" />
      <PageHeading
        title="已删除资产"
        description="查看已删除的资产登记、原文件位置和删除进度。文件仍保留的资产可恢复登记。"
        actions={
          <button type="button" className="button outline" onClick={ledger.refresh}>
            <RefreshCw size={16} />
            刷新列表
          </button>
        }
      />
      {restored && (
        <div className="info-strip" role="status">
          <span>已恢复「{restored.asset.name}」的资产登记。</span>
          <a
            className="text-link"
            href={`#/assets?${new URLSearchParams({ node: restored.asset.node_id, asset: restored.asset.id })}`}
          >
            查看资产
          </a>
        </div>
      )}
      <ErrorNotice error={ledger.error} title="读取已删除资产失败" retry={ledger.refresh} />
      <div className="panel">
        {rows?.length ? (
          <div className="table-scroll">
            <table className="asset-deletions-table" aria-label="已删除资产">
              <thead>
                <tr>
                  <th scope="col">资产名称</th>
                  <th scope="col">机器和原路径</th>
                  <th scope="col">删除登记</th>
                  <th scope="col">文件状态</th>
                  <th scope="col">操作</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((asset) => {
                  const accepted =
                    submitted?.assetId === asset.id && submitted.previous === ledger.data
                      ? submitted.receipt
                      : undefined;
                  const deletion = accepted || asset.file_deletion;
                  const [state, label] = deletion
                    ? fileStates[deletion.status] || ['UNKNOWN', '状态未知']
                    : ['ARCHIVED', '尚未清理'];
                  const failure = deletion?.error_message || deletion?.error_code;
                  return (
                    <tr key={asset.id}>
                      <td>
                        <strong>{asset.name}</strong>
                        <span className="subtext">{assetKinds[asset.kind] || asset.kind}</span>
                      </td>
                      <td>
                        <a
                          className="node-link"
                          href={`#/nodes/${encodeURIComponent(asset.node_id)}`}
                        >
                          <Server size={13} aria-hidden="true" />
                          {asset.node_name || asset.node_id}
                        </a>
                        <code className="asset-purge-path">{asset.path}</code>
                      </td>
                      <td>
                        {asset.archived_at ? (
                          <time dateTime={asset.archived_at} title={asset.archived_at}>
                            {date(asset.archived_at)}
                          </time>
                        ) : (
                          <span>时间未记录</span>
                        )}
                        <span className="subtext">{asset.archived_by || '操作人未记录'}</span>
                      </td>
                      <td>
                        <Badge state={state} label={label} />
                        {deletion && (
                          <time
                            className="subtext"
                            dateTime={deletion.finished_at || deletion.requested_at}
                          >
                            {date(deletion.finished_at || deletion.requested_at)}
                          </time>
                        )}
                        {failure && <span className="subtext">{failure}</span>}
                      </td>
                      <td>
                        <div className="asset-deletion-actions">
                          {!accepted && asset.can_restore === true && (
                            <button
                              type="button"
                              className="button outline small"
                              aria-label={`恢复登记 ${asset.name}`}
                              disabled={Boolean(ledger.error)}
                              onClick={() =>
                                setSelection({
                                  id: asset.id,
                                  path: asset.path,
                                  action: 'restore',
                                  archivedAt: asset.archived_at,
                                })
                              }
                            >
                              <RotateCcw size={14} aria-hidden="true" />
                              恢复登记
                            </button>
                          )}
                          {!accepted && canOpenAssetPurge(asset) && (
                            <button
                              type="button"
                              className="button outline small danger-text"
                              aria-label={`${asset.file_deletion ? '重试删除文件' : '永久删除文件'} ${asset.name}`}
                              disabled={Boolean(ledger.error)}
                              onClick={() =>
                                setSelection({ id: asset.id, path: asset.path, action: 'purge' })
                              }
                            >
                              <Trash2 size={14} aria-hidden="true" />
                              {asset.file_deletion ? '重试删除' : '永久删除文件'}
                            </button>
                          )}
                        </div>
                        {!accepted && !asset.can_restore && asset.restore_block_reason && (
                          <span className="subtext">{asset.restore_block_reason}</span>
                        )}
                        {!accepted && !canOpenAssetPurge(asset) && asset.purge_block_reason && (
                          <span className="subtext">{asset.purge_block_reason}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : ledger.loading && !rows ? (
          <Loading label="正在读取已删除资产…" />
        ) : !ledger.error && rows ? (
          <Empty icon={Archive} title="暂无已删除资产" description="删除登记的资产将在此显示。" />
        ) : null}
      </div>
      {selected && selection?.action === 'purge' && (
        <AssetPurgeDialog
          key={`${selected.id}:${selected.path}`}
          asset={selected}
          available={!ledger.error}
          refresh={ledger.refresh}
          close={() => setSelection(undefined)}
          onSubmitted={onSubmitted}
        />
      )}
      {selected && selection?.action === 'restore' && (
        <AssetRestoreDialog
          key={`${selected.id}:${selected.path}`}
          asset={selected}
          expectedArchivedAt={selection.archivedAt}
          available={!ledger.error}
          refresh={ledger.refresh}
          close={() => setSelection(undefined)}
          onRestored={onRestored}
        />
      )}
    </>
  );
}

function AssetRestoreDialog({
  asset,
  expectedArchivedAt,
  available,
  refresh,
  close,
  onRestored,
}: {
  asset: ArchivedAsset;
  expectedArchivedAt: string | null;
  available: boolean;
  refresh: () => void;
  close: () => void;
  onRestored: (asset: Asset) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const active = useRef<AbortController | undefined>(undefined);
  useEffect(() => () => active.current?.abort(), []);
  const archiveChanged = asset.archived_at !== expectedArchivedAt;
  const mayRestore = available && asset.can_restore === true && !archiveChanged;
  const dismiss = () => {
    if (!active.current) close();
  };
  const restore = async () => {
    if (!mayRestore || active.current) return;
    const request = new AbortController();
    const signal = AbortSignal.any([request.signal, AbortSignal.timeout(20000)]);
    active.current = request;
    setBusy(true);
    setError(undefined);
    try {
      const result = await createOperationClient().post<Asset>(
        `/asset-deletions/${encodeURIComponent(asset.id)}/restore`,
        { expected_archived_at: expectedArchivedAt },
        { signal },
      );
      const state = result?.state || result?.status;
      if (
        result?.id !== asset.id ||
        result.node_id !== asset.node_id ||
        result.path !== asset.path ||
        !state ||
        state === 'ARCHIVED'
      )
        throw new Error('未能确认资产已恢复，请刷新列表后重试。');
      onRestored(result);
    } catch (cause) {
      if (!request.signal.aborted) {
        setError(cause as Error);
        refresh();
      }
    } finally {
      active.current = undefined;
      if (!request.signal.aborted) setBusy(false);
    }
  };
  return (
    <Modal
      title="恢复资产登记"
      close={dismiss}
      footer={
        <>
          <button type="button" className="button outline" onClick={dismiss} disabled={busy}>
            取消
          </button>
          <button
            type="button"
            className="button primary"
            onClick={() => void restore()}
            disabled={!mayRestore || busy}
          >
            {busy && <LoaderCircle size={16} className="spin" aria-hidden="true" />}
            确认恢复
          </button>
        </>
      }
    >
      <p>恢复「{asset.name}」后，它将重新出现在资产目录中。</p>
      {asset.file_deletion?.status === 'PENDING' && mayRestore && (
        <p>尚未执行的文件删除任务也会取消。</p>
      )}
      <p className="asset-purge-path">
        <code>{asset.path}</code>
      </p>
      {!mayRestore && !busy && (
        <p role="status" className="field-hint">
          {archiveChanged
            ? '该资产的删除记录已更新，请关闭窗口后重新选择。'
            : asset.restore_block_reason || '请刷新列表以核对可恢复状态。'}
        </p>
      )}
      <ErrorNotice error={error} title="恢复资产登记失败" />
    </Modal>
  );
}

function AssetPurgeDialog({
  asset,
  available,
  refresh,
  close,
  onSubmitted,
}: {
  asset: ArchivedAsset;
  available: boolean;
  refresh: () => void;
  close: () => void;
  onSubmitted: (assetId: string, receipt: AssetFileDeletion) => void;
}) {
  const [started, setStarted] = useState<{
    receipt: AssetPurgeInspection;
    previousInspectionId?: string;
  }>();
  const [typed, setTyped] = useState<{ path: string; inspectionId?: string }>({ path: '' });
  const [busy, setBusy] = useState<'inspect' | 'purge'>();
  const [error, setError] = useState<Error>();
  const active = useRef<AbortController | undefined>(undefined);
  const autoStarted = useRef(false);
  useEffect(() => () => active.current?.abort(), []);

  const { id, file_deletion: deletion, inspection: recorded } = asset;
  const mayPurge = available && canOpenAssetPurge(asset);
  const inspection =
    started && started.previousInspectionId === recorded?.id && started.receipt.id !== recorded?.id
      ? started.receipt
      : recorded;
  const retry = deletion?.status === 'FAILED';
  const retryPinned = retry && !asset.requires_new_inspection;
  const allowInspection = mayPurge && !retryPinned;
  const canConfirm = available && canConfirmAssetPurge(asset, inspection);
  const payload = canConfirm ? assetPurgeConfirmation(asset, inspection, typed) : undefined;

  const inspect = useCallback(async () => {
    if (active.current || !allowInspection) return;
    const request = new AbortController();
    const signal = AbortSignal.any([request.signal, AbortSignal.timeout(20000)]);
    const operation = createOperationClient();
    active.current = request;
    setBusy('inspect');
    setError(undefined);
    setTyped({ path: '' });
    try {
      const receipt = await operation.post<AssetPurgeInspection>(
        `/assets/${encodeURIComponent(id)}/purge-inspections`,
        {},
        { signal },
      );
      setStarted({ receipt, previousInspectionId: recorded?.id });
      refresh();
    } catch (cause) {
      if (!request.signal.aborted) setError(cause as Error);
    } finally {
      active.current = undefined;
      if (!request.signal.aborted) setBusy(undefined);
    }
  }, [allowInspection, id, recorded?.id, refresh]);

  useEffect(() => {
    if (!mayPurge || deletion || recorded || autoStarted.current) return;
    const timer = setTimeout(() => {
      autoStarted.current = true;
      void inspect();
    }, 0);
    return () => clearTimeout(timer);
  }, [deletion, inspect, mayPurge, recorded]);

  const confirm = async () => {
    if (active.current || !payload) return;
    const request = new AbortController();
    const signal = AbortSignal.any([request.signal, AbortSignal.timeout(120000)]);
    const operation = createOperationClient();
    active.current = request;
    setBusy('purge');
    setError(undefined);
    try {
      const receipt = await operation.post<AssetFileDeletion>(
        `/assets/${encodeURIComponent(id)}/purge`,
        payload,
        { signal },
      );
      onSubmitted(id, receipt);
    } catch (cause) {
      if (!request.signal.aborted) {
        setError(cause as Error);
        refresh();
      }
    } finally {
      active.current = undefined;
      if (!request.signal.aborted) setBusy(undefined);
    }
  };

  const dismiss = () => {
    if (busy !== 'purge') close();
  };
  const inspecting =
    busy === 'inspect' || inspection?.status === 'PENDING' || inspection?.status === 'RUNNING';
  const needsInspection =
    asset.requires_new_inspection &&
    (!inspection?.requested_at || inspection.requested_at <= (deletion?.requested_at || ''));
  return (
    <Modal
      title={retry ? '重试永久删除文件' : '永久删除文件'}
      close={dismiss}
      footer={
        <>
          <button
            type="button"
            className="button outline"
            disabled={busy === 'purge'}
            onClick={dismiss}
          >
            取消
          </button>
          <button
            type="button"
            className="button danger"
            disabled={!payload || Boolean(busy)}
            onClick={() => void confirm()}
          >
            {busy === 'purge' && <LoaderCircle size={16} className="spin" />}
            {inspection?.status === 'MISSING' && !retryPinned
              ? '确认记录文件不存在'
              : '确认永久删除'}
          </button>
        </>
      }
    >
      <p>永久删除「{asset.name}」在下列位置的文件后，文件无法恢复。</p>
      <dl className="detail-list inset">
        <div>
          <dt>所在机器</dt>
          <dd>{asset.node_name || asset.node_id}</dd>
        </div>
        <div>
          <dt>原文件路径</dt>
          <dd>
            <code className="asset-purge-path">{asset.path}</code>
          </dd>
        </div>
      </dl>
      <div className="asset-purge-inspection">
        <p role="status">
          {inspecting && <LoaderCircle size={15} className="spin" aria-hidden="true" />}
          {retryPinned
            ? '使用上次确认的文件检查结果继续删除。'
            : inspecting
              ? inspectionStates[inspection?.status === 'RUNNING' ? 'RUNNING' : 'PENDING']
              : needsInspection
                ? '文件自上次检查后已变化，请重新检查文件。'
                : inspection
                  ? inspectionStates[inspection.status]
                  : '等待启动文件检查'}
        </p>
        {allowInspection && !inspecting && (
          <button
            type="button"
            className="button outline small"
            disabled={Boolean(busy)}
            onClick={() => void inspect()}
          >
            {inspection ? '重新检查文件' : '检查文件'}
          </button>
        )}
      </div>
      {!retryPinned && inspection?.status === 'FAILED' && (
        <ErrorNotice
          title="无法核对原文件"
          error={new Error(inspection.error_message || inspection.error_code || '请重新检查文件。')}
        />
      )}
      {!mayPurge && (
        <p className="field-hint" role="status">
          {available
            ? asset.purge_block_reason || '当前文件状态已更新，请刷新列表。'
            : '请等待列表重新连接。'}
        </p>
      )}
      <Field label="输入完整原文件路径" hint="路径必须与上方显示的内容完全一致。" required>
        <input
          autoComplete="off"
          spellCheck={false}
          disabled={!canConfirm || Boolean(busy)}
          value={typed.inspectionId === inspection?.id ? typed.path : ''}
          onChange={(event) => setTyped({ path: event.target.value, inspectionId: inspection?.id })}
        />
      </Field>
      <ErrorNotice error={error} title="操作未完成" />
    </Modal>
  );
}
