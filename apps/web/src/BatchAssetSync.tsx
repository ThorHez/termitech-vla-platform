import { useEffect, useRef, useState } from 'react';
import { ArrowRight, CheckCircle2, FolderSync, LoaderCircle } from 'lucide-react';
import { api } from './api';
import { useAuth } from './auth';
import { AssetSyncNormOption } from './AssetTransfers';
import { assetTargetPath, assetTransferTaskPath } from './assetTransfer';
import type { AssetTransfer } from './assetTransfer';
import { ErrorNotice, Field, Modal } from './components';
import { assetKinds, useData } from './data';
import { PathAutocomplete } from './PathAutocomplete';
import { nodeAddressText } from './nodeAddress';
import {
  batchSyncError,
  batchSyncIdentity,
  batchSyncReceiptMatches,
  batchSyncRequest,
  batchSyncRows,
  type BatchAssetSyncResult,
  type BatchAssetSyncRow,
} from './batchAssetSyncData';
import './batch-asset-sync.css';

export function BatchAssetSync({
  assetIds,
  close,
  created,
}: {
  assetIds: string[];
  close: () => void;
  created: (transfers: AssetTransfer[]) => void;
}) {
  const { user, can } = useAuth();
  if (!user?.id || !can('assets.write'))
    return (
      <Modal title="无法批量同步" close={close}>
        请使用具有资产编辑权限的个人账户登录后操作。
      </Modal>
    );
  return (
    <BatchAssetSyncContent
      key={user.id}
      owner={user.id}
      assetIds={assetIds}
      close={close}
      created={created}
    />
  );
}

function BatchAssetSyncContent({
  owner,
  assetIds,
  close,
  created,
}: {
  owner: string;
  assetIds: string[];
  close: () => void;
  created: (transfers: AssetTransfer[]) => void;
}) {
  const { user, can } = useAuth();
  const { assets, nodes } = useData();
  const [rows, setRows] = useState<BatchAssetSyncRow[]>(() =>
    batchSyncRows(
      assetIds
        .map((id) => assets.data?.find((asset) => asset.id === id))
        .filter((asset) => !!asset),
      assets.data || [],
    ),
  );
  const [targetId, setTargetId] = useState('');
  const [directory, setDirectory] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [result, setResult] = useState<BatchAssetSyncResult>();
  const requestIdentity = useRef<{ fingerprint: string; key: string } | undefined>(undefined);
  const operation = useRef<AbortController | undefined>(undefined);
  const mounted = useRef(true);
  const pending = useRef(false);
  const assetKey = JSON.stringify(assetIds);
  const current = useRef({ owner, allowed: true, assetKey });
  current.current = { owner: user?.id || '', allowed: can('assets.write'), assetKey };
  const targetNode = nodes.data?.find((node) => node.id === targetId);
  const sourceNodeIds = new Set(
    assetIds.map((id) => assets.data?.find((asset) => asset.id === id)?.node_id),
  );
  const targets = (nodes.data || []).filter((node) => !sourceNodeIds.has(node.id));
  const rowsMatchSelection = JSON.stringify(rows.map((row) => row.assetId)) === assetKey;
  const blocker = !rowsMatchSelection
    ? '资产选择已变化，请重新核对当前项目。'
    : batchSyncError({
        rows,
        assets: assets.data || [],
        nodes: nodes.data || [],
        targetNodeId: targetId,
        directory,
      });

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      operation.current?.abort();
    };
  }, []);
  useEffect(() => {
    operation.current?.abort();
    setBusy(false);
    setRows((previous) =>
      assetIds
        .map(
          (id) =>
            previous.find((row) => row.assetId === id) ||
            batchSyncRows(
              (assets.data || []).filter((asset) => asset.id === id),
              assets.data || [],
            )[0],
        )
        .filter((row) => !!row),
    );
  }, [assetKey]);

  const active = (abort: AbortController, expectedAssets: string) =>
    mounted.current &&
    !abort.signal.aborted &&
    operation.current === abort &&
    current.current.owner === owner &&
    current.current.allowed &&
    current.current.assetKey === expectedAssets;
  const safeClose = () => {
    if (!pending.current) close();
  };
  const update = (index: number, change: Partial<BatchAssetSyncRow>) =>
    setRows((previous) =>
      previous.map((row, position) => (position === index ? { ...row, ...change } : row)),
    );
  const submit = async () => {
    if (
      pending.current ||
      busy ||
      blocker ||
      current.current.owner !== owner ||
      !current.current.allowed
    )
      return;
    const request = batchSyncRequest(rows, targetId, directory);
    const identity = batchSyncIdentity(requestIdentity.current, request, () => crypto.randomUUID());
    requestIdentity.current = identity;
    const abort = new AbortController();
    operation.current?.abort();
    operation.current = abort;
    pending.current = true;
    setBusy(true);
    setError(undefined);
    const expectedAssets = assetKey;
    try {
      const saved = await api<BatchAssetSyncResult>('/asset-transfer-batches', {
        method: 'POST',
        headers: { 'X-Asset-Transfer-Batch-User': owner },
        body: JSON.stringify({ ...request, idempotency_key: identity.key }),
        signal: AbortSignal.any([abort.signal, AbortSignal.timeout(60000)]),
      });
      if (!active(abort, expectedAssets)) return;
      if (!batchSyncReceiptMatches(saved, identity.key, request))
        throw new Error('同步任务返回结果不一致，请保留当前配置并重试。');
      setResult(saved);
      created(saved.transfers);
    } catch (cause) {
      if (active(abort, expectedAssets))
        setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      pending.current = false;
      if (active(abort, expectedAssets)) setBusy(false);
    }
  };

  return (
    <Modal
      wide
      title="批量同步到其他机器"
      subtitle="源机器直传目标机器；控制台负责调度、校验和进度，各项任务可独立取消或重试。"
      close={safeClose}
      className="batch-asset-sync-modal"
      footer={
        <>
          <button type="button" className="button outline" disabled={busy} onClick={safeClose}>
            {result ? '完成' : '取消'}
          </button>
          {!result && (
            <button
              type="button"
              className="button primary"
              disabled={busy || Boolean(blocker)}
              onClick={() => void submit()}
            >
              {busy ? <LoaderCircle size={15} className="spin" /> : <FolderSync size={15} />}
              {busy ? '正在创建后台任务…' : `开始批量同步（${rows.length}）`}
            </button>
          )}
        </>
      }
    >
      <ErrorNotice error={error} title="批量同步未完成" />
      {result ? (
        <section className="batch-asset-sync-result" role="status">
          <CheckCircle2 size={28} />
          <h3>已创建 {result.transfers.length} 个后台同步任务</h3>
          <p>
            关闭窗口不会取消任务，可从任务链接查看直传进度。文件校验完成后才会登记目标副本，训练前仍需预检。
          </p>
          <ul>
            {result.transfers.map((transfer) => (
              <li key={transfer.id}>
                <span>{transfer.name}</span>
                <a href={`#${assetTransferTaskPath(transfer.id)}`}>
                  查看任务 <ArrowRight size={13} />
                </a>
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <>
          <section className="batch-asset-sync-target" aria-label="统一同步目标">
            <Field label="目标机器" required>
              <select
                aria-label="批量同步目标机器"
                value={targetId}
                disabled={busy}
                onChange={(event) => {
                  setTargetId(event.target.value);
                  setDirectory('');
                  setError(undefined);
                }}
              >
                <option value="">选择所有源机器之外的在线机器</option>
                {targets.map((node) => (
                  <option key={node.id} value={node.id} disabled={node.status !== 'ONLINE'}>
                    {node.name}
                    {nodeAddressText(node) ? ` · ${nodeAddressText(node)}` : ''}
                    {node.status === 'ONLINE' ? '' : ' · 离线'}
                  </option>
                ))}
              </select>
            </Field>
            <PathAutocomplete
              key={targetId}
              node={targetNode}
              label="目标父目录"
              value={directory}
              onChange={setDirectory}
              disabled={busy || !targetNode}
              placeholder="例如 /data/shared"
              hint="所有资产会在此目录下创建各自的子目录。已有目标目录不会被覆盖。"
            />
          </section>
          {!targets.some((node) => node.status === 'ONLINE') && (
            <p className="batch-asset-sync-notice">
              暂无同时满足所选资产的目标机器，请恢复其他机器连接或减少源机器范围。
            </p>
          )}
          <section className="batch-asset-sync-items" aria-label="批量同步项目">
            <h3>同步项目 · {rows.length} 项</h3>
            <p>
              目录默认使用源路径的末级名称，文件资产默认使用资产名称；重名时请明确修改，不会自动改名。
            </p>
            {rows.map((row, index) => {
              const asset = assets.data?.find((item) => item.id === row.assetId);
              if (!asset) return null;
              const sourceNode = nodes.data?.find((node) => node.id === asset.node_id);
              return (
                <article
                  key={row.assetId}
                  className="batch-asset-sync-item"
                  aria-label={`同步资产 ${asset.name}`}
                >
                  <div className="batch-asset-sync-source">
                    <strong>{asset.name}</strong>
                    <span>
                      {assetKinds[asset.kind] || asset.kind} · {asset.version} ·{' '}
                      {sourceNode?.name || asset.node_id}
                    </span>
                    <code>{asset.path}</code>
                  </div>
                  <div className="batch-asset-sync-destination">
                    <Field label="目标子目录名" required>
                      <input
                        aria-label={`目标子目录名 ${asset.name}`}
                        maxLength={160}
                        value={row.targetName}
                        disabled={busy}
                        onChange={(event) => update(index, { targetName: event.target.value })}
                      />
                    </Field>
                    <span>目标目录</span>
                    <code>
                      {assetTargetPath(directory, row.targetName) ||
                        '选择父目录并填写子目录名后显示'}
                    </code>
                  </div>
                  <AssetSyncNormOption
                    asset={asset}
                    assets={assets.data || []}
                    checked={row.includeNormStats}
                    disabled={busy}
                    onChange={(value) =>
                      update(index, {
                        includeNormStats: value,
                        expectedNormAssetId: value ? asset.norm_asset_id || undefined : undefined,
                      })
                    }
                  />
                </article>
              );
            })}
          </section>
          <p className="batch-asset-sync-notice">
            一次最多 20 项，目标机器需能访问各源机器的
            SSH。整批校验通过后才创建后台任务；直传过程、取消和重试按每项任务独立管理。
          </p>
          {blocker && (
            <p className="batch-asset-sync-error" role="status">
              {blocker}
            </p>
          )}
        </>
      )}
    </Modal>
  );
}
