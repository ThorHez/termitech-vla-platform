import { useEffect, useState } from 'react';
import { ArrowRight, CheckCircle2, Download, LoaderCircle, RefreshCw, Square } from 'lucide-react';
import { date, post, useApi } from './api';
import { useAuth } from './auth';
import { Badge, ErrorNotice, Panel, ProgressBar } from './components';
import { useData } from './data';
import { PolicyWarnings } from './PolicyWarnings';
import {
  transferBytes,
  transferElapsed,
  transferLabels,
  transferTerminal,
  type CheckpointTransfer,
} from './checkpointTransfer';
import './checkpoint-transfers.css';

export default function CheckpointTransfers({
  useTransfer,
  revision,
}: {
  useTransfer: (item: CheckpointTransfer) => void;
  revision: number;
}) {
  const list = useApi<CheckpointTransfer[]>(`/checkpoint-transfers?revision=${revision}`, 3000);
  if (!list.error && !list.data?.length) return null;
  return (
    <Panel
      title="权重同步"
      subtitle="同步模型参数、assets 与推理配置，不同步 train_state 训练状态目录；校验通过后可用于部署。"
      className="checkpoint-transfers-panel"
    >
      <ErrorNotice error={list.error} retry={list.refresh} />
      <TransferGroups items={list.data || []} changed={list.refresh} useTransfer={useTransfer} />
    </Panel>
  );
}

export function TransferGroups({
  items,
  changed,
  useTransfer,
}: {
  items: CheckpointTransfer[];
  changed: () => void;
  useTransfer?: (item: CheckpointTransfer) => void;
}) {
  const visible = items.filter((item) => item.state !== 'CANCELLED');
  const cancelled = items.filter((item) => item.state === 'CANCELLED');
  return (
    <>
      {visible.map((item) => (
        <TransferCard key={item.id} item={item} changed={changed} useTransfer={useTransfer} />
      ))}
      {!!cancelled.length && (
        <details
          className="checkpoint-transfer-history"
          key={cancelled.map((item) => item.id).join('|')}
        >
          <summary>
            <span>已取消记录（{cancelled.length}）</span>
            <small>已停止，保留记录和部分文件供重试</small>
          </summary>
          {cancelled.map((item) => (
            <TransferCard key={item.id} item={item} changed={changed} useTransfer={useTransfer} />
          ))}
        </details>
      )}
    </>
  );
}

interface TransferCardProps {
  item: CheckpointTransfer;
  changed: () => void;
  useTransfer?: (item: CheckpointTransfer) => void;
  compact?: boolean;
}

export function TransferCard(props: TransferCardProps) {
  if (props.compact && props.item.state === 'CANCELLED')
    return (
      <details className="checkpoint-transfer-cancelled-compact">
        <summary>
          <span>已取消，传输已停止</span>
          <small>保留记录和部分文件；展开查看停止时进度、日志或重试</small>
        </summary>
        <TransferCardContent {...props} compact={false} />
      </details>
    );
  return <TransferCardContent {...props} />;
}

function TransferCardContent({ item, changed, useTransfer, compact = false }: TransferCardProps) {
  const { nodes } = useData();
  const { can } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [now, setNow] = useState(() => Date.now());
  const source = nodes.data?.find((node) => node.id === item.source_node_id);
  const target = nodes.data?.find((node) => node.id === item.target_node_id);
  const terminal = transferTerminal(item.state);
  const cancelled = item.state === 'CANCELLED';
  useEffect(() => {
    if (terminal) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [terminal]);
  const preparing =
    item.bytes_total <= 0 && ['QUEUED', 'EXPORTING', 'PREPARING'].includes(item.state);
  const failedBeforeTransfer = item.bytes_total <= 0 && item.state === 'FAILED';
  const notStarted = failedBeforeTransfer || (cancelled && item.bytes_total <= 0);
  const elapsed =
    !terminal || item.finished_at
      ? transferElapsed(item.created_at, item.finished_at, now)
      : undefined;
  const latestLog = item.logs
    ?.split(/\r?\n/)
    .filter((line) => line.trim())
    .at(-1);
  const writable = can('training.write');
  const perform = async (operation: string) => {
    if (busy) return;
    setBusy(true);
    setError(undefined);
    try {
      await post(`/checkpoint-transfers/${encodeURIComponent(item.id)}/${operation}`);
      changed();
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };
  return (
    <article
      className={`checkpoint-transfer-card ${compact ? 'compact' : ''} ${cancelled ? 'cancelled' : ''}`}
      aria-label={`权重同步 ${item.name}`}
    >
      <div className="checkpoint-transfer-heading">
        <div>
          <strong>
            <Download size={15} />
            {item.name}
          </strong>
          <span>
            {source?.name || item.source_node_id}
            <ArrowRight size={12} />
            {target?.name || item.target_node_id}
          </span>
        </div>
        <Badge
          state={item.state === 'SUCCEEDED' ? 'READY' : item.state}
          label={transferLabels[item.state] || item.state}
        />
      </div>
      {cancelled && (
        <p className="checkpoint-transfer-stopped" role="status">
          已取消，传输已停止。保留记录和部分文件供重试。
        </p>
      )}
      {preparing ? (
        <div className="checkpoint-transfer-preparing" role="status">
          <div className="checkpoint-transfer-progress">
            <div
              className="progress teal checkpoint-transfer-indeterminate"
              role="progressbar"
              aria-label={transferLabels[item.state]}
            >
              <span />
            </div>
            <span>准备中</span>
          </div>
          <p>{transferLabels[item.state]}，尚未开始传输</p>
        </div>
      ) : notStarted ? (
        <p className="checkpoint-transfer-not-started">传输尚未开始</p>
      ) : (
        <div className="checkpoint-transfer-progress" aria-live="polite">
          <ProgressBar
            value={item.state === 'SUCCEEDED' ? 100 : Math.min(99, item.progress_pct || 0)}
          />
          <span>
            {cancelled && '停止时进度 '}
            {item.state === 'SUCCEEDED' ? '100' : Math.min(99, item.progress_pct || 0).toFixed(1)}%
          </span>
        </div>
      )}
      <div className="checkpoint-transfer-metrics">
        {preparing ? (
          <span>文件总大小与数量待统计</span>
        ) : (
          !notStarted && (
            <>
              <span>
                {transferBytes(item.bytes_transferred)} / {transferBytes(item.bytes_total)}
              </span>
              <span>
                {item.files_transferred == null ? '—' : item.files_transferred} /{' '}
                {item.files_total || 0} 个文件
              </span>
            </>
          )
        )}
        {!!item.speed_bytes_per_second && !terminal && !preparing && (
          <span>{transferBytes(item.speed_bytes_per_second)}/s</span>
        )}
        <span>
          第 {item.attempt || 1} 次 · 任务创建于 {date(item.created_at)}
        </span>
        {elapsed && <span>任务耗时 {elapsed}</span>}
        <span>最近状态回执 {date(item.updated_at)}</span>
      </div>
      {latestLog && (
        <div className="checkpoint-transfer-latest-log">
          <span>最近记录</span>
          <code>{latestLog.length > 240 ? `${latestLog.slice(0, 240)}…` : latestLog}</code>
        </div>
      )}
      <p className="checkpoint-transfer-path mono">
        {item.target_path || `${item.target_directory.replace(/\/+$/, '')}/${item.id}`}
      </p>
      <ErrorNotice
        error={
          error ||
          (item.error ? new Error(`${item.error.code} · ${item.error.message}`) : undefined)
        }
        title="同步未完成"
      />
      <PolicyWarnings warnings={item.warnings} />
      {item.state === 'SUCCEEDED' && (
        <div className="checkpoint-transfer-verified">
          <CheckCircle2 size={14} />
          模型参数、assets 与推理配置已同步并通过 SHA-256 校验；未同步 train_state 训练状态目录。
        </div>
      )}
      <div className="checkpoint-transfer-actions">
        {writable && ['FAILED', 'INTERRUPTED', 'CANCELLED'].includes(item.state) && (
          <button
            type="button"
            className="button outline small"
            disabled={busy}
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
            disabled={busy || item.state === 'CANCELLING'}
            onClick={() => void perform('cancel')}
          >
            <Square size={12} />
            取消同步
          </button>
        )}
        {writable && item.state === 'SUCCEEDED' && useTransfer && (
          <button type="button" className="button primary small" onClick={() => useTransfer(item)}>
            使用此权重部署
            <ArrowRight size={13} />
          </button>
        )}
        {busy && <LoaderCircle size={14} className="spin" />}
      </div>
      {!!item.logs && (
        <details>
          <summary>同步日志</summary>
          <pre>{item.logs}</pre>
        </details>
      )}
    </article>
  );
}
