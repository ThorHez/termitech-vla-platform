import { useState } from 'react';
import { AlertTriangle, HardDrive, ShieldCheck } from 'lucide-react';
import { ErrorNotice, Panel, ViewLink } from './components';
import type { Alert, Node } from './types';
import { isDiskCapacityAlert } from './workspaceAlertData';

const previewCount = 4;
const percentFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });

export function focusWorkspaceAlerts() {
  const section = document.getElementById('workspace-alerts');
  section?.focus({ preventScroll: true });
  section?.scrollIntoView({ block: 'start' });
}

const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB'];

function capacity(bytes: number | undefined) {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return undefined;
  const unit = Math.min(units.length - 1, Math.floor(Math.log2(Math.max(bytes, 1)) / 10));
  const value = (bytes / 1024 ** unit).toLocaleString('zh-CN', { maximumFractionDigits: 1 });
  return `${value} ${units[unit]}`;
}

function AlertRow({
  alert,
  nodeName,
  showNodeLink,
}: {
  alert: Alert;
  nodeName?: string;
  showNodeLink: boolean;
}) {
  const disk = isDiskCapacityAlert(alert);
  const Icon = disk ? HardDrive : AlertTriangle;
  const free = disk ? capacity(alert.details?.free_bytes) : undefined;
  const total = disk ? capacity(alert.details?.total_bytes) : undefined;
  const percent = alert.details?.usage_percent;
  const mountpoint = alert.details?.mountpoint;
  const useStructuredDetails =
    disk &&
    nodeName &&
    mountpoint &&
    typeof percent === 'number' &&
    Number.isFinite(percent) &&
    percent >= 0 &&
    percent <= 100;
  return (
    <div className="alert-item">
      <Icon size={17} aria-hidden="true" />
      <div>
        <strong>
          {disk ? alert.title || '磁盘空间告警' : alert.title || alert.code || '运行异常'}
        </strong>
        <p>
          {useStructuredDetails ? (
            <>
              {nodeName} 的磁盘 <code>{mountpoint}</code> 已使用 {percentFormat.format(percent)}%
            </>
          ) : (
            alert.message
          )}
          {free !== undefined && (
            <span className="alert-capacity">
              可用 {free}
              {total && ` / 共 ${total}`}
            </span>
          )}
        </p>
        {alert.run_id ? (
          <ViewLink to={`/runs/${encodeURIComponent(alert.run_id)}`}>查看训练</ViewLink>
        ) : (
          showNodeLink &&
          alert.node_id && (
            <ViewLink to={`/nodes/${encodeURIComponent(alert.node_id)}`}>查看机器</ViewLink>
          )
        )}
      </div>
    </div>
  );
}

export function AlertList({
  alerts,
  nodes = [],
  showNodeLink = true,
  id,
}: {
  alerts: Alert[];
  nodes?: Pick<Node, 'id' | 'name'>[];
  showNodeLink?: boolean;
  id?: string;
}) {
  return (
    <div className="alert-list" id={id}>
      {alerts.map((alert, index) => (
        <AlertRow
          key={alert.id || index}
          alert={alert}
          nodeName={nodes.find((node) => node.id === alert.node_id)?.name}
          showNodeLink={showNodeLink}
        />
      ))}
    </div>
  );
}

export function WorkspaceAlerts({
  alerts,
  nodes,
  enabled,
  error,
  refresh,
}: {
  alerts?: Alert[];
  nodes?: Pick<Node, 'id' | 'name'>[];
  enabled: boolean;
  error?: Error;
  refresh: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const count = alerts?.length ?? 0;
  return (
    <Panel
      id="workspace-alerts"
      tabIndex={-1}
      title="需要关注"
      subtitle="来自机器和训练的实时异常"
      action={
        <div className="workspace-alert-actions">
          <span className="count-pill">{alerts?.length ?? '—'}</span>
          {count > previewCount && (
            <button
              type="button"
              className="text-link"
              aria-controls="workspace-alert-list"
              aria-expanded={expanded}
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? '收起' : '查看全部'}
            </button>
          )}
        </div>
      }
    >
      <ErrorNotice error={error} retry={refresh} />
      {alerts?.length ? (
        <AlertList
          id="workspace-alert-list"
          alerts={expanded ? alerts : alerts.slice(0, previewCount)}
          nodes={nodes}
        />
      ) : (
        <div className="healthy-state">
          <div>
            <ShieldCheck size={26} />
          </div>
          <strong>{enabled && alerts ? '当前没有待处理异常' : '等待资源状态上报'}</strong>
          <p>{enabled && alerts ? '异常会在这里及时展示' : '连接管理服务后即可查看'}</p>
        </div>
      )}
    </Panel>
  );
}
