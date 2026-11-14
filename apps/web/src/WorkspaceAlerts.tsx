import { useState } from 'react';
import { AlertTriangle, HardDrive } from 'lucide-react';
import { ViewLink } from './components';
import type { Alert, Node } from './types';

const regularPreviewCount = 4;
const percentFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });

export const isDiskCapacityAlert = (alert: Alert) => alert.code === 'DISK_CAPACITY_HIGH';

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
}: {
  alerts: Alert[];
  nodes?: Pick<Node, 'id' | 'name'>[];
  showNodeLink?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const disk = alerts.filter(isDiskCapacityAlert);
  const regular = alerts.filter((alert) => !isDiskCapacityAlert(alert));
  const visible = [...disk, ...(expanded ? regular : regular.slice(0, regularPreviewCount))];
  const hidden = regular.length - regularPreviewCount;
  return (
    <div className="alert-list">
      {visible.map((alert, index) => (
        <AlertRow
          key={alert.id || index}
          alert={alert}
          nodeName={nodes.find((node) => node.id === alert.node_id)?.name}
          showNodeLink={showNodeLink}
        />
      ))}
      {hidden > 0 && (
        <button
          type="button"
          className="button outline small"
          aria-expanded={expanded}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? '收起其他异常' : `展开其余 ${hidden} 条异常`}
        </button>
      )}
    </div>
  );
}
