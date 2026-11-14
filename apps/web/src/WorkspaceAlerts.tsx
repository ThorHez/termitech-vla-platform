import { AlertTriangle, HardDrive, ShieldCheck } from 'lucide-react';
import { Back, ErrorNotice, PageHeading, Panel, ViewLink } from './components';
import { useData } from './data';
import type { Alert, Node } from './types';
import { isDiskCapacityAlert } from './workspaceAlertData';

const previewCount = 4;
const percentFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });
export const alertsPath = '/alerts';

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
  return (
    <div className="alert-list">
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

type AlertContentsProps = {
  alerts?: Alert[];
  nodes?: Pick<Node, 'id' | 'name'>[];
  enabled: boolean;
  error?: Error;
  refresh: () => void;
};

function AlertContents({ alerts, nodes, enabled, error, refresh }: AlertContentsProps) {
  return (
    <>
      <ErrorNotice error={error} retry={refresh} />
      {alerts?.length ? (
        <AlertList alerts={alerts} nodes={nodes} />
      ) : (
        <div className="healthy-state">
          <div>
            <ShieldCheck size={26} />
          </div>
          <strong>{enabled && alerts ? '当前没有待处理异常' : '等待资源状态上报'}</strong>
        </div>
      )}
    </>
  );
}

export function WorkspaceAlerts({ alerts, ...props }: AlertContentsProps) {
  return (
    <Panel
      id="workspace-alerts"
      tabIndex={-1}
      title="需要关注"
      subtitle="来自机器和训练的实时异常"
      action={
        <div className="workspace-alert-actions">
          <span className="count-pill">{alerts?.length ?? '—'}</span>
          <ViewLink to={alertsPath}>查看全部</ViewLink>
        </div>
      }
    >
      <AlertContents {...props} alerts={alerts?.slice(0, previewCount)} />
    </Panel>
  );
}

export function AlertsPage() {
  const { alerts, nodes, enabled } = useData();
  return (
    <>
      <Back to="/overview" label="返回训练工作台" />
      <PageHeading
        title="全部异常"
        description="查看当前的机器、磁盘和训练异常。"
        actions={alerts.data && <span className="count-pill">{alerts.data.length} 条异常</span>}
      />
      <div className="panel">
        <AlertContents
          alerts={alerts.data}
          nodes={nodes.data}
          enabled={enabled}
          error={alerts.error}
          refresh={alerts.refresh}
        />
      </div>
    </>
  );
}
