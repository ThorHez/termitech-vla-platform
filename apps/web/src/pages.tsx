import { useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowRight,
  Box,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  FileSliders,
  GitCommitHorizontal,
  HardDrive,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  Network,
  Pencil,
  Plus,
  Search,
  Server,
  Settings2,
  ShieldCheck,
  Sparkles,
  Terminal,
  Trash2,
  Workflow,
} from 'lucide-react';
import { date, displayNumber, navigate, post, useApi } from './api';
import { useAuth } from './auth';
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
  ViewLink,
} from './components';
import { activeStates, assetKinds, gpuAverage, gpuIdle, requiredInputs, useData } from './data';
import type { Asset, Gpu, Node, NodeDetail, Run } from './types';
import { NodeOnboarding, RuntimeCheckView, runtimeCheckFor } from './NodeOnboarding';
import { NodeManagement } from './NodeManagement';
import { NodeRename } from './NodeRename';
import { NodeTerminal } from './NodeTerminal';
import { ResourceHistory } from './ResourceHistory';
import { GpuReservations } from './GpuReservations';
import { NodeGpuResources } from './NodeGpuResources';
import { ProfileRebinding } from './ProfileRebinding';
import { StorageConfiguration } from './StorageConfiguration';
import { matchesNodeQuery, nodeAddressText } from './nodeAddress';
import { PathAutocomplete } from './PathAutocomplete';
import { AssetSyncDialog, AssetSyncBackgroundNotice } from './AssetTransfers';
import type { AssetTransfer } from './assetTransfer';
import { BackgroundTasksPage } from './BackgroundTasks';
import { RunDeleteAction } from './RunDeleteAction';
import { DatasetInspection, useDatasetInspectionDraft } from './DatasetInspection';
import { DatasetNormBindingDialog, DatasetNormField, DatasetNormSummary } from './DatasetNormBinding';
import { AssetDetail } from './AssetDetail';
import { catalogAssetGroups } from './modelAssetGroups';
import { AssetGroupRows } from './AssetGroupRows';
import { groupDatasetNormAssets } from './datasetAssetGroups';
import { ASSET_CATALOG_PAGE_SIZE, paginateAssetCatalog } from './assetCatalogPagination';
import { DatasetValidationBadge } from './DatasetValidation';
import { BatchDatasetRegistration } from './BatchDatasetRegistration';
import { BatchAssetSync } from './BatchAssetSync';
import { BATCH_ASSET_SYNC_LIMIT, batchSyncSourceError, reconcileBatchSyncSelection, toggleBatchSyncVisible } from './batchAssetSyncData';
import { DatasetFormatTemplates, DatasetFormatTemplateDialog } from './DatasetFormatTemplates';
import { datasetNormBindingPayload, datasetNormInputError } from './datasetNormAssets';
import './asset-registration.css';
import './asset-catalog.css';
import {
  DatasetProfileEditor,
  DatasetProfileForm,
  datasetProfileErrors,
  datasetProfileSummary,
  emptyDatasetProfile,
  normalizedDatasetProfile,
  readDatasetProfile,
} from './DatasetProfileForm';

export function RunTable({ runs, compact = false }: { runs: Run[]; compact?: boolean }) {
  const { nodes, templates, refresh } = useData();
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>训练任务</th>
            <th>状态</th>
            {!compact && <th>训练模板</th>}
            <th>运行机器</th>
            <th>创建时间</th>
            <th aria-label="任务操作" />
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td>
                <a className="row-link" href={`#/runs/${run.id}`}>
                  <span className={`row-icon ${run.state === 'RUNNING' ? 'teal' : ''}`}>
                    <Workflow size={16} />
                  </span>
                  <div>
                    <strong>{run.name}</strong>
                    <span className="subtext mono">{run.id.slice(0, 12)}</span>
                  </div>
                </a>
              </td>
              <td>
                <Badge state={run.state} />
              </td>
              {!compact && (
                <td>
                  <span className="tag">
                    {templates.data?.find((item) => item.id === run.template_id)?.name ||
                      run.template_id}
                  </span>
                </td>
              )}
              <td>
                <a className="node-link" href={`#/nodes/${run.node_id}`}>
                  <Server size={13} />
                  {nodes.data?.find((item) => item.id === run.node_id)?.name ||
                    run.node_id.slice(0, 10)}
                </a>
              </td>
              <td className="muted nowrap">{date(run.created_at)}</td>
              <td className="nowrap">
                <a
                  className="icon-button"
                  href={`#/runs/${run.id}`}
                  aria-label={`查看 ${run.name}`}
                >
                  <ChevronRight size={16} />
                </a>
                <RunDeleteAction run={run} compact onDeleted={refresh} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function OverviewPage({ createRun }: { createRun: () => void }) {
  const { can } = useAuth();
  const { enabled, overview, nodes, runs, assets, alerts } = useData();
  const stats = overview.data;
  const recent = [...(runs.data || [])]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .slice(0, 5);
  return (
    <>
      <PageHeading
        eyebrow="WORKSPACE OVERVIEW"
        title="训练工作台"
        description="让每一次训练，都有迹可循。"
        actions={
          <>
            <span className="last-update">
              <span className="live-dot" />
              {overview.updatedAt
                ? `更新于 ${overview.updatedAt.toLocaleTimeString('zh-CN')}`
                : '等待连接'}
            </span>
            {can('training.write') && (
              <button className="button primary" onClick={createRun} disabled={!enabled}>
                <Plus size={17} />
                新建训练
              </button>
            )}
          </>
        }
      />
      {!enabled && (
        <div className="connect-banner">
          <div className="connect-symbol">
            <Network size={24} />
          </div>
          <div>
            <h3>连接你的训练基础设施</h3>
            <p>登录平台账号，查看已接入机器、本地资产和训练任务。</p>
          </div>
          <button className="button primary" onClick={() => navigate('/settings')}>
            连接管理服务
            <ArrowRight size={16} />
          </button>
        </div>
      )}
      <ErrorNotice error={overview.error} retry={overview.refresh} />
      <div className="stat-grid">
        {[
          {
            label: '在线机器',
            value: stats?.nodes_online,
            extra: `共 ${stats?.nodes_total ?? '—'} 台裸金属`,
            icon: Server,
            color: 'blue',
          },
          {
            label: '公共可分配 GPU',
            value: stats?.gpus_available,
            extra: `共 ${stats?.gpus_total ?? '—'} 张 GPU`,
            icon: Cpu,
            color: 'teal',
          },
          {
            label: '进行中的训练',
            value: stats?.runs_active,
            extra: `${stats?.runs_failed ?? '—'} 个任务失败`,
            icon: Activity,
            color: 'purple',
          },
          {
            label: '已登记资产',
            value: stats?.assets_total,
            extra: '本机数据集 · 模型 · 参数',
            icon: Database,
            color: 'orange',
          },
        ].map((item) => (
          <div className="stat-card" key={item.label}>
            <div className="stat-head">
              <span>{item.label}</span>
              <div className={`stat-icon ${item.color}`}>
                <item.icon size={19} />
              </div>
            </div>
            <strong className="stat-number">{item.value ?? '—'}</strong>
            <div className="stat-foot">{item.extra}</div>
          </div>
        ))}
      </div>
      <div className="overview-grid">
        <Panel
          title="最近训练"
          subtitle="从启动到产出，掌握每一步进展"
          action={<ViewLink to="/runs" />}
          className="recent-runs"
        >
          {runs.loading && enabled && !runs.data ? (
            <Loading />
          ) : recent.length ? (
            <RunTable runs={recent} compact />
          ) : (
            <Empty
              icon={Workflow}
              title="你的第一项训练，从这里开始"
              description="接入机器并登记本机资产后，选择模板即可启动训练。"
              action={
                can('training.write') && (
                  <button className="button outline small" disabled={!enabled} onClick={createRun}>
                    <Plus size={15} />
                    创建训练任务
                  </button>
                )
              }
            />
          )}
        </Panel>
        <Panel title="资源概况" subtitle="整机资源与实际占用" action={<ViewLink to="/nodes" />}>
          {nodes.data?.length ? (
            <div className="resource-list">
              {nodes.data.slice(0, 4).map((node) => (
                <button
                  key={node.id}
                  className="resource-item"
                  onClick={() => navigate(`/nodes/${node.id}`)}
                >
                  <div className="resource-top">
                    <span className="resource-name">
                      <Server size={16} />
                      <strong>{node.name}</strong>
                    </span>
                    <Badge state={node.status} />
                  </div>
                  <div className="resource-sub">
                    <span>{node.gpus?.length || 0} 张 GPU</span>
                    <span>
                      {node.status === 'ONLINE' ? node.gpus?.filter(gpuIdle).length || 0 : '—'}{' '}
                      张公共空闲
                    </span>
                  </div>
                  <ProgressBar value={gpuAverage(node.gpus)} />
                </button>
              ))}
            </div>
          ) : (
            <Empty
              icon={Server}
              title="尚未接入机器"
              description="在裸金属安装 Agent，资源将自动出现在这里。"
              action={<ViewLink to="/nodes">查看接入说明</ViewLink>}
            />
          )}
        </Panel>
      </div>
      <div className="overview-bottom">
        <Panel title="开始使用" subtitle="只需三步，连接研究与训练">
          <div className="onboarding-steps">
            {[
              {
                n: '01',
                icon: Server,
                name: '接入计算资源',
                desc: 'Agent 自动报告 GPU 与环境',
                to: '/nodes',
                done: !!nodes.data?.length,
              },
              {
                n: '02',
                icon: Database,
                name: '登记本机资产',
                desc: '建立数据、模型与机器的关联',
                to: '/assets',
                done: !!assets.data?.length,
              },
              {
                n: '03',
                icon: Workflow,
                name: '启动模板训练',
                desc: '自动预检并记录实际代码版本',
                to: '/templates',
                done: !!runs.data?.length,
              },
            ].map((step) => (
              <a className="onboarding-step" href={`#${step.to}`} key={step.n}>
                <span className="step-number">
                  {step.done ? <CheckCircle2 size={17} /> : step.n}
                </span>
                <step.icon size={20} />
                <strong>{step.name}</strong>
                <p>{step.desc}</p>
                <ArrowRight size={16} />
              </a>
            ))}
          </div>
        </Panel>
        <Panel
          title="需要关注"
          subtitle="来自机器和训练的实时异常"
          action={<span className="count-pill">{alerts.data?.length ?? '—'}</span>}
        >
          <ErrorNotice error={alerts.error} retry={alerts.refresh} />
          {alerts.data?.length ? (
            <div className="alert-list">
              {alerts.data.slice(0, 4).map((alert, index) => (
                <div className="alert-item" key={alert.id || index}>
                  <AlertTriangle size={17} />
                  <div>
                    <strong>{alert.title || alert.code || '运行异常'}</strong>
                    <p>{alert.message}</p>
                    {alert.run_id && <ViewLink to={`/runs/${alert.run_id}`}>查看训练</ViewLink>}
                    {!alert.run_id && alert.node_id && (
                      <ViewLink to={`/nodes/${alert.node_id}`}>查看机器</ViewLink>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="healthy-state">
              <div>
                <ShieldCheck size={26} />
              </div>
              <strong>{enabled && alerts.data ? '当前没有待处理异常' : '等待资源状态上报'}</strong>
              <p>{enabled && alerts.data ? '异常会在这里及时展示' : '连接管理服务后即可查看'}</p>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}

function GpuCard({ gpu, index }: { gpu: Gpu; index: number }) {
  const { runs } = useData();
  const reservedRun = runs.data?.find((run) =>
    run.attempts?.some((attempt) => attempt.id === gpu.reserved_by),
  );
  const total = gpu.memory_total_mb,
    used = gpu.memory_used_mb;
  const unknown =
    gpu.processes == null || gpu.utilization_pct == null || total == null || used == null;
  return (
    <div className="gpu-card">
      <div className="gpu-card-title">
        <span className="gpu-index">GPU {index}</span>
        <strong>{gpu.name}</strong>
        <span
          className={`tiny-status ${gpu.reserved_by || gpu.user_reservation ? 'amber' : unknown ? 'gray' : gpuIdle(gpu) ? 'green' : 'blue'}`}
        >
          {gpu.reserved_by
            ? '任务占用'
            : gpu.user_reservation
              ? '用户预留'
              : unknown
              ? '占用未知'
              : gpuIdle(gpu)
                ? '空闲'
                : '资源占用'}
        </span>
      </div>
      <div className="gpu-measures">
        <div>
          <span>GPU 利用率</span>
          <strong>{displayNumber(gpu.utilization_pct, '%')}</strong>
          <ProgressBar value={gpu.utilization_pct} />
        </div>
        <div>
          <span>显存</span>
          <strong>
            {used == null || total == null
              ? '—'
              : `${(used / 1024).toFixed(1)} / ${(total / 1024).toFixed(0)} GB`}
          </strong>
          <ProgressBar
            value={used != null && total ? (used / total) * 100 : undefined}
            tone="blue"
          />
        </div>
      </div>
      <div className="gpu-meta">
        <code title={gpu.uuid}>{gpu.uuid}</code>
        {reservedRun && <a href={`#/runs/${reservedRun.id}`}>查看任务</a>}
      </div>
      {gpu.user_reservation && <div className="gpu-user-reservation">
        <LockKeyhole size={13} />
        预留给 {gpu.user_reservation.display_name || gpu.user_reservation.username}（{gpu.user_reservation.username}）
      </div>}
      {!!gpu.processes?.length && (
        <div className="process-info">
          {gpu.processes.map((process) => (
            <div key={process.pid}>
              PID {process.pid} ·{' '}
              {process.kind === 'graphics'
                ? '图形进程'
                : process.kind === 'compute'
                  ? '计算进程'
                  : process.kind === 'compute_graphics'
                    ? '计算与图形进程'
                    : '类型未知'}{' '}
              ·{' '}
              {process.ownership === 'platform'
                ? '平台训练进程'
                : process.ownership === 'external'
                  ? '外部进程'
                  : '归属未知'}{' '}
              · 显存 {displayNumber(process.memory_mb, ' MiB')}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function NodesPage() {
  const { can } = useAuth();
  const { nodes, assets, profiles, enabled } = useData();
  const [query, setQuery] = useState('');
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [managing, setManaging] = useState<Node>();
  const [terminalNode, setTerminalNode] = useState<Node>();
  const filtered = nodes.data?.filter((node) => matchesNodeQuery(node, query)) || [];
  return (
    <>
      <PageHeading
        eyebrow="COMPUTE RESOURCES"
        title="机器资源"
        description="统一查看裸金属资源、本地资产和运行环境。"
        actions={
          can('machines.manage') && (
            <button
              className="button primary"
              onClick={() => setShowOnboarding(true)}
              disabled={!enabled}
            >
              <Plus size={17} />
              接入机器
            </button>
          )
        }
      />
      <div className="toolbar">
        <div className="segmented">
          <span className="selected">
            全部机器 <b>{nodes.data?.length ?? '—'}</b>
          </span>
        </div>
        <div className="search-field">
          <Search size={16} />
          <input
            aria-label="搜索机器"
            placeholder="搜索机器名称、主机名或 IP"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
      </div>
      <ErrorNotice error={nodes.error} retry={nodes.refresh} />
      {nodes.loading && enabled && !nodes.data ? (
        <Loading />
      ) : filtered.length ? (
        <div className="machine-grid">
          {filtered.map((node) => (
            <div className="machine-card" key={node.id}>
              <a href={`#/nodes/${node.id}`} className="machine-card-main">
                <div className="machine-header">
                  <div className="machine-symbol">
                    <Server size={23} />
                  </div>
                  <div>
                    <h2>{node.name}</h2>
                    <p>{node.hostname}</p>
                  </div>
                  <Badge state={node.status} />
                </div>
                <div className="machine-address">
                  <Network size={14} />
                  <span>IP / 连接地址</span>
                  <strong className="mono">{nodeAddressText(node) || '未记录连接地址'}</strong>
                </div>
                <div className="machine-gpu">
                  <Cpu size={16} />
                  <span>{node.gpus?.[0]?.name || '尚无 GPU 信息'}</span>
                  <strong>× {node.gpus?.length || 0}</strong>
                </div>
                <div className="machine-stats">
                  <div>
                    <strong>
                      {assets.data?.filter(
                        (asset) =>
                          asset.node_id === node.id && (asset.state || asset.status) !== 'ARCHIVED',
                      ).length ?? '—'}
                    </strong>
                    <span>本地资产</span>
                  </div>
                  <div>
                    <strong>
                      {profiles.data?.filter((profile) => profile.node_id === node.id).length ??
                        '—'}
                    </strong>
                    <span>运行环境</span>
                  </div>
                  <div>
                    <strong>
                      {node.status === 'ONLINE' ? node.gpus?.filter(gpuIdle).length || 0 : '—'}
                    </strong>
                    <span>公共空闲 GPU</span>
                  </div>
                </div>
                <div className="machine-footer">
                  <span>
                    <span className={`status-dot ${node.status === 'ONLINE' ? 'online' : ''}`} />
                    最近心跳 {date(node.last_seen)}
                  </span>
                  <ChevronRight size={16} />
                </div>
              </a>
              {can('machines.manage') && (
                <div className="machine-card-management">
                  <button className="text-link" onClick={() => setTerminalNode(node)}>
                    <Terminal size={13} />终端
                  </button>
                  <button className="text-link" onClick={() => setManaging(node)}>
                    <Settings2 size={13} />
                    管理连接
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="panel">
          <Empty
            icon={Server}
            title={query ? '没有匹配的机器' : '接入第一台裸金属'}
            description={
              query
                ? '请尝试其他名称、主机名或 IP。'
                : '在机器上安装 Agent，计算资源、运行环境及在线状态会自动汇报。'
            }
            action={
              !query &&
              can('machines.manage') && (
                <button
                  className="button outline"
                  onClick={() => setShowOnboarding(true)}
                  disabled={!enabled}
                >
                  接入机器
                  <ArrowRight size={16} />
                </button>
              )
            }
          />
        </div>
      )}
      {can('machines.manage') && showOnboarding && (
        <NodeOnboarding close={() => setShowOnboarding(false)} />
      )}
      {can('machines.manage') && managing && (
        <NodeManagement
          node={managing}
          close={() => setManaging(undefined)}
          reconnect={() => {
            setManaging(undefined);
            setShowOnboarding(true);
          }}
        />
      )}
      {can('machines.manage') && terminalNode && (
        <NodeTerminal node={terminalNode} close={() => setTerminalNode(undefined)} />
      )}
    </>
  );
}

export function NodePage({ id, createRun }: { id: string; createRun: (nodeId?: string) => void }) {
  const { can } = useAuth();
  const { nodes, assets, profiles, runs } = useData();
  const details = useApi<NodeDetail | { node: Node; assets: Asset[] }>(`/nodes/${id}`);
  const [showManagement, setShowManagement] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showRename, setShowRename] = useState(false);
  const [showTerminal, setShowTerminal] = useState(false);
  const [showReservations, setShowReservations] = useState(false);
  const [reservationRevision, setReservationRevision] = useState(0);
  const raw = details.data;
  const node =
    (raw && 'node' in raw ? raw.node : (raw as NodeDetail | undefined)) ||
    nodes.data?.find((item) => item.id === id);
  const nodeAssets =
    assets.data?.filter(
      (asset) => asset.node_id === id && (asset.state || asset.status) !== 'ARCHIVED',
    ) || [];
  const nodeProfiles = profiles.data?.filter((profile) => profile.node_id === id) || [];
  const nodeRuns = runs.data?.filter((run) => run.node_id === id) || [];
  return (
    <>
      <Back to="/nodes" label="返回机器资源" />
      <PageHeading
        eyebrow="NODE DETAIL"
        title={node?.name || '机器详情'}
        description={
          node
            ? `${node.hostname} · IP / 连接地址：${nodeAddressText(node) || '未记录连接地址'}`
            : '查看机器资源与本地资产关联'
        }
        actions={
          <>
            {can('machines.manage') && (
              <>
                <button
                  className="button outline"
                  disabled={!node}
                  onClick={() => setShowTerminal(true)}
                >
                  <Terminal size={16} />终端
                </button>
                <button
                  className="button outline"
                  disabled={!node}
                  onClick={() => setShowRename(true)}
                >
                  <Pencil size={16} />
                  重命名
                </button>
                <button
                  className="button outline"
                  disabled={!node}
                  onClick={() => setShowManagement(true)}
                >
                  <Settings2 size={16} />
                  管理连接
                </button>
              </>
            )}
            {can('training.write') && (
              <button
                className="button primary"
                disabled={node?.status !== 'ONLINE'}
                onClick={() => createRun(id)}
              >
                <Plus size={16} />
                在此机器训练
              </button>
            )}
          </>
        }
      />
      <ErrorNotice error={details.error} retry={details.refresh} />
      {!node ? (
        details.loading ? (
          <Loading />
        ) : null
      ) : (
        <>
          {['DISCONNECTED', 'REMOVED'].includes(node.status) && (
            <div className="node-retired-banner">
              <div>
                <strong>
                  {node.status === 'REMOVED' ? '此机器的 Agent 已卸载' : '此机器已断开连接'}
                </strong>
                <p>历史训练及资产关联保留。重新接入后才能创建新的训练任务。</p>
              </div>
              {can('machines.manage') && (
                <button className="button outline small" onClick={() => setShowOnboarding(true)}>
                  <Plus size={14} />
                  重新接入机器
                </button>
              )}
            </div>
          )}
          <div className="node-summary">
            <div>
              <span>连接状态</span>
              <Badge
                state={node.status}
                label={
                  node.status === 'DISCONNECTED'
                    ? '已断开'
                    : node.status === 'REMOVED'
                      ? '已卸载'
                      : undefined
                }
              />
            </div>
            <div>
              <span>最近心跳</span>
              <strong>{date(node.last_seen)}</strong>
            </div>
            <div>
              <span>GPU 数量</span>
              <strong>{node.gpus?.length || 0} 张</strong>
            </div>
            <div>
              <span>活动训练</span>
              <strong>{nodeRuns.filter((run) => activeStates.has(run.state)).length}</strong>
            </div>
          </div>
          <ResourceHistory key={`history-${id}`} nodeId={id} status={node.status} />
          <div className="detail-columns">
            <div>
              <NodeGpuResources key={`gpu-${id}`} node={node} refreshToken={reservationRevision} onManage={() => setShowReservations(true)}>
                {node.gpus?.length ? (
                  <div className="gpu-list">
                    {node.gpus.map((gpu, index) => (
                      <GpuCard key={gpu.uuid} gpu={gpu} index={index} />
                    ))}
                  </div>
                ) : (
                  <Empty
                    icon={Cpu}
                    title="尚无 GPU 指标"
                    description="Agent 需要在 NVIDIA 环境中读取显卡指标，未知值不会按零展示。"
                  />
                )}
              </NodeGpuResources>
              <Panel
                title="本机资产"
                subtitle="仅可供这台机器上的训练使用"
                action={<ViewLink to={`/assets?node=${id}`}>管理资产</ViewLink>}
              >
                {nodeAssets.length ? (
                  <div className="association-list">
                    {nodeAssets.map((asset) => (
                      <div key={asset.id}>
                        <div className="row-icon">
                          {asset.kind === 'DATASET' ? <Database size={16} /> : <Box size={16} />}
                        </div>
                        <div>
                          <strong>{asset.name}</strong>
                          <span className="mono">{asset.path}</span>
                        </div>
                        <span className="tag">{assetKinds[asset.kind]}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <Empty
                    icon={Database}
                    title="还没有登记本机资产"
                    description="添加数据集、模型或归一化参数路径后，即可用于训练。"
                  />
                )}
              </Panel>
            </div>
            <div>
              <Panel title="运行环境" subtitle="来自 Agent 的可信配置">
                {nodeProfiles.length ? (
                  <div className="profile-list">
                    {nodeProfiles.map((profile) => (
                      <div className="profile-item" key={profile.id}>
                        <div>
                          <Terminal size={16} />
                          <strong>{profile.name}</strong>
                          <span className="tag">{profile.execution_mode || 'native'}</span>
                        </div>
                        <dl className="detail-list">
                          <div>
                            <dt>适配器</dt>
                            <dd>{profile.adapter}</dd>
                          </div>
                          <div>
                            <dt>代码目录</dt>
                            <dd>
                              <code>{profile.code_dir}</code>
                            </dd>
                          </div>
                          <div>
                            <dt>Python</dt>
                            <dd>
                              <code>{profile.python}</code>
                            </dd>
                          </div>
                          <div>
                            <dt>默认训练配方</dt>
                            <dd>
                              {profile.recipe ? (
                                <code>{profile.recipe}</code>
                              ) : (
                                '创建训练任务时选择'
                              )}
                            </dd>
                          </div>
                        </dl>
                        <RuntimeCheckView check={runtimeCheckFor(node, profile.id)} />
                        <ProfileRebinding node={node} profile={profile}
                          inUse={nodeRuns.some(run => run.profile_id === profile.id && activeStates.has(run.state))}
                          changed={() => { profiles.refresh(); nodes.refresh(); details.refresh(); }} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <Empty
                    icon={Terminal}
                    title="尚无运行环境"
                    description="在 Agent 配置中登记解释器、代码目录与适配器。"
                  />
                )}
              </Panel>
              <Panel
                title="本地存储"
                action={
                  <StorageConfiguration
                    node={node}
                    inUse={nodeRuns.some((run) => activeStates.has(run.state))}
                    changed={() => {
                      nodes.refresh();
                      details.refresh();
                    }}
                  />
                }
              >
                <dl className="detail-list padded">
                  <div>
                    <dt>允许目录</dt>
                    <dd>
                      {node.roots?.length
                        ? node.roots.map((root) => (
                            <code className="block" key={root}>
                              {root}
                            </code>
                          ))
                        : '—'}
                    </dd>
                  </div>
                  <div>
                    <dt>训练输出目录</dt>
                    <dd>
                      <code>{node.output_root || '—'}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>训练缓存根目录</dt>
                    <dd>
                      {node.cache_root ? (
                        <code>{node.cache_root}</code>
                      ) : (
                        '默认跟随训练项目'
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>CPU 利用率</dt>
                    <dd>
                      {displayNumber(
                        Number(node.metrics?.cpu_percent ?? node.metrics?.cpu_pct ?? NaN),
                        '%',
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>内存利用率</dt>
                    <dd>
                      {displayNumber(
                        Number(node.metrics?.memory_percent ?? node.metrics?.memory_pct ?? NaN),
                        '%',
                      )}
                    </dd>
                  </div>
                </dl>
              </Panel>
            </div>
          </div>
          <Panel title="关联训练" subtitle="使用这台机器的训练历史">
            {nodeRuns.length ? (
              <RunTable runs={nodeRuns} />
            ) : (
              <Empty
                icon={Workflow}
                title="尚无训练记录"
                description="从上方创建训练任务，选择这台机器上的数据与模型。"
              />
            )}
          </Panel>
        </>
      )}
      {can('machines.manage') && showManagement && node && (
        <NodeManagement
          node={node}
          close={() => setShowManagement(false)}
          changed={details.refresh}
          reconnect={() => {
            setShowManagement(false);
            setShowOnboarding(true);
          }}
        />
      )}
      {can('machines.manage') && showOnboarding && (
        <NodeOnboarding close={() => setShowOnboarding(false)} />
      )}
      {can('machines.manage') && showRename && node && (
        <NodeRename node={node} close={() => setShowRename(false)} changed={details.refresh} />
      )}
      {can('machines.manage') && showTerminal && node && (
        <NodeTerminal node={node} close={() => setShowTerminal(false)} />
      )}
      {showReservations && node && <GpuReservations key={node.id} node={node} close={() => setShowReservations(false)} changed={() => { details.refresh(); setReservationRevision((value) => value + 1); }} />}
    </>
  );
}

export function AssetsPage({ nodeFilter }: { nodeFilter?: string }) {
  const { user, can } = useAuth();
  const { assets, nodes, profiles, runs, enabled, refresh } = useData();
  const [kind, setKind] = useState('ALL');
  const [nodeId, setNodeId] = useState(nodeFilter || '');
  const [query, setQuery] = useState('');
  const [register, setRegister] = useState(false);
  const [batchRegister, setBatchRegister] = useState(false);
  const [syncSelection, setSyncSelection] = useState<{ owner: string; ids: string[] }>({
    owner: '',
    ids: [],
  });
  const [batchSyncOwner, setBatchSyncOwner] = useState('');
  const [archiving, setArchiving] = useState<Asset>();
  const [editingProfile, setEditingProfile] = useState<Asset>();
  const [bindingDataset, setBindingDataset] = useState<Asset>();
  const [templateAsset, setTemplateAsset] = useState<Asset>();
  const [detailId, setDetailId] = useState<string>();
  const detailAsset = assets.data?.find((asset) => asset.id === detailId);
  const [syncAsset, setSyncAsset] = useState<Asset>();
  const [backgroundSync, setBackgroundSync] = useState<AssetTransfer>();
  const [assetPage, setAssetPage] = useState({ filters: '', page: 1 });
  const assetTableRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [showArchived, setShowArchived] = useState(false);
  const canWriteAssets = can('assets.write');
  const selectionOwner = user?.id || '';
  const filterKey = JSON.stringify([selectionOwner, kind, nodeId, query, showArchived]);
  useEffect(() => {
    setAssetPage((previous) =>
      previous.filters === filterKey ? previous : { filters: filterKey, page: 1 },
    );
  }, [filterKey]);
  const selectedSyncIds =
    canWriteAssets && selectionOwner && syncSelection.owner === selectionOwner
      ? reconcileBatchSyncSelection(syncSelection.ids, assets.data || [], nodes.data || [])
      : [];
  useEffect(() => {
    setSyncSelection((previous) => {
      const ids =
        canWriteAssets && selectionOwner && previous.owner === selectionOwner
          ? reconcileBatchSyncSelection(previous.ids, assets.data || [], nodes.data || [])
          : [];
      return previous.owner === selectionOwner &&
        JSON.stringify(previous.ids) === JSON.stringify(ids)
        ? previous
        : { owner: selectionOwner, ids };
    });
    if (!canWriteAssets || batchSyncOwner !== selectionOwner) setBatchSyncOwner('');
  }, [selectionOwner, canWriteAssets, assets.data, nodes.data, batchSyncOwner]);
  useEffect(() => {
    if (canWriteAssets) return;
    setBatchRegister(false);
    setEditingProfile(undefined);
    setBindingDataset(undefined);
    setTemplateAsset(undefined);
    setSyncAsset(undefined);
    setArchiving(undefined);
  }, [canWriteAssets]);
  const assetSearchText = (asset: Asset) =>
    `${asset.name} ${asset.path} ${readDatasetProfile(asset.metadata?.dataset_profile)?.robot_type || ''} ${readDatasetProfile(asset.metadata?.dataset_profile)?.task_description || ''}`;
  const scopedAssets = (assets.data || []).filter(
    (asset) =>
      (kind === 'ALL' || asset.kind === kind) &&
      (!nodeId || asset.node_id === nodeId) &&
      (showArchived || (asset.status || asset.state) !== 'ARCHIVED'),
  );
  const scopedEntries = groupDatasetNormAssets(
    catalogAssetGroups(scopedAssets, assets.data || [], runs.data),
  );
  const groupSearchText = new Map<string, string>();
  for (const entry of scopedEntries) {
    if (entry.kind === 'asset') continue;
    const search =
      entry.kind === 'dataset-group'
        ? `${entry.name} ${entry.assets.map(assetSearchText).join(' ')}`
        : entry.name;
    for (const asset of entry.assets) groupSearchText.set(asset.id, search);
  }
  const filtered = scopedAssets.filter((asset) =>
    `${assetSearchText(asset)} ${groupSearchText.get(asset.id) || ''}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const catalogEntries = groupDatasetNormAssets(
    catalogAssetGroups(filtered, assets.data || [], runs.data),
  );
  const catalogPage = paginateAssetCatalog(
    catalogEntries,
    assetPage.filters === filterKey ? assetPage.page : 1,
  );
  useEffect(() => {
    setAssetPage((previous) =>
      previous.filters === filterKey && previous.page !== catalogPage.page
        ? { filters: filterKey, page: catalogPage.page }
        : previous,
    );
  }, [filterKey, catalogPage.page]);
  const changeAssetPage = (page: number) => {
    setAssetPage({ filters: filterKey, page });
    assetTableRef.current?.scrollIntoView({ block: 'start', behavior: 'instant' });
    assetTableRef.current?.focus({ preventScroll: true });
  };
  const modelGroupCount = catalogEntries.filter((entry) => entry.kind === 'model-group').length;
  const datasetGroupCount = catalogEntries.filter((entry) => entry.kind === 'dataset-group').length;
  const groupSummary = [
    modelGroupCount ? `${modelGroupCount} 个训练模型组` : '',
    datasetGroupCount ? `${datasetGroupCount} 个数据集集合` : '',
  ]
    .filter(Boolean)
    .join('，');
  const visibleSyncIds = catalogPage.assetIds.filter((id) =>
    !batchSyncSourceError(assets.data?.find((asset) => asset.id === id), nodes.data || []),
  );
  const selectedVisibleSyncCount = visibleSyncIds.filter((id) =>
    selectedSyncIds.includes(id),
  ).length;
  const allVisibleSyncSelected =
    visibleSyncIds.length > 0 && selectedVisibleSyncCount === visibleSyncIds.length;
  const updateSyncSelection = (visibleIds: string[], checked: boolean) =>
    setSyncSelection((previous) => ({
      owner: selectionOwner,
      ids: toggleBatchSyncVisible(
        previous.owner === selectionOwner
          ? reconcileBatchSyncSelection(previous.ids, assets.data || [], nodes.data || [])
          : [],
        visibleIds,
        checked,
      ),
    }));
  const renderAsset = (asset: Asset, grouped = false) => (
    <tr key={asset.id} className={grouped ? 'asset-group-member-row' : undefined}>
      <td className="asset-selection-column">
        <input
          type="checkbox"
          aria-label={`选择同步资产 ${asset.name}`}
          checked={selectedSyncIds.includes(asset.id)}
          disabled={
            !canWriteAssets ||
            !selectionOwner ||
            Boolean(batchSyncSourceError(asset, nodes.data || []))
          }
          title={batchSyncSourceError(asset, nodes.data || []) || `选择 ${asset.name}`}
          onChange={(event) => updateSyncSelection([asset.id], event.target.checked)}
        />
      </td>
      <td>
        <button
          type="button"
          className="row-link asset-name-link"
          aria-label={`查看资产：${asset.name}`}
          onClick={() => setDetailId(asset.id)}
        >
          <span className={`row-icon ${asset.kind === 'DATASET' ? 'blue' : 'teal'}`}>
            {asset.kind === 'DATASET' ? (
              <Database size={16} />
            ) : asset.kind === 'MODEL' ? (
              <Box size={16} />
            ) : (
              <FileSliders size={16} />
            )}
          </span>
          <strong>{asset.name}</strong>
        </button>
        {asset.kind === 'DATASET' && (
          <>
            <DatasetAssetFeatures asset={asset} />
            <DatasetNormSummary asset={asset} assets={assets.data || []} />
          </>
        )}
      </td>
      <td>
        <span>{assetKinds[asset.kind] || asset.kind}</span>
        <span className="subtext mono">{asset.version}</span>
      </td>
      <td>
        <a className="node-link" href={`#/nodes/${asset.node_id}`}>
          <Server size={13} />
          {nodes.data?.find((node) => node.id === asset.node_id)?.name ||
            asset.node_id.slice(0, 10)}
        </a>
      </td>
      <td>
        <code className="path-truncate" title={asset.path}>
          {asset.path}
        </code>
      </td>
      <td>
        <div className="asset-catalog-validation">
          <Badge
            state={asset.status || asset.state || 'REGISTERED'}
            label={asset.kind === 'DATASET' && (asset.status || asset.state || 'REGISTERED') === 'REGISTERED' ? '已登记' : undefined}
          />
          {asset.kind === 'DATASET' && <DatasetValidationBadge validation={asset.validation} />}
        </div>
      </td>
      <td>
        {can('assets.write') && (asset.status || asset.state) !== 'ARCHIVED' && (
          <button
            type="button"
            className="icon-button"
            aria-label={`删除登记 ${asset.name}`}
            title="删除登记"
            onClick={() => {
              setError(undefined);
              setArchiving(asset);
            }}
          >
            <Trash2 size={16} />
          </button>
        )}
      </td>
    </tr>
  );
  return (
    <>
      <PageHeading
        eyebrow="LOCAL ASSET CATALOG"
        title="数据与模型"
        description="统一登记本地资产，清晰关联机器、训练与产出。"
        actions={
          can('assets.write') && (
            <>
              <button
                type="button"
                className="button outline"
                disabled={!enabled || !nodes.data?.length}
                onClick={() => setBatchRegister(true)}
              >
                <Database size={17} />
                批量登记数据集
              </button>
              <button
                className="button primary"
                disabled={!enabled || !nodes.data?.length}
                onClick={() => setRegister(true)}
              >
                <Plus size={17} />
                登记资产
              </button>
            </>
          )
        }
      />
      <div className="info-strip">
        <HardDrive size={17} />
        <span>
          数据和模型保存在各自机器本地，可同步到其他机器，完成文件校验后登记副本。训练使用所选机器上的资产。
        </span>
      </div>
      {backgroundSync && <AssetSyncBackgroundNotice transfer={backgroundSync} />}
      <div className="toolbar wrap">
        <div className="segmented">
          {[
            ['ALL', '全部'],
            ['DATASET', '数据集'],
            ['MODEL', '模型'],
            ['NORM', '归一化参数'],
          ].map(([value, label]) => (
            <button
              className={kind === value ? 'selected' : ''}
              key={value}
              onClick={() => setKind(value)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="toolbar-filters">
          <select
            aria-label="按机器筛选"
            value={nodeId}
            onChange={(event) => setNodeId(event.target.value)}
          >
            <option value="">全部机器</option>
            {nodes.data?.map((node) => (
              <option value={node.id} key={node.id}>
                {node.name}
              </option>
            ))}
          </select>
          <div className="search-field">
            <Search size={15} />
            <input
              aria-label="搜索资产"
              placeholder="名称、路径、机器人或任务"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
        </div>
      </div>
      {canWriteAssets && (
        <div className="batch-asset-selection" aria-label="资产批量选择">
          <span>
            已选择 {selectedSyncIds.length} 项 · 当前页 {selectedVisibleSyncCount} 项
            {selectedSyncIds.length > selectedVisibleSyncCount
              ? ` · 另有 ${selectedSyncIds.length - selectedVisibleSyncCount} 项来自其他页或筛选`
              : ''}
          </span>
          <button
            type="button"
            className="text-link"
            disabled={!selectedSyncIds.length}
            onClick={() => setSyncSelection({ owner: selectionOwner, ids: [] })}
          >
            清空所有选择
          </button>
          <button
            type="button"
            className="button outline small"
            disabled={
              !enabled ||
              !selectionOwner ||
              !selectedSyncIds.length ||
              selectedSyncIds.length > BATCH_ASSET_SYNC_LIMIT
            }
            onClick={() => setBatchSyncOwner(selectionOwner)}
          >
            批量同步（{selectedSyncIds.length}）
          </button>
          {selectedSyncIds.length > BATCH_ASSET_SYNC_LIMIT && (
            <p className="batch-asset-sync-error" role="status">
              一次最多同步 {BATCH_ASSET_SYNC_LIMIT} 项，请减少选择后分批操作。
            </p>
          )}
        </div>
      )}
      <div className="table-meta">
        <span>
          {groupSummary
            ? `${catalogEntries.length} 项显示 · ${filtered.length} 个资产（${groupSummary}）`
            : `${filtered.length} 项资产`}
        </span>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={showArchived}
            onChange={(event) => setShowArchived(event.target.checked)}
          />
          显示已归档
        </label>
      </div>
      <ErrorNotice error={assets.error || error} retry={assets.refresh} />
      <div className="panel">
        {assets.loading && enabled && !assets.data ? (
          <Loading />
        ) : filtered.length ? (
          <div className="table-scroll" ref={assetTableRef} tabIndex={-1}>
            <table className="asset-catalog-table" aria-label="资产目录">
              <thead>
                <tr>
                  <th className="asset-selection-column">
                    <input
                      type="checkbox"
                      aria-label="全选当前页资产"
                      title="只选择或取消当前页中的可同步资产，其他页的选择会保留"
                      checked={allVisibleSyncSelected}
                      ref={(input) => {
                        if (input)
                          input.indeterminate =
                            selectedVisibleSyncCount > 0 && !allVisibleSyncSelected;
                      }}
                      disabled={!canWriteAssets || !selectionOwner || !visibleSyncIds.length}
                      onChange={(event) =>
                        updateSyncSelection(visibleSyncIds, event.target.checked)
                      }
                    />
                  </th>
                  <th>资产名称</th>
                  <th>类型 / 版本</th>
                  <th>所在机器</th>
                  <th>本地路径</th>
                  <th>状态</th>
                  <th>操作</th>
                </tr>
              </thead>
              {catalogPage.items.map((entry) =>
                entry.kind === 'asset' ? (
                  <tbody key={`asset-${entry.asset.id}`}>{renderAsset(entry.asset)}</tbody>
                ) : (
                  <AssetGroupRows
                    key={entry.key}
                    group={entry}
                    node={nodes.data?.find((node) => node.id === entry.nodeId)}
                    selectedIds={selectedSyncIds}
                    eligibleIds={entry.assets
                      .filter((asset) => visibleSyncIds.includes(asset.id))
                      .map((asset) => asset.id)}
                    canSelect={canWriteAssets && Boolean(selectionOwner)}
                    onSelectionChange={updateSyncSelection}
                    renderAsset={renderAsset}
                    viewDataset={(dataset) => setDetailId(dataset.id)}
                  />
                ),
              )}
            </table>
          </div>
        ) : (
          <Empty
            icon={Database}
            title={
              query || nodeId || kind !== 'ALL' ? '没有符合条件的资产' : '建立你的本地资产目录'
            }
            description={
              query || nodeId || kind !== 'ALL'
                ? '调整搜索条件，或登记新的本地资产。'
                : '登记已有数据集和模型的本机路径。预检时，Agent 会验证它们是否可用。'
            }
            action={
              can('assets.write') && (
                <button
                  className="button outline small"
                  disabled={!enabled || !nodes.data?.length}
                  onClick={() => setRegister(true)}
                >
                  <Plus size={15} />
                  登记资产
                </button>
              )
            }
          />
        )}
      </div>
      {assets.data && (
        <nav className="asset-catalog-pagination" aria-label="资产目录分页">
          <span>
            每页 {ASSET_CATALOG_PAGE_SIZE} 项 · 第 {catalogPage.start}–{catalogPage.end} 项 / 共{' '}
            {catalogPage.total} 项
          </span>
          <div className="asset-catalog-page-controls">
            <button
              type="button"
              className="button outline small"
              disabled={catalogPage.page <= 1}
              onClick={() => changeAssetPage(catalogPage.page - 1)}
            >
              <ChevronLeft size={14} />
              上一页
            </button>
            <span aria-live="polite">第 {catalogPage.page} / {catalogPage.totalPages} 页</span>
            <button
              type="button"
              className="button outline small"
              disabled={catalogPage.page >= catalogPage.totalPages}
              onClick={() => changeAssetPage(catalogPage.page + 1)}
            >
              下一页
              <ChevronRight size={14} />
            </button>
          </div>
          <p>集合按一项计数，展开后可查看组内资产。</p>
        </nav>
      )}
      {detailAsset &&
        !(batchSyncOwner && batchSyncOwner === selectionOwner) &&
        !batchRegister &&
        !syncAsset &&
        !editingProfile &&
        !bindingDataset &&
        !archiving &&
        !templateAsset && (
          <AssetDetail
            asset={detailAsset}
            datasetChoices={detailAsset.kind === 'DATASET' && detailAsset.norm_asset_id
              ? (assets.data || []).filter((asset) =>
                asset.kind === 'DATASET' && asset.node_id === detailAsset.node_id &&
                asset.norm_asset_id === detailAsset.norm_asset_id &&
                (showArchived || (asset.status || asset.state) !== 'ARCHIVED'))
              : undefined}
            selectDataset={setDetailId}
            node={nodes.data?.find((node) => node.id === detailAsset.node_id)}
            assets={assets.data || []}
            profiles={profiles.data}
            validationChanged={assets.refresh}
            close={() => setDetailId(undefined)}
            bindNorm={() => setBindingDataset(detailAsset)}
            editProfile={() => setEditingProfile(detailAsset)}
            sync={() => setSyncAsset(detailAsset)}
            saveTemplate={() => setTemplateAsset(detailAsset)}
            remove={() => {
              setError(undefined);
              setArchiving(detailAsset);
            }}
          />
        )}
      {can('assets.write') && syncAsset && (
        <AssetSyncDialog
          key={syncAsset.id}
          asset={syncAsset}
          close={() => setSyncAsset(undefined)}
          created={(transfer) => {
            setSyncAsset(undefined);
            setBackgroundSync(transfer);
            refresh();
          }}
        />
      )}
      {canWriteAssets && selectionOwner && batchSyncOwner === selectionOwner && (
        <BatchAssetSync
          assetIds={selectedSyncIds}
          close={() => setBatchSyncOwner('')}
          created={() => {
            setSyncSelection({ owner: selectionOwner, ids: [] });
            refresh();
          }}
        />
      )}
      {can('assets.write') && batchRegister && (
        <BatchDatasetRegistration initialNode={nodeId} close={() => setBatchRegister(false)} />
      )}
      {can('assets.write') && register && (
        <RegisterAsset key={nodeId} initialNode={nodeId} close={() => setRegister(false)} />
      )}
      {can('assets.write') && editingProfile && (
        <DatasetProfileEditor asset={editingProfile} close={() => setEditingProfile(undefined)} />
      )}
      {can('assets.write') && bindingDataset && (
        <DatasetNormBindingDialog
          key={bindingDataset.id}
          asset={bindingDataset}
          close={() => setBindingDataset(undefined)}
        />
      )}
      {can('assets.write') && templateAsset && (
        <DatasetFormatTemplateDialog
          key={templateAsset.id}
          asset={templateAsset}
          close={() => setTemplateAsset(undefined)}
        />
      )}
      {can('assets.write') && archiving && (
        <Modal
          title="删除资产登记"
          close={() => {
            if (!busy) setArchiving(undefined);
          }}
          footer={
            <>
              <button
                className="button outline"
                disabled={busy}
                onClick={() => setArchiving(undefined)}
              >
                取消
              </button>
              <button
                className="button danger"
                disabled={busy}
                onClick={async () => {
                  if (!can('assets.write')) return;
                  setBusy(true);
                  setError(undefined);
                  try {
                    await post(`/assets/${archiving.id}/archive`);
                    setArchiving(undefined);
                    setDetailId(undefined);
                    refresh();
                  } catch (err) {
                    setError(err as Error);
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                {busy && <LoaderCircle size={16} className="spin" />}确认删除登记
              </button>
            </>
          }
        >
          <p>
            删除「{archiving.name}
            」的资产登记，将其从新训练的可选资产中移除。历史训练关联和本机文件会继续保留。
          </p>
          {error && <ErrorNotice error={error} />}
        </Modal>
      )}
    </>
  );
}
function DatasetAssetFeatures({ asset }: { asset: Asset }) {
  const profile = readDatasetProfile(asset.metadata?.dataset_profile);
  if (!profile) return <span className="dataset-profile-summary">尚未登记数据特征</span>;
  return (
    <>
      <span className="dataset-profile-summary">{datasetProfileSummary(profile)}</span>
      <span className="dataset-profile-task" title={profile.task_description}>
        {profile.task_description.length > 90
          ? `${profile.task_description.slice(0, 90)}…`
          : profile.task_description}
      </span>
    </>
  );
}

function RegisterAsset(props: { close: () => void; initialNode?: string }) {
  const { can } = useAuth();
  if (!can('assets.write'))
    return (
      <Modal title="权限不足" close={props.close}>
        登记资产需要操作员或管理员权限。
      </Modal>
    );
  return <RegisterAssetContent {...props} />;
}

function assetNameFromPath(path: string) {
  const basename = path.replace(/\/+$/, '').split('/').at(-1) || '';
  if (basename === '.' || basename === '..') return '';
  let name = '';
  for (const character of basename) {
    if (name.length + character.length > 160) break;
    name += character;
  }
  return name;
}

function RegisterAssetContent({ close, initialNode }: { close: () => void; initialNode?: string }) {
  const { nodes, assets, refresh } = useData();
  const [normAssetId, setNormAssetId] = useState('');
  const [normPath, setNormPath] = useState('');
  const [form, setForm] = useState({
    name: '',
    kind: 'DATASET',
    node_id: initialNode || '',
    path: '',
    version: 'v1',
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [datasetProfile, setDatasetProfile] = useState(emptyDatasetProfile);
  const inspectionDraft = useDatasetInspectionDraft(datasetProfile, setDatasetProfile);
  const [deferProfile, setDeferProfile] = useState(false);
  const pending = useRef(false);
  const nameEdited = useRef(false);
  const profileErrors =
    form.kind === 'DATASET' && !deferProfile ? datasetProfileErrors(datasetProfile) : [];
  const normError = form.kind === 'DATASET' ? datasetNormInputError(form.node_id, normAssetId, normPath, assets.data || []) : undefined;
  const set = (key: keyof typeof form, value: string) => {
    if (key === 'name') nameEdited.current = true;
    const fillName = key === 'path' && !nameEdited.current;
    setForm((previous) => ({
      ...previous,
      [key]: value,
      ...(fillName ? { name: assetNameFromPath(value) } : {}),
    }));
    if (key === 'node_id' || key === 'kind') {
      setNormAssetId('');
      setNormPath('');
    }
  };
  const node = nodes.data?.find((item) => item.id === form.node_id);
  const showDatasetConfig = form.kind === 'DATASET' && !deferProfile;
  return (
    <Modal
      title="登记本地资产"
      wide
      className={`asset-register-modal ${showDatasetConfig ? '' : 'asset-register-simple'}`}
      subtitle={form.kind === 'DATASET'
        ? '登记后自动在后台执行基础校验，可在资产列表与详情查看结果。'
        : '建立资产元数据与裸金属本地路径的关联。'}
      close={() => {
        if (!pending.current) close();
      }}
      footer={
        <>
          <button
            className="button outline"
            onClick={() => {
              if (!pending.current) close();
            }}
            disabled={busy}
          >
            取消
          </button>
          <button
            className="button primary"
            form="register-asset"
            type="submit"
            disabled={busy || profileErrors.length > 0 || Boolean(normError)}
          >
            {busy ? <LoaderCircle size={16} className="spin" /> : <Check size={16} />}登记资产
          </button>
        </>
      }
    >
      <form
        id="register-asset"
        className={`asset-registration-layout ${showDatasetConfig ? 'with-dataset-config' : ''}`}
        onSubmit={async (event) => {
          event.preventDefault();
          if (pending.current || profileErrors.length > 0 || normError) return;
          pending.current = true;
          setBusy(true);
          setError(undefined);
          try {
            await post('/assets', {
              ...form,
              ...(form.kind === 'DATASET' ? datasetNormBindingPayload(normAssetId, normPath) : { norm_asset_id: null }),
              metadata:
                form.kind === 'DATASET' && !deferProfile
                  ? { dataset_profile: normalizedDatasetProfile(datasetProfile) }
                  : {},
            });
            refresh();
            close();
          } catch (err) {
            setError(err as Error);
          } finally {
            pending.current = false;
            setBusy(false);
          }
        }}
      >
        <section className="asset-registration-basics" aria-label="资产信息">
          <h3>资产信息</h3>
          <Field label="资产名称" required hint="根据所选目录或文件名自动填写，可手动修改。">
            <input
              required
              disabled={busy}
              maxLength={160}
              placeholder="例如 shellgame-dataset-v1"
              value={form.name}
              onChange={(event) => set('name', event.target.value)}
            />
          </Field>
          <div className="form-grid">
            <Field label="资产类型">
              <select
                disabled={busy}
                value={form.kind}
                onChange={(event) => set('kind', event.target.value)}
              >
                {Object.entries(assetKinds).map(([value, label]) => (
                  <option value={value} key={value}>
                    {label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="资产版本" required>
              <input
                required
                disabled={busy}
                maxLength={80}
                placeholder="v1"
                value={form.version}
                onChange={(event) => set('version', event.target.value)}
              />
            </Field>
          </div>
          <Field label="所在机器" required>
            <select
              required
              disabled={busy}
              value={form.node_id}
              onChange={(event) => set('node_id', event.target.value)}
            >
              <option value="">选择机器</option>
              {nodes.data?.map((item) => (
                <option value={item.id} key={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </Field>
          <PathAutocomplete
            key={form.node_id}
            node={node}
            value={form.path}
            onChange={(value) => set('path', value)}
            disabled={busy}
            hint="可输入完整路径，也可逐级选择目录。"
          />
          {form.kind === 'DATASET' && (
            <DatasetNormField
              nodeId={form.node_id}
              node={node}
              assets={assets.data || []}
              value={normAssetId}
              onChange={setNormAssetId}
              pathValue={normPath}
              onPathChange={setNormPath}
              disabled={busy}
            />
          )}
          {showDatasetConfig && (
            <Field label="任务描述" required>
              <textarea
                required
                disabled={busy}
                maxLength={2000}
                rows={2}
                value={datasetProfile.task_description}
                placeholder="例如 抓取衣物、折叠后放入收纳盒。"
                onChange={(event) => {
                  inspectionDraft.touch('task_description');
                  inspectionDraft.manualChange({
                    ...datasetProfile,
                    task_description: event.target.value,
                  });
                }}
              />
            </Field>
          )}
          {form.kind === 'DATASET' && (
            <>
              <label className="dataset-profile-toggle">
                <input
                  type="checkbox"
                  checked={deferProfile}
                  disabled={busy}
                  onChange={(event) => setDeferProfile(event.target.checked)}
                />
                稍后补录数据集特征
              </label>
              {deferProfile ? (
                <p className="field-hint">
                  仍可登记路径。补录后，新建训练会使用这些特征匹配处理器。
                </p>
              ) : (
                <>
                  <DatasetInspection
                    compact
                    node={node}
                    path={form.path}
                    value={datasetProfile}
                    draft={inspectionDraft}
                    disabled={busy}
                  />
                </>
              )}
            </>
          )}
        </section>
        {showDatasetConfig && (
          <section className="asset-registration-features" aria-label="数据集配置">
            <DatasetFormatTemplates value={datasetProfile} onApply={inspectionDraft.applyTemplate} disabled={busy} />
            <DatasetProfileForm
              compact
              showTaskDescription={false}
              value={datasetProfile}
              onChange={inspectionDraft.manualChange}
              onFieldEdit={inspectionDraft.touch}
              disabled={busy}
            />
            {profileErrors.length > 0 && (
              <p className="field-hint required" role="status">
                {profileErrors[0]}
              </p>
            )}
          </section>
        )}
        <ErrorNotice error={error} />
      </form>
    </Modal>
  );
}

const algorithmInfo: Record<string, { icon: typeof Workflow; color: string; text: string }> = {
  SFT: { icon: Layers3, color: 'blue', text: 'SUPERVISED FINE-TUNING' },
  MEM: { icon: Network, color: 'purple', text: 'MEMORY-AUGMENTED VLA' },
  RECAP: { icon: ArrowDownRight, color: 'orange', text: 'VALUE-GUIDED TRAINING' },
  RLT: { icon: Sparkles, color: 'teal', text: 'REINFORCEMENT LEARNING' },
  SMOKE: { icon: Terminal, color: 'gray', text: 'CONTROL-PLANE SMOKE TEST' },
};
export function TemplatesPage({
  createRun,
}: {
  createRun: (nodeId?: string, templateId?: string) => void;
}) {
  const { can } = useAuth();
  const { templates, enabled } = useData();
  return (
    <>
      <PageHeading
        eyebrow="TRAINING RECIPES"
        title="训练模板"
        description="把实验配方整理成可复用的训练入口。"
      />
      <div className="info-strip">
        <GitCommitHorizontal size={17} />
        <span>每次启动由 Agent 采集实际 commit 和本地修改状态，随训练记录保存。</span>
      </div>
      <ErrorNotice error={templates.error} retry={templates.refresh} />
      {templates.loading && enabled && !templates.data ? (
        <Loading />
      ) : templates.data?.length ? (
        <div className="template-grid">
          {templates.data.map((template) => {
            const info = algorithmInfo[template.algorithm.toUpperCase()] || algorithmInfo.SFT;
            const Icon = info.icon;
            return (
              <article
                className={`template-card ${template.status !== 'AVAILABLE' ? 'planned' : ''}`}
                key={template.id}
              >
                <div className="template-card-top">
                  <div className={`template-icon ${info.color}`}>
                    <Icon size={25} strokeWidth={1.7} />
                  </div>
                  <Badge
                    state={template.status}
                    label={template.status === 'PLANNED' ? '待接入' : undefined}
                  />
                </div>
                <span className="template-eyebrow">{info.text}</span>
                <h2>{template.name}</h2>
                <p>{template.description}</p>
                {template.id === 'recap' && !!template.stages?.length && (
                  <div className="template-stage-list">
                    {template.stages.map((stage, index) => (
                      <span key={stage.id}>
                        {index + 1}. {stage.name}
                      </span>
                    ))}
                  </div>
                )}
                <div className="template-inputs">
                  {requiredInputs(template).map((input) => (
                    <span key={input}>
                      <Check size={13} />
                      {input === 'dataset'
                        ? '本机数据'
                        : input === 'model'
                          ? '基础模型'
                          : input === 'norm'
                            ? '归一化参数'
                            : input}
                    </span>
                  ))}
                </div>
                <footer>
                  <span className="tag mono">{template.id}</span>
                  {can('training.write') && (
                    <button
                      className={`button small ${template.status === 'AVAILABLE' ? 'outline' : 'ghost'}`}
                      disabled={template.status !== 'AVAILABLE'}
                      onClick={() => createRun(undefined, template.id)}
                    >
                      {template.status === 'AVAILABLE' ? (
                        <>
                          使用模板
                          <ArrowRight size={14} />
                        </>
                      ) : (
                        '待接入'
                      )}
                    </button>
                  )}
                </footer>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="panel">
          <Empty
            icon={Layers3}
            title="尚未读取到训练模板"
            description="连接管理服务后，平台会展示已经注册和验证的训练入口。"
          />
        </div>
      )}
      <div className="footnote">
        <ShieldCheck size={15} />
        可使用状态表示已开放该入口。真实算法训练能力仍取决于本机运行环境与配方验收；smoke
        模板只用于链路测试。
      </div>
    </>
  );
}

export function RunsPage({ createRun }: { createRun: () => void }) {
  return <BackgroundTasksPage createRun={createRun} />;
}
