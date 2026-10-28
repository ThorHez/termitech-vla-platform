import { useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Archive,
  ArrowDownRight,
  ArrowRight,
  Box,
  Check,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  FileSliders,
  GitCommitHorizontal,
  HardDrive,
  Layers3,
  LoaderCircle,
  Network,
  Pencil,
  Plus,
  Search,
  Server,
  Settings2,
  ShieldCheck,
  Sparkles,
  Terminal,
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
import { ResourceHistory } from './ResourceHistory';
import { ProfileRebinding } from './ProfileRebinding';
import { StorageConfiguration } from './StorageConfiguration';
import { matchesNodeQuery, nodeAddressText } from './nodeAddress';
import { PathAutocomplete } from './PathAutocomplete';
import { DatasetInspection, useDatasetInspectionDraft } from './DatasetInspection';
import './asset-registration.css';
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
  const { nodes, templates } = useData();
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
            <th aria-label="打开详情" />
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
              <td>
                <a
                  className="icon-button"
                  href={`#/runs/${run.id}`}
                  aria-label={`查看 ${run.name}`}
                >
                  <ChevronRight size={16} />
                </a>
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
            label: '可分配 GPU',
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
                      张空闲
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
          className={`tiny-status ${gpu.reserved_by ? 'amber' : unknown ? 'gray' : gpuIdle(gpu) ? 'green' : 'blue'}`}
        >
          {gpu.reserved_by
            ? '平台已预约'
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
        {reservedRun && <a href={`#/runs/${reservedRun.id}`}>查看预约</a>}
      </div>
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
  const [renaming, setRenaming] = useState<Node>();
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
                    <span>空闲 GPU</span>
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
              <div className="machine-card-management">
                <a className="text-link machine-history-link" href={`#/nodes/${node.id}`}>
                  <Activity size={13} />占用历史
                </a>
                {can('machines.manage') && <>
                  <button className="text-link" onClick={() => setRenaming(node)}>
                    <Pencil size={13} />
                    重命名
                  </button>
                  <button className="text-link" onClick={() => setManaging(node)}>
                    <Settings2 size={13} />
                    管理连接
                  </button>
                </>}
              </div>
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
      {can('machines.manage') && renaming && (
        <NodeRename node={renaming} close={() => setRenaming(undefined)} />
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
          <ResourceHistory key={id} nodeId={id} status={node.status} />
          <div className="detail-columns">
            <div>
              <Panel title="GPU 资源" subtitle="利用率为整卡指标；预约与实测占用分别展示">
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
              </Panel>
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
    </>
  );
}

export function AssetsPage({ nodeFilter }: { nodeFilter?: string }) {
  const { can } = useAuth();
  const { assets, nodes, enabled, refresh } = useData();
  const [kind, setKind] = useState('ALL');
  const [nodeId, setNodeId] = useState(nodeFilter || '');
  const [query, setQuery] = useState('');
  const [register, setRegister] = useState(false);
  const [archiving, setArchiving] = useState<Asset>();
  const [editingProfile, setEditingProfile] = useState<Asset>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [showArchived, setShowArchived] = useState(false);
  const filtered = (assets.data || []).filter(
    (asset) =>
      (kind === 'ALL' || asset.kind === kind) &&
      (!nodeId || asset.node_id === nodeId) &&
      (showArchived || (asset.status || asset.state) !== 'ARCHIVED') &&
      `${asset.name} ${asset.path} ${readDatasetProfile(asset.metadata?.dataset_profile)?.robot_type || ''} ${readDatasetProfile(asset.metadata?.dataset_profile)?.task_description || ''}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <>
      <PageHeading
        eyebrow="LOCAL ASSET CATALOG"
        title="数据与模型"
        description="统一登记本地资产，清晰关联机器、训练与产出。"
        actions={
          can('assets.write') && (
            <button
              className="button primary"
              disabled={!enabled || !nodes.data?.length}
              onClick={() => setRegister(true)}
            >
              <Plus size={17} />
              登记资产
            </button>
          )
        }
      />
      <div className="info-strip">
        <HardDrive size={17} />
        <span>数据和模型保存在各自机器本地。创建训练时，平台只提供同一台机器的可用资产。</span>
      </div>
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
      <div className="table-meta">
        <span>{filtered.length} 项资产</span>
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
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>资产名称</th>
                  <th>类型 / 版本</th>
                  <th>所在机器</th>
                  <th>本地路径</th>
                  <th>状态</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((asset) => (
                  <tr key={asset.id}>
                    <td>
                      <div className="row-link">
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
                      </div>
                      {asset.kind === 'DATASET' && <DatasetAssetFeatures asset={asset} />}
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
                      <Badge state={asset.status || asset.state || 'REGISTERED'} />
                    </td>
                    <td>
                      {can('assets.write') &&
                        asset.kind === 'DATASET' &&
                        (asset.status || asset.state) !== 'ARCHIVED' && (
                          <button
                            type="button"
                            className="button ghost small"
                            onClick={() => setEditingProfile(asset)}
                          >
                            <Pencil size={14} />
                            {readDatasetProfile(asset.metadata?.dataset_profile)
                              ? '编辑数据特征'
                              : '补录数据特征'}
                          </button>
                        )}
                      {can('assets.write') && (asset.status || asset.state) !== 'ARCHIVED' && (
                        <button
                          className="icon-button"
                          aria-label={`归档 ${asset.name}`}
                          title="归档登记"
                          onClick={() => setArchiving(asset)}
                        >
                          <Archive size={16} />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
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
      {can('assets.write') && register && (
        <RegisterAsset key={nodeId} initialNode={nodeId} close={() => setRegister(false)} />
      )}
      {can('assets.write') && editingProfile && (
        <DatasetProfileEditor asset={editingProfile} close={() => setEditingProfile(undefined)} />
      )}
      {can('assets.write') && archiving && (
        <Modal
          title="归档资产登记"
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
                    refresh();
                  } catch (err) {
                    setError(err as Error);
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                {busy && <LoaderCircle size={16} className="spin" />}确认归档
              </button>
            </>
          }
        >
          <p>将「{archiving.name}」从可选资产中归档。历史训练仍保留其关联，本机文件会继续保留。</p>
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

function RegisterAssetContent({ close, initialNode }: { close: () => void; initialNode?: string }) {
  const { nodes, refresh } = useData();
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
  const profileErrors =
    form.kind === 'DATASET' && !deferProfile ? datasetProfileErrors(datasetProfile) : [];
  const set = (key: keyof typeof form, value: string) =>
    setForm((previous) => ({ ...previous, [key]: value }));
  const node = nodes.data?.find((item) => item.id === form.node_id);
  const showDatasetConfig = form.kind === 'DATASET' && !deferProfile;
  return (
    <Modal
      title="登记本地资产"
      wide
      className={`asset-register-modal ${showDatasetConfig ? '' : 'asset-register-simple'}`}
      subtitle="建立资产元数据与裸金属本地路径的关联。"
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
            disabled={busy || profileErrors.length > 0}
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
          if (pending.current || profileErrors.length > 0) return;
          pending.current = true;
          setBusy(true);
          setError(undefined);
          try {
            await post('/assets', {
              ...form,
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
          <Field label="资产名称" required>
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
  const { can } = useAuth();
  const { runs, enabled } = useData();
  const [filter, setFilter] = useState('ALL');
  const [query, setQuery] = useState('');
  const filtered = useMemo(
    () =>
      (runs.data || [])
        .filter(
          (run) =>
            (filter === 'ALL' ||
              (filter === 'ACTIVE' ? activeStates.has(run.state) : run.state === filter)) &&
            run.name.toLowerCase().includes(query.toLowerCase()),
        )
        .sort((a, b) => b.created_at.localeCompare(a.created_at)),
    [runs.data, filter, query],
  );
  return (
    <>
      <PageHeading
        eyebrow="TRAINING RUNS"
        title="训练任务"
        description="追踪训练状态、资源消耗与实际执行代码。"
        actions={
          can('training.write') && (
            <button className="button primary" disabled={!enabled} onClick={createRun}>
              <Plus size={17} />
              新建训练
            </button>
          )
        }
      />
      <div className="toolbar wrap">
        <div className="segmented">
          {[
            ['ALL', '全部任务'],
            ['ACTIVE', '进行中'],
            ['SUCCEEDED', '已完成'],
            ['FAILED', '失败'],
          ].map(([key, label]) => (
            <button
              className={filter === key ? 'selected' : ''}
              key={key}
              onClick={() => setFilter(key)}
            >
              {label}
              {key === 'ALL' && <b>{runs.data?.length ?? '—'}</b>}
            </button>
          ))}
        </div>
        <div className="search-field">
          <Search size={16} />
          <input
            aria-label="搜索训练任务"
            placeholder="搜索训练名称"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
      </div>
      <ErrorNotice error={runs.error} retry={runs.refresh} />
      <div className="panel">
        {runs.loading && enabled && !runs.data ? (
          <Loading />
        ) : filtered.length ? (
          <RunTable runs={filtered} />
        ) : (
          <Empty
            icon={Workflow}
            title={query || filter !== 'ALL' ? '没有匹配的训练任务' : '开始你的第一次训练'}
            description={
              query || filter !== 'ALL'
                ? '尝试调整筛选条件，查看其他训练任务。'
                : '选择机器和训练模板，平台将检查本地资产、运行环境及代码版本。'
            }
            action={
              can('training.write') && (
                <button className="button outline" disabled={!enabled} onClick={createRun}>
                  <Plus size={16} />
                  新建训练
                </button>
              )
            }
          />
        )}
      </div>
    </>
  );
}
