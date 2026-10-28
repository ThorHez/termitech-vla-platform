import { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Copy,
  FlaskConical,
  LoaderCircle,
  Plus,
  Play,
  RefreshCw,
  Rocket,
  Server,
  Square,
} from 'lucide-react';
import { ApiError, date, navigate, post, useApi } from './api';
import { useAuth } from './auth';
import { Badge, Empty, ErrorNotice, Field, Loading, Modal, PageHeading, Panel } from './components';
import { useData } from './data';
import { PathAutocomplete } from './PathAutocomplete';
import { RecipeAutocomplete } from './RecipeAutocomplete';
import { recipeSuggestionScope } from './recipeSuggestions';
import { PolicyWarnings } from './PolicyWarnings';
import { checkpointSourceWarnings } from './policyWarningInfo';
import DataReturnMock from './DataReturnMock';
import { TransferCard, TransferGroups } from './CheckpointTransfers';
import {
  hasTransferCapability,
  checkpointSyncAllowed,
  normalizeTransferPath,
  transferForId,
  transferMatches,
  transferTerminal,
  transferLabels,
  type CheckpointTransfer,
} from './checkpointTransfer';
import {
  deployableOutputs,
  deploymentActive,
  deploymentClientExample,
  deploymentLabels,
  deploymentStartable,
  deploymentStartBlockedReason,
  deploymentDetailReceipt,
  sourceDeploymentBlocker,
  trainingInferenceConfig,
  inferenceBackendError,
  inferenceBackendLabels,
  resolvedInferenceBackendLabel,
  type InferenceBackend,
  type PolicyDeployment,
  type DeploymentRecordSummary,
  type DeploymentRecordDetail,
} from './policyDeployment';
import type { Node, Run } from './types';
import {
  buildDeploymentWorkspaceEntries,
  deploymentRecordLabels,
  deploymentRecordPath,
  parseDeploymentWorkspaceRoute,
  type DeploymentWorkspaceEntry,
} from './deploymentWorkspace';
import {
  inferenceGpuError,
  inferenceGpuMemory,
  inferenceMemoryMb,
  sharedInferenceSupported,
  type InferenceGpuMode,
} from './inferenceGpu';
import './policy-deployments.css';

export default function PolicyDeployments({
  initialRun,
  path = '/deployments',
  tab = 'overview',
}: {
  initialRun?: string;
  path?: string;
  tab?: string;
}) {
  const { can } = useAuth();
  const { nodes } = useData();
  const list = useApi<DeploymentRecordSummary[]>('/deployment-records', 8000);
  const [createdDeployment, setCreatedDeployment] = useState<PolicyDeployment>();
  const [form, setForm] = useState<{
    run?: string;
    transfer?: CheckpointTransfer;
    record?: DeploymentRecordSummary;
  }>();
  const openedRun = useRef('');
  const writable = can('training.write');
  const route = parseDeploymentWorkspaceRoute(path);
  useEffect(() => {
    if (initialRun && openedRun.current !== initialRun && writable) {
      openedRun.current = initialRun;
      setForm({ run: initialRun });
    }
  }, [initialRun, writable]);
  const entries = list.data ? buildDeploymentWorkspaceEntries(list.data) : [];
  const continueConfig = (record: DeploymentRecordDetail, transfer?: CheckpointTransfer) =>
    setForm({ record, transfer, run: record.source_run_id || transfer?.source_run_id });
  return (
    <>
      {route.kind === 'list' ? (
        <>
          <PageHeading
            eyebrow="ROBOT DEPLOYMENT"
            title="真机部署"
            description="查看与管理部署记录。"
            actions={
              writable && (
                <button className="button primary" onClick={() => setForm({})}>
                  <Plus size={16} />
                  新建部署
                </button>
              )
            }
          />
          <ErrorNotice error={list.error} title="部署记录读取失败" retry={list.refresh} />
          {!list.data ? (
            !list.error && <Loading label="正在读取部署记录…" />
          ) : (
            <Panel
              title="部署记录"
              subtitle={`${entries.length} 项部署记录`}
              className="policy-deployment-records"
            >
              {entries.length ? (
                <DeploymentWorkspaceList entries={entries} nodes={nodes.data || []} />
              ) : (
                <Empty
                  icon={Rocket}
                  title="开始第一次真机部署"
                  description="点击新建部署，选择权重和目标机器；同步历史保留在对应部署记录内。"
                />
              )}
            </Panel>
          )}
        </>
      ) : route.kind === 'deployment' ? (
        <DeploymentRecordPage
          key={route.id}
          id={route.id}
          tab={tab}
          initialDeployment={createdDeployment}
          changed={list.refresh}
          continueConfig={continueConfig}
        />
      ) : route.kind === 'preparation' ? (
        <LegacyPreparationLink key={route.id} id={route.id} />
      ) : (
        <div>
          <DeploymentListBack />
          <Panel title="记录不存在">
            <Empty
              icon={AlertCircle}
              title="无法识别此部署记录地址"
              description="请返回部署记录列表，选择已有记录。"
            />
          </Panel>
        </div>
      )}
      {form && writable && (
        <DeploymentForm
          initialRun={form.run}
          initialTransfer={form.transfer}
          initialRecord={form.record}
          synced={list.refresh}
          close={() => {
            setForm(undefined);
            list.refresh();
          }}
          created={(value) => {
            setCreatedDeployment(value);
            setForm(undefined);
            list.refresh();
            navigate(deploymentRecordPath(value.record_id || form.record?.id || value.id));
          }}
        />
      )}
    </>
  );
}

function DeploymentListBack() {
  return (
    <a className="button outline small policy-deployment-back" href="#/deployments">
      <ArrowLeft size={14} />
      返回部署记录
    </a>
  );
}

export function DeploymentWorkspaceList({
  entries,
  nodes,
}: {
  entries: DeploymentWorkspaceEntry[];
  nodes: Node[];
}) {
  return (
    <ul className="policy-deployment-record-list">
      {entries.map(({ key, record }) => {
        const machine =
          nodes.find((node) => node.id === record.node_id)?.name || record.node_id || '未选择机器';
        return (
          <li key={key}>
            <a
              className="policy-deployment-summary-row"
              data-record-kind="record"
              href={'#' + deploymentRecordPath(record.id)}
            >
              <strong>{record.name}</strong>
              <span className="policy-deployment-record-machine">{machine}</span>
              <Badge
                state={record.state === 'READY_TO_DEPLOY' ? 'READY' : record.state}
                label={deploymentRecordLabels[record.state] || record.state}
              />
              <small>{date(record.created_at)}</small>
              <ArrowRight size={14} />
            </a>
          </li>
        );
      })}
    </ul>
  );
}

function LegacyPreparationLink({ id }: { id: string }) {
  const detail = useApi<CheckpointTransfer>(`/checkpoint-transfers/${encodeURIComponent(id)}`, 0);
  const transfer = detail.data?.id === id ? detail.data : undefined;
  useEffect(() => {
    if (transfer?.record_id)
      window.location.replace('#' + deploymentRecordPath(transfer.record_id) + '?tab=sync');
  }, [transfer?.record_id]);
  return (
    <div>
      <DeploymentListBack />
      <ErrorNotice error={detail.error} title="同步记录读取失败" retry={detail.refresh} />
      {transfer && !transfer.record_id ? (
        <ErrorNotice
          error={new Error('此同步记录尚未关联父部署记录，请重试读取。')}
          title="部署归属不可用"
        />
      ) : (
        !detail.error && <Loading label="正在读取所属部署记录…" />
      )}
    </div>
  );
}

function DeploymentRecordPage({
  id,
  tab,
  changed,
  continueConfig,
  initialDeployment,
}: {
  id: string;
  tab: string;
  changed: () => void;
  initialDeployment?: PolicyDeployment;
  continueConfig: (record: DeploymentRecordDetail, transfer?: CheckpointTransfer) => void;
}) {
  const detail = useApi<DeploymentRecordDetail>(
    `/deployment-records/${encodeURIComponent(id)}`,
    3000,
  );
  const record = detail.data;
  useEffect(() => {
    if (record?.id && record.id !== id)
      window.location.replace(
        '#' + deploymentRecordPath(record.id) + '?tab=' + encodeURIComponent(tab),
      );
  }, [record?.id, id, tab]);
  if (!record) {
    if (
      !detail.error &&
      initialDeployment &&
      (initialDeployment.record_id === id || initialDeployment.id === id)
    )
      return (
        <DeploymentDetail
          id={initialDeployment.id}
          recordId={initialDeployment.record_id || id}
          initialItem={initialDeployment}
          tab={tab}
          changed={() => {
            detail.refresh();
            changed();
          }}
        />
      );
    return (
      <div>
        <DeploymentListBack />
        <ErrorNotice error={detail.error} title="部署记录读取失败" retry={detail.refresh} />
        {detail.error ? (
          <Empty
            icon={AlertCircle}
            title={
              detail.error instanceof ApiError && detail.error.status === 404
                ? '找不到此部署记录'
                : '暂时无法读取部署记录'
            }
            description="请检查记录地址或重试读取。"
          />
        ) : (
          <Loading label="正在读取部署记录…" />
        )}
      </div>
    );
  }
  return (
    <>
      <ErrorNotice error={detail.error} title="部署记录刷新失败" retry={detail.refresh} />
      <DeploymentRecordView
        record={record}
        tab={tab}
        changed={() => {
          detail.refresh();
          changed();
        }}
        continueConfig={continueConfig}
      />
    </>
  );
}

export function DeploymentRecordTransfers({
  transfers,
  selectedTransferId,
  changed,
}: {
  transfers: CheckpointTransfer[];
  selectedTransferId?: string | null;
  changed: () => void;
}) {
  const selected = selectedTransferId
    ? transfers.find((item) => item.id === selectedTransferId)
    : undefined;
  const others = transfers.filter((item) => item.id !== selectedTransferId);
  return (
    <Panel
      title="权重同步"
      subtitle="仅展示本部署记录明确关联的同步与历史；不同步 train_state 训练状态目录。"
    >
      {selectedTransferId && !selected && (
        <ErrorNotice
          error={new Error('父记录未返回当前服务采用的同步记录，请重试读取。')}
          title="关联同步不可用"
        />
      )}
      {selected && (
        <>
          <h3 className="policy-record-sync-title">当前服务采用的权重</h3>
          <TransferCard item={selected} changed={changed} />
        </>
      )}
      {!transfers.length && (
        <Empty
          icon={Server}
          title="暂无权重同步记录"
          description="继续配置部署后，可按需要同步权重或使用本机已有 checkpoint。"
        />
      )}
      {!!others.length && (
        <>
          {selected && <h3 className="policy-record-sync-title">其它同步记录</h3>}
          <TransferGroups items={others} changed={changed} />
        </>
      )}
    </Panel>
  );
}

export function DeploymentRecordView({
  record,
  tab = 'overview',
  changed,
  continueConfig,
}: {
  record: DeploymentRecordDetail;
  tab?: string;
  changed: () => void;
  continueConfig: (record: DeploymentRecordDetail, transfer?: CheckpointTransfer) => void;
}) {
  const { can } = useAuth();
  const { nodes, profiles } = useData();
  if (record.deployment_id || record.deployment) {
    if (!record.deployment || record.deployment.id !== record.deployment_id)
      return (
        <div>
          <DeploymentListBack />
          <ErrorNotice
            error={new Error('父记录的服务绑定不可用，请重试读取。')}
            title="服务记录不匹配"
            retry={changed}
          />
        </div>
      );
    return (
      <DeploymentDetail
        key={record.deployment_id}
        id={record.deployment_id}
        recordId={record.id}
        recordName={record.name}
        initialItem={record.deployment}
        recordTransfers={record.transfers}
        tab={tab}
        changed={changed}
      />
    );
  }
  const tabs = deploymentDetailTabs.filter((value) =>
    ['overview', 'sync', 'workflow', 'logs'].includes(value.id),
  );
  const normalizedTab = normalizeDeploymentDetailTab(tab);
  const activeTab = tabs.some((value) => value.id === normalizedTab) ? normalizedTab : 'overview';
  const node = nodes.data?.find((item) => item.id === record.node_id);
  const profile = profiles.data?.find(
    (item) => item.node_id === record.node_id && item.id === record.profile_id,
  );
  return (
    <div className="policy-deployment-detail">
      <DeploymentListBack />
      <PageHeading
        eyebrow="DEPLOYMENT RECORD"
        title={record.name}
        description={`创建于 ${date(record.created_at)} · 尚未启动服务`}
        actions={
          activeTab !== 'workflow' &&
          can('training.write') && (
            <button
              className="button primary"
              onClick={() => continueConfig(record, record.transfers[0])}
            >
              <Plus size={15} />
              继续配置部署
            </button>
          )
        }
      />
      <nav className="tabs policy-deployment-detail-tabs" role="tablist" aria-label="部署详情">
        {tabs.map((value) => (
          <a
            key={value.id}
            role="tab"
            aria-selected={activeTab === value.id}
            className={activeTab === value.id ? 'selected' : ''}
            href={'#' + deploymentRecordPath(record.id) + '?tab=' + value.id}
          >
            {value.label}
          </a>
        ))}
      </nav>
      {activeTab === 'overview' && (
        <Panel
          title="概览"
          action={
            <Badge
              state={record.state}
              label={deploymentRecordLabels[record.state] || record.state}
            />
          }
        >
          <div className="policy-deployment-body">
            <p className="field-hint">
              此部署记录尚未创建推理服务。同步记录和取消历史保留在本记录的“权重同步”中，可继续配置部署。
            </p>
            <dl className="policy-deployment-facts">
              <div>
                <dt>部署机器</dt>
                <dd>{node?.name || record.node_id || '未选择'}</dd>
              </div>
              <div>
                <dt>运行环境</dt>
                <dd>{profile?.name || record.profile_id || '未选择'}</dd>
              </div>
              <div>
                <dt>权重同步记录</dt>
                <dd>{record.transfers.length} 条</dd>
              </div>
              {record.source_run_id && (
                <div>
                  <dt>来源训练</dt>
                  <dd>
                    <a href={'#/runs/' + record.source_run_id}>查看来源训练</a>
                  </dd>
                </div>
              )}
            </dl>
          </div>
        </Panel>
      )}
      {activeTab === 'sync' && (
        <DeploymentRecordTransfers transfers={record.transfers} changed={changed} />
      )}
      {activeTab === 'workflow' && (
        <DataReturnMock
          key={record.id}
          deploymentRecordId={record.id}
          deploymentName={record.name}
          machineName={node?.name || record.node_id || undefined}
        />
      )}
      {activeTab === 'logs' && (
        <Panel title="准备日志" subtitle="本记录尚无服务日志，以下为明确关联的同步记录日志。">
          {!record.transfers.length ? (
            <Empty icon={Server} title="暂无准备日志" description="同步后可在此查看相关记录。" />
          ) : (
            record.transfers.map((item) => (
              <section className="policy-record-preparation-log" key={item.id}>
                <h3>
                  {item.name} · {transferLabels[item.state] || item.state}
                </h3>
                <small>{item.id}</small>
                <pre className="policy-deployment-console">{item.logs || '暂无同步日志'}</pre>
              </section>
            ))
          )}
        </Panel>
      )}
    </div>
  );
}

export const deploymentDetailTabs = [
  { id: 'overview', label: '概览' },
  { id: 'sync', label: '权重同步' },
  { id: 'test', label: '本地测试' },
  { id: 'workflow', label: '真机运行与回流' },
  { id: 'logs', label: '日志' },
  { id: 'client', label: '客户端' },
];

export function normalizeDeploymentDetailTab(tab: string) {
  return tab === 'data-return' || tab === 'mock' ? 'workflow' : tab;
}

export function DeploymentStartActions({
  item,
  node,
  writable,
  busy,
  onStart,
}: {
  item: PolicyDeployment;
  node?: Node;
  writable: boolean;
  busy: string;
  onStart: () => void;
}) {
  const showStart = writable && deploymentStartable(item.state);
  const reason = deploymentStartBlockedReason(item, node, writable, !!busy);
  const starting =
    busy === 'start' ||
    item.pending_command === 'POLICY_DEPLOY_START' ||
    item.state === 'PENDING' ||
    item.state === 'STARTING';
  return (
    <div className="policy-deployment-start-actions">
      <Badge state={item.state} label={deploymentLabels[item.state]} />
      {showStart && (
        <button className="button primary" type="button" disabled={!!reason} onClick={onStart}>
          {starting ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />}
          {starting ? '正在启动…' : item.state === 'FAILED' ? '重试启动' : '启动服务'}
        </button>
      )}
      {starting && (
        <span className="policy-deployment-start-status" role="status">
          {busy === 'start' ? '正在提交启动请求…' : '等待 Agent 启动服务…'}
        </span>
      )}
      {showStart && reason && <span className="policy-deployment-start-reason">{reason}</span>}
    </div>
  );
}

export function DeploymentDetail({
  id,
  changed,
  initialItem,
  recordId,
  recordName,
  recordTransfers,
  tab = 'overview',
}: {
  id: string;
  changed: () => void;
  initialItem?: PolicyDeployment;
  recordId?: string;
  recordName?: string;
  recordTransfers?: CheckpointTransfer[];
  tab?: string;
}) {
  const { can } = useAuth();
  const { nodes, profiles } = useData();
  const detail = useApi<PolicyDeployment>(`/policy-deployments/${encodeURIComponent(id)}`, 3000);
  const [error, setError] = useState<Error>();
  const [busy, setBusy] = useState('');
  const [observationPath, setObservationPath] = useState('');
  const [copied, setCopied] = useState(false);
  const [startReceipt, setStartReceipt] = useState<PolicyDeployment>();
  const running = useRef(false);
  const item = deploymentDetailReceipt(id, detail.data, startReceipt, initialItem);
  const normalizedTab = normalizeDeploymentDetailTab(tab);
  const activeTab = deploymentDetailTabs.some((value) => value.id === normalizedTab)
    ? normalizedTab
    : 'overview';
  const sync = useApi<CheckpointTransfer>(
    recordTransfers === undefined && activeTab === 'sync' && item?.transfer_id
      ? `/checkpoint-transfers/${encodeURIComponent(item.transfer_id)}`
      : null,
    3000,
  );
  const node = nodes.data?.find((value) => value.id === item?.node_id);
  const profile = profiles.data?.find(
    (value) => value.id === item?.profile_id && value.node_id === item?.node_id,
  );
  const writable = can('training.write');
  const request = async (operation: string, body: unknown = {}) => {
    if (running.current) return;
    if (
      operation === 'start' &&
      (!item || deploymentStartBlockedReason(item, node, writable, !!busy))
    )
      return;
    running.current = true;
    setBusy(operation);
    setError(undefined);
    try {
      const receipt = await post<PolicyDeployment>(
        `/policy-deployments/${encodeURIComponent(id)}/${operation}`,
        body,
      );
      if (operation === 'start') {
        if (receipt?.id !== id) throw new Error('启动响应与当前部署记录不匹配，请刷新状态后重试。');
        setStartReceipt(receipt);
      }
      detail.refresh();
      changed();
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      running.current = false;
      setBusy('');
    }
  };
  useEffect(() => {
    if (
      !item ||
      !writable ||
      activeTab === 'workflow' ||
      !deploymentActive(item.state) ||
      node?.status !== 'ONLINE' ||
      item.pending_command
    )
      return;
    const timer = setTimeout(() => {
      void request('refresh');
    }, 8000);
    return () => clearTimeout(timer);
  }, [item?.state, item?.pending_command, item?.updated_at, node?.status, writable, activeTab]);
  if (!item)
    return (
      <div>
        <DeploymentListBack />
        <ErrorNotice error={detail.error} title="部署记录读取失败" retry={detail.refresh} />
        {detail.error ? (
          <Empty
            icon={AlertCircle}
            title={
              detail.error instanceof ApiError && detail.error.status === 404
                ? '找不到此部署记录'
                : '暂时无法读取部署记录'
            }
            description="请检查记录地址或重试读取。"
          />
        ) : (
          <Loading label="正在读取部署记录…" />
        )}
      </div>
    );
  const active = deploymentActive(item.state);
  const pending = !!busy || !!item.pending_command;
  const example = deploymentClientExample(item.port);
  const boundTransfer =
    recordTransfers !== undefined
      ? recordTransfers.find((value) => value.id === item.transfer_id)
      : sync.data?.id === item.transfer_id
        ? sync.data
        : undefined;
  const detailWarnings =
    activeTab === 'sync' && boundTransfer && !sync.error
      ? item.warnings?.filter(
          (warning) => !boundTransfer.warnings?.some((value) => value.code === warning.code),
        )
      : item.warnings;
  return (
    <div className="policy-deployment-detail">
      <DeploymentListBack />
      <PageHeading
        eyebrow="DEPLOYMENT RECORD"
        title={recordName || item.name}
        description={
          recordId
            ? `部署记录 ${recordId} · 服务 ${item.id}`
            : `创建于 ${date(item.created_at)} · ${item.id}`
        }
        actions={
          activeTab === 'workflow' ? (
            <Badge state={item.state} label={deploymentLabels[item.state]} />
          ) : (
            <DeploymentStartActions
              item={item}
              node={node}
              writable={writable}
              busy={busy}
              onStart={() => void request('start')}
            />
          )
        }
      />
      <nav className="tabs policy-deployment-detail-tabs" aria-label="部署详情" role="tablist">
        {deploymentDetailTabs.map((value) => (
          <a
            key={value.id}
            href={'#' + deploymentRecordPath(recordId || item.record_id || id) + '?tab=' + value.id}
            role="tab"
            aria-selected={activeTab === value.id}
            className={activeTab === value.id ? 'selected' : ''}
          >
            {value.label}
          </a>
        ))}
      </nav>
      {activeTab !== 'overview' && activeTab !== 'workflow' && (
        <>
          <ErrorNotice
            error={detail.error || error}
            title="部署操作未完成"
            retry={detail.refresh}
          />
          {item.error && (
            <ErrorNotice
              error={new Error(`${item.error.code} · ${item.error.message}`)}
              title="部署未完成"
            />
          )}
          <PolicyWarnings warnings={detailWarnings} />
        </>
      )}
      {activeTab === 'overview' && (
        <Panel
          title="概览"
          subtitle={`创建于 ${date(item.created_at)} · ${item.id}`}
          action={<Badge state={item.state} label={deploymentLabels[item.state]} />}
        >
          <div className="policy-deployment-body">
            <ErrorNotice
              error={detail.error || error}
              title="部署操作未完成"
              retry={detail.refresh}
            />
            {item.error && (
              <div className="inline-note red" role="alert">
                <AlertCircle size={17} />
                <div>
                  <strong>{item.error.code}</strong>
                  <div>{item.error.message}</div>
                </div>
              </div>
            )}
            <PolicyWarnings warnings={item.warnings} />
            {node?.status !== 'ONLINE' && active && (
              <div className="inline-note amber">
                <AlertCircle size={16} />
                目标机器当前未在线，需等待 Agent 重连后确认服务状态。
              </div>
            )}
            <dl className="policy-deployment-facts">
              <div>
                <dt>部署机器</dt>
                <dd>
                  <a href={`#/nodes/${item.node_id}`}>{node?.name || item.node_id}</a>
                </dd>
              </div>
              <div>
                <dt>运行环境</dt>
                <dd>{profile?.name || item.profile_id}</dd>
              </div>
              <div>
                <dt>推理配置</dt>
                <dd>{item.config_name}</dd>
              </div>
              <div>
                <dt>推理框架</dt>
                <dd>{inferenceBackendLabels[item.inference_backend || 'auto']}</dd>
              </div>
              <div>
                <dt>实际推理框架</dt>
                <dd>{resolvedInferenceBackendLabel(item.resolved_backend)}</dd>
              </div>
              <div>
                <dt>GPU</dt>
                <dd className="mono">{item.gpu_uuid || '未指定'}</dd>
              </div>
              <div>
                <dt>GPU 使用方式</dt>
                <dd>{item.gpu_mode === 'shared' ? '与图形桌面共享' : '独占 GPU'}</dd>
              </div>
              {item.gpu_mode === 'shared' && (
                <div>
                  <dt>预计推理显存</dt>
                  <dd>{inferenceGpuMemory(item.gpu_memory_mb)} · 另留 1 GiB 余量</dd>
                </div>
              )}
              <div className="full">
                <dt>权重目录</dt>
                <dd className="mono">{item.checkpoint_path}</dd>
              </div>
              <div>
                <dt>机器本地地址</dt>
                <dd className="mono">{item.endpoint || `ws://127.0.0.1:${item.port}`}</dd>
              </div>
              <div>
                <dt>最近上报</dt>
                <dd>
                  {date(item.updated_at)}
                  {item.pid ? ` · PID ${item.pid}` : ''}
                </dd>
              </div>
              {item.source_run_id && (
                <div className="full">
                  <dt>来源训练</dt>
                  <dd>
                    <a href={`#/runs/${item.source_run_id}`}>
                      查看训练任务与配置 <ArrowRight size={12} />
                    </a>
                  </dd>
                </div>
              )}
              {item.transfer_id && (
                <div className="full">
                  <dt>权重同步来源</dt>
                  <dd className="mono">{item.transfer_id} · 完整性校验通过</dd>
                </div>
              )}
            </dl>
            <div className="policy-deployment-actions">
              {writable && active && (
                <button
                  className="button outline small"
                  disabled={pending || node?.status !== 'ONLINE'}
                  onClick={() => void request('refresh')}
                >
                  <RefreshCw size={14} className={busy === 'refresh' ? 'spin' : ''} />
                  检查状态
                </button>
              )}
              {writable && active && (
                <button
                  className="button outline small danger-text"
                  disabled={
                    busy === 'stop' ||
                    item.pending_command === 'POLICY_DEPLOY_STOP' ||
                    item.state === 'STOPPING' ||
                    node?.status !== 'ONLINE'
                  }
                  onClick={() => void request('stop')}
                >
                  <Square size={13} />
                  停止服务
                </button>
              )}
              {pending && (
                <span className="field-hint">
                  <LoaderCircle className="spin" size={13} />
                  等待 Agent 上报…
                </span>
              )}
            </div>
            {!!item.checks?.length && (
              <div className="policy-deployment-checks">
                {item.checks.map((check, index) => (
                  <div key={`${check.name}-${index}`} className={check.ok ? 'passed' : 'failed'}>
                    {check.ok ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                    <span>{check.message || check.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Panel>
      )}
      {activeTab === 'sync' &&
        (recordTransfers !== undefined ? (
          <DeploymentRecordTransfers
            transfers={recordTransfers}
            selectedTransferId={item.transfer_id}
            changed={changed}
          />
        ) : (
          <Panel title="权重同步" subtitle="仅展示此部署记录精确关联的同步记录。">
            {!item.transfer_id ? (
              <Empty
                icon={Server}
                title="本记录未关联跨机器同步"
                description="本机权重部署可直接使用 checkpoint；这里不会按训练来源匹配其他同步记录。"
              />
            ) : (
              <>
                <ErrorNotice error={sync.error} title="同步记录读取失败" retry={sync.refresh} />
                {boundTransfer ? (
                  <TransferCard item={boundTransfer} changed={sync.refresh} />
                ) : sync.error ? (
                  <Empty
                    icon={AlertCircle}
                    title="暂时无法读取关联同步记录"
                    description={`关联记录：${item.transfer_id}`}
                  />
                ) : sync.data ? (
                  <ErrorNotice
                    error={new Error('返回的同步记录与此部署绑定不一致。')}
                    title="同步记录不匹配"
                  />
                ) : (
                  <Loading label="正在读取关联同步记录…" />
                )}
              </>
            )}
          </Panel>
        ))}
      {activeTab === 'test' && (
        <Panel
          title="本地测试"
          subtitle="读取部署机器上的观测 JSON，执行一次策略推理并检查动作输出。"
        >
          <div className="policy-deployment-body">
            <div className="inline-note">
              <FlaskConical size={16} />
              测试结果用于验证模型输入输出。机器人执行动作需在本地控制程序中接入策略客户端。
            </div>
            {writable && (
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  void request('test', { observation_path: observationPath.trim() });
                }}
              >
                <PathAutocomplete
                  node={node}
                  value={observationPath}
                  onChange={setObservationPath}
                  label="观测样本 JSON 路径"
                  placeholder="/data/test/observation.json"
                  disabled={pending || item.state !== 'RUNNING'}
                  hint="字段、相机图像数组和状态维度需与训练处理器一致；文件须位于机器允许访问的目录内。"
                />
                <button
                  className="button primary small"
                  type="submit"
                  disabled={
                    pending ||
                    item.state !== 'RUNNING' ||
                    node?.status !== 'ONLINE' ||
                    !observationPath.trim().startsWith('/') ||
                    item.test_result?.status === 'RUNNING'
                  }
                >
                  <FlaskConical size={14} />
                  运行单次推理测试
                </button>
              </form>
            )}
            {item.test_result && (
              <div className="policy-deployment-test-result" aria-live="polite">
                <strong>最近一次测试</strong>
                <pre>{JSON.stringify(item.test_result, null, 2)}</pre>
              </div>
            )}
          </div>
        </Panel>
      )}
      {activeTab === 'logs' && (
        <Panel title="服务日志">
          <pre className="policy-deployment-console" aria-label="部署服务日志">
            {item.logs || '等待 Agent 返回服务日志…'}
          </pre>
        </Panel>
      )}
      {activeTab === 'client' && (
        <Panel
          title="本地客户端"
          action={
            <button
              className="button ghost small"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(example);
                  setCopied(true);
                } catch {
                  setError(new Error('复制失败，请手动选择客户端示例。'));
                }
              }}
            >
              <Copy size={13} />
              {copied ? '已复制' : '复制示例'}
            </button>
          }
        >
          <div className="policy-deployment-body">
            <p className="field-hint">
              在部署机器的 OpenPI 环境中使用 openpi_client。其他电脑通过 SSH 端口转发连接此机器的
              127.0.0.1:{item.port}。
            </p>
            <pre className="policy-deployment-console">{example}</pre>
          </div>
        </Panel>
      )}
      {activeTab === 'workflow' && (
        <DataReturnMock
          key={recordId || item.record_id || item.id}
          deploymentRecordId={recordId || item.record_id || item.id}
          deploymentName={recordName || item.name}
          machineName={node?.name || item.node_id}
          runtimeId={item.id}
        />
      )}
    </div>
  );
}

export function DeploymentBackendField({
  value,
  node,
  onChange,
}: {
  value: InferenceBackend;
  node?: Node;
  onChange: (backend: InferenceBackend) => void;
}) {
  const error = inferenceBackendError(value, node);
  return (
    <>
      <Field
        label="推理框架"
        required
        hint="根据权重格式选择：params/ 使用 JAX，model.safetensors 使用 PyTorch。切换框架不会转换权重。"
      >
        <select
          value={value}
          onChange={(event) => onChange(event.target.value as InferenceBackend)}
        >
          <option value="auto">自动识别</option>
          <option value="jax">JAX</option>
          <option value="pytorch">PyTorch</option>
        </select>
      </Field>
      {error && (
        <div className="inline-note red" role="alert">
          {error}
        </div>
      )}
    </>
  );
}

function DeploymentForm({
  initialRun,
  initialTransfer,
  initialRecord,
  synced,
  close,
  created,
}: {
  initialRun?: string;
  initialTransfer?: CheckpointTransfer;
  initialRecord?: DeploymentRecordSummary;
  synced: () => void;
  close: () => void;
  created: (item: PolicyDeployment) => void;
}) {
  const { nodes, profiles, runs } = useData();
  const [source, setSource] = useState(
    initialRun || initialRecord?.source_run_id || !initialRecord ? 'run' : 'path',
  );
  const [runId, setRunId] = useState(initialRun || initialRecord?.source_run_id || '');
  const [recordId, setRecordId] = useState(initialRecord?.id || initialTransfer?.record_id || '');
  const runDetail = useApi<Run>(
    source === 'run' && runId ? `/runs/${encodeURIComponent(runId)}` : null,
    0,
  );
  const sourceRun = runDetail.data?.id === runId ? runDetail.data : undefined;
  const [name, setName] = useState(initialRecord?.name || '');
  const [nodeId, setNodeId] = useState(initialRecord?.node_id || '');
  const [profileId, setProfileId] = useState(initialRecord?.profile_id || '');
  const [checkpoint, setCheckpoint] = useState('');
  const [directory, setDirectory] = useState(initialTransfer?.target_directory || '');
  const [transferId, setTransferId] = useState(initialTransfer?.id || '');
  const [createdTransfer, setCreatedTransfer] = useState<CheckpointTransfer>();
  const [syncCreating, setSyncCreating] = useState(false);
  const [syncError, setSyncError] = useState<Error>();
  const transferQuery = useApi<CheckpointTransfer>(
    transferId ? `/checkpoint-transfers/${encodeURIComponent(transferId)}` : null,
    3000,
  );
  const transfer = transferForId(transferId, transferQuery.data, createdTransfer, initialTransfer);
  useEffect(() => {
    if (!recordId && transfer?.record_id) setRecordId(transfer.record_id);
  }, [recordId, transfer?.record_id]);
  const [configName, setConfigName] = useState('');
  const [resolvedConfig, setResolvedConfig] = useState<{ scope: string; name: string }>();
  const [inferenceBackend, setInferenceBackend] = useState<InferenceBackend>('auto');
  const [port, setPort] = useState('8000');
  const [gpu, setGpu] = useState('');
  const [gpuMode, setGpuMode] = useState<InferenceGpuMode>('exclusive');
  const [gpuMemoryGiB, setGpuMemoryGiB] = useState('16');
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const initialized = useRef('');
  const submitting = useRef(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (source !== 'run' || !sourceRun || initialized.current === sourceRun.id) return;
    initialized.current = sourceRun.id;
    setName(initialRecord?.name || `${sourceRun.name} · 本地测试`);
    setNodeId(initialTransfer?.target_node_id || initialRecord?.node_id || sourceRun.node_id);
    setProfileId(
      initialTransfer?.target_profile_id || initialRecord?.profile_id || sourceRun.profile_id,
    );
    setCheckpoint(
      initialTransfer?.source_checkpoint_path || deployableOutputs(sourceRun)[0]?.path || '',
    );
    setConfigName('');
    setResolvedConfig(undefined);
    setGpu('');
  }, [sourceRun, source, initialTransfer, initialRecord]);
  const node = nodes.data?.find((item) => item.id === nodeId);
  const sourceNode = nodes.data?.find((item) => item.id === sourceRun?.node_id);
  const crossMachine = source === 'run' && !!sourceRun && sourceRun.node_id !== nodeId;
  const nodeProfiles =
    profiles.data?.filter(
      (item) =>
        item.node_id === nodeId &&
        item.adapter === 'openpi-sft' &&
        (!item.execution_mode || item.execution_mode === 'native'),
    ) || [];
  const profile = nodeProfiles.find((item) => item.id === profileId);
  const configScope = recipeSuggestionScope(node, profile, 'inference-config-suggestions');
  useEffect(() => {
    setConfigName('');
    setResolvedConfig(undefined);
  }, [source, configScope]);
  const outputs = deployableOutputs(sourceRun);
  const blocker = source === 'run' ? sourceDeploymentBlocker(sourceRun) : undefined;
  const trainingConfig = source === 'run' ? trainingInferenceConfig(sourceRun) : undefined;
  const matchingTransfer = transferMatches(transfer, {
    runId,
    checkpoint,
    nodeId,
    profileId,
    directory,
  });
  const sourceWarnings = source === 'run' ? checkpointSourceWarnings(sourceRun, checkpoint) : [];
  const previewWarnings =
    matchingTransfer && transfer?.warnings !== undefined ? [] : sourceWarnings;
  const verifiedTransfer =
    crossMachine && matchingTransfer && transfer?.state === 'SUCCEEDED' ? transfer : undefined;
  const sourceExportReady = hasTransferCapability(sourceNode, 'POLICY_EXPORT');
  const targetPortableReady = hasTransferCapability(node, 'POLICY_DEPLOY_PORTABLE_V1');
  const syncing =
    syncCreating ||
    (!!transferId && (!transfer || (matchingTransfer && !transferTerminal(transfer.state))));
  const syncAllowed = checkpointSyncAllowed({
    sourceRun,
    sourceNode,
    targetNode: node,
    profileSelected: !!profile,
    directory,
    checkpoint,
    blocked: blocker,
    syncing,
    busy,
  });
  const gpuMemoryMb = gpuMode === 'shared' ? inferenceMemoryMb(gpuMemoryGiB) : null;
  const selectedGpu = node?.gpus?.find((item) => item.uuid === gpu);
  const gpuBlocker = inferenceGpuError(selectedGpu, gpuMode, gpuMemoryMb, node);
  const backendBlocker = inferenceBackendError(inferenceBackend, node);
  const valid =
    !!name.trim() &&
    node?.status === 'ONLINE' &&
    !!profile &&
    checkpoint.trim().startsWith('/') &&
    (source === 'run'
      ? !!trainingConfig
      : resolvedConfig?.scope === configScope && resolvedConfig.name === configName) &&
    Number.isInteger(Number(port)) &&
    Number(port) >= 1024 &&
    Number(port) <= 65535 &&
    !!gpu &&
    !gpuBlocker &&
    !backendBlocker &&
    !blocker &&
    (source !== 'run' || !!sourceRun) &&
    (!crossMachine || (!!verifiedTransfer?.target_path && targetPortableReady));
  const clearTransfer = () => {
    setTransferId('');
    setCreatedTransfer(undefined);
    setSyncError(undefined);
  };
  const changeCheckpoint = (value: string) => {
    setCheckpoint(value);
    if (normalizeTransferPath(value) !== normalizeTransferPath(checkpoint)) clearTransfer();
  };
  const changeNode = (value: string) => {
    setNodeId(value);
    const options =
      profiles.data?.filter(
        (item) =>
          item.node_id === value &&
          item.adapter === 'openpi-sft' &&
          (!item.execution_mode || item.execution_mode === 'native'),
      ) || [];
    setProfileId(
      source === 'run' && sourceRun?.node_id === value
        ? sourceRun.profile_id
        : options.length === 1
          ? options[0].id
          : '',
    );
    setGpu('');
    setResolvedConfig(undefined);
    setDirectory(
      (nodes.data?.find((item) => item.id === value)?.output_root || '').replace(/\/+$/, '') +
        '/deployed-models',
    );
    clearTransfer();
    if (source !== 'run') {
      setCheckpoint('');
      setConfigName('');
    }
  };
  const synchronize = async () => {
    if (!syncAllowed || submitting.current) return;
    submitting.current = true;
    setBusy(true);
    setSyncCreating(true);
    setSyncError(undefined);
    setError(undefined);
    try {
      const item = await post<CheckpointTransfer>('/checkpoint-transfers', {
        ...(recordId ? { record_id: recordId } : {}),
        record_name: name.trim() || sourceRun?.name || '部署记录',
        name: `${name.trim() || sourceRun?.name || '模型'} · 权重同步`,
        source_run_id: runId,
        source_checkpoint_path: normalizeTransferPath(checkpoint),
        target_node_id: nodeId,
        target_profile_id: profileId,
        target_directory: normalizeTransferPath(directory),
      });
      if (mounted.current) {
        if (item.record_id) setRecordId(item.record_id);
        setCreatedTransfer(item);
        setTransferId(item.id);
        synced();
      }
    } catch (err) {
      if (mounted.current) setSyncError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      submitting.current = false;
      if (mounted.current) {
        setBusy(false);
        setSyncCreating(false);
      }
    }
  };
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!valid || submitting.current) return;
    submitting.current = true;
    setBusy(true);
    setError(undefined);
    try {
      const result = await post<PolicyDeployment>('/policy-deployments', {
        ...(recordId ? { record_id: recordId } : {}),
        name: name.trim(),
        node_id: nodeId,
        profile_id: profileId,
        checkpoint_path: verifiedTransfer?.target_path || checkpoint.trim(),
        inference_backend: inferenceBackend,
        ...(source === 'run' ? {} : { config_name: configName }),
        port: Number(port),
        gpu_uuid: gpu,
        gpu_mode: gpuMode,
        gpu_memory_mb: gpuMemoryMb,
        default_prompt: prompt.trim() || null,
        ...(source === 'run' ? { source_run_id: runId } : {}),
        ...(verifiedTransfer ? { transfer_id: verifiedTransfer.id } : {}),
      });
      if (mounted.current) created(result);
    } catch (err) {
      if (mounted.current) setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      submitting.current = false;
      if (mounted.current) setBusy(false);
    }
  };
  return (
    <Modal
      title="新建真机部署"
      subtitle="选择模型与运行环境，在目标机器上启动本地策略服务。"
      wide
      close={() => {
        if (!busy) close();
      }}
      footer={
        <>
          <button className="button outline" disabled={busy} onClick={close}>
            取消
          </button>
          <button
            className="button primary"
            type="submit"
            form="policy-deployment-form"
            disabled={!valid || busy}
          >
            {busy && !syncCreating ? (
              <LoaderCircle className="spin" size={15} />
            ) : (
              <Rocket size={15} />
            )}
            部署并启动服务
          </button>
        </>
      }
    >
      <form id="policy-deployment-form" className="policy-deployment-form" onSubmit={submit}>
        <ErrorNotice error={error} title="无法创建部署" />
        <fieldset disabled={busy}>
          <div className="policy-deployment-stage">
            <h3>1. 模型与远程导入</h3>
            <p>模型导入和权重同步不要求空闲 GPU，完成后再配置推理资源。</p>
          </div>
          <Field label="部署名称" required>
            <input
              value={name}
              maxLength={100}
              onChange={(event) => setName(event.target.value)}
              placeholder="例如：叠衣策略 · 本地测试"
              required
            />
          </Field>
          <Field label="模型来源">
            <select
              value={source}
              onChange={(event) => {
                setSource(event.target.value);
                initialized.current = '';
                setRunId('');
                changeNode('');
                setCheckpoint('');
                setConfigName('');
                setResolvedConfig(undefined);
              }}
            >
              <option value="run">训练任务产出</option>
              <option value="path">机器上的本地权重</option>
            </select>
          </Field>
          {source === 'run' && (
            <Field
              label="来源训练任务"
              required
              hint="选择已完成训练的策略权重；部署到其他机器时，会同步推理配置与归一化参数。"
            >
              <select
                value={runId}
                onChange={(event) => {
                  setRunId(event.target.value);
                  initialized.current = '';
                  setCheckpoint('');
                  clearTransfer();
                  setConfigName('');
                  setResolvedConfig(undefined);
                }}
                required
              >
                <option value="">选择 OpenPI 训练任务</option>
                {runs.data
                  ?.filter((item) => item.template_id === 'openpi-sft')
                  .map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
              </select>
            </Field>
          )}
          {source === 'run' && runId && !sourceRun && !runDetail.error && (
            <Loading label="正在读取训练产物…" />
          )}
          {source === 'run' && <ErrorNotice error={runDetail.error} retry={runDetail.refresh} />}
          {blocker && (
            <div className="inline-note red" role="alert">
              <AlertCircle size={16} />
              {blocker}
            </div>
          )}
          <div className="form-grid">
            <Field label="部署机器" required hint="复用机器资源中已在线的机器。">
              <select value={nodeId} required onChange={(event) => changeNode(event.target.value)}>
                <option value="">选择机器</option>
                {nodes.data?.map((item) => (
                  <option key={item.id} value={item.id} disabled={item.status !== 'ONLINE'}>
                    {item.name}
                    {item.status !== 'ONLINE' ? ' · 离线' : ''}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="OpenPI 运行环境" required>
              <select
                value={profileId}
                required
                disabled={source === 'run' && !crossMachine}
                onChange={(event) => {
                  setProfileId(event.target.value);
                  clearTransfer();
                  setConfigName('');
                  setResolvedConfig(undefined);
                }}
              >
                <option value="">选择运行环境</option>
                {nodeProfiles.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          {nodeId && !nodeProfiles.length && (
            <div className="inline-note red">
              此机器没有原生 OpenPI 环境，请先在机器资源中配置运行环境。
            </div>
          )}
          {source === 'run' && !!outputs.length && (
            <Field label="已保存的 checkpoint">
              <select
                value={outputs.some((item) => item.path === checkpoint) ? checkpoint : ''}
                onChange={(event) => {
                  if (event.target.value) changeCheckpoint(event.target.value);
                }}
              >
                <option value="">手动填写目录</option>
                {outputs.map((item) => (
                  <option key={item.id} value={item.path}>
                    {item.name} · {item.path}
                  </option>
                ))}
              </select>
            </Field>
          )}
          <PathAutocomplete
            node={source === 'run' ? sourceNode : node}
            value={checkpoint}
            onChange={changeCheckpoint}
            label={crossMachine ? '源机器 Checkpoint 目录' : 'Checkpoint 目录'}
            required
            placeholder="/data/outputs/checkpoints/config/experiment/10000"
            hint={
              crossMachine
                ? '同步模型参数、assets 与推理配置，不同步 train_state 训练状态目录，也不复制训练数据。'
                : '选择包含模型权重与 assets 归一化参数的目录。'
            }
          />
          <PolicyWarnings warnings={previewWarnings} />
          {crossMachine && (
            <div className="checkpoint-sync-fields">
              <h3>跨机器权重同步</h3>
              <PathAutocomplete
                node={node}
                value={directory}
                onChange={(value) => {
                  setDirectory(value);
                  if (normalizeTransferPath(value) !== normalizeTransferPath(directory))
                    clearTransfer();
                }}
                label="目标保存目录"
                placeholder="/data/deployed-models"
                hint="同步模型参数、assets 与推理配置，不同步 train_state 训练状态目录。每次保存到独立子目录；失败保留部分文件以便重试。"
              />
              {!verifiedTransfer && sourceNode?.status !== 'ONLINE' && (
                <div className="inline-note red">
                  来源机器当前离线，需恢复 Agent 连接后准备推理配置。
                </div>
              )}
              {!verifiedTransfer && sourceNode?.status === 'ONLINE' && !sourceExportReady && (
                <div className="inline-note red">
                  来源机器 Agent 尚不支持推理配置导出，请先在机器资源中更新 Agent。
                </div>
              )}
              {node?.status === 'ONLINE' && !targetPortableReady && (
                <div className="inline-note red">
                  目标机器 Agent 尚不支持跨机器推理配置校验，请先在机器资源中更新 Agent。
                </div>
              )}
              <div className="checkpoint-sync-start">
                <button
                  type="button"
                  className="button outline small"
                  disabled={!syncAllowed || !!verifiedTransfer}
                  onClick={() => void synchronize()}
                >
                  {syncing ? <LoaderCircle className="spin" size={14} /> : <RefreshCw size={14} />}
                  {syncCreating
                    ? '正在创建同步任务…'
                    : verifiedTransfer
                      ? '权重已同步并校验'
                      : syncing
                        ? '同步进行中'
                        : '开始 rsync 同步'}
                </button>
                {syncCreating && <span role="status">正在提交并检查同步条件…</span>}
              </div>
              <ErrorNotice error={syncError} title="无法开始同步" />
              <p>
                复用机器资源中的 SSH
                连接。同步完成后再点击“部署并启动服务”；关闭窗口后仍可在部署页查看进度。
              </p>
              <ErrorNotice
                error={transferId && !transferQuery.loading ? transferQuery.error : undefined}
                title="同步状态读取失败"
                retry={transferQuery.refresh}
              />
              {matchingTransfer && transfer && (
                <TransferCard
                  compact
                  item={transfer}
                  changed={() => {
                    transferQuery.refresh();
                    synced();
                  }}
                />
              )}
              {transfer && !matchingTransfer && (
                <div className="inline-note red" role="alert">
                  同步任务与当前模型或目标设置不匹配，不能用于当前部署。请在“权重同步”中查看该任务，并核对当前选择。
                </div>
              )}
              {transferId && !transfer && <Loading label="正在读取同步任务…" />}
            </div>
          )}
          <div className="policy-deployment-stage">
            <h3>2. 启动推理服务</h3>
            <p>以下 GPU 检查只在启动服务时使用，不影响上方模型同步。</p>
          </div>
          <DeploymentBackendField
            value={inferenceBackend}
            node={node}
            onChange={setInferenceBackend}
          />
          {source === 'run' ? (
            <div className="deployment-config-summary" aria-label="根据训练自动生成推理配置">
              <h3>
                <CheckCircle2 size={16} />
                根据训练自动生成推理配置
              </h3>
              <dl className="policy-deployment-facts">
                <div>
                  <dt>来源训练</dt>
                  <dd>{sourceRun?.name || '请先选择训练任务'}</dd>
                </div>
                <div>
                  <dt>来源配置</dt>
                  <dd>
                    <code>{trainingConfig?.recipe || '尚未确认生成配置'}</code>
                  </dd>
                </div>
                {trainingConfig && (
                  <>
                    <div className="full">
                      <dt>训练配置 SHA-256</dt>
                      <dd>
                        <code>{trainingConfig.sha256}</code>
                      </dd>
                    </div>
                    {trainingConfig.autoConfig && (
                      <div className="full">
                        <dt>冻结的模型参数</dt>
                        <dd>
                          {trainingConfig.autoConfig.model_type.toUpperCase()} · 动作维度{' '}
                          {trainingConfig.autoConfig.action_dim} · Action horizon{' '}
                          {trainingConfig.autoConfig.action_horizon}
                        </dd>
                      </div>
                    )}
                  </>
                )}
                <div className="full">
                  <dt>生成与绑定状态</dt>
                  <dd>
                    {!trainingConfig
                      ? '等待已确认的训练生成配置'
                      : crossMachine
                        ? verifiedTransfer
                          ? '推理配置已随权重同步并校验，部署时直接使用同步产物'
                          : '同步时从已确认的训练配置生成，并与权重一起校验'
                        : '启动时从已确认的训练配置生成，并绑定所选 checkpoint'}
                  </dd>
                </div>
              </dl>
              <p className="field-hint">
                {crossMachine
                  ? '目标环境无需预先登记同名配置；推理配置使用所选权重的归一化参数。'
                  : '自动沿用训练时确认的模型结构、数据处理与动作配置，使用所选权重的归一化参数。'}
              </p>
            </div>
          ) : (
            <RecipeAutocomplete
              key={configScope}
              value={configName}
              onChange={(value) => {
                setConfigName(value);
                setResolvedConfig(undefined);
              }}
              node={node}
              profile={profile}
              namespace="inference-config-suggestions"
              context="inference"
              codeOnly
              disabled={busy}
              onResolvedName={(resolved) =>
                setResolvedConfig(resolved ? { scope: configScope, name: resolved } : undefined)
              }
            />
          )}
          <div className="form-grid">
            <Field label="GPU 使用方式" hint="训练继续独占 GPU；推理可选择与图形桌面共存。">
              <select
                value={gpuMode}
                onChange={(event) => setGpuMode(event.target.value as InferenceGpuMode)}
              >
                <option value="exclusive">独占 GPU</option>
                <option value="shared">与图形桌面共享</option>
              </select>
            </Field>
            {gpuMode === 'shared' && (
              <Field
                label="预计推理显存（GiB）"
                required
                hint="另留 1 GiB 余量。此预算用于启动准入，实际峰值以模型加载为准，并非硬隔离上限。"
              >
                <input
                  type="number"
                  required
                  min={0.25}
                  max={1024}
                  step="any"
                  value={gpuMemoryGiB}
                  onChange={(event) => setGpuMemoryGiB(event.target.value)}
                />
              </Field>
            )}
          </div>
          {gpuMode === 'shared' && node && !sharedInferenceSupported(node) && (
            <div className="inline-note red" role="alert">
              当前 Agent 尚不支持共享 GPU，请先更新 Agent。模型同步可继续进行。
            </div>
          )}
          <div className="form-grid">
            <Field label="推理 GPU" required>
              <select value={gpu} required onChange={(event) => setGpu(event.target.value)}>
                <option value="">选择推理 GPU</option>
                {node?.gpus?.map((item, index) => {
                  const reason = inferenceGpuError(item, gpuMode, gpuMemoryMb, node);
                  return (
                    <option key={item.uuid} value={item.uuid} disabled={!!reason}>
                      GPU {index} · {item.name} · 剩余 {inferenceGpuMemory(item.memory_free_mb)}
                      {reason ? ` · ${reason}` : ''}
                    </option>
                  );
                })}
              </select>
              {gpuBlocker && <p className="field-hint">{gpuBlocker}</p>}
              {selectedGpu && gpuMode === 'shared' && !gpuBlocker && (
                <p className="field-hint">
                  剩余显存 {inferenceGpuMemory(selectedGpu.memory_free_mb)}，预计使用{' '}
                  {inferenceGpuMemory(gpuMemoryMb)}，另留 1 GiB 余量；启动前会再次检查。
                </p>
              )}
            </Field>
            <Field label="服务端口" required hint="监听部署机器的 127.0.0.1。">
              <input
                type="number"
                required
                min={1024}
                max={65535}
                step={1}
                value={port}
                onChange={(event) => setPort(event.target.value)}
              />
            </Field>
          </div>
          <Field label="默认任务指令" hint="观测样本未提供 prompt 时使用。">
            <input
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              maxLength={4096}
              placeholder="例如：将衣服折叠整齐"
            />
          </Field>
          <div className="inline-note">
            <Server size={16} />
            服务由目标机器 Agent
            启动并上报健康状态。部署完成后，可在详情中提交观测样本进行单次推理测试。
          </div>
        </fieldset>
      </form>
    </Modal>
  );
}
