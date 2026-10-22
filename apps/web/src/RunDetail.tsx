import { useMemo, useState } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowRight,
  Box,
  Clock3,
  Code2,
  Database,
  Download,
  ExternalLink,
  GitCommitHorizontal,
  LoaderCircle,
  Server,
  Square,
  Terminal,
  Workflow,
} from 'lucide-react';
import { date, displayNumber, post, useApi, useRunFeed } from './api';
import { useAuth } from './auth';
import {
  Back,
  Badge,
  CodeVersion,
  Empty,
  ErrorNotice,
  MetricChart,
  Modal,
  PageHeading,
  Panel,
} from './components';
import { activeStates, assetKinds, inputRoles, useData } from './data';
import { DatasetBindingSummary } from './DatasetBindings';
import { datasetInputLabel } from './datasetBindingHelpers';
import { stageParameterLabels, trainingStageLabels } from './recap';
import type { Asset, Logs, Run, RunEvent } from './types';

function outputRemoved(output: Asset) {
  return output.state === 'UNAVAILABLE' || output.metadata?.availability === 'removed';
}

function outputStep(output: Asset) {
  const step = output.metadata?.step;
  return typeof step === 'number' && Number.isFinite(step) ? step : -1;
}

export default function RunDetail({ id }: { id: string }) {
  const { can } = useAuth();
  const { nodes, templates, assets, profiles, refresh } = useData();
  const detail = useApi<Run>(`/runs/${id}`, 3000);
  const events = useRunFeed<RunEvent>(`/runs/${id}/events`, 'events');
  const logs = useRunFeed<Logs['items'][number]>(`/runs/${id}/logs`, 'logs');
  const [tab, setTab] = useState('overview');
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const run = detail.data;
  const attempt = run?.attempts?.[run.attempts.length - 1];
  const outputs = useMemo(
    () =>
      [...(run?.outputs || [])].sort((left, right) => {
        const availability = Number(outputRemoved(left)) - Number(outputRemoved(right));
        if (availability) return availability;
        const step = outputStep(right) - outputStep(left);
        if (step) return step;
        return (right.created_at || '').localeCompare(left.created_at || '');
      }),
    [run?.outputs],
  );
  const metricEvents = useMemo(
    () => events.data?.filter((event) => event.type === 'METRIC') || [],
    [events.data],
  );
  const metrics = attempt?.latest_metrics || metricEvents[metricEvents.length - 1]?.data;
  const curve = useMemo(
    () =>
      metricEvents
        .filter(
          (event) =>
            typeof event.data.loss === 'number' &&
            Number.isFinite(event.data.loss) &&
            typeof event.data.step === 'number',
        )
        .map((event) => ({ step: Number(event.data.step), loss: Number(event.data.loss) })),
    [metricEvents],
  );
  const node = nodes.data?.find((item) => item.id === run?.node_id);
  const template = templates.data?.find((item) => item.id === run?.template_id);
  const profile = profiles.data?.find((item) => item.id === run?.profile_id);
  const trainingStage =
    run?.training_stage ||
    attempt?.code_snapshot?.training_stage ||
    run?.code_expectation?.training_stage;
  const stageConfig =
    run?.stage_config ||
    attempt?.code_snapshot?.stage_config ||
    run?.code_expectation?.stage_config;
  const resource = attempt?.resources;
  const recipe = run?.recipe || attempt?.code_snapshot?.recipe || run?.code_expectation?.recipe;
  const requestedEntrypoint =
    run?.entrypoint ||
    attempt?.code_snapshot?.requested_entrypoint ||
    run?.code_expectation?.requested_entrypoint;
  const actualEntrypoint =
    attempt?.code_snapshot?.automatic_preflight?.plan.entrypoint ||
    run?.code_expectation?.automatic_preflight?.plan.entrypoint;
  const recipeDefinition =
    run?.recipe_definition ||
    attempt?.code_snapshot?.recipe_definition ||
    run?.code_expectation?.recipe_definition;
  const downloadLogs = () => {
    const url = URL.createObjectURL(
      new Blob([logs.data.map((item) => `${item.timestamp || ''} ${item.text}`).join('\n') || ''], {
        type: 'text/plain;charset=utf-8',
      }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${run?.name || id}.log`;
    anchor.click();
    URL.revokeObjectURL(url);
  };
  return (
    <>
      <Back to="/runs" label="返回训练任务" />
      <PageHeading
        eyebrow="RUN DETAIL"
        title={run?.name || '训练详情'}
        description={run ? `创建于 ${date(run.created_at)} · ${run.id}` : '正在读取训练信息'}
        actions={
          run && (
            <>
              <Badge state={run.state} />
              {can('training.write') && activeStates.has(run.state) && (
                <button
                  className="button outline danger-text"
                  disabled={run.state === 'CANCELLING'}
                  onClick={() => setConfirmCancel(true)}
                >
                  <Square size={14} />
                  取消训练
                </button>
              )}
            </>
          )
        }
      />
      <ErrorNotice error={detail.error || error} retry={detail.refresh} />
      {detail.loading && !run ? (
        <Loading />
      ) : (
        run && (
          <>
            {run.queue_reason && (
              <div className="info-strip">
                <Clock3 size={17} />
                <span>排队原因：{run.queue_reason}</span>
              </div>
            )}
            {run.state === 'UNKNOWN' && (
              <div className="inline-note amber">
                <AlertCircle size={18} />
                <span>机器失联，当前运行状态未知。GPU 预约继续保留，等待 Agent 重连确认。</span>
              </div>
            )}
            {(run.error || attempt?.error) && (
              <div className="inline-note red">
                <AlertCircle size={18} />
                <div>
                  <strong>{(run.error || attempt?.error)?.code}</strong>
                  <p>{(run.error || attempt?.error)?.message}</p>
                </div>
              </div>
            )}
            <div className="run-summary">
              <div>
                <span>
                  <Server size={14} />
                  计算机器
                </span>
                <a href={`#/nodes/${run.node_id}`}>
                  {node?.name || run.node_id}
                  <ExternalLink size={12} />
                </a>
              </div>
              <div>
                <span>
                  <Workflow size={14} />
                  训练模板
                </span>
                <strong>{template?.name || run.template_id}</strong>
              </div>
              <div>
                <span>
                  <Activity size={14} />
                  GPU 配额
                </span>
                <strong>{run.gpu_count ?? '—'} 张</strong>
              </div>
              <div>
                <span>
                  <GitCommitHorizontal size={14} />
                  实际 commit
                </span>
                <strong className="mono">
                  {attempt?.code_snapshot?.commit_sha?.slice(0, 10) || '等待启动'}
                </strong>
              </div>
            </div>
            <div className="tabs">
              {[
                { id: 'overview', title: '训练概览', icon: Activity },
                { id: 'logs', title: '运行日志', icon: Terminal },
                { id: 'code', title: '代码与配置', icon: Code2 },
                { id: 'assets', title: '输入与产出', icon: Database },
              ].map((item) => (
                <button
                  className={tab === item.id ? 'selected' : ''}
                  key={item.id}
                  onClick={() => setTab(item.id)}
                >
                  <item.icon size={16} />
                  {item.title}
                </button>
              ))}
            </div>
            {tab === 'overview' && (
              <div className="detail-columns">
                <div>
                  <Panel
                    title="训练指标"
                    subtitle="来自训练进程的实际上报"
                    action={
                      <span className="chart-legend">
                        <span />
                        loss
                      </span>
                    }
                  >
                    <ErrorNotice error={events.error} retry={events.refresh} />
                    <div className="training-metrics">
                      <div>
                        <span>当前步数</span>
                        <strong>
                          {displayNumber(typeof metrics?.step === 'number' ? metrics.step : null)}
                          <small>
                            {' '}
                            /{' '}
                            {displayNumber(
                              typeof metrics?.total_steps === 'number'
                                ? metrics.total_steps
                                : run.parameters?.steps,
                            )}
                          </small>
                        </strong>
                      </div>
                      <div>
                        <span>最新 loss</span>
                        <strong>
                          {typeof metrics?.loss === 'number' ? metrics.loss.toFixed(5) : '—'}
                        </strong>
                      </div>
                      <div>
                        <span>指标采样数</span>
                        <strong>{metricEvents.length}</strong>
                      </div>
                    </div>
                    <MetricChart values={curve} />
                  </Panel>
                  <Panel
                    title="最近日志"
                    action={
                      <button className="text-link" onClick={() => setTab('logs')}>
                        全部日志
                        <ArrowRight size={14} />
                      </button>
                    }
                  >
                    <ErrorNotice error={logs.error} retry={logs.refresh} />
                    <LogConsole items={logs.data.slice(-10) || []} compact />
                  </Panel>
                </div>
                <div>
                  <Panel title="执行信息">
                    <dl className="detail-list padded">
                      <div>
                        <dt>运行环境</dt>
                        <dd>{profile?.name || run.profile_id}</dd>
                      </div>
                      {trainingStage && (
                        <div>
                          <dt>训练阶段</dt>
                          <dd>{trainingStageLabels[trainingStage] || trainingStage}</dd>
                        </div>
                      )}
                      <div>
                        <dt>训练配方</dt>
                        <dd>
                          {recipeDefinition && (
                            <span className="recipe-frozen-label">
                              {recipeDefinition.name} · v{recipeDefinition.revision}
                            </span>
                          )}
                          {recipe ? <code>{recipe}</code> : '未记录'}
                        </dd>
                      </div>
                      <div>
                        <dt>执行次数</dt>
                        <dd>{run.attempts?.length || 0}</dd>
                      </div>
                      {(requestedEntrypoint || actualEntrypoint) && (
                        <div>
                          <dt>训练入口</dt>
                          <dd>
                            <code>{actualEntrypoint || requestedEntrypoint}</code>
                          </dd>
                        </div>
                      )}
                      <div>
                        <dt>训练进程</dt>
                        <dd>{attempt?.pid ? `PID ${attempt.pid}` : '—'}</dd>
                      </div>
                      <div>
                        <dt>开始时间</dt>
                        <dd>{date(attempt?.started_at)}</dd>
                      </div>
                      <div>
                        <dt>结束时间</dt>
                        <dd>{date(attempt?.finished_at)}</dd>
                      </div>
                      <div>
                        <dt>产出目录</dt>
                        <dd>
                          <code>{attempt?.output_dir || '等待进程上报'}</code>
                        </dd>
                      </div>
                    </dl>
                  </Panel>
                  <Panel title="进程资源" subtitle="训练进程与子进程的最近一次采样">
                    <dl className="detail-list padded">
                      <div>
                        <dt>进程内存</dt>
                        <dd>
                          {typeof resource?.rss_bytes === 'number'
                            ? `${(resource.rss_bytes / 1024 ** 2).toFixed(1)} MiB`
                            : '—'}
                        </dd>
                      </div>
                      <div>
                        <dt>累计 CPU 时间</dt>
                        <dd>
                          {displayNumber(
                            typeof resource?.cpu_seconds === 'number' ? resource.cpu_seconds : null,
                            ' 秒',
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>采样时间</dt>
                        <dd>
                          {typeof resource?.timestamp === 'string'
                            ? date(resource.timestamp)
                            : '等待采样'}
                        </dd>
                      </div>
                      <div>
                        <dt>分配 GPU</dt>
                        <dd>
                          {attempt?.gpu_uuids?.length
                            ? attempt.gpu_uuids.map((uuid) => (
                                <code className="block" key={uuid}>
                                  {uuid}
                                </code>
                              ))
                            : '未分配 GPU'}
                        </dd>
                      </div>
                    </dl>
                  </Panel>
                  <Panel title="运行事件">
                    <div className="event-timeline">
                      {events.data
                        ?.filter(
                          (event) =>
                            event.type !== 'LOG' &&
                            event.type !== 'METRIC' &&
                            event.type !== 'RESOURCE',
                        )
                        .slice(-8)
                        .reverse()
                        .map((event) => (
                          <div key={`${event.attempt_id}-${event.seq}`}>
                            <span
                              className={`timeline-dot ${event.type === 'FAILED' ? 'red' : ''}`}
                            />
                            <section>
                              <strong>
                                {(
                                  {
                                    STARTED: '训练进程已启动',
                                    SUCCEEDED: '训练已完成',
                                    FAILED: '训练执行失败',
                                    CANCELLED: '训练已取消',
                                  } as Record<string, string>
                                )[event.type] || event.type}
                              </strong>
                              <span>{date(event.timestamp)}</span>
                            </section>
                          </div>
                        ))}
                      <div>
                        <span className="timeline-dot gray" />
                        <section>
                          <strong>任务已创建</strong>
                          <span>{date(run.created_at)}</span>
                        </section>
                      </div>
                    </div>
                  </Panel>
                </div>
              </div>
            )}
            {tab === 'logs' && (
              <Panel
                title="运行日志"
                subtitle="按上报顺序展示 · 自动刷新 · 保留最近 5,000 条"
                action={
                  <button
                    className="button outline small"
                    disabled={!logs.data.length}
                    onClick={downloadLogs}
                  >
                    <Download size={14} />
                    导出当前日志
                  </button>
                }
              >
                <ErrorNotice error={logs.error} retry={logs.refresh} />
                <LogConsole items={logs.data || []} />
              </Panel>
            )}
            {tab === 'code' && (
              <div className="detail-columns">
                <div>
                  <Panel title="实际执行代码" subtitle="每次启动单独采集，包含未提交修改的标识">
                    {run.attempts?.length ? (
                      <div className="attempts-list">
                        {run.attempts.map((item, index) => (
                          <div key={item.id}>
                            <div className="attempt-heading">
                              <strong>执行 #{index + 1}</strong>
                              <Badge state={item.state || item.status || run.state} />
                            </div>
                            <CodeVersion snapshot={item.code_snapshot} />
                            {item.code_changed_during_run && (
                              <div className="inline-note amber">
                                运行期间代码发生变化；此快照记录启动版本，不能代表整个运行期间的全部源文件。
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <CodeVersion />
                    )}
                  </Panel>
                  <Panel title="预检代码基线" subtitle="启动时会与这份记录核对">
                    <CodeVersion snapshot={run.code_expectation} title="提交时绑定的代码版本" />
                  </Panel>
                </div>
                <Panel title="冻结训练配置">
                  <dl className="detail-list padded">
                    {requestedEntrypoint && (
                      <div>
                        <dt>所选训练入口</dt>
                        <dd>
                          <code>{requestedEntrypoint}</code>
                        </dd>
                      </div>
                    )}
                    {actualEntrypoint && (
                      <div>
                        <dt>实际训练入口</dt>
                        <dd>
                          <code>{actualEntrypoint}</code>
                        </dd>
                      </div>
                    )}
                    <div>
                      <dt>训练配方</dt>
                      <dd>
                        {recipeDefinition && (
                          <span className="recipe-frozen-label">
                            {recipeDefinition.name} · v{recipeDefinition.revision}
                          </span>
                        )}
                        {recipe ? <code>{recipe}</code> : '未记录'}
                      </dd>
                    </div>
                    {recipeDefinition && (
                      <>
                        <div>
                          <dt>配方摘要</dt>
                          <dd>
                            <code title={recipeDefinition.digest}>
                              {recipeDefinition.digest.slice(0, 16)}
                            </code>
                          </dd>
                        </div>
                        <div>
                          <dt>配方源文件</dt>
                          <dd>
                            <code>
                              {recipeDefinition.source.path}:{recipeDefinition.source.line}
                            </code>
                          </dd>
                        </div>
                        {Object.entries(recipeDefinition.overrides).map(([path, value]) => (
                          <div key={path}>
                            <dt>
                              <code>{path}</code>
                            </dt>
                            <dd>
                              <code>
                                {typeof value === 'string' ? value : JSON.stringify(value)}
                              </code>
                              <small className="recipe-value-caption">配方版本覆盖值</small>
                            </dd>
                          </div>
                        ))}
                      </>
                    )}
                    {Object.entries(run.parameters || {}).map(([key, value]) => (
                      <div key={key}>
                        <dt>
                          {(
                            {
                              steps: '训练步数',
                              batch_size: '批次大小',
                              learning_rate: '学习率',
                            } as Record<string, string>
                          )[key] || key}
                        </dt>
                        <dd>
                          <code>{typeof value === 'string' ? value : JSON.stringify(value)}</code>
                        </dd>
                      </div>
                    ))}
                    {trainingStage && (
                      <div>
                        <dt>训练阶段</dt>
                        <dd>{trainingStageLabels[trainingStage] || trainingStage}</dd>
                      </div>
                    )}
                    {Object.entries(stageConfig || {}).map(([key, value]) => (
                      <div key={`stage-${key}`}>
                        <dt>{stageParameterLabels[key] || key}</dt>
                        <dd>
                          <code>{JSON.stringify(value)}</code>
                        </dd>
                      </div>
                    ))}
                    <div>
                      <dt>GPU 数量</dt>
                      <dd>{run.gpu_count}</dd>
                    </div>
                    <div>
                      <dt>模板标识</dt>
                      <dd>
                        <code>{run.template_id}</code>
                      </dd>
                    </div>
                    <div>
                      <dt>环境标识</dt>
                      <dd>
                        <code>{run.profile_id}</code>
                      </dd>
                    </div>
                  </dl>
                </Panel>
              </div>
            )}
            {tab === 'assets' && (
              <>
                {!!run.dataset_bindings?.length && (
                  <Panel
                    title="数据集与归一化参数"
                    subtitle="本次训练冻结的配对、采样权重和数据处理配置"
                  >
                    <DatasetBindingSummary
                      bindings={run.dataset_bindings}
                      assets={assets.data || []}
                    />
                  </Panel>
                )}
                <div className="lineage-strip">
                  <span>
                    <Server size={18} />
                    {node?.name || run.node_id}
                  </span>
                  <ArrowRight size={17} />
                  <span>
                    <Database size={18} />
                    本机输入
                  </span>
                  <ArrowRight size={17} />
                  <span>
                    <Workflow size={18} />
                    {run.name}
                  </span>
                  <ArrowRight size={17} />
                  <span>
                    <Box size={18} />
                    本机产出
                  </span>
                </div>
                <div className="two-columns">
                  <Panel title="输入资产" subtitle="冻结本次训练选择的资产引用">
                    {Object.keys(run.inputs || {}).length ? (
                      <div className="run-asset-list">
                        {Object.entries(run.inputs || {}).map(([role, value]) => {
                          const assetId = typeof value === 'string' ? value : value.id;
                          const asset = assets.data?.find((item) => item.id === assetId);
                          return (
                            <div className="run-asset" key={role}>
                              <div className="row-icon">
                                {role === 'dataset' || role.startsWith('dataset_') ? (
                                  <Database size={17} />
                                ) : (
                                  <Box size={17} />
                                )}
                              </div>
                              <div>
                                <span>{inputRoles[role] || datasetInputLabel(role) || role}</span>
                                <strong>
                                  {asset?.name ||
                                    (typeof value === 'object' ? value.name || assetId : value)}
                                </strong>
                                <code>
                                  {asset?.path ||
                                    (typeof value === 'object'
                                      ? value.path
                                      : '路径信息以 Agent 记录为准')}
                                </code>
                                <small>{asset?.version}</small>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <Empty
                        icon={Database}
                        title="此任务没有输入资产"
                        description="smoke 模板使用合成数据，只验证平台控制链路。"
                      />
                    )}
                  </Panel>
                  <Panel
                    title="训练产出"
                    subtitle={
                      activeStates.has(run.state)
                        ? '自动更新已保存的检查点，产出保存在执行机器上'
                        : '产出保存在执行机器上并登记来源关系'
                    }
                  >
                    {outputs.length ? (
                      <div className="run-asset-list" aria-live="polite">
                        {outputs.map((output: Asset, index) => {
                          const step = output.metadata?.step;
                          const savedAt = output.metadata?.saved_at;
                          const removedAt = output.metadata?.removed_at;
                          const removed = outputRemoved(output);
                          return (
                            <div className="run-asset" key={output.id || index}>
                              <div className={`row-icon${removed ? '' : ' teal'}`}>
                                <Box size={17} />
                              </div>
                              <div>
                                <span>
                                  {assetKinds[output.kind] || output.kind}
                                  {typeof step === 'number' && Number.isFinite(step)
                                    ? ` · 第 ${step.toLocaleString('zh-CN')} 步`
                                    : ''}
                                </span>
                                <strong>{output.name}</strong>
                                {removed || output.state === 'READY' ? (
                                  <Badge
                                    state={removed ? 'UNAVAILABLE' : 'READY'}
                                    label={removed ? '已清理' : '可用'}
                                  />
                                ) : null}
                                <code>{output.path}</code>
                                <small>{output.version}</small>
                                {typeof savedAt === 'string' ? (
                                  <small>保存于 {date(savedAt)}</small>
                                ) : output.created_at ? (
                                  <small>登记于 {date(output.created_at)}</small>
                                ) : null}
                                {removed ? (
                                  <small>
                                    {typeof removedAt === 'string'
                                      ? `清理于 ${date(removedAt)}，仅保留产出记录`
                                      : '文件已清理，仅保留产出记录'}
                                  </small>
                                ) : null}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <Empty
                        icon={Box}
                        title={activeStates.has(run.state) ? '等待首个检查点' : '尚无训练产出'}
                        description={
                          run.state === 'RUNNING'
                            ? '训练正在进行，首个检查点保存完成后会自动出现在这里。'
                            : activeStates.has(run.state)
                              ? '训练保存并上报首个检查点后，会自动出现在这里。'
                              : '此任务尚未上报可用的训练产出。'
                        }
                      />
                    )}
                  </Panel>
                </div>
              </>
            )}
          </>
        )
      )}
      {can('training.write') && confirmCancel && (
        <Modal
          title="取消这项训练？"
          close={() => {
            if (!busy) setConfirmCancel(false);
          }}
          footer={
            <>
              <button
                className="button outline"
                disabled={busy}
                onClick={() => setConfirmCancel(false)}
              >
                继续训练
              </button>
              <button
                className="button danger"
                disabled={busy}
                onClick={async () => {
                  if (!can('training.write')) return;
                  setBusy(true);
                  setError(undefined);
                  try {
                    await post(`/runs/${id}/cancel`);
                    setConfirmCancel(false);
                    detail.refresh();
                    refresh();
                  } catch (err) {
                    setError(err as Error);
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                {busy && <LoaderCircle size={16} className="spin" />}确认取消
              </button>
            </>
          }
        >
          <p>
            平台将请求 Agent 终止「{run?.name}」。确认进程退出后，训练状态和 GPU 预约会随之更新。
          </p>
          <ErrorNotice error={error} />
        </Modal>
      )}
    </>
  );
}

function Loading() {
  return (
    <div className="loading">
      <LoaderCircle size={20} className="spin" />
      正在读取训练信息…
    </div>
  );
}
function LogConsole({ items, compact = false }: { items: Logs['items']; compact?: boolean }) {
  return (
    <div className={`log-console ${compact ? 'compact' : ''}`} tabIndex={0} aria-label="训练日志">
      {items.length ? (
        items.map((item, index) => (
          <div className="log-line" key={`${item.seq}-${index}`}>
            <span>
              {item.timestamp
                ? new Date(item.timestamp).toLocaleTimeString('zh-CN')
                : String(item.seq).padStart(4, '0')}
            </span>
            <code>{item.text}</code>
          </div>
        ))
      ) : (
        <div className="log-empty">
          <Terminal size={21} />
          <span>等待训练进程输出日志…</span>
        </div>
      )}
    </div>
  );
}
