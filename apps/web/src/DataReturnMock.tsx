import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import {
  Activity,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Database,
  FileCode2,
  Film,
  Info,
  Hand,
  Layers3,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Search,
  ShieldCheck,
  Square,
  Video,
  Server,
  Terminal,
  Tags,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import {
  dataReturnStats,
  type DataReturnMode,
  type DataReturnEpisodeOutcome,
  type DataReturnSession,
  type DataReturnSessionStatus,
  type DataReturnVersion,
} from './dataReturnSimulation';
import {
  createRobotDataWorkflow,
  robotDataWorkflowReducer,
  workflowCaptureEntry,
  workflowStartReason,
  workflowResumeReason,
  workflowNextEpisodeReason,
  workflowStatus,
  activeWorkflowSession,
  type RobotDataWorkflowAction,
} from './robotDataWorkflowSimulation';
import DataReturnAnnotationEditor, {
  DataReturnAnnotationSummary,
} from './DataReturnAnnotationEditor';
import DataReturnEpisodeControl from './DataReturnEpisodeControl';
import './data-return-mock.css';

type View = 'sessions' | 'review' | 'versions';
type Dialog = 'new' | 'publish' | 'config' | null;
const sessionLabels: Record<DataReturnSessionStatus, string> = {
  draft: '待开始',
  running: '策略执行中',
  paused: '已暂停',
  takeover: '人工接管中',
  between_episodes: '待下一条 Episode',
  completed: '已完成',
};
const reviewLabels = { pending: '待审核', retained: '已保留', excluded: '已排除' };
const episodeOutcomeLabels: Record<DataReturnEpisodeOutcome, string> = {
  success: '成功',
  failure: '失败',
  unknown: '未判定',
};
const pendingExperts = (session: DataReturnSession) =>
  session.episodes.reduce(
    (sum, episode) =>
      sum +
      episode.segments.filter(
        (segment) =>
          segment.source === 'human' && segment.expertValid && segment.review === 'pending',
      ).length,
    0,
  );
const sessionLabel = (session: DataReturnSession) =>
  session.mode === 'shadow' && session.status === 'running'
    ? '人工示教中'
    : sessionLabels[session.status];
const clock = (ms: number) =>
  `${Math.floor(ms / 60000)
    .toString()
    .padStart(2, '0')}:${Math.floor((ms % 60000) / 1000)
    .toString()
    .padStart(2, '0')}`;
const date = (value: number) =>
  new Date(value).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
const preferredEpisode = (session: DataReturnSession) =>
  session.episodes.find((episode) =>
    episode.segments.some(
      (segment) =>
        segment.source === 'human' && segment.expertValid && segment.review === 'pending',
    ),
  ) || session.episodes[0];
const samplePosition = (session: DataReturnSession) => {
  const episode = preferredEpisode(session);
  const human = episode?.segments.find(
    (segment) => segment.source === 'human' && segment.expertValid,
  );
  return human ? Math.min(human.endMs - 1, human.startMs + 1500) : 0;
};

interface DataReturnMockProps {
  deploymentRecordId: string;
  deploymentName: string;
  machineName?: string;
  runtimeId?: string;
}

export default function DataReturnMock(props: DataReturnMockProps) {
  return <DataReturnWorkspace key={props.deploymentRecordId} {...props} />;
}

function DataReturnWorkspace({
  deploymentRecordId,
  deploymentName,
  machineName,
  runtimeId,
}: DataReturnMockProps) {
  const [workflow, setWorkflow] = useState(() =>
    createRobotDataWorkflow({
      deploymentRecordId,
      name: deploymentName,
      machine: machineName || '未指定机器',
      runtimeId,
    }),
  );
  const simulation = workflow.data;
  const robot = workflow.robot;
  const runtimeStatus = workflowStatus(workflow);
  const activeSession = activeWorkflowSession(workflow);
  const [logSource, setLogSource] = useState('ALL');
  const [view, setView] = useState<View>('sessions');
  const [selectedId, setSelectedId] = useState(
    () =>
      simulation.sessions.find((item) => item.status === 'completed')?.id ||
      simulation.sessions[0]?.id ||
      '',
  );
  const selected =
    simulation.sessions.find((item) => item.id === selectedId) || simulation.sessions[0];
  const [episodeId, setEpisodeId] = useState('');
  const [cursorMs, setCursorMs] = useState(() => (selected ? samplePosition(selected) : 0));
  const [playing, setPlaying] = useState(false);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [reviewFilter, setReviewFilter] = useState('all');
  const [annotationFilter, setAnnotationFilter] = useState('all');
  const [annotationTarget, setAnnotationTarget] = useState<{
    sessionId: string;
    episodeId: string;
    segmentId: string;
    annotationId?: string;
  } | null>(null);
  const [dialog, setDialog] = useState<Dialog>(null);
  const [task, setTask] = useState('折叠衣服');
  const [sessionName, setSessionName] = useState('');
  const [mode, setMode] = useState<DataReturnMode>('dagger');
  const [versionName, setVersionName] = useState('');
  const [publishIds, setPublishIds] = useState<string[]>([]);
  const [publishing, setPublishing] = useState(false);
  const [selectedVersionId, setSelectedVersionId] = useState(simulation.versions[0]?.id || '');
  const [toast, setToast] = useState('');
  const captureHintId = useId();
  const publishTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const publishGeneration = useRef(0);
  const send = (action: RobotDataWorkflowAction) =>
    setWorkflow((current) => robotDataWorkflowReducer(current, action));
  const stats = dataReturnStats(simulation);
  const recording = simulation.sessions.some((item) =>
    ['running', 'takeover'].includes(item.status),
  );
  const ticking = recording || !['idle', 'stopped'].includes(robot.phase);
  const sdkReady = ['sdk_ready', 'eval_starting', 'ready', 'running', 'paused'].includes(
    robot.phase,
  );
  const evalReady = ['ready', 'running', 'paused'].includes(robot.phase);
  const telemetry = robot.telemetry;
  const episode =
    selected?.episodes.find((item) => item.id === episodeId) ||
    selected?.episodes.find((item) => item.status === 'recording') ||
    (selected?.status === 'between_episodes'
      ? selected.episodes.at(-1)
      : selected && preferredEpisode(selected));
  const runtimePreview =
    selected &&
    workflow.runtimeSessionId === selected.id &&
    episode?.status === 'recording' &&
    ['running', 'paused', 'takeover'].includes(selected.status);
  const live = runtimePreview && ['running', 'takeover'].includes(selected.status);
  const durationMs = episode?.durationMs || 0;
  const position = runtimePreview ? durationMs : Math.min(cursorMs, durationMs);
  const cameraTime = runtimePreview ? robot.frameMotionMs : position;
  const segment =
    episode?.segments.find((item) => position >= item.startMs && position < item.endMs) ||
    episode?.segments.at(-1);
  const annotationSegment =
    episode?.status === 'sealed' &&
    annotationTarget?.sessionId === selected?.id &&
    annotationTarget?.episodeId === episode.id
      ? episode.segments.find((part) => part.id === annotationTarget.segmentId)
      : undefined;
  const expertValid = segment?.source === 'human' && segment.expertValid;
  const humanControlsPreview = runtimePreview
    ? runtimeStatus.source === 'human'
    : segment?.source === 'human';
  const rows = useMemo(
    () =>
      simulation.sessions.flatMap((session) =>
        session.episodes.flatMap((item) =>
          item.segments.map((part) => ({ session, episode: item, segment: part })),
        ),
      ),
    [simulation.sessions],
  );
  const expertRows = rows.filter(
    (row) =>
      row.segment.source === 'human' && row.segment.expertValid && row.episode.status === 'sealed',
  );
  const retained = expertRows.filter((row) => row.segment.review === 'retained');
  const retainedMs = retained.reduce(
    (sum, row) => sum + row.segment.endMs - row.segment.startMs,
    0,
  );
  const reviewRows = rows.filter(
    (row) =>
      row.episode.status === 'sealed' &&
      (annotationFilter === 'all' ||
        (annotationFilter === 'labeled'
          ? row.segment.annotations.length > 0
          : row.segment.annotations.length === 0)) &&
      (reviewFilter === 'context'
        ? !row.segment.expertValid || row.segment.source !== 'human'
        : row.segment.expertValid &&
          row.segment.source === 'human' &&
          (reviewFilter === 'all' || row.segment.review === reviewFilter)),
  );
  const annotatedCount = rows.filter(
    (row) => row.episode.status === 'sealed' && row.segment.annotations.length > 0,
  ).length;
  const visibleSessions = simulation.sessions.filter(
    (item) =>
      `${item.name} ${item.task}`.toLowerCase().includes(search.toLowerCase()) &&
      (filter === 'all' ||
        (filter === 'completed'
          ? item.status === 'completed'
          : ['draft', 'running', 'paused', 'takeover', 'between_episodes'].includes(item.status))),
  );
  const version =
    simulation.versions.find((item) => item.id === selectedVersionId) || simulation.versions.at(-1);

  useEffect(() => {
    if (!ticking) return;
    const timer = setInterval(
      () =>
        setWorkflow((current) =>
          robotDataWorkflowReducer(current, { type: 'tick', now: Date.now() }),
        ),
      500,
    );
    return () => clearInterval(timer);
  }, [ticking]);
  useEffect(() => {
    const action =
      robot.phase === 'sdk_starting'
        ? 'sdk_ready'
        : robot.phase === 'eval_starting'
          ? 'eval_ready'
          : undefined;
    if (!action) return;
    const generation = workflow.runtimeGeneration;
    const timer = setTimeout(
      () =>
        setWorkflow((current) =>
          robotDataWorkflowReducer(current, {
            type: 'runtime',
            action,
            generation,
            now: Date.now(),
          }),
        ),
      action === 'sdk_ready' ? 1100 : 1700,
    );
    return () => clearTimeout(timer);
  }, [robot.phase, workflow.runtimeGeneration]);
  useEffect(() => {
    if (!playing || live || durationMs <= 0) return;
    const timer = setInterval(
      () => setCursorMs((current) => (current + 150 >= durationMs ? 0 : current + 150)),
      150,
    );
    return () => clearInterval(timer);
  }, [playing, live, durationMs, selectedId, episodeId]);
  useEffect(() => {
    if (simulation.lastPublishedVersionId) setSelectedVersionId(simulation.lastPublishedVersionId);
  }, [simulation.lastPublishedVersionId]);
  useEffect(
    () => () => {
      publishGeneration.current += 1;
      clearTimeout(publishTimer.current);
    },
    [],
  );

  const chooseSession = (session: DataReturnSession) => {
    setAnnotationTarget(null);
    setSelectedId(session.id);
    setEpisodeId('');
    setCursorMs(samplePosition(session));
    setPlaying(false);
  };
  const operate = (type: 'start' | 'pause' | 'takeover' | 'resume' | 'finish') => {
    if (!selected) return;
    const reason =
      type === 'start'
        ? workflowStartReason(workflow, selected.id)
        : type === 'resume'
          ? workflowResumeReason(workflow, selected.id)
          : undefined;
    if (reason) {
      setToast(reason);
      return;
    }
    setPlaying(false);
    setCursorMs(durationMs);
    if (type === 'start' || type === 'resume') {
      setEpisodeId('');
      setAnnotationTarget(null);
    } else if (episode) setEpisodeId(episode.id);
    send({ type, sessionId: selected.id, now: Date.now() });
    if (type === 'finish') setToast('已结束运行并封存当前会话，可继续审核其中的有效专家片段。');
  };
  const finishEpisode = (targetEpisodeId: string, outcome: DataReturnEpisodeOutcome) => {
    if (!selected) return;
    const current = selected.episodes.at(-1);
    if (!current || current.id !== targetEpisodeId || current.status !== 'recording') return;
    setPlaying(false);
    setAnnotationTarget(null);
    setEpisodeId(current.id);
    setCursorMs(current.durationMs);
    send({
      type: 'finish-episode',
      sessionId: selected.id,
      episodeId: current.id,
      outcome,
      now: Date.now(),
    });
    setToast(
      `${current.name} 已结束；本条结果：${outcome === 'success' ? '成功' : outcome === 'failure' ? '失败' : '中止/未判定'}。整理场景后可开始下一条。`,
    );
  };
  const startNextEpisode = (afterEpisodeId: string) => {
    if (!selected) return;
    const reason = workflowNextEpisodeReason(workflow, selected.id, afterEpisodeId);
    if (reason) {
      setToast(reason);
      return;
    }
    setPlaying(false);
    setAnnotationTarget(null);
    setEpisodeId('');
    setCursorMs(0);
    send({ type: 'next-episode', sessionId: selected.id, afterEpisodeId, now: Date.now() });
    setToast('已开始下一条 Episode，继续记录到当前会话。');
  };
  const closeDialog = () => {
    publishGeneration.current += 1;
    clearTimeout(publishTimer.current);
    setPublishing(false);
    setDialog(null);
  };
  const createSession = () => {
    if (!task.trim()) return;
    const next = robotDataWorkflowReducer(workflow, {
      type: 'create-session',
      mode,
      task: task.trim(),
      name: sessionName.trim() || undefined,
      now: Date.now(),
    });
    setWorkflow(next);
    setSelectedId(next.data.activeSessionId);
    setEpisodeId('');
    setCursorMs(0);
    setPlaying(false);
    setDialog(null);
    setView('sessions');
    setFilter('all');
    setSearch('');
    setToast('已新建采集会话；完成 SDK 与 Eval 准备后，即可开始运行与采集。');
    setAnnotationTarget(null);
  };
  const openPublish = () => {
    setPublishIds(retained.map((row) => row.segment.id));
    setVersionName(`cloth_feedback_v${String(simulation.versions.length + 1).padStart(3, '0')}`);
    setDialog('publish');
  };
  const publish = () => {
    if (publishing || !publishIds.length || !versionName.trim()) return;
    setPublishing(true);
    const generation = ++publishGeneration.current;
    const selectedSegments = [...publishIds],
      name = versionName.trim();
    publishTimer.current = setTimeout(() => {
      if (generation !== publishGeneration.current) return;
      setWorkflow((current) =>
        robotDataWorkflowReducer(current, {
          type: 'publish',
          segmentIds: selectedSegments,
          name,
          now: Date.now(),
        }),
      );
      setPublishing(false);
      setDialog(null);
      setView('versions');
      setToast('数据版本已就绪；相同片段与标签的重复发布会复用已有版本。');
    }, 850);
  };
  const reviewSegment = (
    row: (typeof rows)[number],
    review: 'retained' | 'excluded',
    note?: string,
  ) =>
    send({
      type: 'review',
      sessionId: row.session.id,
      episodeId: row.episode.id,
      segmentId: row.segment.id,
      review,
      note,
    });
  const jumpToSegment = (row: (typeof rows)[number], annotate = false) => {
    setSelectedId(row.session.id);
    setEpisodeId(row.episode.id);
    setCursorMs(row.segment.startMs);
    setPlaying(false);
    setAnnotationTarget(
      annotate
        ? { sessionId: row.session.id, episodeId: row.episode.id, segmentId: row.segment.id }
        : null,
    );
    setFilter('all');
    setSearch('');
    setView('sessions');
  };
  const captureEntry = workflowCaptureEntry(workflow, selected?.id);
  const prepareOrStartCapture = () => {
    if (captureEntry.disabled || !captureEntry.action) return;
    if (captureEntry.action === 'start') operate('start');
    else send({ type: 'runtime', action: captureEntry.action, now: Date.now() });
  };
  const captureButton = (overlay = false) => (
    <button
      type="button"
      className={`dr-button primary small${overlay ? ' dr-capture-entry' : ''}`}
      disabled={captureEntry.disabled}
      aria-describedby={captureHintId}
      onClick={prepareOrStartCapture}
    >
      {captureEntry.busy ? <span className="dr-spinner" /> : <Play size={13} />}
      {captureEntry.label}
    </button>
  );
  const resumeReason =
    selected && ['paused', 'takeover'].includes(selected.status)
      ? workflowResumeReason(workflow, selected.id)
      : undefined;
  const runtimeLogs = robot.logs.filter((item) => logSource === 'ALL' || item.source === logSource);
  const policy = actionValues(position, false);
  const human = actionValues(position, true);
  const sent =
    runtimePreview && selected?.status === 'paused'
      ? [0, 0, 0, 0]
      : segment?.source === 'human'
        ? human
        : policy;

  return (
    <section
      className="data-return-mock"
      data-testid="data-return-mock"
      aria-label="真机运行与回流工作区"
    >
      <header className="dr-header">
        <div className="dr-heading">
          <span className="dr-heading-icon">
            <Database size={20} />
          </span>
          <div>
            <h2>真机运行与数据回流</h2>
            <p>SDK / Eval 准备 → 同一会话运行采集 → 打标审核与版本</p>
          </div>
        </div>
        <div className="dr-header-actions">
          <button
            className="dr-button primary"
            type="button"
            onClick={() => {
              setSessionName('');
              setDialog('new');
            }}
          >
            <Plus size={15} />
            新建采集会话
          </button>
        </div>
      </header>
      <div className="dr-notice">
        <Info size={15} />
        <span>功能预览：尚未连接采集设备，操作结果仅保留在本页，不写入资产或提交训练。</span>
        <span className="dr-context">
          {deploymentName} · {machineName || '未指定机器'}
        </span>
      </div>
      <div className="dr-stats">
        {[
          {
            label: '采集会话',
            value: stats.sessionCount,
            suffix: '个',
            icon: Film,
            note: `${stats.completedSessionCount} 个已结束`,
          },
          {
            label: '待审核专家片段',
            value: stats.pendingExpertReviewCount,
            suffix: '段',
            icon: ShieldCheck,
            note: '只统计有效人工纠正',
          },
          {
            label: '已保留专家数据',
            value: clock(retainedMs),
            suffix: '',
            icon: Clock3,
            note: `${retained.length} 段 · 累计`,
          },
          {
            label: '数据版本',
            value: stats.versionCount,
            suffix: '个',
            icon: Layers3,
            note: '本地快照，可预览配置',
          },
        ].map((stat) => (
          <div className="dr-stat" key={stat.label}>
            <span>
              <stat.icon size={15} />
              {stat.label}
            </span>
            <strong>
              {stat.value}
              <small>{stat.suffix}</small>
            </strong>
            <p>{stat.note}</p>
          </div>
        ))}
      </div>
      <div className="dr-runtime-prep" aria-label="运行准备">
        <div className="dr-runtime-prep-top">
          <div className="dr-runtime-summary">
            <span className={`dr-runtime-indicator ${recording ? 'running' : ''}`} />
            <strong>{runtimeStatus.label}</strong>
            {activeSession && <span className="dr-runtime-session">{activeSession.name}</span>}
          </div>
          <div className="dr-runtime-buttons">
            {activeSession && activeSession.id !== selected?.id && (
              <button
                className="dr-button small"
                type="button"
                onClick={() => {
                  chooseSession(activeSession);
                  setView('sessions');
                }}
              >
                查看当前会话
              </button>
            )}
            <button
              className="dr-button small"
              type="button"
              disabled={!!activeSession || !['idle', 'stopped'].includes(robot.phase)}
              onClick={() => send({ type: 'runtime', action: 'start_sdk', now: Date.now() })}
            >
              {robot.phase === 'sdk_starting' ? (
                <span className="dr-spinner" />
              ) : (
                <Server size={13} />
              )}
              {robot.phase === 'sdk_starting' ? 'SDK 准备中…' : '启动 SDK'}
            </button>
            <button
              className="dr-button small"
              type="button"
              disabled={robot.phase !== 'sdk_ready'}
              onClick={() => send({ type: 'runtime', action: 'start_eval', now: Date.now() })}
            >
              {robot.phase === 'eval_starting' ? (
                <span className="dr-spinner" />
              ) : (
                <Activity size={13} />
              )}
              {robot.phase === 'eval_starting' ? 'Eval 初始化中…' : '初始化 Eval'}
            </button>
            <button
              className="dr-button small dr-runtime-reset"
              type="button"
              disabled={robot.phase === 'idle'}
              onClick={() => {
                send({ type: 'reset-runtime', now: Date.now() });
                setPlaying(false);
                setToast('运行准备已重置；当前会话已封存，已有片段与版本仍保留。');
              }}
            >
              <RotateCcw size={12} />
              {activeSession ? '结束并重置准备' : '重置准备'}
            </button>
          </div>
        </div>
        <ol className="dr-workflow-steps" aria-label="采集流程步骤">
          {[
            {
              label: 'SDK 连接',
              detail: sdkReady ? '已连接' : '等待准备',
              ready: sdkReady,
              busy: robot.phase === 'sdk_starting',
            },
            {
              label: 'Eval 初始化',
              detail: evalReady ? '已就绪' : '加载策略',
              ready: evalReady,
              busy: robot.phase === 'eval_starting',
            },
            {
              label: '运行与采集',
              detail: activeSession ? '同一会话关联' : '新建或选择待开始会话',
              ready: !!activeSession,
              busy: false,
            },
            { label: '打标与发布', detail: '片段打标、审核后生成版本', ready: false, busy: false },
          ].map((step, index) => (
            <li
              key={step.label}
              className={`${step.ready ? 'ready' : ''} ${step.busy ? 'busy' : ''}`}
            >
              <span className="dr-workflow-number">
                {step.ready ? (
                  <Check size={12} />
                ) : step.busy ? (
                  <span className="dr-spinner" />
                ) : (
                  index + 1
                )}
              </span>
              <div>
                <strong>{step.label}</strong>
                <small>{step.detail}</small>
              </div>
              {index < 3 && <ChevronRight size={14} />}
            </li>
          ))}
        </ol>
        <p className="dr-runtime-hint">
          新会话先完成 SDK / Eval 准备；已结束的 Episode 可直接回放、打标与审核。
        </p>
      </div>
      <nav className="dr-tabs" aria-label="运行与回流视图">
        {[
          { id: 'sessions', label: '运行与采集', icon: Video, count: stats.sessionCount },
          {
            id: 'review',
            label: '打标与审核',
            icon: ShieldCheck,
            count: stats.pendingExpertReviewCount,
          },
          { id: 'versions', label: '数据版本', icon: Layers3, count: stats.versionCount },
        ].map((item) => (
          <button
            type="button"
            key={item.id}
            className={view === item.id ? 'selected' : ''}
            aria-pressed={view === item.id}
            onClick={() => setView(item.id as View)}
          >
            <item.icon size={15} />
            {item.label}
            <span>{item.count}</span>
          </button>
        ))}
        <button
          type="button"
          className="dr-reset"
          onClick={() => {
            const next = robotDataWorkflowReducer(workflow, { type: 'reset', now: Date.now() });
            const first =
              next.data.sessions.find((item) => item.status === 'completed') ||
              next.data.sessions[0];
            setWorkflow(next);
            setSelectedId(first?.id || '');
            setCursorMs(first ? samplePosition(first) : 0);
            setSelectedVersionId(next.data.versions[0]?.id || '');
            setSearch('');
            setFilter('all');
            setReviewFilter('all');
            setAnnotationFilter('all');
            setAnnotationTarget(null);
            setView('sessions');
            setPlaying(false);
            setEpisodeId('');
            setToast('已重置工作区。');
          }}
        >
          <RotateCcw size={13} />
          重置工作区
        </button>
      </nav>
      {toast && (
        <div className="dr-toast" role="status">
          <CheckCircle2 size={15} />
          <span>{toast}</span>
          <button type="button" aria-label="关闭提示" onClick={() => setToast('')}>
            <X size={14} />
          </button>
        </div>
      )}

      {view === 'sessions' && (
        <div className="dr-session-layout">
          <aside className="dr-session-list" aria-label="采集会话列表">
            <div className="dr-list-title">
              <h3>会话记录</h3>
            </div>
            <label className="dr-search">
              <Search size={14} />
              <input
                aria-label="搜索采集会话"
                placeholder="搜索任务或会话"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
            <div className="dr-filters">
              {[
                ['all', '全部'],
                ['completed', '已结束'],
                ['active', '进行中'],
              ].map(([value, label]) => (
                <button
                  type="button"
                  key={value}
                  className={filter === value ? 'selected' : ''}
                  onClick={() => setFilter(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="dr-session-items">
              {visibleSessions.map((session) => (
                <button
                  type="button"
                  key={session.id}
                  className={`dr-session-item ${selectedId === session.id ? 'selected' : ''}`}
                  onClick={() => chooseSession(session)}
                >
                  <span className="dr-session-item-top">
                    <span className="dr-session-number">
                      S{String(simulation.sessions.indexOf(session) + 1).padStart(2, '0')}
                    </span>
                    <span className={`dr-status ${session.status}`}>{sessionLabel(session)}</span>
                  </span>
                  <strong>{session.name}</strong>
                  <span className="dr-session-task">
                    {session.task} ·{' '}
                    {session.mode === 'dagger' ? '策略 + 人工纠正' : '人工示教 · 影子策略'}
                    {session.status === 'completed' && (
                      <span>
                        {' '}
                        ·{' '}
                        {pendingExperts(session) ? `${pendingExperts(session)} 段待审核` : '已审核'}
                      </span>
                    )}
                  </span>
                  <span className="dr-session-item-bottom">
                    <Clock3 size={11} />
                    {clock(session.durationMs)}
                    <span>{session.takeovers} 次接管</span>
                    <ChevronRight size={12} />
                  </span>
                </button>
              ))}
              {!visibleSessions.length && <p className="dr-empty">没有匹配的采集会话</p>}
            </div>
            <div className="dr-list-foot">
              <span className="dr-dot" />
              本页工作区 · 离开后重置
            </div>
          </aside>
          <div className="dr-session-detail">
            {selected ? (
              <>
                <div className="dr-session-heading">
                  <div>
                    <div className="dr-eyebrow">
                      {runtimePreview
                        ? 'LIVE CAPTURE'
                        : selected.status === 'draft'
                          ? 'SESSION SETUP'
                          : 'SESSION REPLAY'}{' '}
                    </div>
                    <h3>{selected.name}</h3>
                    <p>
                      {date(selected.createdAt)} · {selected.episodes.length} episodes ·{' '}
                      {selected.frames.toLocaleString()} 帧
                    </p>
                  </div>
                  <span className={`dr-status ${selected.status}`}>{sessionLabel(selected)}</span>
                </div>
                <div className="dr-session-controls">
                  <span id={captureHintId} aria-live="polite">
                    <Activity size={13} />
                    {selected.status === 'draft'
                      ? captureEntry.hint
                      : resumeReason
                        ? resumeReason
                        : selected.status === 'between_episodes'
                          ? '本条 Episode 已结束，整理场景后可开始下一条。'
                          : selected.status === 'completed'
                            ? pendingExperts(selected)
                              ? `会话已完成，${pendingExperts(selected)} 段专家片段待审核`
                              : '会话已完成，有效专家片段已审核'
                            : selected.status === 'takeover'
                              ? '人工输入拥有控制权'
                              : selected.status === 'paused'
                                ? '采集已暂停'
                                : selected.mode === 'shadow' && selected.status === 'running'
                                  ? '人工示教 · 影子策略仅作建议'
                                  : '按当前控制模式录制'}
                  </span>
                  <div>
                    {selected.status === 'draft' && captureButton()}
                    {['running', 'takeover'].includes(selected.status) && (
                      <button
                        type="button"
                        className="dr-button small"
                        onClick={() => operate('pause')}
                      >
                        <Pause size={13} />
                        暂停采集
                      </button>
                    )}
                    {selected.status === 'running' && selected.mode === 'dagger' && (
                      <button
                        type="button"
                        className="dr-button amber small"
                        onClick={() => operate('takeover')}
                      >
                        <Hand size={13} />
                        人工接管
                      </button>
                    )}
                    {['paused', 'takeover'].includes(selected.status) && (
                      <button
                        type="button"
                        className="dr-button primary small"
                        disabled={!!resumeReason}
                        onClick={() => operate('resume')}
                      >
                        <Play size={13} />
                        {selected.status === 'paused' ? '继续采集' : '恢复策略'}
                      </button>
                    )}
                    {['running', 'paused', 'takeover', 'between_episodes'].includes(
                      selected.status,
                    ) && (
                      <button
                        type="button"
                        className="dr-button small"
                        title="结束本次全部采集；未单独结束的 Episode 将记为未判定。"
                        onClick={() => operate('finish')}
                      >
                        <Square size={12} />
                        结束整个会话
                      </button>
                    )}
                    {episode?.status === 'sealed' && (
                      <button
                        type="button"
                        className="dr-button primary small"
                        disabled={!segment || segment.endFrame <= segment.startFrame}
                        onClick={() => {
                          if (!episode || !segment) return;
                          setPlaying(false);
                          setAnnotationTarget({
                            sessionId: selected.id,
                            episodeId: episode.id,
                            segmentId: segment.id,
                          });
                        }}
                      >
                        <Tags size={13} />
                        片段打标
                      </button>
                    )}
                    {episode?.status === 'sealed' && (
                      <button
                        type="button"
                        className="dr-button small"
                        onClick={() => setView('review')}
                      >
                        <ShieldCheck size={13} />
                        审核片段
                        <ArrowRight size={12} />
                      </button>
                    )}
                  </div>
                </div>
                <DataReturnEpisodeControl
                  session={selected}
                  selectedEpisodeId={episode?.id}
                  nextBlockedReason={
                    selected.status === 'between_episodes' && selected.episodes.at(-1)
                      ? workflowNextEpisodeReason(
                          workflow,
                          selected.id,
                          selected.episodes.at(-1)!.id,
                        )
                      : undefined
                  }
                  onFinishEpisode={finishEpisode}
                  onStartNext={startNextEpisode}
                  onSelectEpisode={(targetEpisodeId) => {
                    setAnnotationTarget(null);
                    setPlaying(false);
                    setEpisodeId(targetEpisodeId);
                    setCursorMs(0);
                  }}
                />
                <div className={`dr-annotation-workbench${annotationSegment ? ' editing' : ''}`}>
                  <div className="dr-annotation-replay">
                    <div className="dr-preview" aria-label="双相机预览">
                      <div className="dr-preview-toolbar">
                        <span>
                          <Video size={14} />
                          {runtimePreview
                            ? '运行画面'
                            : selected.status === 'draft'
                              ? '采集准备'
                              : '历史回放'}
                        </span>
                        <div className="dr-preview-actions">
                          <button
                            type="button"
                            className={robot.streamConnected ? '' : 'disconnected'}
                            disabled={!evalReady || !runtimePreview}
                            title={
                              runtimePreview
                                ? '影响当前会话的画面与标签有效性'
                                : '开始当前会话后可测试断流；历史回放不受影响'
                            }
                            onClick={() =>
                              send({
                                type: 'runtime',
                                action: robot.streamConnected
                                  ? 'stream_disconnect'
                                  : 'stream_reconnect',
                                now: Date.now(),
                              })
                            }
                          >
                            {robot.streamConnected ? <WifiOff size={11} /> : <Wifi size={11} />}
                            {robot.streamConnected ? '测试断流' : '恢复画面'}
                          </button>
                          <select
                            aria-label="选择 Episode"
                            value={episode?.id || ''}
                            disabled={!selected.episodes.length}
                            onChange={(event) => {
                              setAnnotationTarget(null);
                              setEpisodeId(event.target.value);
                              setCursorMs(0);
                              setPlaying(false);
                            }}
                          >
                            <option value="" disabled>
                              等待新 Episode
                            </option>
                            {selected.episodes.map((item) => (
                              <option key={item.id} value={item.id}>
                                {item.name} ·{' '}
                                {item.status === 'recording'
                                  ? '采集中'
                                  : `已结束 · ${episodeOutcomeLabels[item.outcome]}`}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                      <div className="dr-camera-stage">
                        <div className="dr-camera-grid">
                          <CameraScene
                            view="overhead"
                            time={cameraTime}
                            takeover={!!humanControlsPreview}
                          />
                          <CameraScene
                            view="wrist"
                            time={cameraTime}
                            takeover={!!humanControlsPreview}
                          />
                        </div>
                        {runtimePreview && !robot.streamConnected && (
                          <div className="dr-stream-overlay" role="status">
                            <WifiOff size={21} />
                            <strong>画面中断 · 已冻结最后一帧</strong>
                            <span>执行与录制未自动停止；断流片段不作为有效专家标签。</span>
                          </div>
                        )}
                        {selected.status === 'draft' && (
                          <div
                            className="dr-stream-overlay waiting"
                            role="group"
                            aria-label="当前会话准备与采集"
                          >
                            <strong>
                              {captureEntry.busy
                                ? '正在准备运行环境'
                                : evalReady
                                  ? '准备就绪，可以开始采集'
                                  : '在这里完成准备，开始采集会话'}
                            </strong>
                            <span>{captureEntry.hint}</span>
                            {captureButton(true)}
                          </div>
                        )}
                      </div>
                      <div className="dr-playback">
                        <button
                          type="button"
                          aria-label={playing ? '暂停回放' : '播放回放'}
                          disabled={!!runtimePreview || !durationMs}
                          onClick={() => setPlaying((value) => !value)}
                        >
                          {playing ? <Pause size={15} /> : <Play size={15} />}
                        </button>
                        <span>
                          {clock(position)}
                          <small> / {clock(durationMs)}</small>
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={Math.max(1, durationMs)}
                          step={50}
                          value={position}
                          disabled={!!runtimePreview || !durationMs}
                          aria-label="回放时间"
                          onChange={(event) => setCursorMs(Number(event.target.value))}
                        />
                        <span className="dr-frame">
                          {Math.floor((position / 1000) * (episode?.fps || 20))} f
                        </span>
                      </div>
                    </div>
                    <div className="dr-timeline-section">
                      <div className="dr-section-line">
                        <h4>控制来源时间线</h4>
                        <span>
                          <i className="policy" />
                          策略上下文 <i className="human" />
                          人工纠正 <i className="invalid" />
                          无效标签
                        </span>
                      </div>
                      <div className="dr-timeline" aria-label="控制来源时间线">
                        {episode?.segments.map((part) => (
                          <button
                            type="button"
                            key={part.id}
                            className={`${part.source} ${part.source === 'human' && !part.expertValid ? 'invalid' : ''} ${segment?.id === part.id ? 'active' : ''}`}
                            title={`${part.source === 'human' ? '人工输入' : '策略上下文'} ${clock(part.startMs)}–${clock(part.endMs)}`}
                            style={{
                              left: `${(part.startMs / Math.max(durationMs, 1)) * 100}%`,
                              width: `${Math.max(0.4, ((part.endMs - part.startMs) / Math.max(durationMs, 1)) * 100)}%`,
                            }}
                            onClick={() => {
                              setCursorMs(part.startMs);
                              setPlaying(false);
                              if (annotationTarget && episode.status === 'sealed')
                                setAnnotationTarget({
                                  sessionId: selected.id,
                                  episodeId: episode.id,
                                  segmentId: part.id,
                                });
                            }}
                          />
                        ))}
                        <span
                          className="dr-playhead"
                          style={{
                            left: `${Math.min(100, (position / Math.max(durationMs, 1)) * 100)}%`,
                          }}
                        />
                      </div>
                      <div className="dr-timeline-ticks">
                        <span>00:00</span>
                        <span>{clock(durationMs / 2)}</span>
                        <span>{clock(durationMs)}</span>
                      </div>
                      {episode?.status === 'sealed' &&
                        episode.segments.some((part) => part.annotations.length) && (
                          <div className="dr-annotation-track" aria-label="片段打标时间线">
                            <span>
                              <Tags size={12} /> 语义标签
                            </span>
                            <div>
                              {episode.segments.flatMap((part) =>
                                part.annotations.map((annotation) => (
                                  <button
                                    type="button"
                                    key={annotation.id}
                                    className={`dr-annotation-marker ${annotation.outcome}`}
                                    title={`${annotation.outcome === 'success' ? '成功' : annotation.outcome === 'failure' ? '失败' : '待判定'} · ${annotation.startFrame}–${annotation.endFrame} 帧`}
                                    aria-label={`编辑打标 ${annotation.startFrame}–${annotation.endFrame} 帧`}
                                    style={{
                                      left: `${(annotation.startFrame / Math.max(episode.frames, 1)) * 100}%`,
                                      width: `${Math.max(1, ((annotation.endFrame - annotation.startFrame) / Math.max(episode.frames, 1)) * 100)}%`,
                                    }}
                                    onClick={() => {
                                      setPlaying(false);
                                      setCursorMs((annotation.startFrame / episode.fps) * 1000);
                                      setAnnotationTarget({
                                        sessionId: selected.id,
                                        episodeId: episode.id,
                                        segmentId: part.id,
                                        annotationId: annotation.id,
                                      });
                                    }}
                                  >
                                    <span>
                                      {annotation.outcome === 'success'
                                        ? '成功'
                                        : annotation.outcome === 'failure'
                                          ? '失败'
                                          : '已打标'}
                                    </span>
                                  </button>
                                )),
                              )}
                            </div>
                          </div>
                        )}
                    </div>
                  </div>
                  {annotationSegment && episode && (
                    <DataReturnAnnotationEditor
                      key={`${selected.id}:${episode.id}:${annotationSegment.id}:${annotationTarget?.annotationId || 'new'}`}
                      session={selected}
                      episode={episode}
                      segment={annotationSegment}
                      initialAnnotationId={annotationTarget?.annotationId}
                      cursorFrame={Math.floor((position / 1000) * episode.fps)}
                      onSeek={(frame) => {
                        setPlaying(false);
                        setCursorMs((frame / episode.fps) * 1000);
                      }}
                      onSave={(annotation, annotationId) =>
                        send({
                          type: 'annotate',
                          sessionId: selected.id,
                          episodeId: episode.id,
                          segmentId: annotationSegment.id,
                          annotation,
                          annotationId,
                        })
                      }
                      onRemove={(annotationId) =>
                        send({
                          type: 'remove-annotation',
                          sessionId: selected.id,
                          episodeId: episode.id,
                          segmentId: annotationSegment.id,
                          annotationId,
                        })
                      }
                      onClose={() => setAnnotationTarget(null)}
                    />
                  )}
                </div>
                <div className="dr-section-line dr-action-heading">
                  <h4>
                    动作通道对照 <small>归一化值 · 4 / 14 维切片</small>
                  </h4>
                  <span className={`dr-label-valid ${expertValid ? 'valid' : ''}`}>
                    {expertValid ? <CheckCircle2 size={12} /> : <ShieldCheck size={12} />}
                    expert_label_valid = {expertValid ? 'true' : 'false'}
                  </span>
                </div>
                <div className="dr-action-grid">
                  <ActionCard
                    title="策略建议"
                    tag="POLICY"
                    tone="policy"
                    values={policy}
                    time={position}
                    note="保留为上下文，不自动视为专家标签"
                  />
                  <ActionCard
                    title="专家纠正"
                    tag="HUMAN"
                    tone="human"
                    values={expertValid ? human : undefined}
                    time={position}
                    note={expertValid ? '仅有效人工片段进入候选专家数据' : '当前没有有效专家标签'}
                  />
                  <ActionCard
                    title="下发动作"
                    tag={
                      runtimePreview && selected.status === 'paused'
                        ? 'HOLD'
                        : segment?.source === 'human'
                          ? 'HUMAN'
                          : 'POLICY'
                    }
                    tone="sent"
                    values={sent}
                    time={position}
                    note="按当前控制来源选择动作"
                  />
                </div>
                <details className="dr-telemetry" aria-label="运行遥测">
                  <summary>
                    <span>
                      <Activity size={13} />
                      <strong>运行反馈</strong>
                    </span>
                    <span className="dr-telemetry-pills">
                      <span>
                        推理{' '}
                        <b>
                          {telemetry.inferenceMs === null
                            ? '—'
                            : `${telemetry.inferenceMs.toFixed(0)} ms`}
                        </b>
                      </span>
                      <span>
                        画面 FPS <b>{telemetry.fps || '—'}</b>
                      </span>
                      <span>
                        控制{' '}
                        <b>
                          {
                            { policy: '策略', human: '人工', hold: '保持', none: '待运行' }[
                              runtimeStatus.source
                            ]
                          }
                        </b>
                      </span>
                      <ChevronRight size={13} />
                    </span>
                  </summary>
                  <div className="dr-telemetry-body">
                    <p>
                      {activeSession
                        ? `当前会话：${activeSession.name}${activeSession.status === 'between_episodes' ? ' · 等待开始下一条 Episode' : ''}`
                        : '当前没有正在运行的会话'}{' '}
                      · 画面监控帧率与采集分开，采集固定为 20 FPS。
                    </p>
                    <div className="dr-telemetry-columns">
                      <div>
                        <h4>
                          关节角度 <small>°</small>
                        </h4>
                        <div className="dr-joints">
                          {telemetry.joints.map((value, index) => (
                            <div key={index}>
                              <span>J{index + 1}</span>
                              <strong>
                                {telemetry.feedbackAt === null ? '—' : value.toFixed(1)}
                              </strong>
                              <i
                                style={
                                  {
                                    '--joint-fill': `${Math.min(100, (Math.abs(value) / 180) * 100)}%`,
                                  } as CSSProperties
                                }
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <h4>
                          TCP 位姿 <small>m / rad</small>
                        </h4>
                        <div className="dr-tcp">
                          {['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'].map((label, index) => (
                            <span key={label}>
                              {label}
                              <strong>
                                {telemetry.feedbackAt === null
                                  ? '—'
                                  : telemetry.tcp[index].toFixed(3)}
                              </strong>
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="dr-telemetry-footer">
                      <span>
                        夹爪{' '}
                        <b>
                          {telemetry.feedbackAt === null
                            ? '—'
                            : `${telemetry.gripperMm.toFixed(1)} mm`}
                        </b>
                      </span>
                      <span>
                        反馈历史 <b>{telemetry.historyFrames} 帧</b>
                      </span>
                      <span>
                        下发动作 <b>{telemetry.submittedSteps} 步</b>
                      </span>
                      <span className={workflow.droppedFrameCount ? 'warn' : ''}>
                        断流缺帧 <b>{workflow.droppedFrameCount}</b>
                      </span>
                    </div>
                  </div>
                </details>
              </>
            ) : (
              <p className="dr-empty">新建采集会话以开始录制</p>
            )}
          </div>
        </div>
      )}

      {view === 'review' && (
        <div className="dr-review-view">
          <div className="dr-view-heading">
            <div>
              <div className="dr-eyebrow">CURATION</div>
              <h3>片段打标与审核</h3>
              <p>回放片段并标记结果、问题与纠正意图，再决定保留或排除。策略片段也可打标供分析。</p>
            </div>
            <button
              type="button"
              className="dr-button primary"
              disabled={!retained.length}
              onClick={openPublish}
            >
              <Layers3 size={14} />
              发布版本
            </button>
          </div>
          <div className="dr-review-summary">
            <span>
              <CheckCircle2 size={14} />
              {retained.length} 段已保留
            </span>
            <span>
              <Clock3 size={14} />
              {clock(retainedMs)} 专家数据
            </span>
            <span>
              <ShieldCheck size={14} />
              {stats.pendingExpertReviewCount} 段待审核
            </span>
            <span>
              <Tags size={14} />
              {annotatedCount} 段已打标
            </span>
            <small>按片段来源审核</small>
          </div>
          <div className="dr-filters dr-review-filters">
            {[
              ['all', '全部专家片段'],
              ['pending', '待审核'],
              ['retained', '已保留'],
              ['excluded', '已排除'],
              ['context', '策略 / 无效上下文'],
            ].map(([value, label]) => (
              <button
                type="button"
                key={value}
                className={reviewFilter === value ? 'selected' : ''}
                onClick={() => setReviewFilter(value)}
              >
                {label}
              </button>
            ))}
            <label className="dr-annotation-filter">
              <span>打标状态</span>
              <select
                value={annotationFilter}
                onChange={(event) => setAnnotationFilter(event.target.value)}
              >
                <option value="all">全部打标状态</option>
                <option value="unlabeled">未打标</option>
                <option value="labeled">已打标</option>
              </select>
            </label>
          </div>
          <div className="dr-review-list">
            {reviewRows.map((row, index) => (
              <article className="dr-review-row" key={row.segment.id}>
                <button
                  className="dr-clip-thumb"
                  type="button"
                  aria-label={`回放片段 ${index + 1}`}
                  onClick={() => jumpToSegment(row)}
                >
                  <Film size={20} />
                  <span>CLIP {String(index + 1).padStart(2, '0')}</span>
                  <Play size={12} />
                </button>
                <div className="dr-clip-copy">
                  <div>
                    <strong>{row.session.name}</strong>
                    <span className={`dr-review-state ${row.segment.review}`}>
                      {reviewLabels[row.segment.review]}
                    </span>
                  </div>
                  <p>
                    {row.episode.name} · {clock(row.segment.startMs)} — {clock(row.segment.endMs)} ·{' '}
                    {row.segment.endFrame - row.segment.startFrame} 帧 · 本条结果：
                    {episodeOutcomeLabels[row.episode.outcome]}
                  </p>
                  <span
                    className={`dr-source-tag ${row.segment.source === 'human' && row.segment.expertValid ? 'human' : ''}`}
                  >
                    {row.segment.source === 'human' && row.segment.expertValid
                      ? '有效人工纠正'
                      : '上下文 · 非专家标签'}
                  </span>
                  <DataReturnAnnotationSummary annotations={row.segment.annotations} compact />
                  <label className="dr-note-input">
                    <span>审核备注</span>
                    <input
                      value={row.segment.note}
                      maxLength={240}
                      placeholder="记录纠正意图、质量或排除原因"
                      onChange={(event) =>
                        send({
                          type: 'review',
                          sessionId: row.session.id,
                          episodeId: row.episode.id,
                          segmentId: row.segment.id,
                          review: row.segment.review,
                          note: event.target.value,
                        })
                      }
                    />
                  </label>
                </div>
                <div className="dr-review-actions">
                  <button
                    type="button"
                    className="dr-button small"
                    onClick={() => jumpToSegment(row, true)}
                  >
                    <Tags size={13} />
                    打标
                  </button>
                  <button
                    type="button"
                    className={`dr-button small ${row.segment.review === 'retained' ? 'primary' : ''}`}
                    disabled={!row.segment.expertValid || row.segment.source !== 'human'}
                    onClick={() => reviewSegment(row, 'retained')}
                  >
                    <Check size={13} />
                    保留
                  </button>
                  <button
                    type="button"
                    className="dr-button small"
                    onClick={() => reviewSegment(row, 'excluded')}
                  >
                    <X size={13} />
                    排除
                  </button>
                </div>
              </article>
            ))}
            {!reviewRows.length && <p className="dr-empty">当前筛选下没有片段</p>}
          </div>
        </div>
      )}

      {view === 'versions' && (
        <div className="dr-versions-view">
          <div className="dr-view-heading">
            <div>
              <div className="dr-eyebrow">DATASET VERSIONS</div>
              <h3>每个版本，都有清楚的来处</h3>
              <p>版本快照保留来源片段、打标内容与审核结果。</p>
            </div>
            <button
              type="button"
              className="dr-button primary"
              disabled={!retained.length}
              onClick={openPublish}
            >
              <Plus size={14} />
              发布版本
            </button>
          </div>
          <div className="dr-version-layout">
            <div className="dr-version-list">
              {[...simulation.versions].reverse().map((item) => (
                <button
                  type="button"
                  key={item.id}
                  className={`dr-version-item ${version?.id === item.id ? 'selected' : ''}`}
                  onClick={() => setSelectedVersionId(item.id)}
                >
                  <span className="dr-version-icon">
                    <Layers3 size={18} />
                  </span>
                  <span>
                    <strong>{item.name}</strong>
                    <small>
                      {date(item.createdAt)} · {item.frameCount.toLocaleString()} 帧
                    </small>
                  </span>
                  <CheckCircle2 size={14} />
                </button>
              ))}
            </div>
            {version ? (
              <div className="dr-version-detail">
                <div className="dr-version-heading">
                  <div>
                    <span className="dr-eyebrow">VERSION SNAPSHOT</span>
                    <h4>{version.name}</h4>
                  </div>
                  <span className="dr-status published">已发布</span>
                </div>
                <div className="dr-version-numbers">
                  <span>
                    <strong>{version.sessionIds.length}</strong>来源会话
                  </span>
                  <span>
                    <strong>{version.sourceSegmentIds.length}</strong>专家片段
                  </span>
                  <span>
                    <strong>{version.frameCount.toLocaleString()}</strong>帧
                  </span>
                </div>
                <dl className="dr-version-contract">
                  <div>
                    <dt>来源</dt>
                    <dd>{version.sessionIds.length} 个采集会话，保留有效人工纠正</dd>
                  </div>
                  <div>
                    <dt>数据合同</dt>
                    <dd>
                      <code>bimanual_14d / v1</code>
                      <small>RGB × 2 / state 14D / action 14D / 20 FPS</small>
                    </dd>
                  </div>
                  <div>
                    <dt>归一化参数</dt>
                    <dd>
                      <code>norm_v{version.id.replace('mock-version-', '').padStart(3, '0')}</code>
                      <small>归一化统计待数据处理流程生成</small>
                    </dd>
                  </div>
                  <div>
                    <dt>选择规则</dt>
                    <dd>已封存 + 人工来源 + 标签有效 + 审核保留</dd>
                  </div>
                  <div>
                    <dt>关联上下文</dt>
                    <dd>
                      {deploymentName}
                      <small>
                        record: {deploymentRecordId}
                        {runtimeId ? ` · runtime: ${runtimeId}` : ''}
                      </small>
                    </dd>
                  </div>
                </dl>
                <section className="dr-version-annotations" aria-label="版本打标快照">
                  <div className="dr-section-line">
                    <h4>
                      <Tags size={13} /> 打标快照
                    </h4>
                    <span>
                      {version.segments.reduce((sum, item) => sum + item.annotations.length, 0)}{' '}
                      条标签 · 发布时封存
                    </span>
                  </div>
                  {version.segments.map((item) => (
                    <div className="dr-version-annotation-row" key={item.id}>
                      <strong>
                        {item.sessionName} · {item.episodeName} · 本条结果：
                        {episodeOutcomeLabels[item.episodeOutcome]}
                      </strong>
                      <DataReturnAnnotationSummary annotations={item.annotations} />
                    </div>
                  ))}
                  <p>后续修改工作区标签时，此版本保留发布时的内容。</p>
                </section>
                <div className="dr-version-foot">
                  <span>
                    <Info size={14} />
                    版本内容已封存
                  </span>
                  <button
                    type="button"
                    className="dr-button primary"
                    onClick={() => setDialog('config')}
                  >
                    <FileCode2 size={14} />
                    预览训练配置
                  </button>
                </div>
              </div>
            ) : (
              <p className="dr-empty">审核保留专家片段后，可以发布第一个数据版本。</p>
            )}
          </div>
        </div>
      )}

      <details className="dr-runtime-logs" aria-label="运行日志">
        <summary>
          <span>
            <Terminal size={14} />
            <strong>运行日志</strong>
            <small>本页事件</small>
          </span>
          <span>
            {robot.logs.length} 条<ChevronRight size={14} />
          </span>
        </summary>
        <div className="dr-runtime-log-toolbar">
          <span>SDK / Eval 与当前采集会话的准备、控制及结束记录。</span>
          <label>
            来源
            <select
              aria-label="筛选运行日志"
              value={logSource}
              onChange={(event) => setLogSource(event.target.value)}
            >
              {['ALL', 'SDK', 'EVAL', 'MONITOR'].map((source) => (
                <option value={source} key={source}>
                  {source === 'ALL' ? '全部' : source}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="dr-log-lines">
          {runtimeLogs.length ? (
            [...runtimeLogs].reverse().map((entry) => (
              <div key={entry.id}>
                <time>{new Date(entry.at).toLocaleTimeString('zh-CN', { hour12: false })}</time>
                <span className={`dr-log-source ${entry.source.toLowerCase()}`}>
                  {entry.source}
                </span>
                <code>{workflowLogText(entry.message)}</code>
              </div>
            ))
          ) : (
            <p>尚无此来源的记录。启动 SDK 后将在这里记录准备与会话操作。</p>
          )}
        </div>
      </details>

      {dialog === 'new' && (
        <MockDialog
          title="新建采集会话"
          subtitle="设置本次采集任务与控制模式。"
          close={closeDialog}
          footer={
            <>
              <button type="button" className="dr-button" onClick={closeDialog}>
                取消
              </button>
              <button
                type="button"
                className="dr-button primary"
                disabled={!task.trim()}
                onClick={createSession}
              >
                <Plus size={14} />
                创建会话
              </button>
            </>
          }
        >
          <label className="dr-form-field">
            会话名称
            <input
              value={sessionName}
              maxLength={80}
              placeholder="例如：衣角翻折 · 人工纠正"
              onChange={(event) => setSessionName(event.target.value)}
            />
          </label>
          <label className="dr-form-field">
            任务
            <input value={task} maxLength={100} onChange={(event) => setTask(event.target.value)} />
          </label>
          <label className="dr-form-field">
            控制模式
            <select
              value={mode}
              onChange={(event) => setMode(event.target.value as DataReturnMode)}
            >
              <option value="dagger">DAgger · 策略执行与人工纠正</option>
              <option value="shadow">Shadow · 人工示教与影子策略</option>
            </select>
          </label>
          <div className="dr-modal-note">
            <ShieldCheck size={16} />
            <span>
              {mode === 'dagger'
                ? '接管后的人类输入才可能成为有效专家标签；策略动作始终保留为上下文。'
                : '人工示教控制、策略仅旁路建议；有效示教经审核可进入数据版本。'}
            </span>
          </div>
        </MockDialog>
      )}
      {dialog === 'publish' && (
        <MockDialog
          title="发布数据版本"
          subtitle="确认本次版本名称与保留的专家片段。"
          close={closeDialog}
          footer={
            <>
              <button type="button" className="dr-button" onClick={closeDialog}>
                取消
              </button>
              <button
                type="button"
                className="dr-button primary"
                disabled={publishing || !publishIds.length || !versionName.trim()}
                onClick={publish}
              >
                {publishing ? <span className="dr-mini-spinner" /> : <Layers3 size={14} />}
                {publishing ? '正在生成版本快照…' : '确认发布版本'}
              </button>
            </>
          }
        >
          <label className="dr-form-field">
            版本名称
            <input
              value={versionName}
              maxLength={80}
              disabled={publishing}
              onChange={(event) => setVersionName(event.target.value)}
            />
          </label>
          <div className="dr-publish-summary">
            <span>{publishIds.length} 个已保留片段</span>
            <small>只有有效人工纠正可被选中</small>
          </div>
          <div className="dr-publish-segments">
            {retained.map((row) => (
              <label key={row.segment.id}>
                <input
                  type="checkbox"
                  disabled={publishing}
                  checked={publishIds.includes(row.segment.id)}
                  onChange={(event) =>
                    setPublishIds((ids) =>
                      event.target.checked
                        ? [...ids, row.segment.id]
                        : ids.filter((id) => id !== row.segment.id),
                    )
                  }
                />
                <span>
                  <strong>{row.session.name}</strong>
                  <small>
                    {row.episode.name} · {clock(row.segment.startMs)}—{clock(row.segment.endMs)}
                  </small>
                </span>
                <span>{row.segment.endFrame - row.segment.startFrame} 帧</span>
              </label>
            ))}
          </div>
          <div className="dr-modal-note">
            <Info size={16} />
            <span>归一化统计需由数据处理流程生成；取消将终止本次版本生成。</span>
          </div>
        </MockDialog>
      )}
      {dialog === 'config' && version && (
        <MockDialog
          title="训练配置预览"
          subtitle="查看版本对应的训练配置，当前未提交训练。"
          close={closeDialog}
          footer={
            <button type="button" className="dr-button primary" onClick={closeDialog}>
              关闭预览
            </button>
          }
        >
          <div className="dr-config-summary">
            <Info size={16} />
            来自数据版本 {version.name}
          </div>
          <pre className="dr-config-code">
            {JSON.stringify(mockTrainingConfig(version), null, 2)}
          </pre>
          <p className="dr-config-hint">
            训练前需核对数据合同并计算归一化统计；此处仅提供配置预览。
          </p>
        </MockDialog>
      )}
    </section>
  );
}

export function workflowLogText(message: string) {
  return message
    .replace(/^\[MOCK\]\s*/, '')
    .replace('SDK 模拟就绪', 'SDK 已就绪')
    .replace('模拟观测', '观测')
    .replace('策略模拟执行', '策略执行')
    .replace('模拟位姿保持', '位姿保持')
    .replace('模拟 SDK / Eval', 'SDK / Eval')
    .replace('模拟画面断流', '画面连接中断')
    .replace('模拟画面恢复', '画面恢复');
}

function actionValues(ms: number, human: boolean) {
  const t = ms / 1000;
  return [
    Math.sin(t * 0.51) * 0.16 + (human ? 0.024 : 0),
    Math.cos(t * 0.37) * 0.11 - (human ? 0.018 : 0),
    Math.sin(t * 0.23 + 1) * 0.08,
    0.68 + Math.sin(t * 0.18) * 0.16,
  ];
}

function ActionCard({
  title,
  tag,
  tone,
  values,
  note,
  time,
}: {
  title: string;
  tag: string;
  tone: string;
  values?: number[];
  note: string;
  time: number;
}) {
  const points = Array.from(
    { length: 36 },
    (_, index) =>
      `${index * 5},${24 - Math.sin(index * 0.29 + time / 3000 + (tone === 'human' ? 0.6 : 0)) * 11}`,
  ).join(' ');
  return (
    <article className={`dr-action-card ${tone}`}>
      <header>
        <strong>{title}</strong>
        <span>{tag}</span>
      </header>
      <div className="dr-action-values">
        {['Δx', 'Δy', 'Δz', 'grip'].map((label, index) => (
          <span key={label}>
            <small>{label}</small>
            <code>
              {values ? `${values[index] >= 0 ? '+' : ''}${values[index].toFixed(3)}` : '—'}
            </code>
          </span>
        ))}
      </div>
      <svg viewBox="0 0 175 48" aria-hidden="true">
        <path d="M0 24H175" className="baseline" />
        {values && <polyline points={points} />}
      </svg>
      <p>{note}</p>
    </article>
  );
}

function mockTrainingConfig(version: DataReturnVersion) {
  return {
    preview_only: true,
    dataset_version: version.name,
    source: 'browser_preview',
    deployment_record_id: version.context.deploymentRecordId,
    contract: 'bimanual_14d/v1',
    action_dim: 14,
    action_horizon: 16,
    fps: 20,
    expert_only: true,
    source_segments: version.sourceSegmentIds.map((id) => id.replace(/^mock-/, '')),
    norm_stats: 'RECOMPUTE_BEFORE_TRAINING',
    training_submitted: false,
  };
}

function MockDialog({
  title,
  subtitle,
  close,
  children,
  footer,
}: {
  title: string;
  subtitle: string;
  close: () => void;
  children: ReactNode;
  footer: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(close);
  closeRef.current = close;
  const id = useId();
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    ref.current?.focus();
    const keys = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeRef.current();
      if (event.key !== 'Tab') return;
      const items = ref.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled)',
      );
      if (!items?.length) {
        event.preventDefault();
        return;
      }
      const first = items[0],
        last = items[items.length - 1];
      if (
        event.shiftKey &&
        (document.activeElement === first || document.activeElement === ref.current)
      ) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', keys);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener('keydown', keys);
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  return (
    <div
      className="dr-modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <div
        className="dr-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={id}
        tabIndex={-1}
        ref={ref}
      >
        <header>
          <div>
            <h3 id={id}>{title}</h3>
            <p>{subtitle}</p>
          </div>
          <button type="button" className="dr-modal-close" aria-label="关闭对话框" onClick={close}>
            <X size={18} />
          </button>
        </header>
        <div className="dr-modal-body">{children}</div>
        <footer>{footer}</footer>
      </div>
    </div>
  );
}

function CameraScene({
  view,
  time,
  takeover,
}: {
  view: 'overhead' | 'wrist';
  time: number;
  takeover: boolean;
}) {
  const id = useId().replace(/:/g, '');
  const shift = Math.sin(time / 3400) * 8;
  const color = takeover ? '#e6b568' : '#67c5b0';
  return (
    <div className={`dr-camera ${view}`}>
      <div className="dr-camera-label">
        <span>
          <i />
          {view === 'overhead' ? 'TOP · 俯视相机' : 'WRIST · 腕部相机'}
        </span>
      </div>
      <svg
        viewBox="0 0 480 300"
        preserveAspectRatio="xMidYMid slice"
        role="img"
        aria-label={view === 'overhead' ? '俯视相机画面' : '腕部相机画面'}
      >
        <defs>
          <linearGradient id={`${id}-bg`} x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="#16343b" />
            <stop offset="1" stopColor="#0c2028" />
          </linearGradient>
          <linearGradient id={`${id}-cloth`} x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="#c1cfc4" />
            <stop offset=".5" stopColor="#9dafaa" />
            <stop offset="1" stopColor="#728e89" />
          </linearGradient>
          <linearGradient id={`${id}-arm`} x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="#d5dedd" />
            <stop offset=".45" stopColor="#88a4a6" />
            <stop offset="1" stopColor="#486a73" />
          </linearGradient>
          <pattern id={`${id}-grid`} width="32" height="32" patternUnits="userSpaceOnUse">
            <path d="M32 0H0V32" stroke="#53847f" strokeOpacity=".12" fill="none" />
          </pattern>
          <pattern id={`${id}-weave`} width="5" height="5" patternUnits="userSpaceOnUse">
            <path d="M0 0V5M0 0H5" stroke="#f1f5ed" strokeOpacity=".15" fill="none" />
          </pattern>
        </defs>
        <rect width="480" height="300" fill={`url(#${id}-bg)`} />
        <rect width="480" height="300" fill={`url(#${id}-grid)`} />
        {view === 'overhead' ? (
          <>
            <path d="M50 61L407 44L448 252L30 270Z" fill="#1d3c43" stroke="#3a6269" />
            <path d="M63 73L397 57L434 239L45 256Z" fill="#26464c" opacity=".8" />
            <ellipse cx="241" cy="200" rx="106" ry="35" fill="#07191f" opacity=".45" />
            <g transform={`translate(${shift} 0)`}>
              <path
                d="M164 111L302 107L330 216L154 224L137 184Z"
                fill={`url(#${id}-cloth)`}
                stroke="#c6d5ca"
                strokeOpacity=".4"
              />
              <path d="M164 111L302 107L330 216L154 224L137 184Z" fill={`url(#${id}-weave)`} />
              <path
                d="M164 111L205 171L154 224M302 107L264 173L330 216M205 171L264 173"
                fill="none"
                stroke="#e1e7dc"
                strokeOpacity=".37"
                strokeWidth="2"
              />
              <path
                d="M145 175L170 180M295 160L316 155"
                stroke={color}
                strokeWidth="3"
                strokeLinecap="round"
              />
            </g>
            <g transform={`rotate(${shift / 2} 89 258)`}>
              <path
                d="M74 275L91 209L142 178"
                fill="none"
                stroke="#09232a"
                strokeWidth="27"
                strokeLinecap="round"
              />
              <path
                d="M74 270L91 209L142 178"
                fill="none"
                stroke={`url(#${id}-arm)`}
                strokeWidth="19"
                strokeLinecap="round"
              />
              <circle cx="91" cy="209" r="15" fill="#4a707a" stroke="#9ab3b8" strokeWidth="3" />
              <path
                d="M135 169L154 174L150 187M135 169L136 184L150 187"
                fill="none"
                stroke="#b4c9cb"
                strokeWidth="6"
                strokeLinejoin="round"
              />
            </g>
            <g transform={`rotate(${-shift / 2} 401 65)`}>
              <path
                d="M413 30L388 107L323 151"
                fill="none"
                stroke="#09232a"
                strokeWidth="27"
                strokeLinecap="round"
              />
              <path
                d="M413 30L388 107L323 151"
                fill="none"
                stroke={`url(#${id}-arm)`}
                strokeWidth="19"
                strokeLinecap="round"
              />
              <circle cx="388" cy="107" r="15" fill="#4a707a" stroke="#9ab3b8" strokeWidth="3" />
              <path
                d="M317 142L307 154L321 165M317 142L332 151L321 165"
                fill="none"
                stroke="#b4c9cb"
                strokeWidth="6"
                strokeLinejoin="round"
              />
            </g>
            <path
              d="M129 91v-8h23M321 83h23v18M344 230v14h-24M149 244h-20v-15"
              fill="none"
              stroke={color}
              strokeOpacity=".65"
            />
            <circle
              cx="241"
              cy="171"
              r="23"
              fill="none"
              stroke={color}
              strokeDasharray="3 6"
              strokeOpacity=".55"
            />
          </>
        ) : (
          <>
            <path d="M-28 190L285 94L520 218V330H-28Z" fill="#28484d" />
            <g transform={`translate(${shift / 2} 0)`}>
              <path d="M62 198L278 105L414 228L343 302L152 300Z" fill={`url(#${id}-cloth)`} />
              <path d="M62 198L278 105L414 228L343 302L152 300Z" fill={`url(#${id}-weave)`} />
              <path
                d="M278 105L244 223L152 300M244 223L414 228"
                fill="none"
                stroke="#d5e1d8"
                strokeOpacity=".4"
                strokeWidth="3"
              />
            </g>
            <path
              d="M189 14L199 76L224 121M293 14L283 76L259 121"
              stroke="#102a33"
              strokeWidth="28"
              fill="none"
              strokeLinecap="round"
            />
            <path
              d="M189 8L199 76L224 121M293 8L283 76L259 121"
              stroke={`url(#${id}-arm)`}
              strokeWidth="19"
              fill="none"
              strokeLinecap="round"
            />
            <path
              d="M218 116L225 149M265 116L257 149"
              stroke="#c4d7d4"
              strokeWidth="9"
              strokeLinecap="round"
            />
            <rect
              x="210"
              y="139"
              width="59"
              height="37"
              rx="5"
              fill="none"
              stroke={color}
              strokeDasharray="4 3"
            />
            <path d="M241 176v17M233 184h16" stroke={color} strokeOpacity=".7" />
          </>
        )}
        <path
          d="M17 45V29h16M447 29h16v16M463 259v16h-16M33 275H17v-16"
          stroke="#84b3ab"
          strokeOpacity=".4"
          fill="none"
        />
        <text x="29" y="281" fontSize="9" fill="#7baba6" fontFamily="monospace">
          FRAME ·{' '}
          {Math.floor(time / 50)
            .toString()
            .padStart(5, '0')}
        </text>
      </svg>
      <div className="dr-camera-meta">
        <span>
          {view === 'overhead' ? 'observation.images.top_head' : 'observation.images.wrist'}
        </span>
        <small>RGB · HWC</small>
      </div>
    </div>
  );
}
