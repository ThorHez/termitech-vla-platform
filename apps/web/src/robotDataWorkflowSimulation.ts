/** One browser-only run owns preparation, capture and its resulting data. */
import {
  createDataReturnSimulation,
  dataReturnReducer,
  type DataReturnAction,
  type DataReturnContext,
  type DataReturnSession,
  type DataReturnSimulation,
} from './dataReturnSimulation';
import {
  createRobotMock,
  robotMockPhaseLabels,
  tickRobotMock,
  transitionRobotMock,
  ROBOT_MOCK_LOG_LIMIT,
  type RobotMockAction,
  type RobotMockState,
} from './robotDeploymentSimulation';

export interface WorkflowStreamGap {
  sessionId: string;
  episodeId: string;
  startMs: number;
  endMs?: number;
}

export interface RobotDataWorkflowState {
  data: DataReturnSimulation;
  robot: RobotMockState;
  /** The same session receives control events and later supplies review data. */
  runtimeSessionId: string | null;
  runtimeGeneration: number;
  droppedFrameCount: number;
  streamGaps: WorkflowStreamGap[];
}

type RuntimeInput = 'start_sdk' | 'start_eval' | 'stream_disconnect' | 'stream_reconnect';
export type RobotDataWorkflowAction =
  | DataReturnAction
  | { type: 'runtime'; action: RuntimeInput; now: number }
  | { type: 'runtime'; action: 'sdk_ready' | 'eval_ready'; generation: number; now: number }
  | { type: 'reset-runtime'; now: number };

const occupiesControl = (session: DataReturnSession) =>
  ['running', 'takeover', 'paused', 'between_episodes'].includes(session.status);
const recording = (session: DataReturnSession) =>
  session.status === 'running' || session.status === 'takeover';
const readyPhases = ['ready', 'running', 'paused'];

export function createRobotDataWorkflow(
  context: DataReturnContext,
  now = Date.now(),
): RobotDataWorkflowState {
  return {
    data: createDataReturnSimulation(context, now),
    robot: createRobotMock(now),
    runtimeSessionId: null,
    runtimeGeneration: 0,
    droppedFrameCount: 0,
    streamGaps: [],
  };
}

export function activeWorkflowSession(state: RobotDataWorkflowState) {
  return (
    state.data.sessions.find(
      (session) => session.id === state.runtimeSessionId && occupiesControl(session),
    ) || state.data.sessions.find(occupiesControl)
  );
}

function workflowCaptureSessionReason(state: RobotDataWorkflowState, sessionId?: string) {
  const session = state.data.sessions.find((item) => item.id === sessionId);
  if (!session) return '请选择一个采集会话。';
  if (session.status !== 'draft') return '该会话已开始，请继续当前会话或新建会话。';
  if (state.data.sessions.some((item) => item.id !== sessionId && occupiesControl(item)))
    return '请先结束当前会话，再开始另一个会话。';
  return undefined;
}

export function workflowStartReason(state: RobotDataWorkflowState, sessionId: string) {
  const reason = workflowCaptureSessionReason(state, sessionId);
  if (reason) return reason;
  if (state.robot.phase !== 'ready') return '请先完成 SDK 与 Eval 的准备。';
  return undefined;
}

/** The next explicit local action is available beside the capture preview. */
export function workflowCaptureEntry(
  state: RobotDataWorkflowState,
  sessionId?: string,
): {
  label: string;
  hint: string;
  disabled: boolean;
  busy: boolean;
  action: 'start_sdk' | 'start_eval' | 'start' | null;
} {
  const reason = workflowCaptureSessionReason(state, sessionId);
  if (reason) return { label: '开始采集', hint: reason, disabled: true, busy: false, action: null };
  switch (state.robot.phase) {
    case 'idle':
    case 'stopped':
      return {
        label: '启动 SDK',
        hint: '先在此启动 SDK，完成后在同一位置初始化 Eval。',
        disabled: false,
        busy: false,
        action: 'start_sdk',
      };
    case 'sdk_starting':
      return {
        label: 'SDK 准备中…',
        hint: 'SDK 正在连接；完成后可在此初始化 Eval。',
        disabled: true,
        busy: true,
        action: null,
      };
    case 'sdk_ready':
      return {
        label: '初始化 Eval',
        hint: 'SDK 已就绪，在此初始化 Eval；完成后即可开始当前会话。',
        disabled: false,
        busy: false,
        action: 'start_eval',
      };
    case 'eval_starting':
      return {
        label: 'Eval 初始化中…',
        hint: 'Eval 正在加载策略；完成后可在此开始采集。',
        disabled: true,
        busy: true,
        action: null,
      };
    case 'ready':
      return {
        label: '开始采集',
        hint: '准备已完成，点击后开始当前会话的采集。',
        disabled: false,
        busy: false,
        action: 'start',
      };
    default:
      return {
        label: '开始采集',
        hint: '运行尚未就绪，请先重置准备。',
        disabled: true,
        busy: false,
        action: null,
      };
  }
}

export function workflowResumeReason(state: RobotDataWorkflowState, sessionId: string) {
  const session = state.data.sessions.find((item) => item.id === sessionId);
  if (!session || !['paused', 'takeover'].includes(session.status)) return '该会话当前不能恢复。';
  if (state.runtimeSessionId !== sessionId) return '当前会话与运行身份不一致，请先结束会话。';
  if (state.data.sessions.some((item) => item.id !== sessionId && occupiesControl(item)))
    return '另一个会话正在使用控制权。';
  if (!['paused', 'running'].includes(state.robot.phase))
    return '运行环境未就绪，请先结束当前会话并重新准备。';
  return undefined;
}

export function workflowNextEpisodeReason(
  state: RobotDataWorkflowState,
  sessionId: string,
  afterEpisodeId: string,
): string | undefined {
  const session = state.data.sessions.find((item) => item.id === sessionId);
  if (!session || session.status !== 'between_episodes') return '请先结束当前 Episode。';
  const previous = session.episodes.at(-1);
  if (!previous || previous.id !== afterEpisodeId || previous.status !== 'sealed')
    return 'Episode 已更新，请从当前会话的最新记录开始下一轮。';
  if (state.runtimeSessionId !== sessionId) return '当前会话与运行身份不一致，请先结束会话。';
  if (state.data.sessions.some((item) => item.id !== sessionId && occupiesControl(item)))
    return '另一个会话正在使用控制权。';
  if (state.robot.phase !== 'paused') return '运行环境未保持就绪，请先结束会话并重新准备。';
  return undefined;
}

export function workflowStatus(state: RobotDataWorkflowState): {
  label: string;
  source: 'policy' | 'human' | 'hold' | 'none';
  ready: boolean;
  activeSessionId: string | null;
} {
  const session = activeWorkflowSession(state);
  const ready = readyPhases.includes(state.robot.phase);
  if (!session)
    return {
      label:
        state.robot.phase === 'ready'
          ? '准备就绪，等待开始运行与采集'
          : state.robot.phase === 'stopped'
            ? '运行已结束'
            : robotMockPhaseLabels[state.robot.phase],
      source: 'none',
      ready,
      activeSessionId: null,
    };
  if (session.status === 'between_episodes')
    return {
      label: '本轮已结束，等待复位场景并开始下一 Episode',
      source: 'hold',
      ready,
      activeSessionId: session.id,
    };
  if (session.status === 'paused')
    return {
      label: '采集已暂停，位姿保持',
      source: 'hold',
      ready,
      activeSessionId: session.id,
    };
  const human = session.mode === 'shadow' || session.status === 'takeover';
  return {
    label: human
      ? session.mode === 'shadow'
        ? '人工示教与影子推理'
        : '人工接管与纠正'
      : '策略运行与采集',
    source: human ? 'human' : 'policy',
    ready,
    activeSessionId: session.id,
  };
}

function appendLog(robot: RobotMockState, message: string): RobotMockState {
  return {
    ...robot,
    logs: [
      ...robot.logs,
      {
        id: (robot.logs.at(-1)?.id || 0) + 1,
        at: robot.lastTickAt,
        source: 'MONITOR' as const,
        message,
      },
    ].slice(-ROBOT_MOCK_LOG_LIMIT),
  };
}

function contextualLog(robot: RobotMockState, message: string): RobotMockState {
  if (!robot.logs.length) return appendLog(robot, message);
  return {
    ...robot,
    logs: robot.logs.map((entry, index) =>
      index !== robot.logs.length - 1 ? entry : { ...entry, message },
    ),
  };
}

function closeGap(state: RobotDataWorkflowState): RobotDataWorkflowState {
  const session = state.data.sessions.find((item) => item.id === state.runtimeSessionId);
  return {
    ...state,
    streamGaps: state.streamGaps.map((gap) =>
      gap.endMs !== undefined || gap.sessionId !== session?.id
        ? gap
        : {
            ...gap,
            endMs:
              session.episodes.find((episode) => episode.id === gap.episodeId)?.durationMs ||
              gap.startMs,
          },
    ),
  };
}

/** Capture time advances during an outage; missing image frames stay explicit. */
function advance(state: RobotDataWorkflowState, now: number): RobotDataWorkflowState {
  const before = activeWorkflowSession(state);
  const data = dataReturnReducer(state.data, { type: 'tick', now });
  const after = before && data.sessions.find((session) => session.id === before.id);
  const missing =
    !state.robot.streamConnected && before && after && recording(before)
      ? Math.max(0, after.frames - before.frames)
      : 0;
  const robot = tickRobotMock(state.robot, now);
  return {
    ...state,
    data,
    robot,
    droppedFrameCount: state.droppedFrameCount + missing,
  };
}

/** Split the label at a sensor transition rather than blessing a mixed span. */
function splitSensorBoundary(state: RobotDataWorkflowState): RobotDataWorkflowState {
  const session = activeWorkflowSession(state);
  if (!session) return state;
  const episode = session.episodes.at(-1);
  const last = episode?.segments.at(-1);
  if (!episode || episode.status !== 'recording' || !last) return state;
  const expertValid = last.source === 'human' && state.robot.streamConnected;
  const piece = {
    ...last,
    id: `${episode.id}-segment-${episode.segments.length + 1}`,
    startMs: episode.durationMs,
    endMs: episode.durationMs,
    startFrame: episode.frames,
    endFrame: episode.frames,
    expertValid,
    review: 'pending' as const,
    annotations: [],
    note: state.robot.streamConnected
      ? '画面恢复，重新开始有效片段。'
      : '画面断流，此片段不作为有效专家标签。',
  };
  return {
    ...state,
    data: {
      ...state.data,
      sessions: state.data.sessions.map((item) =>
        item.id !== session.id
          ? item
          : {
              ...item,
              episodes: item.episodes.map((part) =>
                part.id !== episode.id ? part : { ...part, segments: [...part.segments, piece] },
              ),
            },
      ),
    },
  };
}

function invalidateLatestLabel(state: RobotDataWorkflowState): RobotDataWorkflowState {
  if (state.robot.streamConnected) return state;
  const session = activeWorkflowSession(state);
  const episode = session?.episodes.at(-1);
  if (!session || !episode || episode.status !== 'recording') return state;
  return {
    ...state,
    data: {
      ...state.data,
      sessions: state.data.sessions.map((item) =>
        item.id !== session.id
          ? item
          : {
              ...item,
              episodes: item.episodes.map((part) =>
                part.id !== episode.id
                  ? part
                  : {
                      ...part,
                      segments: part.segments.map((segment, index) =>
                        index !== part.segments.length - 1
                          ? segment
                          : {
                              ...segment,
                              expertValid: false,
                              note: '画面断流，此片段不作为有效专家标签。',
                            },
                      ),
                    },
              ),
            },
      ),
    },
  };
}

export function robotDataWorkflowReducer(
  state: RobotDataWorkflowState,
  action: RobotDataWorkflowAction,
): RobotDataWorkflowState {
  if ('now' in action) {
    const minimum = Math.max(
      state.robot.lastTickAt,
      ...state.data.sessions.map((session) => session.lastUpdatedAt),
    );
    action = {
      ...action,
      now: Math.max(minimum, Number.isFinite(action.now) ? action.now : minimum),
    };
  }
  if (action.type === 'reset') {
    const next = createRobotDataWorkflow(state.data.context, action.now);
    return { ...next, runtimeGeneration: state.runtimeGeneration + 1 };
  }
  if (action.type === 'reset-runtime') {
    let next = advance(state, action.now);
    const session = activeWorkflowSession(next);
    if (session)
      next = {
        ...next,
        data: dataReturnReducer(next.data, {
          type: 'finish',
          sessionId: session.id,
          now: action.now,
        }),
      };
    next = closeGap(next);
    return {
      ...next,
      robot: appendLog(
        createRobotMock(next.robot.lastTickAt),
        '运行准备已重置，既有会话和数据版本保留。',
      ),
      runtimeSessionId: null,
      runtimeGeneration: state.runtimeGeneration + 1,
    };
  }
  if (action.type === 'runtime') {
    const completion = action.action === 'sdk_ready' || action.action === 'eval_ready';
    if (completion && (!('generation' in action) || action.generation !== state.runtimeGeneration))
      return state;
    if (action.action === 'start_sdk' && activeWorkflowSession(state)) return state;
    let base = state;
    if (action.action === 'start_sdk' && state.robot.phase === 'stopped') {
      base = {
        ...state,
        robot: createRobotMock(state.robot.lastTickAt),
        runtimeSessionId: null,
      };
    }
    const candidate = transitionRobotMock(base.robot, action.action, action.now);
    if (candidate === base.robot) return state;
    let next = advance(base, action.now);
    next = {
      ...next,
      robot: candidate,
      runtimeGeneration:
        action.action === 'start_sdk' ? state.runtimeGeneration + 1 : state.runtimeGeneration,
    };
    if (action.action === 'stream_disconnect' || action.action === 'stream_reconnect') {
      next = splitSensorBoundary(next);
      const session = activeWorkflowSession(next);
      if (
        action.action === 'stream_disconnect' &&
        session?.episodes.at(-1)?.status === 'recording'
      ) {
        const episode = session.episodes.at(-1)!;
        next = {
          ...next,
          streamGaps: [
            ...next.streamGaps,
            { sessionId: session.id, episodeId: episode.id, startMs: episode.durationMs },
          ],
        };
      } else if (action.action === 'stream_reconnect') next = closeGap(next);
    }
    return next;
  }
  if (action.type === 'tick') return advance(state, action.now);
  if (
    action.type === 'create-session' ||
    action.type === 'review' ||
    action.type === 'annotate' ||
    action.type === 'remove-annotation' ||
    action.type === 'publish'
  ) {
    const data = dataReturnReducer(state.data, action);
    return data === state.data ? state : { ...state, data };
  }
  if (action.type === 'start' && workflowStartReason(state, action.sessionId)) return state;
  if (action.type === 'resume' && workflowResumeReason(state, action.sessionId)) return state;
  if (
    action.type === 'next-episode' &&
    workflowNextEpisodeReason(state, action.sessionId, action.afterEpisodeId)
  )
    return state;
  if (action.type !== 'start' && action.sessionId !== state.runtimeSessionId) return state;
  const before = state.data.sessions.find((session) => session.id === action.sessionId);
  if (!before) return state;
  const probe = dataReturnReducer(state.data, action);
  if (probe === state.data) return state;
  let next = advance(state, action.now);
  next = {
    ...next,
    data: dataReturnReducer(next.data, action),
    runtimeSessionId: action.sessionId,
  };
  let robotAction: RobotMockAction | undefined;
  if (action.type === 'start' || action.type === 'resume' || action.type === 'next-episode')
    robotAction = 'start';
  else if (action.type === 'pause' || action.type === 'finish-episode') robotAction = 'pause';
  else if (action.type === 'finish') robotAction = 'stop';
  if (robotAction)
    next = { ...next, robot: transitionRobotMock(next.robot, robotAction, action.now) };
  const sessionLabel = `会话 ${before.name}`;
  if (action.type === 'takeover')
    next = {
      ...next,
      robot: appendLog(next.robot, `${sessionLabel} 已由人工接管，继续录制纠正片段。`),
    };
  else if (action.type === 'finish-episode')
    next = {
      ...closeGap(next),
      robot: appendLog(
        next.robot,
        `${sessionLabel} 的 ${before.episodes.at(-1)?.name} 已封存，结果：${action.outcome}；等待复位场景并开始下一轮。`,
      ),
    };
  else if (action.type === 'next-episode')
    next = {
      ...next,
      robot: contextualLog(
        next.robot,
        `${sessionLabel} 开始 ${next.data.sessions.find((item) => item.id === before.id)?.episodes.at(-1)?.name}；复用 SDK/Eval，采集帧从零开始。`,
      ),
    };
  else if (action.type === 'finish')
    next = {
      ...closeGap(next),
      robot: contextualLog(
        next.robot,
        `${sessionLabel} 已封存，SDK/Eval 已停止，可继续审核并发布版本。`,
      ),
    };
  else if (action.type === 'pause')
    next = {
      ...next,
      robot: contextualLog(next.robot, `${sessionLabel} 已暂停采集，位姿保持。`),
    };
  else if (action.type === 'start' || action.type === 'resume')
    next = {
      ...next,
      robot: contextualLog(
        next.robot,
        `${sessionLabel} ${action.type === 'start' ? '开始' : '恢复'}${
          before.mode === 'shadow' ? '人工示教，策略仅提供影子建议' : '策略运行与采集'
        }。`,
      ),
    };
  if ((action.type === 'start' || action.type === 'next-episode') && !next.robot.streamConnected) {
    const episode = next.data.sessions.find((session) => session.id === before.id)?.episodes.at(-1);
    if (episode)
      next = {
        ...next,
        streamGaps: [
          ...next.streamGaps,
          { sessionId: before.id, episodeId: episode.id, startMs: episode.durationMs },
        ],
      };
  }
  return invalidateLatestLabel(next);
}
