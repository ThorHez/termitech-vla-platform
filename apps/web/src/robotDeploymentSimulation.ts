/** Browser-only robot deployment demonstration. No network or hardware access. */
export type RobotMockPhase =
  | 'idle'
  | 'sdk_starting'
  | 'sdk_ready'
  | 'eval_starting'
  | 'ready'
  | 'running'
  | 'paused'
  | 'stopped';

export type RobotMockAction =
  | 'start_sdk'
  | 'sdk_ready'
  | 'start_eval'
  | 'eval_ready'
  | 'start'
  | 'pause'
  | 'stop'
  | 'reset'
  | 'stream_disconnect'
  | 'stream_reconnect';

export interface RobotMockTelemetry {
  /** Six simulated joint angles, in degrees. */
  joints: number[];
  /** xyz in meters followed by a rotation vector in radians. */
  tcp: number[];
  gripperMm: number;
  inferenceMs: number | null;
  fps: number;
  historyFrames: number;
  submittedSteps: number;
  feedbackAt: number | null;
}

export interface RobotMockLog {
  id: number;
  at: number;
  source: 'SDK' | 'EVAL' | 'MONITOR';
  message: string;
}

export interface RobotMockState {
  phase: RobotMockPhase;
  phaseStartedAt: number;
  streamConnected: boolean;
  lastTickAt: number;
  frameAt: number | null;
  /** Render mock camera movement from this value; it freezes on video loss. */
  frameMotionMs: number;
  /** Simulated execution time; video loss does not stop the executor. */
  motionMs: number;
  telemetry: RobotMockTelemetry;
  logs: RobotMockLog[];
}

export const robotMockPhaseLabels: Record<RobotMockPhase, string> = {
  idle: '等待启动 SDK',
  sdk_starting: 'SDK 启动中',
  sdk_ready: 'SDK 已就绪',
  eval_starting: 'Eval 初始化中',
  ready: '等待开始执行',
  running: '策略执行中',
  paused: '策略已暂停',
  stopped: '模拟测试已结束',
};

export const ROBOT_MOCK_LOG_LIMIT = 48;

const cameraActive = (phase: RobotMockPhase) =>
  phase === 'ready' || phase === 'running' || phase === 'paused';
const feedbackActive = (phase: RobotMockPhase) =>
  phase === 'sdk_ready' || phase === 'eval_starting' || cameraActive(phase);
const timeAt = (state: RobotMockState, nowMs: number) =>
  Number.isFinite(nowMs) ? Math.max(state.lastTickAt, nowMs) : state.lastTickAt;
const rounded = (value: number, digits = 3) => Number(value.toFixed(digits));

export function createRobotMock(nowMs = 0): RobotMockState {
  const now = Number.isFinite(nowMs) ? nowMs : 0;
  return {
    phase: 'idle',
    phaseStartedAt: now,
    streamConnected: true,
    lastTickAt: now,
    frameAt: null,
    frameMotionMs: 0,
    motionMs: 0,
    telemetry: {
      joints: [0, -24, 48, 0, 68, 0],
      tcp: [0.36, 0, 0.23, 3.142, 0, 0],
      gripperMm: 68,
      inferenceMs: null,
      fps: 0,
      historyFrames: 0,
      submittedSteps: 0,
      feedbackAt: null,
    },
    logs: [],
  };
}

/** Advancing the demo clock never starts components or changes lifecycle stages. */
export function tickRobotMock(state: RobotMockState, nowMs: number): RobotMockState {
  const now = timeAt(state, nowMs);
  const moving = state.phase === 'running';
  const motionMs = state.motionMs + (moving ? now - state.lastTickAt : 0);
  const t = motionMs / 1000;
  const video = cameraActive(state.phase) && state.streamConnected;
  const telemetry: RobotMockTelemetry = {
    ...state.telemetry,
    feedbackAt: feedbackActive(state.phase) ? now : state.telemetry.feedbackAt,
    fps: video ? 12 : 0,
  };
  if (state.phase === 'eval_starting') {
    telemetry.historyFrames = Math.min(240, Math.floor((now - state.phaseStartedAt) / 7));
  }
  if (moving) {
    telemetry.joints = [
      12 * Math.sin(t * 0.6),
      -24 + 5 * Math.sin(t * 0.5),
      48 + 8 * Math.sin(t * 0.7),
      4 * Math.sin(t * 0.4),
      68 + 6 * Math.sin(t * 0.5),
      8 * Math.sin(t * 0.6),
    ].map((value) => rounded(value, 1));
    telemetry.tcp = [
      0.36 + 0.035 * Math.sin(t * 0.6),
      0.025 * Math.sin(t * 0.4),
      0.23 + 0.018 * Math.sin(t * 0.7),
      Math.PI,
      0.04 * Math.sin(t * 0.4),
      0.06 * Math.sin(t * 0.6),
    ].map((value) => rounded(value));
    telemetry.gripperMm = rounded(48 + 20 * Math.cos(t * 0.8), 1);
    telemetry.inferenceMs = rounded(86 + 9 * Math.sin(t * 0.7), 1);
    telemetry.submittedSteps = Math.floor(motionMs / 400) * 8;
  }
  return {
    ...state,
    lastTickAt: now,
    motionMs,
    frameAt: video ? now : state.frameAt,
    frameMotionMs: video ? motionMs : state.frameMotionMs,
    telemetry,
  };
}

function log(
  state: RobotMockState,
  source: RobotMockLog['source'],
  message: string,
): RobotMockState {
  const entry: RobotMockLog = {
    id: (state.logs.at(-1)?.id || 0) + 1,
    at: state.lastTickAt,
    source,
    message,
  };
  return { ...state, logs: [...state.logs, entry].slice(-ROBOT_MOCK_LOG_LIMIT) };
}

/** Invalid or duplicate commands are no-ops, including delayed startup callbacks. */
export function transitionRobotMock(
  state: RobotMockState,
  action: RobotMockAction,
  nowMs: number,
): RobotMockState {
  if (action === 'reset') return createRobotMock(timeAt(state, nowMs));
  const phase = state.phase;
  const accepted =
    (action === 'start_sdk' && phase === 'idle') ||
    (action === 'sdk_ready' && phase === 'sdk_starting') ||
    (action === 'start_eval' && phase === 'sdk_ready') ||
    (action === 'eval_ready' && phase === 'eval_starting') ||
    (action === 'start' && (phase === 'ready' || phase === 'paused')) ||
    (action === 'pause' && phase === 'running') ||
    (action === 'stop' && phase !== 'idle' && phase !== 'stopped') ||
    (action === 'stream_disconnect' && cameraActive(phase) && state.streamConnected) ||
    (action === 'stream_reconnect' && cameraActive(phase) && !state.streamConnected);
  if (!accepted) return state;
  let next = tickRobotMock(state, nowMs);
  if (action === 'stream_disconnect' || action === 'stream_reconnect') {
    next = tickRobotMock(
      { ...next, streamConnected: action === 'stream_reconnect' },
      next.lastTickAt,
    );
    return log(
      next,
      'MONITOR',
      action === 'stream_disconnect'
        ? '[MOCK] 模拟画面断流；执行状态不变，末帧已冻结。'
        : '[MOCK] 模拟画面恢复，显示最新帧。',
    );
  }
  const transitions: Partial<
    Record<RobotMockAction, [RobotMockPhase, RobotMockLog['source'], string]>
  > = {
    start_sdk: ['sdk_starting', 'SDK', '[MOCK] 启动 arx5-sdk；未连接 CAN 或机械臂。'],
    sdk_ready: ['sdk_ready', 'SDK', '[MOCK] SDK 模拟就绪，反馈通道已准备。'],
    start_eval: ['eval_starting', 'EVAL', '[MOCK] 初始化 Eval、相机及历史观测。'],
    eval_ready: ['ready', 'EVAL', '[MOCK] 初始化完成，模拟观测 241/241；等待手动开始。'],
    start: [
      'running',
      'EVAL',
      phase === 'paused' ? '[MOCK] 恢复策略模拟执行。' : '[MOCK] 开始策略模拟执行。',
    ],
    pause: ['paused', 'EVAL', '[MOCK] 策略已暂停，模拟位姿保持；相机预览继续。'],
    stop: ['stopped', 'EVAL', '[MOCK] 结束测试，模拟 SDK / Eval 已停止；末次反馈保留。'],
  };
  const transition = transitions[action];
  if (!transition) return state;
  next = {
    ...next,
    phase: transition[0],
    phaseStartedAt: next.lastTickAt,
    telemetry: {
      ...next.telemetry,
      historyFrames: action === 'eval_ready' ? 241 : next.telemetry.historyFrames,
    },
  };
  return log(tickRobotMock(next, next.lastTickAt), transition[1], transition[2]);
}
