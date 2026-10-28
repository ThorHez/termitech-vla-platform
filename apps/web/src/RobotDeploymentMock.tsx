import { useEffect, useId, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Camera,
  Check,
  Circle,
  FlaskConical,
  Hand,
  LoaderCircle,
  Maximize2,
  Pause,
  Play,
  Radio,
  RefreshCw,
  RotateCcw,
  Server,
  Square,
  Terminal,
  Video,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import { Panel } from './components';
import {
  createRobotMock,
  robotMockPhaseLabels,
  tickRobotMock,
  transitionRobotMock,
  type RobotMockAction,
  type RobotMockPhase,
} from './robotDeploymentSimulation';
import './robot-deployment-mock.css';

type CameraView = 'wrist' | 'overhead';
type LogSource = 'ALL' | 'SDK' | 'EVAL' | 'MONITOR';

const cameraLabels: Record<CameraView, string> = {
  wrist: '腕部相机',
  overhead: '俯视相机',
};
const activePhases: RobotMockPhase[] = ['ready', 'running', 'paused'];
const sdkPhases: RobotMockPhase[] = ['sdk_ready', 'eval_starting', ...activePhases];
const clock = (value: number | null) =>
  value === null ? '—' : new Date(value).toLocaleTimeString('zh-CN', { hour12: false });

export default function RobotDeploymentMock({
  deploymentId,
  machineName = '模拟部署机器',
  policyLabel = 'OpenPI · serve-policy',
  onClose,
}: {
  deploymentId?: string;
  machineName?: string;
  policyLabel?: string;
  onClose?: () => void;
}) {
  const [state, setState] = useState(() => createRobotMock(Date.now()));
  const [camera, setCamera] = useState<CameraView>('wrist');
  const [logSource, setLogSource] = useState<LogSource>('ALL');
  const [largeView, setLargeView] = useState(false);
  const perform = (action: RobotMockAction) => {
    setState((previous) => transitionRobotMock(previous, action, Date.now()));
  };
  useEffect(() => {
    if (state.phase === 'idle' || state.phase === 'stopped') return;
    const timer = window.setInterval(() => {
      setState((previous) => tickRobotMock(previous, Date.now()));
    }, 200);
    return () => window.clearInterval(timer);
  }, [state.phase]);
  useEffect(() => {
    const action =
      state.phase === 'sdk_starting'
        ? 'sdk_ready'
        : state.phase === 'eval_starting'
          ? 'eval_ready'
          : undefined;
    if (!action) return;
    const timer = window.setTimeout(
      () => setState((previous) => transitionRobotMock(previous, action, Date.now())),
      state.phase === 'sdk_starting' ? 1100 : 1700,
    );
    return () => window.clearTimeout(timer);
  }, [state.phase]);

  const evalActive = activePhases.includes(state.phase);
  const sdkReady = sdkPhases.includes(state.phase);
  const hasFrame = state.frameAt !== null;
  const live = evalActive && state.streamConnected;
  const feedbackLive = sdkReady;
  const hasFeedback = telemetryAvailable(state.telemetry.feedbackAt);
  const historyAvailable = state.phase === 'eval_starting' || hasFrame;
  const moving = state.phase === 'running';
  const phaseBusy = state.phase === 'sdk_starting' || state.phase === 'eval_starting';
  const telemetry = state.telemetry;
  const logs = state.logs.filter((entry) => logSource === 'ALL' || entry.source === logSource);
  const steps = [
    { title: '策略服务', detail: '已就绪（模拟前提）', done: true, current: false },
    {
      title: '启动 SDK',
      detail: 'arx5-sdk',
      done: sdkReady,
      current: state.phase === 'idle' || state.phase === 'sdk_starting',
    },
    {
      title: '初始化 Eval',
      detail: 'umi-arx-deploy',
      done: evalActive,
      current: state.phase === 'sdk_ready' || state.phase === 'eval_starting',
    },
    {
      title: '真机测试',
      detail: '画面与状态',
      done: state.phase === 'running' || state.phase === 'paused',
      current: evalActive,
    },
  ];
  const stageHint: Record<RobotMockPhase, string> = {
    idle: '策略服务已模拟就绪，接下来启动机器人 SDK。',
    sdk_starting: '正在模拟连接 CAN 与读取机械臂反馈…',
    sdk_ready: 'SDK 已模拟就绪，接下来初始化位姿、相机与 Eval。',
    eval_starting: '正在模拟初始化位姿、采集历史帧与策略预热…',
    ready: 'Eval 已模拟就绪，可查看画面并开始策略执行。',
    running: '策略正在模拟执行，画面与反馈持续更新。',
    paused: '策略已模拟暂停，机械臂保持当前位姿，画面继续更新。',
    stopped: '模拟测试已结束，画面和反馈保留在最后一次更新。',
  };

  return (
    <Panel
      title="真机测试流程"
      subtitle="从策略服务继续，体验 SDK、Eval 与实时监控。"
      className="robot-mock"
      action={
        <div className="robot-mock-heading-actions">
          <span className="robot-mock-badge">
            <FlaskConical size={12} /> MOCK
          </span>
          {onClose && (
            <button className="button ghost small" aria-label="关闭模拟流程" onClick={onClose}>
              <X size={16} />
            </button>
          )}
        </div>
      }
    >
      <div className="robot-mock-notice">
        <FlaskConical size={16} />
        <span>
          模拟演示 · 策略服务按已就绪演示；画面和状态均为 mock，不连接硬件，也不启停真实服务。
        </span>
      </div>
      <ol className="robot-mock-steps" aria-label="模拟真机部署步骤">
        {steps.map((step, index) => (
          <li
            key={step.title}
            className={`${step.done ? 'done' : ''} ${step.current ? 'current' : ''}`}
          >
            <span className="robot-mock-step-number">
              {step.done ? <Check size={15} /> : index + 1}
            </span>
            <span>
              <strong>{step.title}</strong>
              <small>{step.detail}</small>
            </span>
            {index < steps.length - 1 && <ArrowRight className="robot-mock-step-arrow" size={14} />}
          </li>
        ))}
      </ol>
      <div className="robot-mock-session">
        <div className="robot-mock-session-copy">
          <span className={`robot-mock-phase ${moving ? 'running' : ''}`} role="status">
            {phaseBusy ? <LoaderCircle size={14} className="spin" /> : <Activity size={14} />}
            {robotMockPhaseLabels[state.phase]}
          </span>
          <p>{stageHint[state.phase]}</p>
          <small>
            <Server size={12} /> {machineName} <span>·</span> {policyLabel}
          </small>
          {deploymentId && <small className="robot-mock-source">关联部署：{deploymentId}</small>}
        </div>
        <div className="robot-mock-controls" aria-label="模拟流程操作">
          <button
            className="button primary"
            disabled={state.phase !== 'idle'}
            onClick={() => perform('start_sdk')}
          >
            {state.phase === 'sdk_starting' ? (
              <LoaderCircle size={14} className="spin" />
            ) : (
              <Server size={14} />
            )}
            启动 SDK（模拟）
          </button>
          <button
            className="button primary"
            disabled={state.phase !== 'sdk_ready'}
            onClick={() => perform('start_eval')}
          >
            {state.phase === 'eval_starting' ? (
              <LoaderCircle size={14} className="spin" />
            ) : (
              <Video size={14} />
            )}
            初始化并启动 Eval
          </button>
          <button
            className="button primary"
            disabled={!['ready', 'running', 'paused'].includes(state.phase)}
            onClick={() => perform(moving ? 'pause' : 'start')}
          >
            {moving ? <Pause size={14} /> : <Play size={14} />}
            {moving ? '暂停策略' : state.phase === 'paused' ? '继续执行' : '开始执行'}
          </button>
          <button
            className="button outline danger-text"
            disabled={state.phase === 'idle' || state.phase === 'stopped'}
            onClick={() => perform('stop')}
          >
            <Square size={13} />
            结束测试
          </button>
          <button
            className="button ghost small"
            disabled={state.phase === 'idle'}
            onClick={() => perform('reset')}
          >
            <RotateCcw size={13} />
            重新演示
          </button>
        </div>
      </div>
      <div className={`robot-mock-monitor ${largeView ? 'large-view' : ''}`}>
        <section className="robot-mock-camera" aria-label="模拟相机监控">
          <div className="robot-mock-camera-toolbar">
            <span>
              <Camera size={14} />
              <strong>Eval 视觉</strong>
              <span className="robot-mock-tiny-tag">模拟画面</span>
            </span>
            <button
              className="robot-mock-camera-size"
              aria-label={largeView ? '恢复画面大小' : '放大模拟画面'}
              aria-pressed={largeView}
              onClick={() => setLargeView((value) => !value)}
            >
              <Maximize2 size={14} />
            </button>
          </div>
          <div className="robot-mock-camera-tabs" role="group" aria-label="模拟相机选择">
            {(['wrist', 'overhead'] as const).map((value) => (
              <button
                key={value}
                className={camera === value ? 'selected' : ''}
                aria-pressed={camera === value}
                onClick={() => setCamera(value)}
              >
                <Video size={12} />
                {cameraLabels[value]}
                <span>CAM {value === 'wrist' ? '01' : '02'}</span>
              </button>
            ))}
          </div>
          <div className={`robot-mock-camera-frame ${!live ? 'not-live' : ''}`}>
            <CameraScene view={camera} motionMs={state.frameMotionMs} available={hasFrame} />
            <div className="robot-mock-frame-hud">
              <span className={`robot-mock-live-tag ${live ? 'online' : ''}`}>
                <Circle size={7} fill="currentColor" />
                {live
                  ? 'MOCK LIVE'
                  : state.phase === 'stopped'
                    ? '已结束'
                    : hasFrame
                      ? '画面中断'
                      : '等待画面'}
              </span>
              <span>
                {live ? `${telemetry.fps} FPS` : '— FPS'}
                <span> · </span>640 × 400
              </span>
            </div>
            <div className="robot-mock-frame-bottom">
              <span>
                {cameraLabels[camera]}
                <small>FRAME {clock(state.frameAt)}</small>
              </span>
              <strong>MOCK</strong>
            </div>
            {!hasFrame && (
              <div className="robot-mock-frame-empty">
                <Camera size={32} strokeWidth={1.2} />
                <strong>等待 Eval 画面</strong>
                <span>初始化并启动 Eval 后显示模拟相机画面</span>
              </div>
            )}
            {hasFrame && !state.streamConnected && evalActive && (
              <div className="robot-mock-frame-interrupted" role="status">
                <WifiOff size={20} />
                <strong>模拟画面中断</strong>
                <span>已冻结最后一帧 · {clock(state.frameAt)}</span>
              </div>
            )}
          </div>
          <div className="robot-mock-camera-footer">
            <span className={live ? 'connected' : ''}>
              {live ? <Wifi size={12} /> : <WifiOff size={12} />}
              {live
                ? '模拟监控通道在线'
                : state.phase === 'stopped'
                  ? '模拟监控已关闭'
                  : hasFrame
                    ? '模拟监控断开'
                    : '等待模拟监控'}
            </span>
            <button
              disabled={!evalActive}
              onClick={() =>
                perform(state.streamConnected ? 'stream_disconnect' : 'stream_reconnect')
              }
            >
              {state.streamConnected ? <WifiOff size={12} /> : <RefreshCw size={12} />}
              {state.streamConnected ? '模拟断流' : '恢复画面'}
            </button>
          </div>
        </section>
        <section className="robot-mock-feedback" aria-label="模拟机械臂状态">
          <div className="robot-mock-feedback-heading">
            <h3>
              <Activity size={15} />
              机械臂反馈
            </h3>
            <span className={feedbackLive ? 'connected' : ''}>
              {feedbackLive ? '模拟在线' : hasFeedback ? '反馈已过期' : '未连接'}
            </span>
          </div>
          <div className="robot-mock-components">
            <ComponentStatus label="Policy" ready busy={false} detail="模拟就绪" />
            <ComponentStatus
              label="SDK"
              ready={sdkReady}
              busy={state.phase === 'sdk_starting'}
              detail={sdkReady ? '模拟就绪' : state.phase === 'stopped' ? '已停止' : '待启动'}
            />
            <ComponentStatus
              label="Eval"
              ready={evalActive}
              busy={state.phase === 'eval_starting'}
              detail={
                evalActive
                  ? moving
                    ? '模拟执行'
                    : state.phase === 'paused'
                      ? '模拟暂停'
                      : '模拟就绪'
                  : state.phase === 'stopped'
                    ? '已停止'
                    : '待启动'
              }
            />
          </div>
          <div className="robot-mock-telemetry">
            <div>
              <span>
                <Hand size={12} />
                夹爪开合
              </span>
              <strong>
                {hasFeedback ? telemetry.gripperMm.toFixed(1) : '—'}
                <small>mm</small>
              </strong>
            </div>
            <div>
              <span>
                <Activity size={12} />
                推理耗时
              </span>
              <strong>
                {hasFrame && telemetry.inferenceMs !== null
                  ? telemetry.inferenceMs.toFixed(0)
                  : '—'}
                <small>ms</small>
              </strong>
            </div>
          </div>
          <div className="robot-mock-pose">
            <h4>
              TCP 实际位姿 <span>模拟反馈</span>
            </h4>
            <div className="robot-mock-pose-grid">
              {['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'].map((label, index) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{hasFeedback ? telemetry.tcp[index].toFixed(3) : '—'}</strong>
                  <small>{index < 3 ? 'm' : 'rad'}</small>
                </div>
              ))}
            </div>
          </div>
          <div className="robot-mock-joints">
            <h4>
              关节位置 <span>°</span>
            </h4>
            <div>
              {telemetry.joints.map((value, index) => (
                <div key={index}>
                  <span>J{index + 1}</span>
                  <strong>{hasFeedback ? value.toFixed(1) : '—'}</strong>
                </div>
              ))}
            </div>
          </div>
          <div className="robot-mock-history">
            <span>
              历史观测<span>{historyAvailable ? telemetry.historyFrames : 0} / 241 帧</span>
            </span>
            <div>
              <span
                style={{
                  width: `${historyAvailable ? (telemetry.historyFrames / 241) * 100 : 0}%`,
                }}
              />
            </div>
            <small>
              {hasFrame
                ? `${telemetry.submittedSteps} 步模拟动作已提交`
                : '等待 Eval 初始化与历史采集'}
            </small>
          </div>
          <div className="robot-mock-feedback-time">
            <Radio size={11} />
            反馈更新 {clock(telemetry.feedbackAt)}
            {hasFeedback && !feedbackLive && ' · 数据已停止更新'}
          </div>
        </section>
      </div>
      <section className="robot-mock-logs" aria-label="模拟运行日志">
        <div className="robot-mock-log-heading">
          <h3>
            <Terminal size={14} />
            运行日志<span>MOCK</span>
          </h3>
          <div role="group" aria-label="模拟日志来源">
            {(['ALL', 'SDK', 'EVAL', 'MONITOR'] as const).map((source) => (
              <button
                key={source}
                className={logSource === source ? 'selected' : ''}
                aria-pressed={logSource === source}
                onClick={() => setLogSource(source)}
              >
                {source === 'ALL' ? '全部' : source === 'MONITOR' ? '监控' : source}
              </button>
            ))}
          </div>
        </div>
        <div className="robot-mock-log-content" tabIndex={0}>
          {logs.length ? (
            logs
              .slice()
              .reverse()
              .map((entry) => (
                <div key={entry.id}>
                  <time>{clock(entry.at)}</time>
                  <span className={`source-${entry.source.toLowerCase()}`}>[{entry.source}]</span>
                  <span>{entry.message}</span>
                </div>
              ))
          ) : (
            <p>等待模拟操作，日志将在这里显示。</p>
          )}
        </div>
      </section>
    </Panel>
  );
}

function telemetryAvailable(value: number | null) {
  return value !== null;
}

function ComponentStatus({
  label,
  ready,
  busy,
  detail,
}: {
  label: string;
  ready: boolean;
  busy: boolean;
  detail: string;
}) {
  return (
    <div className={ready ? 'ready' : busy ? 'busy' : ''}>
      {busy ? (
        <LoaderCircle size={11} className="spin" />
      ) : (
        <span className="robot-mock-component-dot" />
      )}
      <strong>{label}</strong>
      <small>{busy ? '模拟启动中' : detail}</small>
    </div>
  );
}

function CameraScene({
  view,
  motionMs,
  available,
}: {
  view: CameraView;
  motionMs: number;
  available: boolean;
}) {
  const id = useId().replace(/:/g, '');
  const motion = motionMs / 1000;
  const dx = Math.sin(motion * 0.65) * 40;
  const dy = Math.sin(motion * 0.4) * 14;
  const opening = 15 + (48 + 20 * Math.cos(motion * 0.8)) * 0.28;
  const gridId = `${id}-grid`;
  const tableId = `${id}-table`;
  const metalId = `${id}-metal`;
  const matId = `${id}-mat`;
  return (
    <svg
      className={`robot-mock-scene ${!available ? 'waiting' : ''}`}
      viewBox="0 0 640 400"
      role="img"
      aria-label={`${cameraLabels[view]}模拟画面`}
    >
      <defs>
        <linearGradient id={tableId} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#243d43" />
          <stop offset="1" stopColor="#172c34" />
        </linearGradient>
        <linearGradient id={metalId} x1="0" y1="0" x2="1" y2="0">
          <stop stopColor="#5f7881" />
          <stop offset=".4" stopColor="#c9d7d8" />
          <stop offset=".65" stopColor="#9eafb5" />
          <stop offset="1" stopColor="#546e78" />
        </linearGradient>
        <linearGradient id={matId} x1="0" y1="0" x2="0" y2="1">
          <stop stopColor="#60847d" />
          <stop offset="1" stopColor="#385f5e" />
        </linearGradient>
        <pattern id={gridId} width="32" height="32" patternUnits="userSpaceOnUse">
          <path d="M32 0H0V32" fill="none" stroke="#a0bfba" strokeOpacity=".055" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="640" height="400" fill={`url(#${tableId})`} />
      {view === 'wrist' ? (
        <>
          <path d="M0 101L640 114V400H0Z" fill="#6e8280" />
          <path d="M0 105L640 118V400H0Z" fill={`url(#${gridId})`} />
          <path d="M112 150L523 157L604 400H31Z" fill="#192d32" opacity=".3" />
          <path d="M120 143L516 150L586 400H43Z" fill={`url(#${matId})`} />
          <path
            d="M138 161L500 168L558 387H70Z"
            fill="none"
            stroke="#c9d8c9"
            strokeWidth="1"
            strokeOpacity=".25"
          />
          <path d="M0 109L640 122" stroke="#adbdba" strokeOpacity=".2" strokeWidth="2" />
          <g transform={`translate(${dx * -0.35} ${dy * -0.35})`}>
            <ellipse cx="374" cy="307" rx="52" ry="13" fill="#0a1c23" opacity=".3" />
            <path d="M348 259L394 251L407 298L357 309Z" fill="#b95d3a" />
            <path d="M348 259L365 246L411 239L394 251Z" fill="#ed9d68" />
            <path d="M394 251L411 239L421 285L407 298Z" fill="#d5784a" />
            <path d="M352 262L389 255" stroke="#f1b58c" strokeWidth="2" strokeOpacity=".6" />
            <path d="M204 311L253 304L271 339L217 349Z" fill="#7bafb0" />
            <path d="M204 311L219 302L266 295L253 304Z" fill="#a6c7bc" />
            <path d="M253 304L266 295L284 330L271 339Z" fill="#507f82" />
            <path
              d="M330 232h-14v14M433 232h14v14M316 314v14h14M433 328h14v-14"
              fill="none"
              stroke="#7bddc7"
              strokeOpacity=".7"
              strokeWidth="1.5"
            />
            <text x="316" y="222" fontSize="9" fill="#a2dfd1" letterSpacing="1">
              TARGET · 01
            </text>
          </g>
          <g transform={`translate(${dx} ${dy})`}>
            <ellipse cx="327" cy="255" rx="75" ry="19" fill="#112a2c" opacity=".18" />
            <path d="M278 -10H365L355 111L288 110Z" fill={`url(#${metalId})`} />
            <path d="M286 19L356 19L353 87L290 87Z" fill="#243c46" />
            <path d="M294 25L348 25" stroke="#a4bcbf" strokeOpacity=".4" strokeWidth="2" />
            <path d="M281 95H361V138H281Z" fill="#152d38" />
            <rect x="291" y="105" width="60" height="22" rx="4" fill="#38545e" />
            <path d={`M${310 - opening} 132h17v92l-7 10h-10Z`} fill={`url(#${metalId})`} />
            <path d={`M${328 + opening} 132h-17v92l7 10h10Z`} fill={`url(#${metalId})`} />
            <path
              d={`M${312 - opening} 203h10v35h-10ZM${317 + opening} 203h10v35h-10Z`}
              fill="#1d3038"
            />
            <circle cx="298" cy="115" r="3" fill="#7ed7c1" />
            <circle cx="342" cy="115" r="3" fill="#a3b7b9" />
          </g>
        </>
      ) : (
        <>
          <rect x="61" y="57" width="519" height="302" rx="7" fill="#607b79" />
          <rect x="74" y="69" width="494" height="278" rx="4" fill={`url(#${gridId})`} />
          <rect x="106" y="93" width="358" height="230" rx="7" fill={`url(#${matId})`} />
          <rect
            x="119"
            y="106"
            width="332"
            height="204"
            rx="4"
            fill="none"
            stroke="#b8ccc1"
            strokeOpacity=".2"
          />
          <rect
            x="252"
            y="201"
            width="41"
            height="43"
            rx="3"
            fill="#182d2e"
            opacity=".3"
            transform="rotate(-12 272 222)"
          />
          <rect
            x="247"
            y="196"
            width="41"
            height="43"
            rx="3"
            fill="#df9160"
            stroke="#f0ad7f"
            strokeWidth="2"
            transform="rotate(-12 267 217)"
          />
          <rect
            x="155"
            y="261"
            width="37"
            height="32"
            rx="3"
            fill="#96bdb7"
            transform="rotate(15 173 277)"
          />
          <path
            d="M227 175h-15v15M303 175h15v15M212 245v15h15M303 260h15v-15"
            fill="none"
            stroke="#8ee3cf"
            strokeOpacity=".65"
            strokeWidth="1.5"
          />
          <text x="211" y="165" fontSize="9" fill="#a2dfd1" letterSpacing="1">
            TARGET · 01
          </text>
          <ellipse cx="534" cy="213" rx="40" ry="42" fill="#102d34" opacity=".5" />
          <circle cx="526" cy="203" r="35" fill="#48616c" stroke="#7d969a" strokeWidth="5" />
          <path
            d={`M526 203L${455 + dx * 0.3} ${136 + dy}L${336 + dx} ${173 + dy}`}
            stroke="#172d36"
            strokeWidth="39"
            strokeLinecap="round"
            fill="none"
            opacity=".45"
            transform="translate(5 9)"
          />
          <path
            d={`M526 203L${455 + dx * 0.3} ${136 + dy}L${336 + dx} ${173 + dy}`}
            stroke="#9aaeb2"
            strokeWidth="28"
            strokeLinecap="round"
            fill="none"
          />
          <path
            d={`M526 203L${455 + dx * 0.3} ${136 + dy}L${336 + dx} ${173 + dy}`}
            stroke="#c1cdcc"
            strokeWidth="9"
            strokeLinecap="round"
            fill="none"
            opacity=".55"
          />
          <circle
            cx={455 + dx * 0.3}
            cy={136 + dy}
            r="21"
            fill="#3c5761"
            stroke="#92abad"
            strokeWidth="4"
          />
          <g transform={`translate(${336 + dx} ${173 + dy}) rotate(22)`}>
            <rect
              x="-18"
              y="-23"
              width="36"
              height="47"
              rx="7"
              fill="#294650"
              stroke="#9cafb4"
              strokeWidth="2"
            />
            <path d={`M-5 ${-opening}h-30v8h30ZM-5 ${opening}h-30v-8h30Z`} fill="#bdcaca" />
            <circle r="6" fill="#55bba8" />
          </g>
        </>
      )}
      <path d="M312 200h16M320 192v16" stroke="#c8e3d9" strokeWidth="1" strokeOpacity=".5" />
      <path
        d="M18 65V49h16M606 49h16v16M18 333v16h16M606 349h16v-16"
        fill="none"
        stroke="#c0ded5"
        strokeOpacity=".35"
      />
      <text x="25" y="377" fill="#d4e5df" fillOpacity=".55" fontSize="8" letterSpacing="1.4">
        SIMULATED ROBOT VISION · NO HARDWARE CONNECTED
      </text>
    </svg>
  );
}
