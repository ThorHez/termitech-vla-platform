import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  AlertCircle,
  ArrowRight,
  Check,
  ChevronLeft,
  Copy,
  GitBranch,
  GitCommitHorizontal,
  LoaderCircle,
  RefreshCw,
  X,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { ApiError, date, navigate } from './api';
import type { CodeSnapshot } from './types';
import {
  buildMetricChartModel,
  formatMetricValue,
  metricFraction,
  metricInterpolate,
  nearestMetricPoint,
} from './metricChartData';
import type { MetricChartInput, MetricChartPoint } from './metricChartData';

const states: Record<string, [string, string]> = {
  ONLINE: ['在线', 'green'],
  OFFLINE: ['离线', 'gray'],
  DISCONNECTED: ['已断开', 'gray'],
  REMOVED: ['已卸载', 'gray'],
  UNINSTALLED: ['已卸载', 'gray'],
  DEGRADED: ['异常', 'amber'],
  QUEUED: ['排队中', 'amber'],
  STARTING: ['启动中', 'blue'],
  RUNNING: ['训练中', 'green'],
  SUCCEEDED: ['已完成', 'green'],
  FAILED: ['失败', 'red'],
  CANCELLING: ['取消中', 'amber'],
  CANCELLED: ['已取消', 'gray'],
  UNKNOWN: ['状态未知', 'amber'],
  AVAILABLE: ['可使用', 'green'],
  PLANNED: ['待验收', 'gray'],
  READY: ['已就绪', 'green'],
  REGISTERED: ['待校验', 'blue'],
  ARCHIVED: ['已归档', 'gray'],
  MISSING: ['路径不存在', 'red'],
  PENDING: ['校验中', 'blue'],
  VALID: ['已验证', 'green'],
  INVALID: ['无效', 'red'],
};
export function Badge({ state, label }: { state?: string; label?: string }) {
  const [text, color] = states[state || ''] || [state || '未知', 'gray'];
  return (
    <span className={`badge ${color}`}>
      <span className="badge-dot" />
      {label || text}
    </span>
  );
}
export function Empty({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Icon size={25} strokeWidth={1.5} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}
export function ErrorNotice({
  error,
  retry,
  title = '数据获取失败',
  hint,
}: {
  error?: Error;
  retry?: () => void;
  title?: string;
  hint?: string;
}) {
  if (!error) return null;
  return (
    <div className="error-notice" role="alert">
      <AlertCircle size={18} />
      <div>
        <strong>{title}</strong>
        <p>
          {error instanceof ApiError && <code>{error.code} · </code>}
          {error.message}
        </p>
        {hint && <p>{hint}</p>}
      </div>
      {retry && (
        <button className="button small ghost" onClick={retry}>
          <RefreshCw size={14} />
          重试
        </button>
      )}
    </div>
  );
}
export function Loading({ label = '正在读取数据…' }: { label?: string }) {
  return (
    <div className="loading" role="status">
      <LoaderCircle size={20} className="spin" />
      <span>{label}</span>
    </div>
  );
}
export function PageHeading({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="heading-actions">{actions}</div>}
    </div>
  );
}
export function Panel({
  title,
  subtitle,
  action,
  children,
  className = '',
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
export function ViewLink({ to, children = '查看全部' }: { to: string; children?: ReactNode }) {
  return (
    <a className="text-link" href={`#${to}`}>
      {children}
      <ArrowRight size={14} />
    </a>
  );
}
export function Back({ to, label }: { to: string; label: string }) {
  return (
    <button className="back-link" onClick={() => navigate(to)}>
      <ChevronLeft size={16} />
      {label}
    </button>
  );
}
export function ProgressBar({ value, tone = 'teal' }: { value?: number | null; tone?: string }) {
  return (
    <div
      className={`progress ${tone}`}
      role="progressbar"
      aria-valuenow={typeof value === 'number' ? value : undefined}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <span style={{ width: `${Math.min(100, Math.max(0, value || 0))}%` }} />
    </div>
  );
}
export function Field({
  label,
  children,
  hint,
  required,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
  required?: boolean;
}) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        {required && <span className="required"> *</span>}
      </span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}
export function Modal({
  title,
  subtitle,
  close,
  children,
  footer,
  wide = false,
  className = '',
}: {
  title: string;
  subtitle?: string;
  close: () => void;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(close);
  closeRef.current = close;
  useEffect(() => {
    const last = document.activeElement as HTMLElement | null;
    document.body.style.overflow = 'hidden';
    ref.current?.focus();
    const handleKey = (event: KeyboardEvent) => {
      const dialogs = document.querySelectorAll<HTMLElement>('[role="dialog"][aria-modal="true"]');
      if (dialogs[dialogs.length - 1] !== ref.current) return;
      if (event.key === 'Escape') closeRef.current();
      if (event.key === 'Tab') {
        const elements = ref.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex="0"]',
        );
        if (!elements?.length) {
          event.preventDefault();
          return;
        }
        const first = elements[0],
          lastElement = elements[elements.length - 1];
        if (
          event.shiftKey &&
          (document.activeElement === first || document.activeElement === ref.current)
        ) {
          event.preventDefault();
          lastElement.focus();
        } else if (!event.shiftKey && document.activeElement === lastElement) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => {
      const remainingDialog = document.querySelector<HTMLElement>(
        '[role="dialog"][aria-modal="true"]',
      );
      document.body.style.overflow = remainingDialog ? 'hidden' : '';
      window.removeEventListener('keydown', handleKey);
      if (last?.isConnected) last.focus();
      else remainingDialog?.focus();
    };
  }, []);
  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <div
        className={`modal ${wide ? 'wide' : ''} ${className}`}
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
      >
        <header className="modal-heading">
          <div>
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <button className="icon-button" onClick={close} aria-label="关闭弹窗">
            <X size={20} />
          </button>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-footer">{footer}</footer>}
      </div>
    </div>
  );
}
export function CopyButton({ text, label = '复制' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="icon-button compact"
      aria-label={label}
      title={label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1800);
        } catch {
          setCopied(false);
        }
      }}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}
export function CodeVersion({
  snapshot,
  title = '实际执行代码版本',
}: {
  snapshot?: CodeSnapshot;
  title?: string;
}) {
  if (!snapshot?.commit_sha)
    return (
      <div className="quiet-placeholder">
        <GitCommitHorizontal size={18} />
        <span>训练启动后展示实际代码版本</span>
      </div>
    );
  return (
    <div className="code-version">
      <div className="code-version-top">
        <div className="mini-icon">
          <GitCommitHorizontal size={20} />
        </div>
        <div>
          <strong>{title}</strong>
          <span>由裸金属 Agent 采集 · {date(snapshot.captured_at)}</span>
        </div>
        <Badge
          state={snapshot.dirty ? 'DEGRADED' : 'READY'}
          label={snapshot.dirty ? '有本地修改' : '无本地修改'}
        />
      </div>
      <div className="commit-line">
        <code title={snapshot.commit_sha}>{snapshot.commit_sha || '未读取到 commit'}</code>
        {snapshot.commit_sha && (
          <CopyButton text={snapshot.commit_sha} label="复制完整 commit SHA" />
        )}
      </div>
      <div className="code-meta">
        <span>
          <GitBranch size={13} />
          {snapshot.branch || 'detached HEAD'}
        </span>
        <span>{snapshot.commit_message || '无提交说明'}</span>
      </div>
      <dl className="detail-list code-paths">
        <div>
          <dt>代码目录</dt>
          <dd>
            <code>{snapshot.code_dir || '—'}</code>
          </dd>
        </div>
        <div>
          <dt>实际模块路径</dt>
          <dd>
            <code>{snapshot.module_origin || '—'}</code>
          </dd>
        </div>
      </dl>
      {snapshot.complete === false && (
        <div className="inline-note amber">代码快照不完整，当前记录不能保证精确复现。</div>
      )}
      {snapshot.dirty && (
        <div className="inline-note amber">存在本地修改，完整差异记录保留在该次训练目录中。</div>
      )}
    </div>
  );
}
export function MetricChart({
  values,
  metricName = 'loss',
}: {
  values: readonly MetricChartInput[];
  metricName?: string;
}) {
  const gradientId = useId();
  const descriptionId = useId();
  const [selected, setSelected] = useState<number | null>(null);
  const model = useMemo(() => buildMetricChartModel(values), [values]);
  if (!model.knownCount)
    return (
      <div className="chart-empty">
        <div className="chart-grid" />
        <span>等待训练上报 {metricName} 指标</span>
      </div>
    );
  const width = 700,
    height = 206;
  const left = 65,
    right = 22,
    top = 24,
    bottom = 34;
  const chartWidth = width - left - right;
  const chartHeight = height - top - bottom;
  const x = (point: MetricChartPoint) =>
    left + metricFraction(point.step, model.first, model.last) * chartWidth;
  const y = (value: number) =>
    top + (1 - metricFraction(value, model.domain.low, model.domain.high)) * chartHeight;
  const pointCoordinates = (point: MetricChartPoint) =>
    `${x(point).toFixed(2)},${y(point.value as number).toFixed(2)}`;
  const selectedPoint =
    selected === null ? undefined : model.points[Math.min(selected, model.points.length - 1)];
  return (
    <div className="metric-chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        tabIndex={0}
        aria-label={`${metricName} 训练指标曲线，共 ${model.knownCount} 个有效采样点，最近有效观测值 ${formatMetricValue(model.latest?.value)}`}
        aria-describedby={descriptionId}
        onPointerMove={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          const fraction = Math.max(
            0,
            Math.min(
              1,
              (((event.clientX - bounds.left) / bounds.width) * width - left) / chartWidth,
            ),
          );
          setSelected(nearestMetricPoint(model.points, fraction, model.first, model.last));
        }}
        onPointerLeave={() => setSelected(null)}
        onFocus={() => setSelected(model.points.length - 1)}
        onBlur={() => setSelected(null)}
        onKeyDown={(event) => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          setSelected((current) =>
            event.key === 'Home'
              ? 0
              : event.key === 'End'
                ? model.points.length - 1
                : Math.max(
                    0,
                    Math.min(
                      model.points.length - 1,
                      (current ?? model.points.length - 1) + (event.key === 'ArrowLeft' ? -1 : 1),
                    ),
                  ),
          );
        }}
      >
        <desc id={descriptionId}>
          横轴为训练步数。无效或缺失指标保留断点。聚焦后使用左右方向键查看采样，Home 和 End
          跳转首尾。
        </desc>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-gradient, #13a595)" stopOpacity=".22" />
            <stop offset="100%" stopColor="var(--chart-gradient, #13a595)" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 1, 2, 3].map((tick) => (
          <g key={tick}>
            <line
              x1={left}
              x2={width - right}
              y1={top + (tick * chartHeight) / 3}
              y2={top + (tick * chartHeight) / 3}
              stroke="var(--chart-grid, #e9eef0)"
              strokeDasharray="3 4"
            />
            <text x={left - 8} y={top + (tick * chartHeight) / 3 + 4} textAnchor="end">
              {formatMetricValue(
                metricInterpolate(model.domain.high, model.domain.low, tick / 3),
                4,
              )}
            </text>
          </g>
        ))}
        {model.segments.map((segment, index) => {
          const line = segment.map(pointCoordinates).join(' ');
          const lastPoint = segment[segment.length - 1];
          return (
            <g key={index}>
              {segment.length > 1 && (
                <>
                  <polygon
                    points={`${x(segment[0]).toFixed(2)},${height - bottom} ${line} ${x(lastPoint).toFixed(2)},${height - bottom}`}
                    fill={`url(#${gradientId})`}
                  />
                  <polyline
                    points={line}
                    fill="none"
                    stroke="var(--chart-teal, #0b9b8e)"
                    strokeWidth="2.4"
                    strokeLinejoin="round"
                  />
                </>
              )}
              {(segment.length === 1 || model.constant || model.first === model.last) && (
                <circle
                  cx={x(lastPoint)}
                  cy={y(lastPoint.value as number)}
                  r="4"
                  fill="var(--chart-teal, #0b9b8e)"
                />
              )}
            </g>
          );
        })}
        {selectedPoint && (
          <g>
            <line
              x1={x(selectedPoint)}
              x2={x(selectedPoint)}
              y1={top}
              y2={height - bottom}
              stroke="var(--text-muted, #9daeb9)"
              strokeDasharray="3 4"
            />
            {selectedPoint.value !== null && (
              <circle
                cx={x(selectedPoint)}
                cy={y(selectedPoint.value)}
                r="4"
                fill="var(--chart-teal, #0b9b8e)"
                stroke="var(--bg-overlay, #fff)"
                strokeWidth="2"
              />
            )}
          </g>
        )}
        <text x={left} y={height - 9}>
          {formatMetricValue(model.first)}
        </text>
        <text x={width - right} y={height - 9} textAnchor="end">
          {formatMetricValue(model.last)} step
        </text>
      </svg>
      <div
        aria-live="polite"
        style={{
          minHeight: 27,
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: '5px 12px',
          padding: '3px 8px',
          color: 'var(--text-muted, #9daeb9)',
          fontSize: 11,
        }}
      >
        {selectedPoint ? (
          <>
            <span>step {formatMetricValue(selectedPoint.step)}</span>
            <span>
              {metricName} <strong>{formatMetricValue(selectedPoint.value)}</strong>
            </span>
            {selectedPoint.timestamp && (
              <time dateTime={selectedPoint.timestamp}>{date(selectedPoint.timestamp)}</time>
            )}
            {selectedPoint.value === null && <span>此采样未上报有效数值</span>}
          </>
        ) : (
          <span>移动指针或聚焦后使用方向键查看数值</span>
        )}
      </div>
    </div>
  );
}
