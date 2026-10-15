import { useEffect, useRef, useState } from 'react';
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
import { date, navigate } from './api';
import type { CodeSnapshot } from './types';

const states: Record<string, [string, string]> = {
  ONLINE: ['在线', 'green'],
  OFFLINE: ['离线', 'gray'],
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
export function ErrorNotice({ error, retry }: { error?: Error; retry?: () => void }) {
  if (!error) return null;
  return (
    <div className="error-notice" role="alert">
      <AlertCircle size={18} />
      <div>
        <strong>数据获取失败</strong>
        <p>{error.message}</p>
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
}: {
  title: string;
  subtitle?: string;
  close: () => void;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(close);
  closeRef.current = close;
  useEffect(() => {
    const last = document.activeElement as HTMLElement | null;
    document.body.style.overflow = 'hidden';
    ref.current?.focus();
    const handleKey = (event: KeyboardEvent) => {
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
      document.body.style.overflow = '';
      window.removeEventListener('keydown', handleKey);
      last?.focus();
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
        className={`modal ${wide ? 'wide' : ''}`}
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
export function MetricChart({ values }: { values: { step: number; loss: number }[] }) {
  if (!values.length)
    return (
      <div className="chart-empty">
        <div className="chart-grid" />
        <span>等待训练上报 loss 指标</span>
      </div>
    );
  const width = 700,
    height = 206,
    pad = 35;
  const losses = values.map((item) => item.loss),
    steps = values.map((item) => item.step);
  const low = Math.min(...losses),
    high = Math.max(...losses),
    first = Math.min(...steps),
    last = Math.max(...steps);
  const range = high - low || Math.abs(high) * 0.1 || 1;
  const yMin = low - range * 0.08,
    yMax = high + range * 0.08;
  const point = (item: { step: number; loss: number }) =>
    `${pad + ((item.step - first) / (last - first || 1)) * (width - pad * 2)},${height - pad - ((item.loss - yMin) / (yMax - yMin)) * (height - pad * 2)}`;
  const line = values.map(point).join(' ');
  const area = `${pad},${height - pad} ${line} ${point(values[values.length - 1]).split(',')[0]},${height - pad}`;
  return (
    <div className="metric-chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`训练 loss 曲线，共 ${values.length} 个采样点，最新 ${values[values.length - 1].loss}`}
      >
        <defs>
          <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#13a595" stopOpacity=".22" />
            <stop offset="100%" stopColor="#13a595" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 1, 2, 3].map((tick) => (
          <g key={tick}>
            <line
              x1={pad}
              x2={width - pad}
              y1={pad + (tick * (height - 2 * pad)) / 3}
              y2={pad + (tick * (height - 2 * pad)) / 3}
              stroke="#e9eef0"
              strokeDasharray="3 4"
            />
            <text x={pad - 8} y={pad + (tick * (height - 2 * pad)) / 3 + 4} textAnchor="end">
              {(yMax - ((yMax - yMin) * tick) / 3).toPrecision(3)}
            </text>
          </g>
        ))}
        <polygon points={area} fill="url(#lossFill)" />
        <polyline
          points={line}
          fill="none"
          stroke="#0b9b8e"
          strokeWidth="2.4"
          strokeLinejoin="round"
        />
        {values.length === 1 && (
          <circle cx={pad} cy={Number(point(values[0]).split(',')[1])} r="4" fill="#0b9b8e" />
        )}
        <text x={pad} y={height - 9}>
          {first}
        </text>
        <text x={width - pad} y={height - 9} textAnchor="end">
          {last} step
        </text>
      </svg>
    </div>
  );
}
