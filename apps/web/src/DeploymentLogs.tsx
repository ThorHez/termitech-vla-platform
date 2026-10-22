import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Clock3, Download, LoaderCircle } from 'lucide-react';
import { authenticatedFetch } from './api';
import type { RemoteDeployment } from './types';

const stageLabels: Record<string, string> = {
  QUEUED: '等待部署',
  CONNECTING: '连接机器',
  CHECKING: '检查安装条件',
  TUNNEL: '建立 SSH 隧道',
  UPLOADING: '上传安装文件',
  VERIFYING: '校验安装文件',
  INSTALLING: '安装 Agent',
  INSTALL_CHECKING: '检查安装包',
  INSTALL_ENVIRONMENT: '创建 Agent Python 环境',
  INSTALL_DEPENDENCIES: '下载并安装依赖',
  INSTALL_PACKAGE: '安装 Agent 程序',
  INSTALL_VERIFY: '验证 Agent 安装',
  INSTALL_CONFIG: '写入 Agent 配置',
  INSTALL_SERVICE: '配置系统服务',
  STARTING: '启动服务',
  STARTED: '服务已启动，等待上线',
  WAITING_AGENT: '等待心跳与环境检查',
  ONLINE: '接入完成',
  FAILED: '执行失败',
  INTERRUPTED: '执行中断',
  UNINSTALLING: '卸载 Agent',
  UNINSTALLED: 'Agent 已卸载',
  DISCONNECTED: '已断开连接',
};

export const deploymentStageLabel = (stage: string) => stageLabels[stage] || stage;
export const deploymentMainStage = (stage: string) =>
  stage.startsWith('INSTALL_')
    ? 'INSTALLING'
    : stage === 'CHECKING' || stage === 'TUNNEL'
      ? 'CONNECTING'
      : stage === 'VERIFYING'
        ? 'UPLOADING'
        : stage === 'STARTED'
          ? 'WAITING_AGENT'
          : stage;

const timestamp = (value: string) => {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString('zh-CN', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
};

const duration = (seconds: number) => {
  if (!Number.isFinite(seconds)) return '—';
  const value = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  return `${hours ? `${hours} 小时 ` : ''}${hours || minutes ? `${minutes} 分 ` : ''}${value % 60} 秒`;
};

export function DeploymentLogs({ job, active }: { job: RemoteDeployment; active: boolean }) {
  const [now, setNow] = useState(Date.now());
  const [follow, setFollow] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState('');
  const scroll = useRef<HTMLOListElement>(null);
  const downloadController = useRef<AbortController | null>(null);
  const last = job.logs[job.logs.length - 1];
  const start = job.started_at ? Date.parse(job.started_at) : null;
  const end = job.finished_at ? Date.parse(job.finished_at) : now;
  const lastTime = last ? Date.parse(last.timestamp) : NaN;
  const silentSeconds = Number.isFinite(lastTime) ? Math.max(0, (now - lastTime) / 1000) : 0;

  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  useEffect(() => {
    if (follow && scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [job.logs, follow]);
  useEffect(() => () => downloadController.current?.abort(), []);

  const download = async () => {
    const controller = new AbortController();
    downloadController.current = controller;
    setDownloading(true);
    setDownloadError('');
    try {
      const response = await authenticatedFetch(
        `/remote-deployments/${encodeURIComponent(job.id)}/logs/download`,
        {
          cache: 'no-store',
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30000)]),
        },
      );
      if (!response.ok) {
        let message = `日志下载失败（${response.status}），请稍后重试。`;
        try {
          const detail = (await response.json()) as {
            error?: { message?: string };
            detail?: string;
          };
          message = detail.error?.message || detail.detail || message;
        } catch {
          // Keep the HTTP error when the service did not return a JSON error.
        }
        throw new Error(message);
      }
      const blob = await response.blob();
      if (controller.signal.aborted) return;
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = objectUrl;
      anchor.download = `agent-deployment-${job.id.replace(/[^a-zA-Z0-9_-]/g, '_')}.jsonl`;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (cause) {
      if (!controller.signal.aborted) {
        setDownloadError(
          cause instanceof Error
            ? `无法下载日志：${cause.message}`
            : '无法下载日志，请检查管理服务连接后重试。',
        );
      }
    } finally {
      if (!controller.signal.aborted) setDownloading(false);
    }
  };

  return (
    <section className="onboard-deploy-log" aria-label="部署日志">
      <div className="onboard-section-heading onboard-log-heading">
        <h3>部署日志</h3>
        <button
          type="button"
          className="button outline small"
          disabled={downloading}
          onClick={() => void download()}
        >
          {downloading ? <LoaderCircle size={13} className="spin" /> : <Download size={13} />}
          {downloading ? '正在下载' : '下载完整日志'}
        </button>
      </div>
      <dl className="onboard-log-summary">
        <div>
          <dt>总耗时</dt>
          <dd>
            {start === null
              ? job.state === 'QUEUED'
                ? '等待开始'
                : '未记录'
              : duration((end - start) / 1000)}
          </dd>
        </div>
        <div>
          <dt>{active ? '当前步骤' : '最后步骤'}</dt>
          <dd>{deploymentStageLabel(job.stage)}</dd>
        </div>
        <div>
          <dt>最近输出</dt>
          <dd>{last ? timestamp(last.timestamp) : '暂无输出'}</dd>
        </div>
      </dl>
      {active && silentSeconds >= 30 && (
        <p className="onboard-log-wait">
          <Clock3 size={13} />已 {duration(silentSeconds)}{' '}
          未收到新输出，正在等待输出；网络下载可能较慢。
        </p>
      )}
      <div className="onboard-log-toolbar">
        <small>最近 {job.logs.length} 条 · 页面最多保留 300 条，完整记录可下载</small>
        <label>
          <input
            type="checkbox"
            checked={follow}
            onChange={(event) => setFollow(event.target.checked)}
          />
          自动跟随
        </label>
      </div>
      {job.logs.length ? (
        <ol
          ref={scroll}
          tabIndex={0}
          aria-label="部署日志内容"
          aria-live="off"
          onScroll={(event) => {
            const list = event.currentTarget;
            if (list.scrollHeight - list.scrollTop - list.clientHeight >= 24) setFollow(false);
          }}
        >
          {job.logs.map((entry, index) => (
            <li key={`${entry.timestamp}-${index}`}>
              <time dateTime={entry.timestamp}>{timestamp(entry.timestamp)}</time>
              <span className="onboard-log-stage">{deploymentStageLabel(entry.stage)}</span>
              <span className="onboard-log-message">{entry.message}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p>等待中台开始执行。</p>
      )}
      {downloadError && (
        <div className="inline-note red" role="alert">
          <AlertTriangle size={14} />
          {downloadError}
        </div>
      )}
    </section>
  );
}
