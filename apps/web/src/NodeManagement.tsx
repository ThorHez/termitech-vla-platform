import { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  LoaderCircle,
  RefreshCw,
  Server,
  Trash2,
  Unplug,
} from 'lucide-react';
import { api, date, useApi } from './api';
import { useAuth } from './auth';
import { Badge, ErrorNotice, Loading, Modal } from './components';
import { activeStates, useData } from './data';
import type { Node, NodeManagementInfo, RemoteDeployment } from './types';

const complete = (job: RemoteDeployment) =>
  ['UNINSTALLED', 'DISCONNECTED', 'FAILED', 'INTERRUPTED'].includes(job.state);
const failed = (job: RemoteDeployment) => ['FAILED', 'INTERRUPTED'].includes(job.state);

export function NodeManagement(props: Parameters<typeof NodeManagementContent>[0]) {
  const { can } = useAuth();
  if (!can('machines.manage'))
    return (
      <Modal title="权限不足" close={props.close}>
        管理机器连接需要管理员权限。
      </Modal>
    );
  return <NodeManagementContent {...props} />;
}

function NodeManagementContent({
  node,
  close,
  changed,
  reconnect,
}: {
  node: Node;
  close: () => void;
  changed?: () => void;
  reconnect: () => void;
}) {
  const { runs, refresh } = useData();
  const management = useApi<NodeManagementInfo>(`/nodes/${node.id}/management`, 0);
  const [action, setAction] = useState<'disconnect' | 'uninstall'>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [pollError, setPollError] = useState<Error>();
  const [job, setJob] = useState<RemoteDeployment>();
  const request = useRef<{ action: string; key: string } | undefined>(undefined);
  const retryKey = useRef<{ id: string; key: string } | undefined>(undefined);
  const changedRef = useRef({ refresh, changed });
  changedRef.current = { refresh, changed };
  const activeRuns = (runs.data || []).filter(
    (run) => run.node_id === node.id && activeStates.has(run.state),
  );
  const trainingUnknown = runs.loading && !runs.data;
  const info = management.data;
  const canDisconnect = info?.can_disconnect && !activeRuns.length && !trainingUnknown;
  const canUninstall = info?.can_uninstall && !activeRuns.length && !trainingUnknown;
  const active = job && !complete(job);
  const reason = activeRuns.length
    ? `此机器有 ${activeRuns.length} 个活动训练，请结束训练后再操作。`
    : trainingUnknown
      ? '正在确认训练状态。'
      : '';

  const updated = () => {
    changedRef.current.refresh();
    changedRef.current.changed?.();
  };
  useEffect(() => {
    if (!job || complete(job)) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await api<RemoteDeployment>(`/remote-deployments/${job.id}`, {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
          cache: 'no-store',
        });
        if (!controller.signal.aborted) {
          setJob(result);
          setPollError(undefined);
          if (complete(result)) updated();
        }
      } catch (cause) {
        if (!controller.signal.aborted) setPollError(cause as Error);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 2000);
      }
    };
    timer = setTimeout(poll, 1000);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [job?.id, job?.state]);

  const execute = async () => {
    if (!action || busy || (action === 'uninstall' ? !canUninstall : !canDisconnect)) return;
    if (request.current?.action !== action) request.current = { action, key: crypto.randomUUID() };
    setBusy(true);
    setError(undefined);
    try {
      const result = await api<RemoteDeployment>(`/nodes/${node.id}/${action}`, {
        method: 'POST',
        headers: { 'Idempotency-Key': request.current.key },
      });
      setJob(result);
      if (complete(result)) updated();
    } catch (cause) {
      setError(cause as Error);
    } finally {
      setBusy(false);
    }
  };
  const retry = async () => {
    if (!job || busy) return;
    if (retryKey.current?.id !== job.id)
      retryKey.current = { id: job.id, key: crypto.randomUUID() };
    setBusy(true);
    setError(undefined);
    try {
      const result = await api<RemoteDeployment>(`/remote-deployments/${job.id}/retry`, {
        method: 'POST',
        headers: { 'Idempotency-Key': retryKey.current.key },
      });
      setJob(result);
      setPollError(undefined);
    } catch (cause) {
      setError(cause as Error);
    } finally {
      setBusy(false);
    }
  };
  const safeClose = () => {
    if (!busy) close();
  };

  return (
    <Modal
      wide
      title={
        job
          ? job.operation === 'disconnect'
            ? '连接已断开'
            : 'Agent 卸载状态'
          : action
            ? action === 'disconnect'
              ? '确认断开连接'
              : '确认卸载 Agent'
            : '管理机器连接'
      }
      subtitle={node.name}
      close={safeClose}
      footer={
        <>
          {!job && action && (
            <button
              className="button ghost"
              disabled={busy}
              onClick={() => {
                setAction(undefined);
                setError(undefined);
              }}
            >
              <ArrowLeft size={15} />
              返回
            </button>
          )}
          {!job && (
            <button className="button outline" disabled={busy} onClick={safeClose}>
              取消
            </button>
          )}
          {!job && action && (
            <button
              className="button danger"
              disabled={busy || (action === 'uninstall' ? !canUninstall : !canDisconnect)}
              onClick={() => void execute()}
            >
              {busy && <LoaderCircle size={15} className="spin" />}
              {action === 'disconnect' ? '确认断开连接' : '确认卸载 Agent'}
            </button>
          )}
          {job && failed(job) && job.operation === 'uninstall' && (
            <button className="button outline" disabled={busy} onClick={() => void retry()}>
              {busy ? <LoaderCircle size={15} className="spin" /> : <RefreshCw size={15} />}重试卸载
            </button>
          )}
          {job && ['DISCONNECTED', 'UNINSTALLED'].includes(job.state) && (
            <button className="button outline" onClick={reconnect}>
              重新接入机器
            </button>
          )}
          {job && (
            <button className="button primary" disabled={busy} onClick={safeClose}>
              {active ? '后台继续卸载' : '完成'}
            </button>
          )}
        </>
      }
    >
      <ErrorNotice error={error} />
      {!job && (
        <>
          <ErrorNotice error={management.error} retry={management.refresh} />
          {management.loading && !info ? (
            <Loading label="正在检查机器管理状态…" />
          ) : (
            info && (
              <>
                <div className="node-management-target">
                  <Server size={20} />
                  <div>
                    <strong>{node.name}</strong>
                    <span>
                      {info.host
                        ? `${info.username}@${info.host}:${info.port || 22}`
                        : node.hostname}
                    </span>
                    {info.install_dir && <code>{info.install_dir}</code>}
                  </div>
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
                {reason && (
                  <div className="inline-note amber" role="status">
                    <AlertTriangle size={16} />
                    {reason}
                  </div>
                )}
                {!action ? (
                  <div className="node-management-actions">
                    <section>
                      <Unplug size={21} />
                      <div>
                        <h3>断开连接</h3>
                        <p>
                          注销中台登记并关闭平台维护的 SSH 隧道。远端 Agent
                          程序和服务保留，后续心跳将被拒绝；历史训练记录保留。
                        </p>
                        {!canDisconnect && !reason && (
                          <small>
                            {info.disconnect_reason || info.reason || '当前状态不支持断开。'}
                          </small>
                        )}
                      </div>
                      <button
                        className="button outline small"
                        disabled={!canDisconnect}
                        onClick={() => setAction('disconnect')}
                      >
                        断开连接
                      </button>
                    </section>
                    <section>
                      <Trash2 size={21} />
                      <div>
                        <h3>卸载 Agent</h3>
                        <p>
                          通过已保存的 SSH 连接远程停止服务、卸载 Agent
                          并重置机器身份。训练代码、数据、模型和训练输出保留。
                        </p>
                        {!canUninstall && !reason && (
                          <small>
                            {info.uninstall_reason ||
                              info.reason ||
                              (info.managed
                                ? '当前状态不支持远程卸载。'
                                : '没有可用的 SSH 管理连接，请重新通过 SSH 接入后卸载。')}
                          </small>
                        )}
                      </div>
                      <button
                        className="button outline small danger-text"
                        disabled={!canUninstall}
                        onClick={() => setAction('uninstall')}
                      >
                        卸载 Agent
                      </button>
                    </section>
                  </div>
                ) : (
                  <div className="node-management-confirm">
                    <div className="inline-note amber">
                      <AlertTriangle size={18} />
                      {action === 'disconnect'
                        ? '断开后，此机器将从可用机器列表移除，不能再创建训练。远端 Agent 仍安装在机器上。'
                        : '中台将远程停止并卸载此机器的 Agent 服务，重置本次接入的机器身份。此机器随后从可用机器列表移除。'}
                    </div>
                    <p>
                      {action === 'disconnect'
                        ? '已保存的 SSH 凭据将被清除。断开后如需远程卸载，必须重新填写 SSH 信息接入。'
                        : '训练代码、数据、模型和训练输出会保留；历史训练记录及关联关系可继续查看。'}
                    </p>
                    {action === 'uninstall' && info.install_dir && (
                      <p>
                        Agent 安装目录：<code>{info.install_dir}</code>
                      </p>
                    )}
                    {!canDisconnect && action === 'disconnect' && (
                      <p className="danger-text">
                        {reason || info.disconnect_reason || info.reason}
                      </p>
                    )}
                    {!canUninstall && action === 'uninstall' && (
                      <p className="danger-text">
                        {reason || info.uninstall_reason || info.reason}
                      </p>
                    )}
                  </div>
                )}
              </>
            )
          )}
        </>
      )}
      {job && (
        <>
          <div
            className={`onboard-deploy-heading ${['DISCONNECTED', 'UNINSTALLED'].includes(job.state) ? 'success' : failed(job) ? 'failed' : ''}`}
          >
            {['DISCONNECTED', 'UNINSTALLED'].includes(job.state) ? (
              <CheckCircle2 size={25} />
            ) : failed(job) ? (
              <AlertTriangle size={25} />
            ) : (
              <LoaderCircle size={25} className="spin" />
            )}
            <div>
              <h3>
                {job.state === 'DISCONNECTED'
                  ? '连接已断开'
                  : job.state === 'UNINSTALLED'
                    ? 'Agent 已卸载'
                    : failed(job)
                      ? '卸载未完成'
                      : '正在远程卸载 Agent'}
              </h3>
              <p>
                {job.host || node.hostname} · {job.install_dir}
              </p>
            </div>
          </div>
          {job.error && (
            <div className="inline-note red" role="alert">
              <AlertTriangle size={16} />
              <div>
                {job.error.message}
                <p className="onboard-caption">错误代码：{job.error.code}</p>
              </div>
            </div>
          )}
          {pollError && (
            <div className="inline-note amber" role="alert">
              状态读取失败：{pollError.message}。页面将自动重试。
            </div>
          )}
          {job.state === 'DISCONNECTED' && (
            <p className="node-management-result">
              中台登记和 SSH 隧道已关闭，远端 Agent 保留。已保存的 SSH
              凭据已清除，需要重新接入后才能远程卸载。
            </p>
          )}
          {job.state === 'UNINSTALLED' && (
            <>
              <p className="node-management-result">
                Agent
                服务已停止并卸载，机器身份已重置。训练代码、数据、模型、输出和历史训练记录保留。
              </p>
              {job.operation_result?.backup_dir && (
                <p className="onboard-caption">
                  Agent 状态备份：<code>{job.operation_result.backup_dir}</code>
                </p>
              )}
            </>
          )}
          {['DISCONNECTED', 'UNINSTALLED'].includes(job.state) && (
            <a className="text-link" href={`#/nodes/${node.id}`} onClick={safeClose}>
              查看机器历史详情
            </a>
          )}
          <section className="onboard-deploy-log" aria-label="机器管理日志">
            <div className="onboard-section-heading">
              <h3>操作日志</h3>
            </div>
            {job.logs.length ? (
              <ol>
                {job.logs.map((entry, index) => (
                  <li key={`${entry.timestamp}-${index}`}>
                    <time>{date(entry.timestamp)}</time>
                    <span>{entry.message}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p>操作已提交。</p>
            )}
          </section>
          {active && (
            <p className="onboard-caption">
              关闭窗口后中台继续执行，可在「接入机器 → 最近部署」查看结果。
            </p>
          )}
        </>
      )}
    </Modal>
  );
}
