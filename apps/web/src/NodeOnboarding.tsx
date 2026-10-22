import { useCallback, useEffect, useId, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  History,
  LoaderCircle,
  Plus,
  RefreshCw,
  Server,
  ShieldCheck,
  Terminal,
  Trash2,
} from 'lucide-react';
import { api, date, post } from './api';
import { useAuth } from './auth';
import { Badge, Field, Modal, ViewLink } from './components';
import { useData } from './data';
import { DeploymentLogs, deploymentMainStage, deploymentStageLabel } from './DeploymentLogs';
import { SSHPathAutocomplete, type SshPathConnection } from './SSHPathAutocomplete';
import type {
  AgentConfigDefaults,
  AgentConfigRequest,
  AgentProfileInput,
  Node,
  Profile,
  RemoteDeployment,
  RuntimeCheck,
  SshConnectionResult,
} from './types';

const newProfile = (n: number): AgentProfileInput => ({
  id: `openpi-native-${n}`,
  name: `OpenPI 训练环境 ${n}`,
  python: '',
  code_dir: '',
  adapter: 'openpi-sft',
  execution_mode: 'native',
});
const linuxPath = (path: string) =>
  path.startsWith('/') && !/[\0\r\n]/.test(path) && !path.split('/').includes('..');
const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));
const terminal = (job: RemoteDeployment) =>
  ['SUCCEEDED', 'FAILED', 'INTERRUPTED', 'DISCONNECTED', 'UNINSTALLED'].includes(job.state);
const retired = (job: RemoteDeployment) =>
  ['DISCONNECTED', 'UNINSTALLED'].includes(job.state) ||
  ['DISCONNECTED', 'UNINSTALLED'].includes(job.tunnel_state);
const completed = (job: RemoteDeployment) => job.state === 'SUCCEEDED' || retired(job);
const deploymentStages = [
  ['CONNECTING', '连接机器'],
  ['UPLOADING', '上传安装文件'],
  ['INSTALLING', '安装 Agent'],
  ['STARTING', '启动服务'],
  ['WAITING_AGENT', '确认机器上线'],
] as const;
const checkLabels: Record<string, string> = {
  code_directory: '代码目录',
  python_interpreter: '训练解释器',
  smoke_entrypoint: '模拟测试入口',
  git_repository: 'Git 仓库',
  execution_mode: '执行方式',
  python_runtime: 'Python 运行环境',
  module_origin: '实际模块来源',
  interpreter_probe: 'Python 执行检查',
  openpi_import: 'OpenPI 模块来源',
  jax_import: 'JAX 模块定位',
};
const deploymentLabel = (job: RemoteDeployment): string => {
  if (job.state === 'UNINSTALLED' || job.tunnel_state === 'UNINSTALLED') return 'Agent 已卸载';
  if (job.state === 'DISCONNECTED' || job.tunnel_state === 'DISCONNECTED') return '已断开连接';
  if (job.operation === 'uninstall')
    return ['FAILED', 'INTERRUPTED'].includes(job.state) ? '卸载未完成' : '正在卸载 Agent';
  return job.state === 'SUCCEEDED'
    ? '接入完成'
    : job.state === 'FAILED'
      ? '部署失败'
      : job.state === 'INTERRUPTED'
        ? '部署中断'
        : deploymentStageLabel(job.stage) || '准备部署';
};

export function runtimeCheckFor(node: Node | undefined, profileId: string) {
  const checks = node?.metrics?.runtime_checks;
  return Array.isArray(checks)
    ? (checks as RuntimeCheck[]).find((item) => item.profile_id === profileId)
    : undefined;
}

export function RuntimeCheckView({ check }: { check?: RuntimeCheck }) {
  return (
    <section className="runtime-check">
      <div className="runtime-check-title">
        <span>训练 Python 环境检查</span>
        <Badge
          state={check?.status === 'CHECKING' ? 'PENDING' : check?.status || 'REGISTERED'}
          label={
            !check
              ? '待 Agent 检查'
              : check.status === 'READY'
                ? '环境探测通过'
                : check.status === 'CHECKING'
                  ? '正在检查'
                  : '检查失败'
          }
        />
      </div>
      {check?.python_version && <p>Python {check.python_version}</p>}
      {check?.executable && <code className="block">{check.executable}</code>}
      {check?.module_origin && <p className="runtime-module">模块来源：{check.module_origin}</p>}
      {!!check?.checks?.length && (
        <ul className="runtime-check-list">
          {check.checks.map((item, index) => (
            <li className={item.ok ? 'passed' : 'failed'} key={`${item.name}-${index}`}>
              {item.ok ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
              <span>
                {checkLabels[item.name] || item.name}：{item.message}
              </span>
            </li>
          ))}
        </ul>
      )}
      {check?.error && <p className="runtime-error">{check.error}</p>}
      <small>
        {check?.checked_at ? `检查时间 ${date(check.checked_at)} · ` : ''}
        {check?.status === 'READY'
          ? '环境探测已通过；创建训练时，Agent 会自动检查所选配方与本机资产。'
          : !check
            ? '等待支持环境探测的 Agent 上报，不代表环境已可用。'
            : '检查使用此训练环境的解释器执行。'}
      </small>
    </section>
  );
}

export function NodeOnboarding({ close }: { close: () => void }) {
  const { can } = useAuth();
  if (!can('machines.manage'))
    return (
      <Modal title="权限不足" close={close}>
        接入与部署机器需要管理员权限。
      </Modal>
    );
  return <NodeOnboardingContent close={close} />;
}

function NodeOnboardingContent({ close }: { close: () => void }) {
  const { refresh } = useData();
  const [step, setStep] = useState(0);
  const [host, setHost] = useState('');
  const [port, setPort] = useState('22');
  const [username, setUsername] = useState('root');
  const [password, setPassword] = useState('');
  const [installDir, setInstallDir] = useState('/data2/termitech-vla-agent');
  const [connection, setConnection] = useState<SshConnectionResult>();
  const [browseConnection, setBrowseConnection] = useState<SshPathConnection>();
  const [sshAction, setSshAction] = useState<'connect' | 'test'>('test');
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [customUrl, setCustomUrl] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const advancedId = useId();
  const [defaults, setDefaults] = useState<AgentConfigDefaults>();
  const [roots, setRoots] = useState(['/data2']);
  const [outputRoot, setOutputRoot] = useState('/data2/termitech-vla-agent/outputs');
  const [stateDir, setStateDir] = useState('/data2/termitech-vla-agent/state');
  const [profiles, setProfiles] = useState<AgentProfileInput[]>([newProfile(1)]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [job, setJob] = useState<RemoteDeployment>();
  const [pollError, setPollError] = useState('');
  const [recent, setRecent] = useState<RemoteDeployment[]>([]);
  const [showRecent, setShowRecent] = useState(false);
  const [recentError, setRecentError] = useState('');
  const [matchedNode, setMatchedNode] = useState<Node>();
  const [matchedProfiles, setMatchedProfiles] = useState<Profile[]>([]);
  const [checkError, setCheckError] = useState('');
  const dirty = useRef(new Set<string>());
  const submission = useRef<{ key: string; body: string } | undefined>(undefined);
  const retrySubmission = useRef<{ id: string; key: string } | undefined>(undefined);
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  const rootPaths = roots.map((item) => item.trim());
  const isActive = job && !terminal(job);
  const managementHistory = step === 2 && job?.operation && job.operation !== 'install';

  useEffect(() => {
    const controller = new AbortController();
    void api<AgentConfigDefaults>('/agent-config-defaults', { signal: controller.signal })
      .then(setDefaults)
      .catch(() => {
        /* SSH deployment can establish its own callback tunnel. */
      });
    void api<RemoteDeployment[]>('/remote-deployments', { signal: controller.signal })
      .then(setRecent)
      .catch((cause) => {
        if (!controller.signal.aborted) setRecentError(errorMessage(cause));
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!browseConnection) return;
    const connectionId = browseConnection.connection_id;
    const expiresAt = Date.parse(browseConnection.expires_at);
    const timer = setTimeout(
      () => {
        setBrowseConnection((current) =>
          current?.connection_id === connectionId ? undefined : current,
        );
        setConnection((current) => (current?.connection_id === connectionId ? undefined : current));
        submission.current = undefined;
        if (step < 2) setError('SSH 连接凭据已过期，请重新填写密码并连接后继续。');
      },
      Number.isFinite(expiresAt) ? Math.max(0, expiresAt - Date.now()) : 0,
    );
    return () => clearTimeout(timer);
  }, [browseConnection?.connection_id, browseConnection?.expires_at, step]);

  useEffect(() => {
    if (!job?.id || terminal(job)) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await api<RemoteDeployment>(`/remote-deployments/${job.id}`, {
          cache: 'no-store',
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
        });
        if (!controller.signal.aborted) {
          setJob(result);
          setPollError('');
          if (terminal(result)) refreshRef.current();
        }
      } catch (cause) {
        if (!controller.signal.aborted) setPollError(errorMessage(cause));
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

  const checkRuntime = useCallback(async () => {
    if (!job?.node_id) return;
    try {
      const [nodes, allProfiles] = await Promise.all([
        api<Node[]>('/nodes'),
        api<Profile[]>('/profiles'),
      ]);
      setMatchedNode(nodes.find((node) => node.id === job.node_id));
      setMatchedProfiles(
        allProfiles.filter(
          (profile) =>
            profile.node_id === job.node_id &&
            profile.metadata?.onboarding_id === job.onboarding_id,
        ),
      );
      setCheckError('');
    } catch (cause) {
      setCheckError(errorMessage(cause));
    }
  }, [job?.node_id, job?.onboarding_id]);
  useEffect(() => {
    if (!job?.node_id) return;
    void checkRuntime();
    const timer = setInterval(() => void checkRuntime(), 8000);
    return () => clearInterval(timer);
  }, [job?.node_id, checkRuntime]);

  const edited = (field: string) => {
    dirty.current.add(field);
    submission.current = undefined;
    setError('');
  };
  const invalidateConnection = () => {
    setConnection(undefined);
    setBrowseConnection(undefined);
    submission.current = undefined;
    setError('');
  };
  const validateLogin = () => {
    if (!host.trim() || /[\s/\\@]/.test(host.trim()))
      return '请填写裸金属 IP 或主机名，不包含协议或端口。';
    if (!/^\d+$/.test(port) || +port < 1 || +port > 65535) return 'SSH 端口须为 1–65535 的整数。';
    if (!username.trim() || /[\s\0\r\n]/.test(username)) return '请填写有效的 SSH 用户名。';
    return '';
  };
  const validateInstallDir = () => {
    const prefix = installDir.trim();
    if (
      !linuxPath(prefix) ||
      !/^\/[A-Za-z0-9_./-]+$/.test(prefix) ||
      prefix.split('/').filter((part) => part && part !== '.').length < 2
    )
      return 'Agent 安装目录须为具体的 Linux 绝对路径，例如 /data2/termitech-vla-agent。';
    return '';
  };
  const connectSsh = async () => {
    const message = validateLogin() || (!password ? '请填写 SSH 登录密码。' : '');
    if (message) {
      setError(message);
      return;
    }
    const body = {
      host: host.trim(),
      port: +port,
      username: username.trim(),
      password,
    };
    setPassword('');
    setConnection(undefined);
    setBrowseConnection(undefined);
    setSshAction('connect');
    setBusy(true);
    setError('');
    try {
      const result = await api<SshPathConnection>('/ssh-connections/connect', {
        method: 'POST',
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(90000),
      });
      setBrowseConnection(result);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };
  const testSsh = async () => {
    const usableConnection =
      browseConnection && Date.parse(browseConnection.expires_at) > Date.now();
    const message =
      validateLogin() ||
      validateInstallDir() ||
      (!usableConnection && !password
        ? browseConnection
          ? 'SSH 连接凭据已过期，请重新填写密码并连接。'
          : '请填写 SSH 登录密码，或先连接并读取目录。'
        : '');
    if (message) {
      setError(message);
      return;
    }
    const endpoint = usableConnection
      ? `/ssh-connections/${browseConnection!.connection_id}/test`
      : '/ssh-connections/test';
    const body = usableConnection
      ? { install_dir: installDir.trim() }
      : {
          host: host.trim(),
          port: +port,
          username: username.trim(),
          password,
          install_dir: installDir.trim(),
        };
    setPassword('');
    setConnection(undefined);
    setSshAction('test');
    setBusy(true);
    setError('');
    try {
      const result = await api<SshConnectionResult>(endpoint, {
        method: 'POST',
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(90000),
      });
      setConnection(result);
      setBrowseConnection(result);
      const existing = result.existing_agent;
      if (!dirty.current.has('name')) setName(existing?.name || result.hostname || host.trim());
      if (existing) {
        if (!dirty.current.has('roots')) setRoots(existing.roots.length ? existing.roots : ['']);
        if (!dirty.current.has('outputRoot')) setOutputRoot(existing.output_root);
        if (!dirty.current.has('stateDir')) setStateDir(existing.state_dir);
        if (!dirty.current.has('profiles')) setProfiles(existing.profiles);
      }
      if (!result.can_install)
        setError('SSH 已连接，但安装条件未满足。请根据检查结果调整后重新测试。');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };
  const validateConfig = () => {
    if (!connection?.can_install) return '请先测试 SSH 连接并确认安装条件通过。';
    if (Date.parse(connection.expires_at) <= Date.now())
      return 'SSH 连接凭据已过期，请返回重新测试连接。';
    if (!name.trim()) return '请填写机器名称。';
    if (customUrl) {
      try {
        const parsed = new URL(url.trim());
        if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password)
          return '中台地址须为 HTTP 或 HTTPS 地址，且不能包含账号密码。';
        if (
          parsed.protocol === 'http:' &&
          !['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname)
        )
          return '远程中台请使用 HTTPS；HTTP 仅支持 SSH 隧道的本机地址。';
        if (
          url.includes('?') ||
          url.includes('#') ||
          url.includes('\\') ||
          url
            .replace(/^https?:\/\/[^/]*/i, '')
            .split('/')
            .includes('..')
        )
          return '中台地址不能包含查询参数、锚点、反斜杠或 .. 路径段。';
      } catch {
        return '请填写有效的中台完整地址。';
      }
    }
    if (!rootPaths.length || rootPaths.some((path) => !linuxPath(path)))
      return '请为每个受管目录填写 Linux 绝对路径，不能留空或包含 ..。';
    if (!linuxPath(outputRoot.trim()) || !linuxPath(stateDir.trim()))
      return '输出目录和 Agent 状态目录须为 Linux 绝对路径。';
    const ids = new Set<string>();
    for (const [index, profile] of profiles.entries()) {
      if (!profile.name.trim() || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(profile.id.trim()))
        return `环境 ${index + 1}：请填写名称和有效标识（字母、数字、连字符或下划线）。`;
      if (ids.has(profile.id.trim())) return '同一台机器的训练环境标识不能重复。';
      ids.add(profile.id.trim());
      if (!linuxPath(profile.python.trim()) || !linuxPath(profile.code_dir.trim()))
        return `环境 ${index + 1}：训练 Python 解释器和代码目录均须填写 Linux 绝对路径。`;
    }
    return '';
  };
  const updateProfile = (index: number, patch: Partial<AgentProfileInput>) => {
    edited('profiles');
    setProfiles((items) => items.map((item, i) => (i === index ? { ...item, ...patch } : item)));
  };
  const deploy = async () => {
    const message = validateConfig();
    if (message) {
      setError(message);
      return;
    }
    const agent: AgentConfigRequest = {
      name: name.trim(),
      ...(customUrl ? { control_plane_url: url.trim().replace(/\/$/, '') } : {}),
      roots: rootPaths,
      output_root: outputRoot.trim(),
      state_dir: stateDir.trim(),
      profiles: profiles.map((profile) => ({
        id: profile.id.trim(),
        name: profile.name.trim(),
        python: profile.python.trim(),
        code_dir: profile.code_dir.trim(),
        adapter: profile.adapter,
        ...(profile.recipe?.trim() ? { recipe: profile.recipe.trim() } : {}),
        execution_mode: 'native',
      })),
    };
    const body = JSON.stringify({
      connection_id: connection!.connection_id,
      install_dir: installDir.trim(),
      agent,
    });
    if (!submission.current || submission.current.body !== body)
      submission.current = { key: crypto.randomUUID(), body };
    setBusy(true);
    setError('');
    try {
      const result = await api<RemoteDeployment>('/remote-deployments', {
        method: 'POST',
        body,
        headers: { 'Idempotency-Key': submission.current.key },
      });
      setJob(result);
      setMatchedNode(undefined);
      setMatchedProfiles([]);
      setStep(2);
      setRecent((items) => [result, ...items.filter((item) => item.id !== result.id)]);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };
  const retry = async () => {
    if (!job) return;
    if (retrySubmission.current?.id !== job.id)
      retrySubmission.current = { id: job.id, key: crypto.randomUUID() };
    setBusy(true);
    setError('');
    try {
      const result = await post<RemoteDeployment>(
        `/remote-deployments/${job.id}/retry`,
        {},
        {
          'Idempotency-Key': retrySubmission.current.key,
        },
      );
      setJob(result);
      setPollError('');
      setMatchedNode(undefined);
      setMatchedProfiles([]);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };
  const editDeployment = () => {
    if (!job) return;
    const prefix = job.install_dir || '/data2/termitech-vla-agent';
    setHost(job.host || '');
    setPort(String(job.port || 22));
    setUsername(job.username || 'root');
    setInstallDir(prefix);
    setName(job.agent.name || '');
    setRoots(job.agent.roots?.length ? job.agent.roots : ['/data2']);
    setOutputRoot(job.agent.output_root || `${prefix}/outputs`);
    setStateDir(job.agent.state_dir || `${prefix}/state`);
    setProfiles(job.agent.profiles?.length ? job.agent.profiles : [newProfile(1)]);
    setCustomUrl(!!job.agent.control_plane_url);
    setUrl(job.agent.control_plane_url || '');
    dirty.current = new Set(['name', 'roots', 'outputRoot', 'stateDir', 'profiles']);
    setPassword('');
    invalidateConnection();
    setJob(undefined);
    setStep(0);
    setMatchedNode(undefined);
    setMatchedProfiles([]);
  };

  return (
    <Modal
      wide
      title={managementHistory ? '机器管理记录' : '接入裸金属机器'}
      subtitle={
        managementHistory
          ? '查看远程操作的进度、日志与结果。'
          : '填写 SSH 连接信息，由中台远程安装并启动 Agent。'
      }
      close={close}
      footer={
        <>
          {step === 1 && (
            <button
              className="button ghost"
              disabled={busy}
              onClick={() => {
                setStep(0);
                setError('');
              }}
            >
              <ArrowLeft size={15} />
              上一步
            </button>
          )}
          {step === 0 &&
            (connection?.can_install ? (
              <button
                className="button primary"
                onClick={() => {
                  setStep(1);
                  setError('');
                }}
              >
                配置训练环境
                <ArrowRight size={15} />
              </button>
            ) : (
              <button className="button primary" disabled={busy} onClick={() => void testSsh()}>
                {busy ? <LoaderCircle className="spin" size={15} /> : <ShieldCheck size={15} />}
                {busy
                  ? sshAction === 'connect'
                    ? '正在连接 SSH'
                    : '正在测试 SSH'
                  : '测试 SSH 连接'}
              </button>
            ))}
          {step === 1 && (
            <button className="button primary" disabled={busy} onClick={() => void deploy()}>
              {busy ? <LoaderCircle className="spin" size={15} /> : <Server size={15} />}
              {busy ? '正在提交部署' : '远程安装并启动'}
            </button>
          )}
          {step === 2 && job && (
            <>
              {['FAILED', 'INTERRUPTED'].includes(job.state) &&
                !retired(job) &&
                (!job.operation || job.operation === 'install') && (
                  <button className="button outline" disabled={busy} onClick={editDeployment}>
                    修改连接配置
                  </button>
                )}
              {['FAILED', 'INTERRUPTED'].includes(job.state) &&
                !retired(job) &&
                job.operation !== 'disconnect' && (
                  <button className="button outline" disabled={busy} onClick={() => void retry()}>
                    {busy ? <LoaderCircle className="spin" size={15} /> : <RefreshCw size={15} />}
                    {job.operation === 'uninstall' ? '重试卸载' : '重试部署'}
                  </button>
                )}
              {retired(job) && (
                <button
                  className="button outline"
                  onClick={() => {
                    editDeployment();
                    setCustomUrl(false);
                    setUrl('');
                  }}
                >
                  重新接入机器
                </button>
              )}
              <button
                className="button primary"
                onClick={() => {
                  refresh();
                  close();
                }}
              >
                {completed(job)
                  ? retired(job)
                    ? '完成'
                    : '完成接入'
                  : isActive
                    ? job.operation === 'uninstall'
                      ? '后台继续卸载'
                      : '后台继续部署'
                    : '关闭'}
              </button>
            </>
          )}
        </>
      }
    >
      {!managementHistory && (
        <div className="wizard-steps">
          {['SSH 连接', '训练环境', '远程部署'].map((label, index) => (
            <div
              className={`wizard-step ${step === index ? 'current' : step > index ? 'done' : ''}`}
              key={label}
            >
              <span>{step > index ? <Check size={14} /> : index + 1}</span>
              <strong>{label}</strong>
            </div>
          ))}
        </div>
      )}
      {error && (
        <div className="inline-note red" role="alert">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}
      {step === 0 && (
        <div className="onboard-form">
          <div className="onboard-section-heading">
            <h3>远程登录信息</h3>
            <button
              className="text-link"
              onClick={() => setShowRecent((value) => !value)}
              aria-expanded={showRecent}
            >
              <History size={14} />
              最近部署
            </button>
          </div>
          {showRecent && (
            <section className="onboard-recent">
              {recentError ? (
                <p role="alert">{recentError}</p>
              ) : !recent.length ? (
                <p>暂无部署记录。</p>
              ) : (
                recent.slice(0, 10).map((item) => (
                  <button
                    key={item.id}
                    className="onboard-recent-item"
                    onClick={() => {
                      setJob(item);
                      setStep(2);
                      setError('');
                      setPassword('');
                    }}
                  >
                    <span>
                      <strong>
                        {item.agent.name || item.host || item.node_id || '机器管理记录'}
                      </strong>
                      <small>
                        {item.host ? `${item.username}@${item.host}:${item.port} · ` : ''}
                        {date(item.created_at)}
                      </small>
                    </span>
                    <Badge
                      state={
                        item.state === 'RUNNING' || item.state === 'WAITING_AGENT'
                          ? 'STARTING'
                          : item.state
                      }
                      label={deploymentLabel(item)}
                    />
                    <ChevronRight size={14} />
                  </button>
                ))
              )}
            </section>
          )}
          <div className="form-grid onboard-ssh-address">
            <Field label="裸金属 IP / 主机名" required>
              <input
                autoComplete="off"
                value={host}
                disabled={busy}
                onChange={(e) => {
                  setHost(e.target.value);
                  invalidateConnection();
                }}
                placeholder="例如 115.190.212.169"
              />
            </Field>
            <Field label="SSH 端口" required>
              <input
                inputMode="numeric"
                value={port}
                disabled={busy}
                onChange={(e) => {
                  setPort(e.target.value);
                  invalidateConnection();
                }}
              />
            </Field>
          </div>
          <div className="form-grid">
            <Field label="SSH 用户名" required>
              <input
                autoComplete="off"
                value={username}
                disabled={busy}
                onChange={(e) => {
                  setUsername(e.target.value);
                  invalidateConnection();
                }}
              />
            </Field>
            <Field
              label="SSH 登录密码"
              required={!browseConnection}
              hint={
                browseConnection
                  ? '密码已从表单清除，目录浏览和安装检查使用已保存的 SSH 连接。'
                  : '仅发送至中台，不写入浏览器存储。'
              }
            >
              <input
                type="password"
                autoComplete="new-password"
                value={password}
                disabled={busy}
                onChange={(e) => {
                  setPassword(e.target.value);
                  invalidateConnection();
                }}
                placeholder={browseConnection ? 'SSH 已连接' : '填写远程登录密码'}
              />
            </Field>
          </div>
          <div className="onboard-section-heading">
            <button
              type="button"
              className="button outline small"
              disabled={busy || !!browseConnection}
              onClick={() => void connectSsh()}
            >
              {busy && sshAction === 'connect' ? (
                <LoaderCircle className="spin" size={15} />
              ) : browseConnection ? (
                <CheckCircle2 size={15} />
              ) : (
                <Server size={15} />
              )}
              {browseConnection ? 'SSH 已连接，可联想目录' : '连接并读取目录'}
            </button>
            {browseConnection && (
              <small>{browseConnection.hostname || browseConnection.host || host}</small>
            )}
          </div>
          <SSHPathAutocomplete
            label="Agent 安装目录"
            value={installDir}
            connection={browseConnection}
            disabled={busy}
            hint="先连接 SSH 即可联想目录，也可手动填写新目录。优先创建 Agent 专用 Conda 环境；没有 Conda 时创建 venv。"
            onChange={(value) => {
              setInstallDir(value);
              if (!dirty.current.has('outputRoot'))
                setOutputRoot(`${value.replace(/\/$/, '')}/outputs`);
              if (!dirty.current.has('stateDir')) setStateDir(`${value.replace(/\/$/, '')}/state`);
              setConnection(undefined);
              submission.current = undefined;
              setError('');
            }}
          />
          <div className="inline-note">
            <Terminal size={16} />
            Agent 将使用 SSH 登录用户 {username || '（待填写）'}{' '}
            运行。中台会检查安装条件，再上传程序并启动服务。
          </div>
          {connection && (
            <section className="onboard-ssh-result" aria-label="SSH 检查结果">
              <div className="onboard-section-heading">
                <strong>{connection.hostname || connection.host}</strong>
                <Badge
                  state={connection.can_install ? 'READY' : 'FAILED'}
                  label={connection.can_install ? 'SSH 连接及安装检查通过' : '安装检查未通过'}
                />
              </div>
              <p className="onboard-fingerprint">
                主机指纹 <code>{connection.fingerprint}</code>
              </p>
              <ul className="runtime-check-list">
                {connection.checks.map((check, index) => (
                  <li className={check.ok ? 'passed' : 'failed'} key={`${check.name}-${index}`}>
                    {check.ok ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
                    <span>{check.message}</span>
                  </li>
                ))}
              </ul>
              {connection.existing_agent && (
                <div className="inline-note green">
                  <CheckCircle2 size={15} />
                  检测到已有 Agent。将更新程序并保留机器身份，已读取原有目录和训练环境配置。
                </div>
              )}
            </section>
          )}
        </div>
      )}
      {step === 1 && (
        <>
          <div className="onboard-target">
            <Server size={16} />
            <span>
              {username}@{host}:{port}
              <small>Agent 安装至 {installDir}</small>
            </span>
            <Badge state="READY" label="SSH 已验证" />
          </div>
          <Field label="机器名称" required hint="用于机器资源列表展示。">
            <input
              value={name}
              onChange={(e) => {
                edited('name');
                setName(e.target.value);
              }}
              placeholder="例如 server-169"
            />
          </Field>
          <div className="inline-note">
            <Terminal size={16} />
            指定该机器已有训练环境中的 Python 可执行文件。Agent 使用独立环境，训练按这里的路径启动。
          </div>
          <div className="onboard-profiles">
            {profiles.map((profile, index) => (
              <section className="onboard-profile" key={index}>
                <div className="onboard-section-heading">
                  <h3>训练环境 {index + 1}</h3>
                  {profiles.length > 1 && (
                    <button
                      className="icon-button"
                      aria-label={`移除训练环境 ${index + 1}`}
                      onClick={() => {
                        edited('profiles');
                        setProfiles((items) => items.filter((_, i) => i !== index));
                      }}
                    >
                      <Trash2 size={15} />
                    </button>
                  )}
                </div>
                <div className="form-grid">
                  <Field label="环境名称" required>
                    <input
                      value={profile.name}
                      onChange={(e) => updateProfile(index, { name: e.target.value })}
                    />
                  </Field>
                  <Field
                    label="环境标识"
                    required
                    hint="同机唯一，使用字母、数字、连字符或下划线。"
                  >
                    <input
                      value={profile.id}
                      onChange={(e) => updateProfile(index, { id: e.target.value })}
                    />
                  </Field>
                </div>
                <SSHPathAutocomplete
                  label="训练 Python 解释器绝对路径"
                  value={profile.python}
                  connection={browseConnection}
                  directoryOnly={false}
                  onChange={(value) => updateProfile(index, { python: value })}
                  placeholder="/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python"
                  hint="选择 bin/python 可执行文件，不能只填虚拟环境目录。"
                />
                <SSHPathAutocomplete
                  label="训练代码目录"
                  value={profile.code_dir}
                  connection={browseConnection}
                  onChange={(value) => updateProfile(index, { code_dir: value })}
                  placeholder="/data2/hzl_workspace_for_pi/openpi-umi"
                  hint="Agent 将从该目录读取训练代码版本并检查实际模块来源。"
                />
                <Field
                  label="训练适配器"
                  required
                  hint="训练配方在创建训练任务时选择，接入机器无需填写。"
                >
                  <select
                    value={profile.adapter}
                    onChange={(e) =>
                      updateProfile(index, {
                        adapter: e.target.value as AgentProfileInput['adapter'],
                        ...(e.target.value !== profile.adapter ? { recipe: undefined } : {}),
                      })
                    }
                  >
                    <option value="openpi-sft">OpenPI SFT</option>
                    <option value="smoke">平台链路测试（模拟）</option>
                  </select>
                </Field>
                {profile.adapter === 'smoke' && (
                  <div className="inline-note amber">
                    仅执行 CPU 模拟负载，用于验证平台链路。代码目录需包含
                    smoke_training.py，不会训练 VLA 模型。
                  </div>
                )}
              </section>
            ))}
          </div>
          <button
            className="button outline small"
            onClick={() => {
              edited('profiles');
              setProfiles((items) => {
                let next = items.length + 1;
                while (items.some((item) => item.id === `openpi-native-${next}`)) next++;
                return [...items, newProfile(next)];
              });
            }}
          >
            <Plus size={15} />
            添加另一个训练环境
          </button>
          <p className="onboard-caption">
            同一台机器可登记多个 Python 环境，创建训练时再选择。环境路径会由 Agent
            实际检查；所选训练配方与本机资产将在训练预检时自动验证。
          </p>

          <section className="onboard-managed-roots" aria-label="受管本地目录">
            {roots.map((path, index) => (
              <div className="onboard-managed-root" key={index}>
                <SSHPathAutocomplete
                  label={`受管本地目录 ${index + 1}`}
                  value={path}
                  connection={browseConnection}
                  onChange={(value) => {
                    edited('roots');
                    setRoots((items) => items.map((item, i) => (i === index ? value : item)));
                  }}
                  hint={
                    index === 0
                      ? '选择机器上已存在的目录，目录可手动编辑。训练代码、数据、模型与输出须在这些目录内。'
                      : '选择机器上已存在的目录，目录可手动编辑。'
                  }
                />
                {roots.length > 1 && (
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`移除受管本地目录 ${index + 1}`}
                    onClick={() => {
                      edited('roots');
                      setRoots((items) => items.filter((_, i) => i !== index));
                    }}
                  >
                    <Trash2 size={15} />
                  </button>
                )}
              </div>
            ))}
            <button
              type="button"
              className="button outline small"
              onClick={() => {
                edited('roots');
                setRoots((items) => [...items, '']);
              }}
            >
              <Plus size={15} />
              添加受管目录
            </button>
          </section>
          <div className="form-grid">
            <SSHPathAutocomplete
              label="训练输出目录"
              value={outputRoot}
              connection={browseConnection}
              onChange={(value) => {
                edited('outputRoot');
                setOutputRoot(value);
              }}
            />
            <SSHPathAutocomplete
              label="Agent 状态目录"
              value={stateDir}
              connection={browseConnection}
              hint="保存机器身份与任务记录；更新时保留原目录。"
              onChange={(value) => {
                edited('stateDir');
                setStateDir(value);
              }}
            />
          </div>
          <div className="onboard-connection-status">
            <CheckCircle2 size={15} />
            <div>
              <strong>{customUrl ? '已启用自定义中台地址' : '中台连接由平台自动管理'}</strong>
              <p>需要 SSH 隧道时由中台建立并维护，无需手动配置。</p>
            </div>
          </div>
          <button
            className="onboard-advanced-toggle"
            aria-expanded={advanced}
            aria-controls={advancedId}
            onClick={() => setAdvanced((value) => !value)}
          >
            {advanced ? <ChevronDown size={15} /> : <ChevronRight size={15} />}高级连接设置
          </button>
          {advanced && (
            <section className="onboard-advanced" id={advancedId}>
              {defaults?.control_plane_url && (
                <p className="onboard-default-address">
                  平台默认地址：<code>{defaults.control_plane_url}</code>
                </p>
              )}
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={customUrl}
                  onChange={(e) => {
                    setCustomUrl(e.target.checked);
                    edited('url');
                  }}
                />
                自定义中台连接地址
              </label>
              {customUrl && (
                <Field
                  label="Agent 可访问的中台地址"
                  required
                  hint="仅在目标机器使用特殊 HTTPS 网络入口时修改，可包含反向代理路径前缀。"
                >
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => {
                      setUrl(e.target.value);
                      edited('url');
                    }}
                    placeholder="https://vla.example.com"
                  />
                </Field>
              )}
              <p className="onboard-url-help">
                本机地址会由中台通过 SSH 隧道转发；远程 HTTPS 地址由 Agent 直接连接。
              </p>
            </section>
          )}
        </>
      )}
      {step === 2 && job && (
        <div className="onboard-deployment">
          <div
            className={`onboard-deploy-heading ${completed(job) ? 'success' : ['FAILED', 'INTERRUPTED'].includes(job.state) ? 'failed' : ''}`}
          >
            {completed(job) ? (
              <CheckCircle2 size={25} />
            ) : ['FAILED', 'INTERRUPTED'].includes(job.state) ? (
              <AlertTriangle size={25} />
            ) : (
              <LoaderCircle size={25} className="spin" />
            )}
            <div>
              <h3>{deploymentLabel(job)}</h3>
              <p>
                {job.agent.name || job.host || job.node_id || '机器管理记录'}
                {job.host ? ` · ${job.username}@${job.host}:${job.port}` : ''}
              </p>
            </div>
            <Badge
              state={
                job.state === 'RUNNING' || job.state === 'WAITING_AGENT' ? 'STARTING' : job.state
              }
              label={deploymentLabel(job)}
            />
          </div>
          <p className="onboard-caption">
            {job.install_dir ? `安装目录：${job.install_dir} · ` : ''}创建于 {date(job.created_at)}
          </p>
          {(!job.operation || job.operation === 'install') && !retired(job) && (
            <ol className="onboard-deploy-stages">
              {deploymentStages.map(([stage, label], index) => {
                const current = deploymentStages.findIndex(
                  ([value]) => value === deploymentMainStage(job.stage),
                );
                const passed = job.state === 'SUCCEEDED' || current > index;
                const active = current === index && job.state !== 'SUCCEEDED';
                return (
                  <li key={stage} className={passed ? 'done' : active ? 'current' : ''}>
                    <span>{passed ? <Check size={13} /> : index + 1}</span>
                    {label}
                  </li>
                );
              })}
            </ol>
          )}
          {retired(job) && (
            <div className="inline-note green">
              <CheckCircle2 size={16} />
              {job.state === 'UNINSTALLED' || job.tunnel_state === 'UNINSTALLED'
                ? 'Agent 已卸载，训练代码、数据、模型、输出和历史记录保留。'
                : '中台连接和 SSH 隧道已关闭，远端 Agent 保留。再次管理此机器需重新填写 SSH 信息。'}
            </div>
          )}
          {job.operation_result?.backup_dir && (
            <p className="onboard-caption">
              Agent 状态备份：<code>{job.operation_result.backup_dir}</code>
            </p>
          )}
          {retired(job) && job.node_id && (
            <a className="text-link" href={`#/nodes/${job.node_id}`} onClick={close}>
              查看机器历史详情
              <ArrowRight size={14} />
            </a>
          )}
          {job.error && (
            <div className="inline-note red" role="alert">
              <AlertTriangle size={16} />
              <div>
                <strong>{job.error.message}</strong>
                <p className="onboard-caption">
                  错误代码：{job.error.code}。
                  {job.operation === 'uninstall'
                    ? '可重试卸载，或查看日志排查原因。'
                    : '可重试部署，或修改连接与环境配置。'}
                </p>
              </div>
            </div>
          )}
          {pollError && (
            <div className="inline-note amber" role="alert">
              <AlertTriangle size={16} />
              状态读取失败：{pollError} 页面将自动重试，部署仍在中台执行。
            </div>
          )}
          {!retired(job) && job.operation !== 'uninstall' && (
            <div className="onboard-deploy-meta">
              <span>连接方式</span>
              <strong>
                {job.connection_mode === 'ssh-tunnel' ? 'SSH 加密隧道' : '直接连接中台'}
              </strong>
              {job.connection_mode === 'ssh-tunnel' && (
                <Badge
                  state={
                    job.tunnel_state === 'HEALTHY'
                      ? 'ONLINE'
                      : job.tunnel_state === 'FAILED'
                        ? 'FAILED'
                        : 'STARTING'
                  }
                  label={
                    job.tunnel_state === 'HEALTHY'
                      ? '隧道正常'
                      : job.tunnel_state === 'RECONNECTING'
                        ? '隧道重连中'
                        : job.tunnel_state === 'FAILED'
                          ? '隧道异常'
                          : '建立隧道中'
                  }
                />
              )}
            </div>
          )}
          <DeploymentLogs key={job.id} job={job} active={Boolean(isActive)} />
          {isActive && (
            <p className="onboard-caption">
              关闭窗口后中台会继续执行。可从「接入机器 → 最近部署」重新查看进度。
            </p>
          )}
          {job.node_id && !retired(job) && (!job.operation || job.operation === 'install') && (
            <section className="onboard-runtime-results">
              <div className="onboard-connected">
                <strong>{matchedNode?.name || job.agent.name}</strong>
                {matchedNode && <Badge state={matchedNode.status} />}
                <ViewLink to={`/nodes/${job.node_id}`}>查看机器详情</ViewLink>
              </div>
              <p className="onboard-caption">
                {matchedNode?.last_seen && `最近心跳：${date(matchedNode.last_seen)}。`}
                {job.state === 'SUCCEEDED'
                  ? '本次部署已确认上线并通过环境检查。'
                  : '机器当前在线状态不代表本次部署已完成，部署确认以上方进度为准。'}
              </p>
              {checkError && (
                <div className="inline-note amber" role="alert">
                  训练环境状态读取失败：{checkError}
                </div>
              )}
              {!matchedProfiles.length && (
                <p className="onboard-caption">等待本次部署的环境检查。</p>
              )}
              {matchedProfiles.map((profile) => (
                <div className="onboard-profile-check" key={profile.id}>
                  <strong>{profile.name}</strong>
                  <RuntimeCheckView check={runtimeCheckFor(matchedNode, profile.id)} />
                </div>
              ))}
            </section>
          )}
        </div>
      )}
    </Modal>
  );
}
