import { useEffect, useId, useRef, useState } from 'react';
import { CheckCircle2, LoaderCircle, RefreshCw } from 'lucide-react';
import { api, ApiError } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Modal } from './components';
import { PathAutocomplete } from './PathAutocomplete';
import type { Node, Profile, RuntimeCheck } from './types';
import './profile-rebinding.css';

interface Rebinding {
  id: string;
  node_id: string;
  profile_id: string;
  status: 'PENDING' | 'READY' | 'FAILED';
  requested?: { code_dir: string; python: string };
  profile: Profile | null;
  runtime_check: RuntimeCheck | null;
  error: { code: string; message: string } | null;
}

const sshMetadataConnection = (node: Node) =>
  node.metrics?.connection_type === 'local_ssh_metadata_gateway';

export function runtimeBindingSupported(node: Node) {
  const metrics = node.metrics || {};
  if (!Object.hasOwn(metrics, 'capabilities')) return true;
  return Array.isArray(metrics.capabilities) && metrics.capabilities.includes('PROFILE_REBIND');
}

export function runtimeBindingUnchanged(
  node: Node,
  baseline: Profile,
  codeDir: string,
  python: string,
  confirmed?: Profile,
) {
  const samePaths =
    codeDir.trim().replace(/\/+$/, '') === baseline.code_dir.replace(/\/+$/, '') &&
    python.trim() === baseline.python;
  if (!samePaths || !sshMetadataConnection(node)) return samePaths;
  const checks = node.metrics?.runtime_checks;
  const ready =
    Array.isArray(checks) &&
    checks.some(
      (check) =>
        check &&
        typeof check === 'object' &&
        check.profile_id === baseline.id &&
        check.status === 'READY',
    );
  const justConfirmed =
    confirmed?.id === baseline.id &&
    confirmed.node_id === node.id &&
    confirmed.code_dir === baseline.code_dir &&
    confirmed.python === baseline.python;
  return ready || justConfirmed;
}

function unsupportedBindingMessage(node: Node) {
  return sshMetadataConnection(node)
    ? '当前 SSH 数据连接未开放环境重绑，原连接绑定保持不变。'
    : '当前连接不支持重新绑定项目与环境，原绑定保持不变。';
}

const readOnlyBindingError = (cause: unknown) =>
  cause instanceof ApiError && cause.code === 'READ_ONLY_REAL_AGENT';

export function runtimeBindingError(cause: unknown, node: Node): Error {
  return readOnlyBindingError(cause)
    ? new Error(unsupportedBindingMessage(node))
    : cause instanceof Error
      ? cause
      : new Error(String(cause));
}

export function ProfileBindingModeNotice({ node }: { node: Node }) {
  return (
    <>
      <p className="field-hint">
        {sshMetadataConnection(node)
          ? '通过 SSH 校验机器上已存在的 OpenPI 项目和训练 Python 解释器。校验通过后，将绑定保存到当前控制台的数据连接。'
          : '选择机器上已存在的 OpenPI 项目和训练 Python 解释器。Agent 校验通过后保存，新训练使用新的绑定。'}
      </p>
      <p className="field-hint">
        {sshMetadataConnection(node)
          ? '请使用已安装依赖的现有环境。此处保存当前控制台连接的项目与 Python 路径；运行中的任务会阻止修改，绑定变化后需要重新预检。'
          : '训练环境必须已安装依赖；本操作不拉取代码或创建环境。运行中的任务会阻止修改，绑定变化后需要重新预检。'}
      </p>
    </>
  );
}

export function runtimeBindingPathError(value: string) {
  const path = value.trim();
  return !path.startsWith('/') ||
    path === '/' ||
    path.length > 4096 ||
    /[\u0000-\u001f\u007f-\u009f]/.test(path) ||
    path.split('/').includes('..')
    ? '请填写完整的绝对路径，不能使用根目录或包含上级目录、控制字符。'
    : undefined;
}

function checkedOperation(value: Rebinding, node: Node, profile: Profile, operationId?: string) {
  if (
    !value ||
    value.node_id !== node.id ||
    value.profile_id !== profile.id ||
    !value.id ||
    (operationId && value.id !== operationId) ||
    !['PENDING', 'READY', 'FAILED'].includes(value.status) ||
    !value.requested ||
    typeof value.requested.code_dir !== 'string' ||
    typeof value.requested.python !== 'string' ||
    runtimeBindingPathError(value.requested.code_dir) ||
    runtimeBindingPathError(value.requested.python)
  )
    throw new Error(
      sshMetadataConnection(node)
        ? '连接绑定结果与当前机器或运行环境不一致，请重新读取状态。'
        : 'Agent 绑定结果与当前机器或运行环境不一致，请重新读取状态。',
    );
  if (
    value.status === 'READY' &&
    (!value.profile ||
      value.profile.node_id !== node.id ||
      value.profile.id !== profile.id ||
      !value.requested ||
      value.profile.code_dir !== value.requested.code_dir ||
      value.profile.python !== value.requested.python)
  )
    throw new Error(
      sshMetadataConnection(node)
        ? 'SSH 数据连接返回的项目或解释器与本次绑定请求不一致，请重新读取状态。'
        : 'Agent 返回的项目或解释器与本次绑定请求不一致，请重新读取状态。',
    );
  return value;
}

export function ProfileRebinding(props: {
  node: Node;
  profile: Profile;
  inUse?: boolean;
  changed?: () => void;
}) {
  const { can } = useAuth();
  if (
    !can('machines.manage') ||
    props.profile.adapter !== 'openpi-sft' ||
    (props.profile.execution_mode || 'native') !== 'native'
  )
    return null;
  return <ProfileRebindingForm key={`${props.node.id}:${props.profile.id}`} {...props} />;
}

function ProfileRebindingForm({
  node,
  profile,
  inUse = false,
  changed,
}: Parameters<typeof ProfileRebinding>[0]) {
  const [open, setOpen] = useState(false);
  const [baseline, setBaseline] = useState(profile);
  const [codeDir, setCodeDir] = useState(profile.code_dir);
  const [python, setPython] = useState(profile.python);
  const [operation, setOperation] = useState<Rebinding>();
  const [error, setError] = useState<Error>();
  const [reading, setReading] = useState(false);
  const [statusKnown, setStatusKnown] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [unsupportedByServer, setUnsupportedByServer] = useState(false);
  const [reload, setReload] = useState(0);
  const inFlight = useRef(false);
  const alive = useRef(true);
  const callback = useRef(changed);
  callback.current = changed;
  const notified = useRef<string | undefined>(undefined);
  const formId = useId();
  const pending = operation?.status === 'PENDING';
  const sshConnection = sshMetadataConnection(node);
  const capabilityUnsupported = !runtimeBindingSupported(node);
  const bindingUnsupported = capabilityUnsupported || unsupportedByServer;
  const capabilitySignature = Object.hasOwn(node.metrics || {}, 'capabilities')
    ? JSON.stringify(node.metrics?.capabilities)
    : 'legacy';
  const pathError = runtimeBindingPathError(codeDir) || runtimeBindingPathError(python);
  const unchanged = runtimeBindingUnchanged(
    node,
    baseline,
    codeDir,
    python,
    operation?.status === 'READY' && operation.profile ? operation.profile : undefined,
  );
  const unavailable = node.status !== 'ONLINE' || inUse;
  const endpoint = `/nodes/${encodeURIComponent(node.id)}/profile-rebindings`;

  useEffect(() => {
    setUnsupportedByServer(false);
  }, [capabilitySignature, node.metrics?.connection_type]);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const accept = (value: Rebinding, id?: string) => {
    const next = checkedOperation(value, node, profile, id);
    setOperation(next);
    if (next.status === 'FAILED')
      setError(new Error(next.error?.message || '重新绑定失败，原配置继续保留。'));
    if (next.status === 'READY' && next.profile) {
      setBaseline(next.profile);
      setCodeDir(next.profile.code_dir);
      setPython(next.profile.python);
      setError(undefined);
      if (notified.current !== next.id) {
        notified.current = next.id;
        callback.current?.();
      }
    }
    return next;
  };

  // Reopen after navigation or a lost response by asking the server which operation owns this profile.
  useEffect(() => {
    if (!open) return;
    let live = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    setReading(true);
    setStatusKnown(false);
    api<{ operation: Rebinding | null }>(
      `${endpoint}?profile_id=${encodeURIComponent(profile.id)}`,
      { signal: controller.signal },
    )
      .then((value) => {
        if (!live) return;
        if (!value || !Object.hasOwn(value, 'operation'))
          throw new Error('无法确认当前绑定状态，请重新读取。');
        const latest = value.operation ? checkedOperation(value.operation, node, profile) : null;
        if (latest?.status === 'PENDING') {
          const next = accept(latest);
          if (next.requested) {
            setCodeDir(next.requested.code_dir);
            setPython(next.requested.python);
          }
        }
        setStatusKnown(true);
      })
      .catch((cause) => {
        if (live) {
          if (readOnlyBindingError(cause)) setUnsupportedByServer(true);
          setError(runtimeBindingError(cause, node));
        }
      })
      .finally(() => {
        clearTimeout(timeout);
        if (live) setReading(false);
      });
    return () => {
      live = false;
      clearTimeout(timeout);
      controller.abort();
    };
  }, [open, endpoint, profile.id, reload]);

  useEffect(() => {
    if (!open || !operation || operation.status !== 'PENDING') return;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController | undefined;
    const poll = async () => {
      controller = new AbortController();
      const timeout = setTimeout(() => controller?.abort(), 10000);
      try {
        const result = await api<Rebinding>(`${endpoint}/${encodeURIComponent(operation.id)}`, {
          signal: controller.signal,
        });
        if (!live) return;
        setError(undefined);
        if (accept(result, operation.id).status === 'PENDING') timer = setTimeout(poll, 1200);
      } catch (cause) {
        if (live) {
          if (readOnlyBindingError(cause)) setUnsupportedByServer(true);
          setError(runtimeBindingError(cause, node));
        }
      } finally {
        clearTimeout(timeout);
      }
    };
    timer = setTimeout(poll, 500);
    return () => {
      live = false;
      clearTimeout(timer);
      controller?.abort();
    };
  }, [open, operation?.id, operation?.status, endpoint, reload]);

  const submit = async () => {
    if (
      inFlight.current ||
      pending ||
      reading ||
      !statusKnown ||
      pathError ||
      unchanged ||
      unavailable ||
      bindingUnsupported
    )
      return;
    inFlight.current = true;
    setSubmitting(true);
    setError(undefined);
    try {
      const result = await api<Rebinding>(endpoint, {
        method: 'POST',
        body: JSON.stringify({
          profile_id: profile.id,
          expected_code_dir: baseline.code_dir,
          expected_python: baseline.python,
          code_dir: codeDir.trim().replace(/\/+$/, ''),
          python: python.trim(),
        }),
      });
      if (alive.current) accept(result);
    } catch (cause) {
      if (alive.current) {
        if (readOnlyBindingError(cause)) setUnsupportedByServer(true);
        setError(runtimeBindingError(cause, node));
        setStatusKnown(false);
        setReload((value) => value + 1);
      }
    } finally {
      inFlight.current = false;
      if (alive.current) setSubmitting(false);
    }
  };
  const close = () => {
    if (!inFlight.current) setOpen(false);
  };
  return (
    <>
      <button
        type="button"
        className="button outline small"
        disabled={unavailable || bindingUnsupported || submitting || pending}
        title={
          inUse
            ? '该环境有未结束的训练任务，结束后才能重新绑定。'
            : node.status !== 'ONLINE'
              ? '机器在线后才能重新绑定。'
              : bindingUnsupported
                ? unsupportedBindingMessage(node)
                : undefined
        }
        onClick={() => {
          setBaseline(profile);
          setCodeDir(profile.code_dir);
          setPython(profile.python);
          setError(undefined);
          setReading(true);
          setStatusKnown(false);
          setOpen(true);
        }}
      >
        <RefreshCw size={14} />
        {pending ? '环境绑定处理中…' : '重新绑定项目与环境'}
      </button>
      {bindingUnsupported && !open && (
        <p className="field-hint" role="status">
          {unsupportedBindingMessage(node)}
        </p>
      )}
      {pending && !open && (
        <button
          type="button"
          className="button outline small"
          onClick={() => {
            setReading(true);
            setStatusKnown(false);
            setOpen(true);
          }}
        >
          查看绑定状态
        </button>
      )}
      {open && (
        <Modal
          title="重新绑定 OpenPI 项目与环境"
          subtitle={`${node.name} · ${profile.name}`}
          close={close}
          className="profile-rebind-modal"
          footer={
            <>
              <button
                type="button"
                className="button outline"
                disabled={submitting}
                onClick={close}
              >
                {pending ? '关闭并稍后查看' : '关闭'}
              </button>
              <button
                type="submit"
                form={formId}
                className="button primary"
                disabled={
                  submitting ||
                  reading ||
                  !statusKnown ||
                  pending ||
                  unavailable ||
                  bindingUnsupported ||
                  !!pathError ||
                  unchanged
                }
              >
                {(submitting || pending) && <LoaderCircle size={15} className="spin" />}
                {pending
                  ? sshConnection
                    ? '等待 SSH 连接确认…'
                    : '等待 Agent 确认…'
                  : '校验并重新绑定'}
              </button>
            </>
          }
        >
          <ProfileBindingModeNotice node={node} />
          {capabilityUnsupported && (
            <p className="inline-note" role="status">
              {unsupportedBindingMessage(node)}
            </p>
          )}
          <ErrorNotice
            error={error}
            title="环境绑定未完成"
            retry={() => {
              setError(undefined);
              setUnsupportedByServer(false);
              setReload((value) => value + 1);
            }}
          />
          {pending && (
            <div className="inline-note" role="status">
              <LoaderCircle size={16} className="spin" />
              <div>
                <strong>
                  {sshConnection
                    ? '正在通过 SSH 校验并保存当前控制台连接绑定'
                    : 'Agent 正在校验并保存绑定'}
                </strong>
                <p>
                  {operation.error?.message ||
                    (sshConnection
                      ? '当前连接绑定保持锁定，等待 SSH 校验与保存结果。可关闭窗口后重新查看状态。'
                      : '原运行环境保持锁定，等待 Agent 确认结果。可关闭窗口后重新查看状态。')}
                </p>
              </div>
            </div>
          )}
          {operation?.status === 'READY' && (
            <div className="inline-note green" role="status">
              <CheckCircle2 size={16} />
              <strong>
                {sshConnection
                  ? '项目与 Python 路径已保存到当前控制台连接，后续训练需要重新预检。'
                  : '项目与训练环境已重新绑定，后续训练需要重新预检。'}
              </strong>
            </div>
          )}
          {unavailable && (
            <p className="inline-note red" role="alert">
              {inUse
                ? '该环境有未结束的训练任务，当前不能修改绑定。'
                : '机器已离线，等待恢复在线后再操作。'}
            </p>
          )}
          <form
            id={formId}
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
          >
            <PathAutocomplete
              label="OpenPI 项目目录"
              node={node}
              value={codeDir}
              onChange={setCodeDir}
              disabled={submitting || pending || reading || bindingUnsupported}
              placeholder="例如 /data4/openpi-umi"
              hint="选择项目根目录，需包含 src/openpi/training/config.py。"
            />
            <PathAutocomplete
              label="OpenPI 环境 Python 解释器"
              node={node}
              value={python}
              onChange={setPython}
              disabled={submitting || pending || reading || bindingUnsupported}
              placeholder="例如 /data4/openpi-umi/.venv/bin/python"
              hint="选择现有虚拟环境或 Conda 环境的 bin/python，可直接输入完整路径。"
            />
            {pathError && <p className="field-hint">{pathError}</p>}
          </form>
        </Modal>
      )}
    </>
  );
}
