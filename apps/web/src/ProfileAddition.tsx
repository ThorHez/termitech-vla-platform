import { useEffect, useId, useRef, useState } from 'react';
import { CheckCircle2, LoaderCircle, Plus } from 'lucide-react';
import { api, ApiError } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Field, Modal } from './components';
import { PathAutocomplete } from './PathAutocomplete';
import { runtimeBindingPathError } from './ProfileRebinding';
import type { Node, Profile, RuntimeCheck } from './types';
import './profile-rebinding.css';

export interface ProfileAdditionRequest {
  profile_id: string;
  name: string;
  code_dir: string;
  python: string;
}

export interface ProfileAdditionOperation {
  id: string;
  node_id: string;
  profile_id: string;
  action: 'add';
  status: 'PENDING' | 'READY' | 'FAILED';
  requested: { name: string; code_dir: string; python: string };
  profile: Profile | null;
  runtime_check: RuntimeCheck | null;
  error: { code: string; message: string } | null;
}

const pendingRequests = new Map<string, ProfileAdditionRequest>();
const storageKey = (key: string) => `termitech.profile-addition.${key}`;

function rememberRequest(key: string, request?: ProfileAdditionRequest) {
  if (request) pendingRequests.set(key, request);
  else pendingRequests.delete(key);
  try {
    if (request) sessionStorage.setItem(storageKey(key), JSON.stringify(request));
    else sessionStorage.removeItem(storageKey(key));
  } catch {
    // The tab can still recover an interrupted request when storage is unavailable.
  }
}

function readRequest(key: string) {
  if (pendingRequests.has(key)) return pendingRequests.get(key);
  try {
    const raw = sessionStorage.getItem(storageKey(key));
    if (!raw) return;
    const value = JSON.parse(raw) as ProfileAdditionRequest;
    if (
      /^[A-Za-z0-9_-]{1,128}$/.test(value.profile_id) &&
      typeof value.name === 'string' &&
      !!value.name.trim() &&
      typeof value.code_dir === 'string' &&
      typeof value.python === 'string' &&
      !runtimeBindingPathError(value.code_dir) &&
      !runtimeBindingPathError(value.python)
    ) {
      pendingRequests.set(key, value);
      return value;
    }
  } catch {
    // A damaged or unavailable browser draft does not affect Agent configuration.
  }
}

export function profileAdditionSupported(node: Node) {
  return (
    Array.isArray(node.metrics?.capabilities) && node.metrics.capabilities.includes('PROFILE_ADD')
  );
}

export function profileAdditionRequest(
  profileId: string,
  name: string,
  codeDir: string,
  python: string,
): ProfileAdditionRequest {
  const cleanPath = (value: string) =>
    `/${value
      .trim()
      .split('/')
      .filter((part) => part && part !== '.')
      .join('/')}`;
  return {
    profile_id: profileId,
    name: name.trim(),
    code_dir: cleanPath(codeDir),
    python: cleanPath(python),
  };
}

export function checkedProfileAdditionOperation(
  value: ProfileAdditionOperation,
  node: Node,
  request: ProfileAdditionRequest,
  operationId?: string,
) {
  const failure = () => new Error('新增环境的回执与当前机器或提交内容不一致，请重新读取状态。');
  if (
    !value ||
    typeof value.id !== 'string' ||
    !value.id ||
    (operationId && value.id !== operationId) ||
    value.node_id !== node.id ||
    value.profile_id !== request.profile_id ||
    value.action !== 'add' ||
    !['PENDING', 'READY', 'FAILED'].includes(value.status) ||
    !value.requested ||
    value.requested.name !== request.name ||
    value.requested.code_dir !== request.code_dir ||
    value.requested.python !== request.python
  )
    throw failure();
  if (
    value.status === 'READY' &&
    (!value.profile ||
      value.profile.node_id !== node.id ||
      value.profile.id !== request.profile_id ||
      value.profile.name !== request.name ||
      value.profile.code_dir !== request.code_dir ||
      value.profile.python !== request.python ||
      value.profile.adapter !== 'openpi-sft' ||
      (value.profile.execution_mode || 'native') !== 'native' ||
      value.runtime_check?.profile_id !== request.profile_id ||
      value.runtime_check?.status !== 'READY')
  )
    throw failure();
  return value;
}

export function ProfileAddition(props: { node: Node; changed?: () => void }) {
  const { can, user } = useAuth();
  if (!can('machines.manage')) return null;
  const recoveryKey = `${user?.id || 'session'}:${props.node.id}`;
  return <ProfileAdditionForm key={recoveryKey} {...props} recoveryKey={recoveryKey} />;
}

function ProfileAdditionForm({
  node,
  changed,
  recoveryKey,
}: Parameters<typeof ProfileAddition>[0] & { recoveryKey: string }) {
  const [open, setOpen] = useState(false);
  const [request, setRequest] = useState(() => readRequest(recoveryKey));
  const [name, setName] = useState(request?.name || '');
  const [codeDir, setCodeDir] = useState(request?.code_dir || '');
  const [python, setPython] = useState(request?.python || '');
  const [operation, setOperation] = useState<ProfileAdditionOperation>();
  const [error, setError] = useState<Error>();
  const [reading, setReading] = useState(false);
  const [statusKnown, setStatusKnown] = useState(!request);
  const [submitting, setSubmitting] = useState(false);
  const [reload, setReload] = useState(0);
  const inFlight = useRef(false);
  const alive = useRef(true);
  const notified = useRef<string | undefined>(undefined);
  const callback = useRef(changed);
  callback.current = changed;
  const formId = useId();
  const endpoint = `/nodes/${encodeURIComponent(node.id)}/profile-rebindings`;
  const pending = operation?.status === 'PENDING';
  const complete = operation?.status === 'READY';
  const unresolved = !!request && !complete && operation?.status !== 'FAILED';
  const unsupported = !profileAdditionSupported(node);
  const unavailable = node.status !== 'ONLINE';
  const pathError = runtimeBindingPathError(codeDir) || runtimeBindingPathError(python);
  const invalidName =
    !name.trim() || name.trim().length > 200 || /[\u0000-\u001f\u007f-\u009f]/.test(name);
  const fieldsDisabled = submitting || reading || unresolved || complete;

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const accept = (
    value: ProfileAdditionOperation,
    submitted: ProfileAdditionRequest,
    operationId?: string,
  ) => {
    const next = checkedProfileAdditionOperation(value, node, submitted, operationId);
    setOperation(next);
    setStatusKnown(true);
    setError(
      next.status === 'FAILED'
        ? new Error(next.error?.message || '新增环境失败，请核对目录和解释器后重试。')
        : undefined,
    );
    if (next.status !== 'PENDING') rememberRequest(recoveryKey);
    if (next.status === 'READY' && notified.current !== next.id) {
      notified.current = next.id;
      callback.current?.();
    }
    return next;
  };

  // Recover by the stable profile ID even if the POST response was lost or this page was reopened.
  useEffect(() => {
    if (!open || !request || complete || inFlight.current) return;
    let live = true;
    const controller = new AbortController();
    setReading(true);
    setStatusKnown(false);
    api<{ operation: ProfileAdditionOperation | null }>(
      `${endpoint}?profile_id=${encodeURIComponent(request.profile_id)}`,
      { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]) },
    )
      .then((value) => {
        if (!live) return;
        if (!value || !Object.hasOwn(value, 'operation'))
          throw new Error('无法确认新增环境的状态，请重新读取。');
        if (value.operation) accept(value.operation, request);
        else setStatusKnown(true);
      })
      .catch((cause) => {
        if (live) setError(cause instanceof Error ? cause : new Error(String(cause)));
      })
      .finally(() => {
        if (live) setReading(false);
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [open, request, endpoint, reload, complete]);

  useEffect(() => {
    if (!open || !pending || !operation || !request) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const value = await api<ProfileAdditionOperation>(
          `${endpoint}/${encodeURIComponent(operation.id)}`,
          { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(10000)]) },
        );
        if (live && accept(value, request, operation.id).status === 'PENDING')
          timer = setTimeout(poll, 1200);
      } catch (cause) {
        if (live) setError(cause instanceof Error ? cause : new Error(String(cause)));
      }
    };
    timer = setTimeout(poll, 500);
    return () => {
      live = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [open, operation?.id, pending, request, endpoint, reload]);

  const submit = async () => {
    if (
      inFlight.current ||
      pending ||
      complete ||
      reading ||
      !statusKnown ||
      unsupported ||
      unavailable ||
      pathError ||
      invalidName
    )
      return;
    const submitted =
      unresolved && request
        ? request
        : profileAdditionRequest(
            request?.profile_id || `openpi-${crypto.randomUUID()}`,
            name,
            codeDir,
            python,
          );
    inFlight.current = true;
    setSubmitting(true);
    setError(undefined);
    rememberRequest(recoveryKey, submitted);
    setRequest(submitted);
    try {
      const value = await api<ProfileAdditionOperation>(
        `/nodes/${encodeURIComponent(node.id)}/profiles`,
        {
          method: 'POST',
          body: JSON.stringify(submitted),
        },
      );
      if (alive.current) accept(value, submitted);
    } catch (cause) {
      if (alive.current) {
        setError(cause instanceof Error ? cause : new Error(String(cause)));
        // An explicit API rejection cannot have committed this request. Network failures must recover first.
        if (cause instanceof ApiError && cause.status >= 400 && cause.status < 500) {
          rememberRequest(recoveryKey);
          setRequest(undefined);
          setStatusKnown(true);
        } else {
          setStatusKnown(false);
          setReload((value) => value + 1);
        }
      }
    } finally {
      inFlight.current = false;
      if (alive.current) setSubmitting(false);
    }
  };

  const close = () => {
    if (!inFlight.current) setOpen(false);
  };
  const begin = () => {
    if (complete) {
      setRequest(undefined);
      setOperation(undefined);
      setName('');
      setCodeDir('');
      setPython('');
      setStatusKnown(true);
    }
    setError(undefined);
    setOpen(true);
  };

  return (
    <>
      <button
        type="button"
        className="button outline small"
        disabled={!request && (unavailable || unsupported)}
        title={
          unsupported
            ? '请更新 Agent 后新增训练环境。'
            : unavailable
              ? '机器在线后才能新增训练环境。'
              : undefined
        }
        onClick={begin}
      >
        <Plus size={14} />
        {unresolved ? '查看新增状态' : '添加训练环境'}
      </button>
      {open && (
        <Modal
          title="添加训练环境"
          subtitle={node.name}
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
              {!complete && (
                <button
                  type="submit"
                  form={formId}
                  className="button primary"
                  disabled={
                    submitting ||
                    pending ||
                    reading ||
                    !statusKnown ||
                    unsupported ||
                    unavailable ||
                    !!pathError ||
                    invalidName
                  }
                >
                  {(submitting || pending) && <LoaderCircle size={15} className="spin" />}
                  {pending ? '正在检查并保存…' : unresolved ? '重试提交' : '检查并添加'}
                </button>
              )}
            </>
          }
        >
          <p className="field-hint">
            登记机器上已有的代码目录和虚拟环境。添加后，创建训练时选择对应环境；现有环境继续保留。
          </p>
          <ErrorNotice
            error={error}
            title="新增训练环境未完成"
            retry={() => setReload((value) => value + 1)}
          />
          {reading && (
            <p className="field-hint" role="status">
              正在读取新增状态…
            </p>
          )}
          {pending && (
            <div className="inline-note" role="status">
              <LoaderCircle size={16} className="spin" />
              <div>
                <strong>Agent 正在检查代码与 Python 并保存环境</strong>
                <p>{operation.error?.message || '可以关闭窗口，稍后继续查看结果。'}</p>
              </div>
            </div>
          )}
          {complete && (
            <div className="inline-note green" role="status">
              <CheckCircle2 size={16} />
              <strong>训练环境已添加，创建训练时可以选择“{operation.profile?.name}”。</strong>
            </div>
          )}
          {unsupported && (
            <p className="inline-note red" role="alert">
              当前 Agent 尚不支持新增训练环境，请先升级 Agent。
            </p>
          )}
          {unavailable && (
            <p className="inline-note red" role="alert">
              机器已离线，恢复在线后可以继续操作。
            </p>
          )}
          <form
            id={formId}
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
          >
            <Field label="环境名称" required hint="使用容易辨认的名称，训练记录会保留所选环境。">
              <input
                required
                maxLength={200}
                value={name}
                onChange={(event) => setName(event.target.value)}
                disabled={fieldsDisabled}
                placeholder="例如：我的 OpenPI 实验环境"
              />
            </Field>
            <PathAutocomplete
              label="OpenPI 代码目录"
              node={node}
              value={codeDir}
              onChange={setCodeDir}
              disabled={fieldsDisabled}
              placeholder="例如 /data/openpi-experiment"
              hint="填写机器上已有的项目根目录，需位于 Agent 受管目录内。"
            />
            <PathAutocomplete
              label="虚拟环境 Python 解释器"
              node={node}
              value={python}
              onChange={setPython}
              disabled={fieldsDisabled}
              placeholder="例如 /data/openpi-experiment/.venv/bin/python"
              hint="支持虚拟环境或 Conda 环境的 bin/python，环境应已安装训练依赖。"
            />
            {!fieldsDisabled && (codeDir || python) && pathError && (
              <p className="field-hint">{pathError}</p>
            )}
          </form>
        </Modal>
      )}
    </>
  );
}
