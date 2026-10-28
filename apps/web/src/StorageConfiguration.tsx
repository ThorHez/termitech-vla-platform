import { useEffect, useId, useRef, useState } from 'react';
import { CheckCircle2, HardDrive, LoaderCircle, Settings2 } from 'lucide-react';
import { api, ApiError } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Modal } from './components';
import { PathAutocomplete } from './PathAutocomplete';
import type { Node } from './types';
import './storage-configuration.css';

interface DirectorySpace {
  path: string;
  free_bytes: number;
  total_bytes: number;
}

export interface StorageSettings {
  output_root: string;
  cache_root: string | null;
}

export interface StorageOperation {
  id: string;
  node_id: string;
  status: 'PENDING' | 'READY' | 'FAILED';
  requested: StorageSettings;
  storage:
    | (StorageSettings & {
        directories: { output: DirectorySpace; cache: DirectorySpace | null };
      })
    | null;
  error: { code: string; message: string } | null;
}

export function storageConfigurationSupported(node: Node) {
  const metrics = node.metrics || {};
  if (!Object.hasOwn(metrics, 'capabilities')) return true;
  return Array.isArray(metrics.capabilities) && metrics.capabilities.includes('STORAGE_CONFIGURE');
}

const cleanPath = (value: string) => value.trim().replace(/\/+/g, '/').replace(/\/+$/, '');

export function storageDirectoryError(value: string, optional = false) {
  const path = value.trim();
  if (optional && !path) return undefined;
  return !path.startsWith('/') ||
    !cleanPath(path) ||
    path.length > 4096 ||
    /[\\\u0000-\u001f\u007f-\u009f]/.test(path) ||
    path.split('/').some((part) => part === '..' || part === '.')
    ? '请填写完整的本地绝对目录，不能使用根目录、上级目录或控制字符。'
    : undefined;
}

export function storageSettings(outputRoot: string, cacheRoot: string): StorageSettings {
  return { output_root: cleanPath(outputRoot), cache_root: cleanPath(cacheRoot) || null };
}

const sameSettings = (left: StorageSettings, right: StorageSettings) =>
  left.output_root === right.output_root && left.cache_root === right.cache_root;

export function checkedStorageOperation(value: StorageOperation, node: Node, operationId?: string) {
  const failure = () => new Error('目录设置回执与当前机器或请求不一致，请重新读取状态。');
  if (
    !value ||
    value.node_id !== node.id ||
    typeof value.id !== 'string' ||
    !value.id ||
    (operationId && value.id !== operationId) ||
    !['PENDING', 'READY', 'FAILED'].includes(value.status) ||
    !value.requested ||
    typeof value.requested.output_root !== 'string' ||
    storageDirectoryError(value.requested.output_root) ||
    (value.requested.cache_root !== null &&
      (typeof value.requested.cache_root !== 'string' ||
        !value.requested.cache_root ||
        storageDirectoryError(value.requested.cache_root)))
  )
    throw failure();
  if (value.status === 'READY') {
    const saved = value.storage;
    const directoryValid = (directory: DirectorySpace | null | undefined, path: string) =>
      !!directory &&
      directory.path === path &&
      Number.isSafeInteger(directory.free_bytes) &&
      directory.free_bytes >= 0 &&
      Number.isSafeInteger(directory.total_bytes) &&
      directory.total_bytes >= directory.free_bytes;
    if (
      !saved ||
      !sameSettings(saved, value.requested) ||
      !directoryValid(saved.directories?.output, saved.output_root) ||
      (saved.cache_root === null
        ? saved.directories.cache !== null
        : !directoryValid(saved.directories?.cache, saved.cache_root))
    )
      throw failure();
  }
  return value;
}

export function storageConfigurationError(cause: unknown): Error {
  if (
    cause instanceof ApiError &&
    ['READ_ONLY_REAL_AGENT', 'AGENT_STORAGE_CONFIGURE_UNSUPPORTED', 'COMMAND_UNSUPPORTED'].includes(
      cause.code,
    )
  )
    return new Error('当前连接尚不支持修改存储目录，请更新 Agent 后重试。');
  return cause instanceof Error ? cause : new Error(String(cause));
}

export function StorageConfiguration(props: { node: Node; inUse?: boolean; changed?: () => void }) {
  const { can } = useAuth();
  if (!can('machines.manage')) return null;
  return <StorageConfigurationForm key={props.node.id} {...props} />;
}

function StorageConfigurationForm({
  node,
  inUse = false,
  changed,
}: Parameters<typeof StorageConfiguration>[0]) {
  const [open, setOpen] = useState(false);
  const [baseline, setBaseline] = useState<StorageSettings>({
    output_root: node.output_root || '',
    cache_root: node.cache_root || null,
  });
  const [outputRoot, setOutputRoot] = useState(node.output_root || '');
  const [cacheRoot, setCacheRoot] = useState(node.cache_root || '');
  const [operation, setOperation] = useState<StorageOperation>();
  const [error, setError] = useState<Error>();
  const [reading, setReading] = useState(false);
  const [statusKnown, setStatusKnown] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [reload, setReload] = useState(0);
  const alive = useRef(true);
  const inFlight = useRef(false);
  const notified = useRef<string | undefined>(undefined);
  const callback = useRef(changed);
  callback.current = changed;
  const formId = useId();
  const pending = operation?.status === 'PENDING';
  const unsupported = !storageConfigurationSupported(node);
  const unavailable = node.status !== 'ONLINE' || inUse;
  const pathError = storageDirectoryError(outputRoot) || storageDirectoryError(cacheRoot, true);
  const target = storageSettings(outputRoot, cacheRoot);
  const unchanged = sameSettings(target, baseline);
  const endpoint = `/nodes/${encodeURIComponent(node.id)}/storage-configurations`;

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const accept = (value: StorageOperation, id?: string) => {
    const next = checkedStorageOperation(value, node, id);
    setOperation(next);
    if (next.status === 'FAILED')
      setError(new Error(next.error?.message || '目录设置未完成，请检查目标目录后重试。'));
    if (next.status === 'READY' && next.storage) {
      setBaseline(next.storage);
      setOutputRoot(next.storage.output_root);
      setCacheRoot(next.storage.cache_root || '');
      setError(undefined);
      if (notified.current !== next.id) {
        notified.current = next.id;
        callback.current?.();
      }
    }
    return next;
  };

  // A lost POST response or closed dialog must recover the operation before another save.
  useEffect(() => {
    if (!open) return;
    let live = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    setReading(true);
    setStatusKnown(false);
    api<{ operation: StorageOperation | null }>(endpoint, { signal: controller.signal })
      .then((value) => {
        if (!live) return;
        if (!value || !Object.hasOwn(value, 'operation'))
          throw new Error('无法确认当前目录设置状态，请重新读取。');
        const latest = value.operation ? checkedStorageOperation(value.operation, node) : null;
        if (latest?.status === 'PENDING') {
          accept(latest);
          setOutputRoot(latest.requested.output_root);
          setCacheRoot(latest.requested.cache_root || '');
        } else if (
          latest &&
          ((operation?.status === 'PENDING' && latest.id === operation.id) ||
            (latest.status === 'READY' &&
              sameSettings(latest.requested, storageSettings(outputRoot, cacheRoot))))
        ) {
          accept(latest);
        }
        setStatusKnown(true);
      })
      .catch((cause) => {
        if (live) setError(storageConfigurationError(cause));
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
  }, [open, endpoint, reload]);

  useEffect(() => {
    if (!open || !operation || operation.status !== 'PENDING') return;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController | undefined;
    const poll = async () => {
      controller = new AbortController();
      const timeout = setTimeout(() => controller?.abort(), 10000);
      try {
        const result = await api<StorageOperation>(
          `${endpoint}/${encodeURIComponent(operation.id)}`,
          {
            signal: controller.signal,
          },
        );
        if (!live) return;
        setError(undefined);
        if (accept(result, operation.id).status === 'PENDING') timer = setTimeout(poll, 1200);
      } catch (cause) {
        if (live) setError(storageConfigurationError(cause));
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
  }, [open, endpoint, operation?.id, operation?.status, reload]);

  const submit = async () => {
    if (
      inFlight.current ||
      pending ||
      reading ||
      !statusKnown ||
      pathError ||
      unchanged ||
      unavailable ||
      unsupported
    )
      return;
    inFlight.current = true;
    setSubmitting(true);
    setError(undefined);
    try {
      const result = await api<StorageOperation>(endpoint, {
        method: 'POST',
        body: JSON.stringify({
          expected_output_root: baseline.output_root,
          expected_cache_root: baseline.cache_root,
          ...target,
        }),
      });
      checkedStorageOperation(result, node);
      if (!sameSettings(result.requested, target))
        throw new Error('机器返回的目录与本次保存内容不一致，请重新读取状态。');
      if (alive.current) accept(result);
    } catch (cause) {
      if (alive.current) {
        setError(storageConfigurationError(cause));
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
  const disabled = submitting || pending || reading || unsupported || unavailable;
  return (
    <>
      <div className="storage-configuration-entry">
        <button
          type="button"
          className="button outline small"
          disabled={unavailable || unsupported || submitting}
          title={
            inUse
              ? '机器有未结束的训练任务，结束后才能修改存储目录。'
              : node.status !== 'ONLINE'
                ? '机器在线后才能修改存储目录。'
                : unsupported
                  ? '请更新 Agent 以启用存储目录修改。'
                  : undefined
          }
          onClick={() => {
            const current = {
              output_root: node.output_root || '',
              cache_root: node.cache_root || null,
            };
            setBaseline(current);
            setOutputRoot(current.output_root);
            setCacheRoot(current.cache_root || '');
            setError(undefined);
            setReading(true);
            setStatusKnown(false);
            setOpen(true);
          }}
        >
          <Settings2 size={14} />
          {pending ? '查看目录设置状态' : '修改存储目录'}
        </button>
        {unsupported && (
          <p className="field-hint storage-unsupported" role="status">
            当前连接尚不支持修改存储目录，请更新 Agent 后重试。
          </p>
        )}
      </div>
      {open && (
        <Modal
          title="修改训练存储目录"
          subtitle={node.name}
          close={close}
          className="storage-configuration-modal"
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
                disabled={disabled || !statusKnown || !!pathError || unchanged}
              >
                {(submitting || pending) && <LoaderCircle size={15} className="spin" />}
                {pending ? '等待机器确认…' : '校验并保存'}
              </button>
            </>
          }
        >
          <p className="field-hint">
            新目录用于后续训练。已有输出和缓存保留在原位置，不会自动迁移；切换缓存目录后可能需要重新生成缓存。
          </p>
          <p className="field-hint">
            保存前检查目标目录可写性与可用空间，保存后需要重新预检。机器有未结束的训练或部署时不能修改。
          </p>
          <ErrorNotice
            error={error}
            title="目录设置未完成"
            retry={() => {
              setError(undefined);
              setReload((value) => value + 1);
            }}
          />
          {pending && (
            <div className="inline-note" role="status">
              <LoaderCircle size={16} className="spin" />
              <div>
                <strong>正在检查并保存目录</strong>
                <p>{operation.error?.message || '等待机器确认，可关闭窗口后重新查看状态。'}</p>
              </div>
            </div>
          )}
          {operation?.status === 'READY' && (
            <div className="inline-note green" role="status">
              <CheckCircle2 size={16} />
              <strong>存储目录已保存，后续训练请重新预检。</strong>
            </div>
          )}
          {unavailable && (
            <p className="inline-note red" role="alert">
              {inUse
                ? '机器有未结束的训练任务，当前不能修改目录。'
                : '机器已离线，恢复在线后才能保存。'}
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
              label="训练输出目录"
              node={node}
              value={outputRoot}
              onChange={setOutputRoot}
              disabled={disabled}
              placeholder="例如 /data3/vla/outputs"
              hint={`当前目录：${baseline.output_root || '未配置'}`}
            />
            <PathAutocomplete
              label="训练缓存根目录"
              node={node}
              value={cacheRoot}
              onChange={setCacheRoot}
              disabled={disabled}
              required={false}
              placeholder="留空使用训练项目的默认缓存目录"
              hint={`当前设置：${baseline.cache_root || '默认跟随训练项目'}。此目录统一存放数据集、模型、编译和临时缓存。`}
            />
            {pathError && (
              <p className="field-hint" role="alert">
                {pathError}
              </p>
            )}
          </form>
          {operation?.status === 'READY' && operation.storage && (
            <div className="storage-directory-space" aria-label="校验时目标目录磁盘空间">
              {[operation.storage.directories.output, operation.storage.directories.cache]
                .filter((directory): directory is DirectorySpace => !!directory)
                .map((directory, index) => (
                  <div key={`${index}:${directory.path}`}>
                    <HardDrive size={16} />
                    <div>
                      <strong>{index === 0 ? '训练输出目录' : '训练缓存目录'}</strong>
                      <code>{directory.path}</code>
                    </div>
                    <span>校验时可用 {(directory.free_bytes / 2 ** 30).toFixed(2)} GiB</span>
                  </div>
                ))}
            </div>
          )}
        </Modal>
      )}
    </>
  );
}
