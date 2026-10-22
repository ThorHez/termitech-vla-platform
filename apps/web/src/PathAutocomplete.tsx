import { useEffect, useId, useRef, useState } from 'react';
import { ChevronRight, File, Folder, LoaderCircle, RefreshCw } from 'lucide-react';
import { api, ApiError } from './api';
import type { Node, PathSuggestion, PathSuggestions, Profile } from './types';

export interface SshPathConnection {
  connection_id: string;
  expires_at: string;
  host?: string;
  hostname?: string;
  port?: number;
  username?: string;
  fingerprint?: string;
}

interface SshPathSuggestions {
  connection_id: string;
  status: 'READY';
  prefix: string;
  parent?: string | null;
  entries: PathSuggestion[];
  truncated: boolean;
}

function validProjectPrefix(value: string) {
  return (
    !value.startsWith('/') &&
    !/[\\\u0000-\u001f\u007f]/.test(value) &&
    value
      .split('/')
      .every((part, index, parts) =>
        part ? part !== '.' && part !== '..' : index === parts.length - 1,
      )
  );
}

export function validTrainingEntrypoint(value: string) {
  return validProjectPrefix(value) && value.endsWith('.py');
}

function failureMessage(error: unknown) {
  if (error instanceof ApiError) {
    if (['AGENT_PATH_SUGGEST_UNSUPPORTED', 'COMMAND_UNSUPPORTED'].includes(error.code))
      return '这台机器的 Agent 尚不支持路径联想，请更新 Agent。也可以直接填写完整路径。';
    if (['NODE_OFFLINE', 'NODE_UNAVAILABLE', 'NODE_NOT_ONLINE'].includes(error.code))
      return '机器当前离线，暂时无法读取目录。可以继续手动填写路径。';
  }
  return error instanceof Error ? error.message : '目录读取失败，请重试或手动填写路径。';
}

export function PathAutocomplete({
  value,
  onChange,
  node,
  disabled = false,
  profile,
  projectRelative = false,
  ssh,
  required = true,
  hint,
  label = '本地绝对路径',
  placeholder = '输入路径，自动读取所选机器的目录',
}: {
  value: string;
  onChange: (value: string) => void;
  node?: Node;
  disabled?: boolean;
  profile?: Profile;
  projectRelative?: boolean;
  ssh?: { connection?: SshPathConnection; directoryOnly?: boolean };
  required?: boolean;
  hint?: string;
  label?: string;
  placeholder?: string;
}) {
  const id = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState<PathSuggestions | SshPathSuggestions>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [active, setActive] = useState(-1);
  const [revision, setRevision] = useState(0);
  const basePath = profile?.code_dir.replace(/\/+$/, '') || '';
  const projectReady = !!profile && profile.node_id === node?.id && basePath.startsWith('/');
  const validPrefix = !projectRelative || validProjectPrefix(value);
  const prefix = projectRelative ? `${basePath}/${value}` : value;
  const connection = ssh?.connection;
  const directoryOnly = ssh?.directoryOnly ?? true;
  const expiry = Date.parse(connection?.expires_at || '');
  const sshExpired = !!connection && (!Number.isFinite(expiry) || expiry <= Date.now());
  const canQuery = ssh
    ? !!connection && !sshExpired
    : !!node && node.status === 'ONLINE' && (!projectRelative || (projectReady && validPrefix));
  // Bind visible results to their input and machine, including between React effects.
  const current =
    result &&
    (ssh
      ? 'connection_id' in result && result.connection_id === connection?.connection_id
      : 'node_id' in result && result.node_id === node?.id) &&
    result?.prefix === prefix &&
    (!projectRelative ||
      ('profile_id' in result &&
        result.profile_id === profile?.id &&
        result.base_path?.replace(/\/+$/, '') === basePath))
      ? result
      : undefined;
  const entries = (current?.entries || []).filter(
    (entry) =>
      (!ssh || !directoryOnly || entry.kind === 'directory') &&
      (!projectRelative ||
        (entry.path.startsWith(`${basePath}/`) &&
          validProjectPrefix(entry.path.slice(basePath.length + 1)) &&
          (entry.kind === 'directory' || entry.path.endsWith('.py')))),
  );
  const displayPath = (path: string) =>
    projectRelative ? path.slice(basePath.length + 1) || './' : path;

  useEffect(() => {
    if (!connection || sshExpired) return;
    const timer = setTimeout(
      () => setRevision((value) => value + 1),
      Math.min(expiry - Date.now() + 1, 2147483647),
    );
    return () => clearTimeout(timer);
  }, [connection?.connection_id, expiry, sshExpired]);

  useEffect(() => {
    const closeOutside = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as globalThis.Node)) setOpen(false);
    };
    // Close after the target receives its click. Closing on blur/pointerdown
    // can shrink the form and move that target before mouseup loses the click.
    document.addEventListener('click', closeOutside);
    return () => document.removeEventListener('click', closeOutside);
  }, []);

  useEffect(() => {
    if (open && active >= 0)
      document.getElementById(`${id}-option-${active}`)?.scrollIntoView({ block: 'nearest' });
  }, [active, id, open]);

  useEffect(() => {
    setResult(undefined);
    setError(undefined);
    setActive(-1);
    if (!open || !canQuery || disabled) {
      setLoading(false);
      return;
    }
    const nodeId = node?.id;
    const connectionId = connection?.connection_id;
    let live = true;
    let timedOut = false;
    let pollTimer: ReturnType<typeof setTimeout>;
    let deadlineTimer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setLoading(true);
    const debounce = setTimeout(async () => {
      deadlineTimer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, 15000);
      try {
        let response = await api<PathSuggestions | SshPathSuggestions>(
          ssh
            ? `/ssh-connections/${encodeURIComponent(connectionId!)}/path-suggestions`
            : `/nodes/${encodeURIComponent(nodeId!)}/path-suggestions`,
          {
            method: 'POST',
            body: JSON.stringify({
              prefix,
              limit: 30,
              ...(ssh ? { directory_only: directoryOnly } : {}),
              ...(projectRelative ? { profile_id: profile?.id } : {}),
            }),
            signal: controller.signal,
          },
        );
        while ('id' in response && response.status === 'PENDING') {
          await new Promise<void>((resolve, reject) => {
            const abort = () => {
              clearTimeout(pollTimer);
              reject(new DOMException('Aborted', 'AbortError'));
            };
            if (controller.signal.aborted) return abort();
            controller.signal.addEventListener('abort', abort, { once: true });
            pollTimer = setTimeout(() => {
              controller.signal.removeEventListener('abort', abort);
              resolve();
            }, 500);
          });
          response = await api<PathSuggestions>(
            `/nodes/${encodeURIComponent(nodeId!)}/path-suggestions/${encodeURIComponent(response.id)}`,
            { signal: controller.signal },
          );
        }
        if (response.status === 'FAILED') {
          const detail = response.error;
          throw new ApiError(
            typeof detail === 'object' && detail ? detail.code : 'PATH_SUGGEST_FAILED',
            typeof detail === 'string' ? detail : detail?.message || '目录读取失败，请重试。',
            0,
          );
        }
        if (live) setResult(response);
      } catch (err) {
        if (live) {
          setError(
            timedOut
              ? `目录读取超过 15 秒，请重试或直接填写${projectRelative ? '项目内相对' : '完整'}路径。`
              : failureMessage(err),
          );
        }
      } finally {
        clearTimeout(deadlineTimer);
        if (live) setLoading(false);
      }
    }, 300);
    return () => {
      live = false;
      clearTimeout(debounce);
      clearTimeout(pollTimer);
      clearTimeout(deadlineTimer);
      controller.abort();
    };
  }, [
    node?.id,
    !!ssh,
    connection?.connection_id,
    directoryOnly,
    canQuery,
    disabled,
    open,
    prefix,
    projectRelative,
    profile?.id,
    basePath,
    revision,
  ]);

  const choose = (entry: PathSuggestion) => {
    const pathValue = projectRelative ? entry.path.slice(basePath.length + 1) : entry.path;
    const path = entry.kind === 'directory' ? `${pathValue.replace(/\/$/, '')}/` : pathValue;
    onChange(path);
    setResult(undefined);
    setActive(-1);
    setOpen(entry.kind === 'directory');
    if (path === value && entry.kind === 'directory') setRevision((previous) => previous + 1);
    inputRef.current?.focus();
  };

  return (
    <div className="field path-autocomplete" ref={rootRef}>
      <label className="field-label" htmlFor={id}>
        {label}
        {required && <span className="required"> *</span>}
      </label>
      <input
        ref={inputRef}
        id={id}
        required={required}
        pattern={projectRelative ? undefined : '/.*'}
        maxLength={4096}
        disabled={disabled || (projectRelative && !projectReady)}
        autoComplete="off"
        spellCheck={false}
        className="mono"
        placeholder={placeholder}
        value={value}
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls={open ? `${id}-options` : undefined}
        aria-activedescendant={open && active >= 0 ? `${id}-option-${active}` : undefined}
        aria-describedby={`${id}-hint`}
        onFocus={() => setOpen(true)}
        onChange={(event) => {
          setResult(undefined);
          setActive(-1);
          onChange(event.target.value);
          setOpen(true);
        }}
        onKeyDown={(event) => {
          if (event.nativeEvent.isComposing) return;
          if (event.key === 'Escape' && open) {
            event.preventDefault();
            event.stopPropagation();
            setOpen(false);
          } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            setOpen(true);
            if (entries.length) {
              setActive((previous) =>
                event.key === 'ArrowDown'
                  ? (previous + 1) % entries.length
                  : (previous <= 0 ? entries.length : previous) - 1,
              );
            }
          } else if (event.key === 'Enter' && open) {
            // Choosing a path must never also submit the asset registration form.
            event.preventDefault();
            if (active >= 0 && entries[active]) choose(entries[active]);
          } else if (event.key === 'Tab') {
            setOpen(false);
          }
        }}
      />
      {open && (
        <div className="path-suggestion-panel">
          <div role="status" className="path-suggestion-status" aria-live="polite">
            {ssh && !connection ? (
              '先连接 SSH 即可联想远程目录，也可以直接填写路径。'
            ) : sshExpired ? (
              'SSH 连接凭据已过期，请重新连接后读取目录；路径仍可手动填写。'
            ) : projectRelative && !projectReady ? (
              '请先选择代码项目对应的本机运行环境。'
            ) : !validPrefix ? (
              '请填写项目内相对路径，不能使用绝对路径或 ..。'
            ) : !ssh && !node ? (
              '选择所在机器后可以联想目录；也可以先手动填写路径。'
            ) : !canQuery ? (
              '机器当前离线，暂时无法读取目录。可以继续手动填写路径。'
            ) : loading ? (
              <>
                <LoaderCircle size={14} className="spin" />
                正在读取机器上的目录…
              </>
            ) : error ? (
              <>
                <span>{error}</span>
                <button
                  type="button"
                  className="path-suggestion-retry"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => setRevision((previous) => previous + 1)}
                >
                  <RefreshCw size={12} />
                  重试
                </button>
              </>
            ) : entries.length ? (
              <>
                {current?.parent ? (
                  <span className="mono">{displayPath(current.parent)}</span>
                ) : projectRelative ? (
                  '当前代码项目'
                ) : ssh ? (
                  'SSH 用户可访问的目录'
                ) : (
                  '允许访问的目录'
                )}
                <span>↑ ↓ 选择 · Enter 填入</span>
              </>
            ) : (
              `没有匹配的路径，可以继续输入或手动填写${projectRelative ? '项目内 .py 文件的相对' : '完整'}路径。`
            )}
          </div>
          <div
            id={`${id}-options`}
            role="listbox"
            aria-label={
              ssh ? `${label}建议` : projectRelative ? '项目内训练脚本建议' : '本机路径建议'
            }
            className="path-suggestion-list"
            aria-busy={loading}
          >
            {entries.map((entry, index) => (
              <button
                type="button"
                role="option"
                aria-selected={active === index}
                tabIndex={-1}
                id={`${id}-option-${index}`}
                key={entry.path}
                className={`path-suggestion-option ${active === index ? 'active' : ''}`}
                title={entry.path}
                onMouseDown={(event) => event.preventDefault()}
                onMouseMove={() => setActive(index)}
                onClick={() => choose(entry)}
              >
                {entry.kind === 'directory' ? <Folder size={15} /> : <File size={15} />}
                <span className="mono">
                  {entry.name}
                  {entry.kind === 'directory' && !entry.name.endsWith('/') && '/'}
                </span>
                {entry.kind === 'directory' && <ChevronRight size={13} />}
              </button>
            ))}
          </div>
          {current?.truncated && (
            <p className="path-suggestion-truncated">仅显示前 30 项，继续输入可缩小范围。</p>
          )}
        </div>
      )}
      <span id={`${id}-hint`} className="field-hint">
        {hint ||
          (ssh
            ? '通过 SSH 读取当前用户可访问的目录。选择目录可继续展开，路径也可手动编辑。'
            : projectRelative
              ? '从当前代码项目内选择 .py 文件，填写相对路径；Agent 预检时会验证文件和训练入口。'
              : node?.roots?.length
                ? `允许目录：${node.roots.join('、')}。选择目录后可继续展开，路径仍可手动编辑。`
                : '填写机器上已经存在的完整路径，Agent 将在预检时验证。')}
      </span>
    </div>
  );
}
