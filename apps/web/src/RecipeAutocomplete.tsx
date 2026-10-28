import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { BookOpen, LoaderCircle, RefreshCw } from 'lucide-react';
import { api, ApiError } from './api';
import type { Node, Profile, RecipeSuggestion, RecipeSuggestions } from './types';
import {
  currentRecipeEntries,
  recipeSuggestionScope,
  type RecipeSuggestionNamespace,
} from './recipeSuggestions';

const recipeName = /^[A-Za-z0-9_-]{1,255}$/;
const warningMessages: Record<string, string> = {
  DYNAMIC_REGISTRY_UNSUPPORTED: '部分动态生成的配方无法静态列出，可手动填写。',
  INVALID_RECIPE_NAME: '已忽略不符合命名规则的配方。',
  DISCOVERY_LIMIT_REACHED: '部分配方超出扫描上限，可继续输入筛选或手动填写。',
};

function registeredRecipes(profile?: Profile): RecipeSuggestion[] {
  if (!profile) return [];
  const validated = new Set(
    Array.isArray(profile.metadata?.validated_recipes)
      ? profile.metadata.validated_recipes.filter(
          (value): value is string => typeof value === 'string' && recipeName.test(value),
        )
      : [],
  );
  const legacy = profile.recipe;
  if (legacy && recipeName.test(legacy) && profile.metadata?.validated === true)
    validated.add(legacy);
  const names = new Set(validated);
  if (legacy && recipeName.test(legacy)) names.add(legacy);
  return [...names].sort().map((name) => ({
    name,
    validated: validated.has(name),
    source: 'registered',
  }));
}

function failureMessage(error: unknown, codeOnly: boolean) {
  if (error instanceof ApiError) {
    if (['AGENT_RECIPE_SUGGEST_UNSUPPORTED', 'COMMAND_UNSUPPORTED'].includes(error.code))
      return codeOnly
        ? '这台机器的 Agent 尚不支持配置联想，请更新 Agent 后重试。'
        : '这台机器的 Agent 尚不支持配方联想，请更新 Agent。也可以手动填写。';
    if (['NODE_OFFLINE', 'NODE_UNAVAILABLE', 'NODE_NOT_ONLINE'].includes(error.code))
      return codeOnly
        ? '机器当前离线，恢复连接后才能选择代码中的推理配置。'
        : '机器当前离线，暂时无法读取代码中的配方。可以继续手动填写。';
  }
  return error instanceof Error
    ? error.message
    : codeOnly
      ? '推理配置读取失败，请重试。'
      : '配方读取失败，请重试或手动填写。';
}

export function RecipeAutocomplete({
  value,
  onChange,
  node,
  profile,
  onResolvedName,
  disabled = false,
  namespace = 'recipe-suggestions',
  context = 'training',
  codeOnly = false,
  label,
  hint,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  node?: Node;
  profile?: Profile;
  onResolvedName?: (name: string | undefined) => void;
  disabled?: boolean;
  namespace?: RecipeSuggestionNamespace;
  context?: 'training' | 'inference';
  codeOnly?: boolean;
  label?: string;
  hint?: string;
  placeholder?: string;
}) {
  const id = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState<{ scope: string; response: RecipeSuggestions }>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ scope: string; prefix: string; message: string }>();
  const [active, setActive] = useState(-1);
  const [revision, setRevision] = useState(0);
  const [resolved, setResolved] = useState<{ scope: string; name: string }>();
  const resolvedCallback = useRef(onResolvedName);
  resolvedCallback.current = onResolvedName;
  // Include profile contents so a changed checkout or validation list cannot retain old suggestions.
  const profileKey = JSON.stringify([
    profile?.id,
    profile?.node_id,
    profile?.code_dir,
    profile?.python,
    profile?.adapter,
    profile?.execution_mode,
    profile?.recipe,
    profile?.metadata?.validated,
    profile?.metadata?.validated_recipes,
  ]);
  const scope = recipeSuggestionScope(node, profile, namespace);
  const inference = context === 'inference';
  const noun = inference ? '推理配置' : '配方';
  const validPrefix = /^[A-Za-z0-9_-]{0,255}$/.test(value);
  const matchingProfile = !!profile && profile.node_id === node?.id;
  const canQuery = !!node && node.status === 'ONLINE' && matchingProfile && validPrefix;
  const current =
    result?.scope === scope &&
    result.response.node_id === node?.id &&
    result.response.profile_id === profile?.id &&
    result.response.prefix === value
      ? result.response
      : undefined;
  const currentError = error?.scope === scope && error.prefix === value ? error.message : undefined;
  const fallback = useMemo(() => registeredRecipes(profile), [profileKey]);
  const usingRegistered = !codeOnly && !current && (!canQuery || !!currentError) && matchingProfile;
  const keyword = value.toLowerCase();
  const matchingRegistered = fallback
    .filter((entry) => entry.name.toLowerCase().includes(keyword))
    .sort((left, right) => {
      const a = left.name.toLowerCase(),
        b = right.name.toLowerCase();
      const priority = Number(!a.startsWith(keyword)) - Number(!b.startsWith(keyword));
      return (
        priority ||
        (a < b ? -1 : a > b ? 1 : left.name < right.name ? -1 : left.name > right.name ? 1 : 0)
      );
    });
  const discovered = currentRecipeEntries(current, node, profile, value, codeOnly);
  const entries = discovered || (usingRegistered ? matchingRegistered.slice(0, 30) : []);
  const truncated = current?.truncated || (usingRegistered && matchingRegistered.length > 30);
  const exactName =
    discovered?.find((entry) => entry.name === value)?.name ||
    (!codeOnly ? fallback.find((entry) => entry.name === value)?.name : undefined);
  const resolvedName =
    resolved?.scope === scope && resolved.name === value ? resolved.name : undefined;
  useEffect(() => {
    setResolved((previous) =>
      exactName
        ? { scope, name: exactName }
        : previous?.scope === scope && previous.name === value
          ? previous
          : undefined,
    );
  }, [scope, value, exactName]);
  useEffect(() => {
    resolvedCallback.current?.(resolvedName);
  }, [scope, resolvedName]);

  useEffect(() => {
    const closeOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as globalThis.Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', closeOutside);
    return () => document.removeEventListener('pointerdown', closeOutside);
  }, []);

  useEffect(() => {
    if (open && active >= 0)
      document.getElementById(`${id}-option-${active}`)?.scrollIntoView({ block: 'nearest' });
  }, [active, id, open]);

  useEffect(() => {
    setResult(undefined);
    setError(undefined);
    setActive(-1);
    if (!open || !canQuery || disabled || !node || !profile) {
      setLoading(false);
      return;
    }
    const nodeId = node.id;
    const profileId = profile.id;
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
        let response = await api<RecipeSuggestions>(
          `/nodes/${encodeURIComponent(nodeId)}/${namespace}`,
          {
            method: 'POST',
            body: JSON.stringify({ profile_id: profileId, prefix: value, limit: 30 }),
            signal: controller.signal,
          },
        );
        while (response.status === 'PENDING') {
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
          response = await api<RecipeSuggestions>(
            `/nodes/${encodeURIComponent(nodeId)}/${namespace}/${encodeURIComponent(response.id)}`,
            { signal: controller.signal },
          );
        }
        if (response.status === 'FAILED')
          throw new ApiError(
            response.error?.code || 'RECIPE_SUGGEST_FAILED',
            response.error?.message || `${noun}读取失败，请重试。`,
            0,
          );
        if (
          response.node_id !== nodeId ||
          response.profile_id !== profileId ||
          response.prefix !== value
        )
          throw new Error(`${noun}响应与当前机器或运行环境不匹配，请重试。`);
        if (live) setResult({ scope, response });
      } catch (err) {
        if (live)
          setError({
            scope,
            prefix: value,
            message: timedOut
              ? `${noun}读取超过 15 秒，请重试${codeOnly ? '。' : '或手动填写。'}`
              : failureMessage(err, codeOnly),
          });
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
  }, [scope, canQuery, disabled, open, value, revision, namespace, noun, codeOnly]);

  const choose = (entry: RecipeSuggestion) => {
    setResolved({ scope, name: entry.name });
    onChange(entry.name);
    resolvedCallback.current?.(entry.name);
    setOpen(false);
    setResult(undefined);
    setActive(-1);
    inputRef.current?.focus();
  };

  return (
    <div className="field recipe-autocomplete" ref={rootRef}>
      <label className="field-label" htmlFor={id}>
        {label || (inference ? '推理配置名称' : '训练配方')}
        <span className="required"> *</span>
      </label>
      <input
        id={id}
        ref={inputRef}
        required
        maxLength={255}
        pattern="[A-Za-z0-9_-]+"
        disabled={disabled}
        autoComplete="off"
        spellCheck={false}
        className="mono"
        placeholder={placeholder || `输入 config_name，联想所选环境中的${noun}`}
        value={value}
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls={open ? `${id}-options` : undefined}
        aria-activedescendant={
          open && active >= 0 && entries[active] ? `${id}-option-${active}` : undefined
        }
        aria-describedby={`${id}-hint`}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onChange={(event) => {
          setResolved(undefined);
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
            if (entries.length)
              setActive((previous) =>
                event.key === 'ArrowDown'
                  ? (previous + 1) % entries.length
                  : (previous <= 0 ? entries.length : previous) - 1,
              );
          } else if (event.key === 'Enter' && open) {
            event.preventDefault();
            if (active >= 0 && entries[active]) choose(entries[active]);
          } else if (event.key === 'Tab') setOpen(false);
        }}
      />
      {open && (
        <div className="path-suggestion-panel recipe-suggestion-panel">
          <div role="status" className="path-suggestion-status" aria-live="polite">
            {!node ? (
              `选择机器和运行环境后可以联想${noun}${codeOnly ? '。' : '；也可以手动填写。'}`
            ) : !matchingProfile ? (
              '请先选择这台机器上的运行环境。'
            ) : !validPrefix ? (
              `${noun}名称仅支持字母、数字、下划线和连字符。`
            ) : node.status !== 'ONLINE' ? (
              `机器当前离线，暂时无法读取代码中的${noun}${codeOnly ? '。' : '。可以继续手动填写。'}`
            ) : loading ? (
              <>
                <LoaderCircle size={14} className="spin" />
                正在读取所选环境中的{inference ? '推理配置' : '训练配方'}…
              </>
            ) : currentError ? (
              <>
                <span>{currentError}</span>
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
                <span>
                  匹配 {entries.length} 个{noun}
                </span>
                <span>↑ ↓ 选择 · Enter 填入</span>
              </>
            ) : (
              `没有匹配的${noun}，${codeOnly ? '请继续输入并选择当前代码中的已有配置。' : '可继续输入或手动填写代码中的 config_name。'}`
            )}
          </div>
          {current?.source && (
            <div className="recipe-suggestion-context">
              代码来源：<code>{`${profile?.code_dir.replace(/\/$/, '')}/${current.source}`}</code>
            </div>
          )}
          {usingRegistered && entries.length > 0 && (
            <div className="recipe-suggestion-context">以下仅为已登记配方，尚未核对本机代码。</div>
          )}
          <div
            id={`${id}-options`}
            role="listbox"
            aria-label={inference ? '推理配置建议' : '训练配方建议'}
            className="path-suggestion-list recipe-suggestion-list"
            aria-busy={loading}
          >
            {entries.map((entry, index) => (
              <button
                type="button"
                role="option"
                aria-selected={active === index}
                tabIndex={-1}
                id={`${id}-option-${index}`}
                key={entry.name}
                className={`path-suggestion-option recipe-suggestion-option ${active === index ? 'active' : ''}`}
                onMouseDown={(event) => event.preventDefault()}
                onMouseMove={() => setActive(index)}
                onClick={() => choose(entry)}
              >
                <BookOpen size={15} />
                <span className="recipe-suggestion-name">
                  <code>{entry.name}</code>
                  <small>{entry.source === 'registered' ? '已登记' : '代码中发现'}</small>
                </span>
                <span className={`recipe-validation ${entry.validated ? 'validated' : ''}`}>
                  {entry.validated ? '已有验收记录' : inference ? '部署时验证' : '预检时验证'}
                </span>
              </button>
            ))}
          </div>
          {truncated && (
            <p className="path-suggestion-truncated">仅显示前 30 项，继续输入可缩小范围。</p>
          )}
          {!inference &&
            current?.warnings.map((warning) =>
              warningMessages[warning] ? (
                <p className="path-suggestion-truncated" key={warning}>
                  {warningMessages[warning]}
                </p>
              ) : null,
            )}
        </div>
      )}
      <span id={`${id}-hint`} className="field-hint">
        {hint ||
          (inference
            ? '输入 config_name 的任意片段搜索，选择目标运行环境代码中的已有配置；应与 checkpoint 的模型结构一致。'
            : '支持输入 config_name 的任意片段，不区分大小写。所选配方由 Agent 在预检时自动验证。')}
        {inference && resolvedName && (
          <span>
            {' '}
            已确认现有配置：<code>{resolvedName}</code>。
          </span>
        )}
      </span>
    </div>
  );
}
