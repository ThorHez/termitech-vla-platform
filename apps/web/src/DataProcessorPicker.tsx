import { useEffect, useId, useState } from 'react';
import { LoaderCircle } from 'lucide-react';
import { api, ApiError } from './api';
import { Field } from './components';
import { assetManagedOverridePaths } from './recipeFieldPolicy';
import { composedProcessingErrors } from './composedProcessing';
import type {
  DataProcessorCatalog,
  DataProcessorCatalogValue,
  DataProcessorChoice,
  DataProcessorEntry,
  DataProcessorParameter,
  DataProcessorSchema,
  DatasetBinding,
  Node,
  Profile,
  RecipeDatasetSlot,
  RecipeLiteral,
} from './types';

function literal(value: unknown, depth = 0): value is RecipeLiteral {
  if (depth > 4) return false;
  if (value === null || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value === 'string')
    return value.length <= 1024 && !/[\u0000-\u0008\u000b-\u001f]/.test(value);
  if (Array.isArray(value))
    return value.length <= 64 && value.every((item) => literal(item, depth + 1));
  return (
    typeof value === 'object' &&
    Object.entries(value).length <= 64 &&
    Object.entries(value).every(
      ([key, item]) => key.length <= 128 && !key.startsWith('_') && literal(item, depth + 1),
    )
  );
}

export function processorDefaults(
  entry: DataProcessorEntry,
  managedParameters: Record<string, RecipeLiteral> = {},
): DataProcessorChoice {
  return {
    name: entry.name,
    parameters: Object.fromEntries(
      entry.parameters
        .filter(
          (parameter) =>
            Object.hasOwn(parameter, 'default') &&
            !Object.hasOwn(managedParameters, parameter.path),
        )
        .map((parameter) => [parameter.path, parameter.default!]),
    ),
  };
}

export function selectBindingProcessor(
  binding: DatasetBinding,
  processor: DataProcessorEntry | number,
  managedParameters: Record<string, RecipeLiteral> = {},
): DatasetBinding {
  const { recipe_index: _recipeIndex, processor: _processor, ...assets } = binding;
  return typeof processor === 'number'
    ? { ...assets, recipe_index: processor }
    : { ...assets, processor: processorDefaults(processor, managedParameters) };
}

export function filterDataProcessors(
  entries: DataProcessorEntry[],
  search: string,
  selected?: string,
) {
  return entries.filter(
    (entry) =>
      (entry.name !== 'ComposableDataConfig' || entry.name === selected) &&
      (entry.name === selected ||
        `${entry.name} ${entry.description}`.toLowerCase().includes(search.trim().toLowerCase())),
  );
}

function matchesSchema(value: RecipeLiteral, schema: DataProcessorSchema, depth = 0): boolean {
  if (depth > 4) return false;
  if (value === null) return !!schema.nullable || schema.type === 'null';
  if (
    schema.choices &&
    !schema.choices.some((option) => JSON.stringify(option) === JSON.stringify(value))
  )
    return false;
  if (schema.type === 'boolean') return typeof value === 'boolean';
  if (schema.type === 'integer') return typeof value === 'number' && Number.isSafeInteger(value);
  if (schema.type === 'number') return typeof value === 'number' && Number.isFinite(value);
  if (schema.type === 'string') return typeof value === 'string';
  if (schema.type === 'array')
    return (
      Array.isArray(value) &&
      (!schema.items || value.every((item) => matchesSchema(item, schema.items!, depth + 1)))
    );
  if (schema.type === 'object')
    return (
      typeof value === 'object' &&
      !Array.isArray(value) &&
      (!schema.values ||
        Object.values(value).every((item) => matchesSchema(item, schema.values!, depth + 1)))
    );
  return false;
}

export function processorParameterErrors(
  choice: DataProcessorChoice,
  entry?: DataProcessorEntry,
  managedParameters: Record<string, RecipeLiteral> = {},
): string[] {
  if (!entry) return ['处理器未出现在当前环境目录中，请重新读取并选择。'];
  if (!entry.available) return [entry.reason || '此处理器有无法从页面提供的必填配置。'];
  if (choice.name === 'ComposableDataConfig') {
    if (entry.name !== choice.name || entry.composition_schema_version !== 1)
      return ['当前训练环境不支持版本 1 的组件组合处理，请更新 Agent 和训练代码。'];
    if (
      Object.keys(choice.parameters).length !== 1 ||
      !Object.hasOwn(choice.parameters, 'composition')
    )
      return ['组件组合只接受本次训练保存的 composition，训练阶段由任务注入。'];
    return composedProcessingErrors(choice.parameters.composition);
  }
  const errors: string[] = [];
  const managed = assetManagedOverridePaths(
    Object.fromEntries(
      Object.entries(choice.parameters).map(([path, value]) => [`data.${path}`, value]),
    ),
  );
  if (managed.length) errors.push('数据、归一化、模型目录及混采权重请在训练资产绑定中设置。');
  const declared = new Map(entry.parameters.map((parameter) => [parameter.path, parameter]));
  for (const path of Object.keys(choice.parameters))
    if (!declared.has(path)) errors.push(`${path} 已不在当前处理器参数中，请重新选择处理器。`);
  for (const parameter of entry.parameters) {
    const present = Object.hasOwn(choice.parameters, parameter.path);
    if (Object.hasOwn(managedParameters, parameter.path)) {
      if (present)
        errors.push(`${parameter.path} 已由阶段参数统一设置，请使用阶段参数并移除重复配置。`);
      continue;
    }
    const value = choice.parameters[parameter.path];
    if (!present) {
      if (parameter.required && !Object.hasOwn(parameter, 'default'))
        errors.push(`请填写 ${parameter.path}。`);
      continue;
    }
    if (
      !matchesSchema(value, parameter) ||
      !literal(value) ||
      JSON.stringify(value).length > 8192 ||
      (parameter.required && typeof value === 'string' && !value.trim())
    )
      errors.push(`请检查 ${parameter.path} 的类型或可选值。`);
  }
  return errors;
}

export function useDataProcessorCatalog(node?: Node, profile?: Profile) {
  const scope = `${node?.id || ''}\0${profile?.id || ''}\0${profile?.code_dir || ''}\0${profile?.python || ''}`;
  const [result, setResult] = useState<{ scope: string; value: DataProcessorCatalogValue }>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(false);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    setResult(undefined);
    setError(undefined);
    if (!node || !profile || profile.node_id !== node.id) {
      setLoading(false);
      return;
    }
    if (node.status !== 'ONLINE') {
      setLoading(false);
      setError(new Error('机器离线，无法读取当前代码项目的数据处理器目录。'));
      return;
    }
    let live = true;
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), 60000);
    setLoading(true);
    void (async () => {
      try {
        let response = await api<DataProcessorCatalog>(
          `/nodes/${encodeURIComponent(node.id)}/data-processors`,
          {
            method: 'POST',
            body: JSON.stringify({ profile_id: profile.id }),
            signal: controller.signal,
          },
        );
        while (response.status === 'PENDING') {
          await new Promise<void>((resolve, reject) => {
            const abort = () => {
              clearTimeout(timer);
              reject(new DOMException('Aborted', 'AbortError'));
            };
            const timer = setTimeout(() => {
              controller.signal.removeEventListener('abort', abort);
              resolve();
            }, 700);
            if (controller.signal.aborted) abort();
            else controller.signal.addEventListener('abort', abort, { once: true });
          });
          response = await api<DataProcessorCatalog>(
            `/nodes/${encodeURIComponent(node.id)}/data-processors/${encodeURIComponent(response.id)}`,
            { signal: controller.signal },
          );
        }
        if (response.node_id !== node.id || response.profile_id !== profile.id)
          throw new Error('数据处理器目录与当前环境不一致，请重新读取。');
        if (response.status !== 'READY' || !response.catalog)
          throw new ApiError(
            response.error?.code || 'DATA_PROCESSORS_FAILED',
            response.error?.message ||
              '无法读取数据处理器目录；请确认 Agent 已更新并检查训练环境。',
            0,
          );
        if (response.catalog.profile_id !== profile.id)
          throw new Error('目录环境不匹配，请重新读取。');
        if (live) setResult({ scope, value: response.catalog });
      } catch (err) {
        if (live)
          setError(
            controller.signal.aborted
              ? new Error('读取处理器目录超时，请检查机器 Agent 与训练环境后重试。')
              : err instanceof ApiError && (err.code.includes('UNSUPPORTED') || err.status === 404)
                ? new ApiError(
                    err.code,
                    '当前服务或机器 Agent 尚不支持数据处理器目录，请更新后重试；也可明确沿用当前配方数据项。',
                    err.status,
                  )
                : err instanceof Error
                  ? err
                  : new Error(String(err)),
          );
      } finally {
        clearTimeout(deadline);
        if (live) setLoading(false);
      }
    })();
    return () => {
      live = false;
      clearTimeout(deadline);
      controller.abort();
    };
  }, [scope, node?.status, revision]);
  return {
    catalog: result?.scope === scope ? result.value : undefined,
    error,
    loading,
    refresh: () => setRevision((value) => value + 1),
  };
}

export function DataProcessorCatalogStatus({
  loading,
  error,
  refresh,
  catalog,
}: ReturnType<typeof useDataProcessorCatalog>) {
  return (
    <>
      {loading && (
        <p className="field-hint" role="status">
          <LoaderCircle className="spin" size={12} /> 正在从当前训练项目读取已注册的数据处理器…
        </p>
      )}
      {error && (
        <div className="error-notice" role="alert">
          <div>
            <strong>数据处理器目录读取失败</strong>
            <p>
              {error instanceof ApiError ? `${error.code} · ` : ''}
              {error.message}
            </p>
          </div>
          <button type="button" className="button small ghost" onClick={refresh}>
            重试读取目录
          </button>
        </div>
      )}
      {!loading && catalog && (
        <p className="field-hint">
          已读取 {catalog.entries.length}{' '}
          个处理器；目录来自所选训练环境。预检核对配置与资产，实际数据转换和训练仍需运行验证。
          <button type="button" className="text-link" onClick={refresh}>
            重新读取目录
          </button>
        </p>
      )}
      {catalog?.source && (
        <code className="processor-source">
          目录来源：
          {typeof catalog.source === 'string'
            ? catalog.source
            : `${catalog.source.code_dir || ''}/${catalog.source.path || ''}${catalog.source.commit_sha ? ` · ${String(catalog.source.commit_sha).slice(0, 12)}` : ''}`}
        </code>
      )}
    </>
  );
}

export function DataProcessorPicker({
  binding,
  slots,
  entries,
  index,
  managedParameters = {},
  allowRecipeDefault = true,
  onChange,
}: {
  binding: DatasetBinding;
  slots: RecipeDatasetSlot[];
  entries: DataProcessorEntry[];
  index: number;
  managedParameters?: Record<string, RecipeLiteral>;
  allowRecipeDefault?: boolean;
  onChange: (binding: DatasetBinding) => void;
}) {
  const id = useId();
  const [search, setSearch] = useState('');
  const choice = binding.processor;
  const selected = choice ? entries.find((entry) => entry.name === choice.name) : undefined;
  const errors = choice ? processorParameterErrors(choice, selected, managedParameters) : [];
  const legacy = allowRecipeDefault
    ? slots.length
      ? slots
      : [{ recipe_index: 0, processing: '未静态展开的当前配方数据处理', weight: 1 }]
    : [];
  const filtered = filterDataProcessors(entries, search, choice?.name);
  const changeParameter = (parameter: DataProcessorParameter, text: string, json = false) => {
    if (!choice) return;
    const parameters = { ...choice.parameters };
    if (text === '' && (parameter.type !== 'string' || json)) delete parameters[parameter.path];
    else if (json) {
      try {
        parameters[parameter.path] = JSON.parse(text);
      } catch {
        parameters[parameter.path] = text;
      }
    } else if (parameter.type === 'string') parameters[parameter.path] = text;
    else if (parameter.type === 'integer' || parameter.type === 'number')
      parameters[parameter.path] = Number(text);
    else {
      try {
        parameters[parameter.path] = JSON.parse(text);
      } catch {
        parameters[parameter.path] = text;
      }
    }
    const { recipe_index: _recipeIndex, ...selectedBinding } = binding;
    onChange({ ...selectedBinding, processor: { name: choice.name, parameters } });
  };
  const text = (value: RecipeLiteral | undefined) =>
    typeof value === 'string' ? value : value === undefined ? '' : JSON.stringify(value);
  return (
    <div className="processor-picker">
      <Field
        label={`搜索数据处理器 ${index + 1}`}
        hint="按当前项目实际注册的处理器类名或说明搜索，与当前配方中的数据项相互独立。"
      >
        <input
          type="search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="搜索处理器类名或说明"
        />
      </Field>
      <Field
        label={`数据处理配置 ${index + 1}`}
        required
        hint={
          allowRecipeDefault
            ? '可选择项目中的处理器，或明确沿用当前配方数据项；选择后仍需训练预检。'
            : '自动组装需要明确的数据处理配置；可使用资产组合配置或选择已注册处理器。'
        }
      >
        <select
          required
          value={
            choice
              ? `class:${choice.name}`
              : allowRecipeDefault
                ? `legacy:${binding.recipe_index}`
                : ''
          }
          onChange={(event) => {
            if (event.target.value.startsWith('legacy:'))
              onChange(selectBindingProcessor(binding, Number(event.target.value.slice(7))));
            else {
              const entry = entries.find((item) => item.name === event.target.value.slice(6));
              if (entry?.available)
                onChange(selectBindingProcessor(binding, entry, managedParameters));
            }
          }}
        >
          {!allowRecipeDefault && <option value="">选择数据处理器</option>}
          {allowRecipeDefault && (
            <optgroup label="当前配方数据项（兼容旧任务）">
              {legacy.map((slot) => (
                <option key={slot.recipe_index} value={`legacy:${slot.recipe_index}`}>
                  沿用当前配方数据项 {slot.recipe_index + 1} · {slot.processing}
                </option>
              ))}
            </optgroup>
          )}
          <optgroup label="当前项目已注册的处理器">
            {choice && !selected && (
              <option value={`class:${choice.name}`}>{choice.name}（等待目录核对）</option>
            )}
            {filtered.map((entry) => (
              <option
                key={entry.name}
                value={`class:${entry.name}`}
                disabled={!entry.available}
                title={entry.reason || entry.description}
              >
                {entry.name}
                {!entry.available
                  ? `（不可用：${(entry.reason || '有未支持的必填配置').slice(0, 80)}）`
                  : ''}
              </option>
            ))}
          </optgroup>
        </select>
      </Field>
      {selected && choice && (
        <div className="processor-settings">
          <strong>{selected.name}</strong>
          <p className="field-hint">{selected.description || '项目注册的数据处理器。'}</p>
          {selected.source && (
            <code className="processor-source">
              {selected.source.path} · SHA256 {selected.source.sha256.slice(0, 12)}
            </code>
          )}
          {selected.parameters.map((parameter) => {
            const value = Object.hasOwn(choice.parameters, parameter.path)
              ? choice.parameters[parameter.path]
              : undefined;
            const inputId = `${id}-${parameter.path}`;
            if (Object.hasOwn(managedParameters, parameter.path))
              return (
                <Field
                  key={parameter.path}
                  label={parameter.path}
                  hint="由本次 ReCap 阶段参数统一设置，请在下方阶段参数中修改。"
                >
                  <code>{JSON.stringify(managedParameters[parameter.path])}</code>
                </Field>
              );
            return (
              <Field
                key={parameter.path}
                label={parameter.path}
                required={parameter.required}
                hint={
                  Object.hasOwn(parameter, 'default')
                    ? `默认值：${text(parameter.default)}`
                    : parameter.required
                      ? '此参数没有默认值，请填写。'
                      : '可选参数；留空使用处理器默认配置。'
                }
              >
                {parameter.choices ? (
                  <select
                    id={inputId}
                    value={value === undefined ? '' : JSON.stringify(value)}
                    onChange={(event) => changeParameter(parameter, event.target.value, true)}
                  >
                    <option value="">使用默认值</option>
                    {parameter.choices.map((option) => (
                      <option key={JSON.stringify(option)} value={JSON.stringify(option)}>
                        {text(option)}
                      </option>
                    ))}
                  </select>
                ) : parameter.type === 'boolean' || parameter.type === 'null' ? (
                  <select
                    id={inputId}
                    value={value === undefined ? '' : JSON.stringify(value)}
                    onChange={(event) => changeParameter(parameter, event.target.value)}
                  >
                    <option value="">{parameter.required ? '请选择' : '使用默认值'}</option>
                    {parameter.type === 'boolean' && (
                      <>
                        <option value="true">True</option>
                        <option value="false">False</option>
                      </>
                    )}
                    {(parameter.nullable || parameter.type === 'null') && (
                      <option value="null">None</option>
                    )}
                  </select>
                ) : parameter.type === 'array' || parameter.type === 'object' ? (
                  <textarea
                    id={inputId}
                    rows={2}
                    spellCheck={false}
                    value={text(value)}
                    onChange={(event) => changeParameter(parameter, event.target.value)}
                    placeholder={
                      parameter.type === 'array' ? 'JSON 列表，例如 ["actions"]' : 'JSON 对象'
                    }
                  />
                ) : (
                  <input
                    id={inputId}
                    type={parameter.type === 'string' ? 'text' : 'number'}
                    step={parameter.type === 'integer' ? 1 : 'any'}
                    value={value === null ? '' : text(value)}
                    disabled={value === null}
                    onChange={(event) => changeParameter(parameter, event.target.value)}
                  />
                )}
                {parameter.nullable &&
                  ['string', 'integer', 'number'].includes(parameter.type) &&
                  !parameter.choices && (
                    <button
                      type="button"
                      className="text-link"
                      onClick={() => changeParameter(parameter, value === null ? '' : 'null', true)}
                    >
                      {value === null ? '当前为 None；改为填写值' : '使用 None'}
                    </button>
                  )}
              </Field>
            );
          })}
          {Object.keys(choice.parameters).some((path) =>
            Object.hasOwn(managedParameters, path),
          ) && (
            <button
              type="button"
              className="button outline small"
              onClick={() => {
                const { recipe_index: _recipeIndex, ...selectedBinding } = binding;
                onChange({
                  ...selectedBinding,
                  processor: {
                    name: choice.name,
                    parameters: Object.fromEntries(
                      Object.entries(choice.parameters).filter(
                        ([path]) => !Object.hasOwn(managedParameters, path),
                      ),
                    ),
                  },
                });
              }}
            >
              使用阶段参数，移除重复配置
            </button>
          )}
          {!!selected.unsupported_parameters.length && (
            <details>
              <summary>沿用处理器默认值的复杂参数（只读）</summary>
              <ul>
                {selected.unsupported_parameters.map((parameter) => (
                  <li key={parameter.path}>
                    <code>{parameter.path}</code>：{parameter.reason}
                    {parameter.required ? '（必填，当前处理器不可选择）' : ''}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
      {errors.length > 0 && (
        <div className="inline-note red training-validation-error" role="alert">
          <div>
            <strong>处理器配置错误</strong>
            <ul className="training-validation-errors">
              {errors.map((error) => <li key={error}>{error}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
