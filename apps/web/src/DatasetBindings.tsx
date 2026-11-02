import { ChevronDown, Database, Plus, Trash2 } from 'lucide-react';
import { useEffect, useId, useLayoutEffect, useReducer, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { Field } from './components';
import {
  appendDatasetBinding,
  bindSharedNormalization,
  datasetNormalizationMode,
  normalizedAssetPath,
  selectDatasetBinding,
  sharedNormalizationAssetId,
} from './datasetBindingHelpers';
import {
  DataProcessorCatalogStatus,
  processorParameterErrors,
  useDataProcessorCatalog,
} from './DataProcessorPicker';
import { DatasetProcessing, type DatasetProcessingMode } from './DatasetProcessing';
import { composedChoice, trainingProfileSeed } from './ComposedDatasetProcessing';
import { trainingCompositionDraft } from './composedProcessing';
import {
  createDatasetBindingDisclosure,
  datasetBindingDisclosureReducer,
} from './datasetBindingDisclosure';
import type { ImageLayoutEdits } from './imageLayout';
import { trainingNormalizationLabel } from './trainingNormalization';
import { datasetProfileSummary, type DatasetProfile } from './DatasetProfileForm';
import type {
  Asset,
  DatasetBinding,
  Node,
  Profile,
  RecipeDatasetSlot,
  RecipeLiteral,
  TrainingNormalizationMethod,
} from './types';

export function DatasetBindings({
  bindings,
  slots,
  assets,
  node,
  profile,
  onValidityChange,
  managedParameters = {},
  trainingStage = 'sft',
  normalizationOverride,
  automatic = false,
  showProcessing = true,
  readOnly = false,
  onEditProcessing,
  preserveProcessingOnDatasetChange = false,
  imageLayoutEdits,
  onChange,
}: {
  bindings: DatasetBinding[];
  slots: RecipeDatasetSlot[];
  assets: Asset[];
  node?: Node;
  profile?: Profile;
  onValidityChange?: (valid: boolean) => void;
  managedParameters?: Record<string, RecipeLiteral>;
  trainingStage?: 'sft' | 'value' | 'offline_rl';
  normalizationOverride?: TrainingNormalizationMethod;
  automatic?: boolean;
  showProcessing?: boolean;
  readOnly?: boolean;
  onEditProcessing?: () => void;
  preserveProcessingOnDatasetChange?: boolean;
  imageLayoutEdits?: ImageLayoutEdits;
  onChange: (bindings: DatasetBinding[]) => void;
}) {
  const disclosureId = useId();
  const [disclosure, dispatchDisclosure] = useReducer(
    datasetBindingDisclosureReducer,
    bindings.length,
    createDatasetBindingDisclosure,
  );
  const lastDatasetSelect = useRef<HTMLSelectElement>(null);
  const pendingDatasetFocus = useRef<number | null>(null);
  useLayoutEffect(() => {
    const index = pendingDatasetFocus.current;
    const select = lastDatasetSelect.current;
    if (
      index === null ||
      index !== bindings.length - 1 ||
      !disclosure.expanded.includes(index) ||
      !select
    )
      return;
    pendingDatasetFocus.current = null;
    (select.closest('.field') || select).scrollIntoView({
      block: 'start',
      inline: 'nearest',
      behavior: 'instant',
    });
    select.focus({ preventScroll: true });
  }, [bindings.length, disclosure.expanded]);
  useEffect(() => {
    dispatchDisclosure({ type: 'sync', count: bindings.length });
  }, [bindings.length]);
  const datasets = assets.filter((asset) => asset.kind === 'DATASET');
  const norms = assets.filter(
    (asset) =>
      asset.kind === 'NORM' &&
      (!node || asset.node_id === node.id) &&
      asset.state !== 'ARCHIVED' &&
      asset.state !== 'UNAVAILABLE',
  );
  const normalizationMode = datasetNormalizationMode(bindings);
  const sharedNormalization = normalizationMode === 'shared';
  const sharedNormId = sharedNormalizationAssetId(bindings);
  const sharedNorm = norms.find((asset) => asset.id === sharedNormId);
  const processingChoices = slots.length
    ? slots
    : [{ recipe_index: 0, weight: 1, processing: '配方默认处理' }];
  const catalogState = useDataProcessorCatalog(
    showProcessing ? node : undefined,
    showProcessing ? profile : undefined,
  );
  const entries = catalogState.catalog?.entries || [];
  const [automaticValidity, setAutomaticValidity] = useState<
    Record<string, { token: string; valid: boolean }>
  >({});
  const [processingModes, setProcessingModes] = useState<Record<string, DatasetProcessingMode>>({});
  const bindingKey = (binding: DatasetBinding, index: number) =>
    `${node?.id}:${profile?.id}:${trainingStage}:${binding.dataset_asset_id}:${index}`;
  const changeRef = useRef(onChange);
  changeRef.current = onChange;
  useEffect(() => {
    if (readOnly || !showProcessing) return;
    let changed = false;
    const next = bindings.map((binding, index) => {
      if (binding.processor) return binding;
      const mode = processingModes[bindingKey(binding, index)];
      if (mode && mode !== 'composed') return binding;
      if (!mode && !automatic && binding.recipe_index !== undefined) return binding;
      const dataset = assets.find(
        (asset) => asset.kind === 'DATASET' && asset.id === binding.dataset_asset_id,
      );
      if (!dataset) return binding;
      const { recipe_index: _recipe, ...selection } = binding;
      changed = true;
      return {
        ...selection,
        processor: composedChoice(trainingCompositionDraft(trainingProfileSeed(dataset))),
      };
    });
    if (changed) changeRef.current(next);
  }, [
    bindings,
    assets,
    automatic,
    readOnly,
    showProcessing,
    processingModes,
    node?.id,
    profile?.id,
    trainingStage,
  ]);
  const validityRef = useRef(onValidityChange);
  validityRef.current = onValidityChange;
  const validationTokens = bindings.map((binding, index) =>
    JSON.stringify([
      binding.processor,
      binding.recipe_index,
      trainingStage,
      managedParameters,
      automatic,
      normalizationOverride,
      datasets.find((asset) => asset.id === binding.dataset_asset_id)?.metadata?.dataset_profile,
      entries.find((entry) => entry.name === binding.processor?.name),
      slots.find((slot) => slot.recipe_index === binding.recipe_index) || slots[index] || slots[0],
    ]),
  );
  const valid =
    !showProcessing ||
    bindings.every(
      (binding, index) =>
        automaticValidity[bindingKey(binding, index)]?.token === validationTokens[index] &&
        automaticValidity[bindingKey(binding, index)]?.valid === true &&
        (!automatic || !!binding.processor) &&
        (!binding.processor ||
          processorParameterErrors(
            binding.processor,
            entries.find((entry) => entry.name === binding.processor!.name),
            managedParameters,
          ).length === 0),
    );
  useEffect(() => {
    validityRef.current?.(valid);
  }, [valid, onValidityChange]);
  const totalWeight = bindings.reduce(
    (sum, binding) => sum + (Number.isFinite(binding.weight) ? binding.weight : 0),
    0,
  );
  const update = (
    index: number,
    value: Partial<Pick<DatasetBinding, 'dataset_asset_id' | 'norm_asset_id' | 'weight'>>,
  ) =>
    onChange(bindings.map((binding, row) => (row === index ? { ...binding, ...value } : binding)));
  return (
    <section className="dataset-bindings" aria-label="数据集与归一化参数绑定">
      <div className="dataset-bindings-heading">
        <div>
          <Database size={16} />
          <h3>训练数据集</h3>
          <span>{bindings.length} 个</span>
        </div>
      </div>
      <p className="field-hint">
        本次训练只使用这里选择的已登记资产；配方中的目录仅作源码参考。
        {bindings.length > 1
          ? '多数据集推荐使用一份共享归一化参数，按全部源数据与本次混采权重重新统计。'
          : '请选择此数据集对应的归一化参数。'}
        新增数据集会复制第 1 组的处理配置与混采权重；共享归一化时也会保留同一份参数。
      </p>
      {normalizationMode === 'legacy' && (
        <div className="inline-note" role="status">
          <strong>逐组归一化（旧配置）</strong>
          <span>
            已保留草稿或模板中原有的逐组选择。切换后会清空这些归一化选择，请重新选择一份共享参数。
            多源统计应按原始样本及混采权重重新计算，不能平均或合并旧 norm 文件。
          </span>
          {!readOnly && (
            <button
              type="button"
              className="button outline small"
              onClick={() => onChange(bindSharedNormalization(bindings, ''))}
            >
              切换为共享归一化
            </button>
          )}
        </div>
      )}
      {sharedNormalization && (
        <div className="form-grid">
          <Field
            label="共享归一化参数"
            required
            hint={
              sharedNorm?.path ||
              (norms.length
                ? '为所有数据集选择同一份按本次数据与混采权重重新统计的参数。'
                : '请先在「数据与模型」登记当前机器的共享归一化参数。')
            }
          >
            {readOnly ? (
              <span>{sharedNorm?.name || sharedNormId || '未选择共享归一化参数'}</span>
            ) : (
              <select
                required
                value={sharedNormId}
                onChange={(event) =>
                  onChange(bindSharedNormalization(bindings, event.target.value))
                }
              >
                <option value="">选择共享归一化参数</option>
                {norms.map((asset) => (
                  <option value={asset.id} key={asset.id}>
                    {asset.name} · {asset.version}
                  </option>
                ))}
              </select>
            )}
          </Field>
          <p className="field-hint">
            所有数据集共用此参数。新增或更换数据集、修改混采权重或处理布局后需要重新统计；
            预检会核对统计来源，单源参数不能冒充多源统计。此处只选择已有参数，不会启动全量统计任务。
          </p>
        </div>
      )}
      {showProcessing && <DataProcessorCatalogStatus {...catalogState} />}
      {showProcessing && !automatic && !slots.length && (
        <p className="recipe-detail-warning">
          未能静态读取数据处理配置，Agent 将在预检时核对配方。
        </p>
      )}
      {bindings.map((binding, index) => {
        const dataset = datasets.find((asset) => asset.id === binding.dataset_asset_id);
        const norm = norms.find((asset) => asset.id === binding.norm_asset_id);
        const share =
          totalWeight > 0 && Number.isFinite(binding.weight)
            ? (100 * binding.weight) / totalWeight
            : undefined;
        const expanded = disclosure.expanded.includes(index);
        const contentId = `${disclosureId}-dataset-${index}`;
        const selectionIssues = [
          !dataset ? '请选择数据集' : '',
          !norm ? (sharedNormalization ? '请选择共享归一化参数' : '请选择归一化参数') : '',
          !Number.isFinite(binding.weight) || binding.weight <= 0 ? '混采权重需大于 0' : '',
        ].filter(Boolean);
        const validation = automaticValidity[bindingKey(binding, index)];
        const currentValidation = validation?.token === validationTokens[index];
        const parameterErrors = binding.processor
          ? processorParameterErrors(
              binding.processor,
              entries.find((entry) => entry.name === binding.processor!.name),
              managedParameters,
            )
          : [];
        const processingIncomplete =
          showProcessing &&
          ((automatic && !binding.processor) ||
            (currentValidation && !validation.valid) ||
            parameterErrors.length > 0);
        const processingPending = showProcessing && !currentValidation && !processingIncomplete;
        const issues = [
          ...selectionIssues,
          processingIncomplete ? '处理配置待完善，请展开检查' : '',
        ].filter(Boolean);
        const recipeSlot =
          slots.find((slot) => slot.recipe_index === binding.recipe_index) ||
          slots[index] ||
          slots[0];
        const processingLabel = binding.processor
          ? binding.processor.name === 'ComposableDataConfig'
            ? '自定义字段组合'
            : binding.processor.name
          : automatic
            ? '待配置'
            : recipeSlot?.processing || `沿用配方数据项 ${(binding.recipe_index ?? 0) + 1}`;
        return (
          <div className="dataset-binding" key={index}>
            <div className="dataset-binding-title">
              <button
                type="button"
                className="dataset-binding-toggle"
                aria-expanded={expanded}
                aria-controls={contentId}
                onClick={() => dispatchDisclosure({ type: 'toggle', index })}
              >
                <ChevronDown size={15} className={expanded ? '' : 'collapsed'} />
                <strong>数据集 {index + 1}</strong>
                <span>{expanded ? '收起配置' : '展开配置'}</span>
              </button>
              {!readOnly && (
                <button
                  type="button"
                  className="icon-button"
                  disabled={bindings.length <= 1}
                  aria-label={`移除数据集 ${index + 1}`}
                  title="移除数据集"
                  onClick={() => {
                    dispatchDisclosure({ type: 'remove', index });
                    onChange(bindings.filter((_, row) => row !== index));
                  }}
                >
                  <Trash2 size={15} />
                </button>
              )}
            </div>
            <div className="dataset-binding-overview">
              <strong>{dataset ? `${dataset.name} · ${dataset.version}` : '尚未选择数据集'}</strong>
              <span>归一化：{norm ? `${norm.name} · ${norm.version}` : '未选择'}</span>
              <span>
                混采权重：{Number.isFinite(binding.weight) ? binding.weight : '未填写'}
                {share !== undefined && binding.weight > 0
                  ? ` · 权重占比（未考虑样本数）约 ${share.toFixed(2)}%`
                  : ''}
              </span>
              <span>处理方案：{processingLabel}</span>
              <span
                className={`dataset-binding-status${issues.length ? ' error' : processingPending ? ' pending' : ''}`}
              >
                {issues.length
                  ? issues.join('；')
                  : processingPending
                    ? '处理配置校验中'
                    : '本组配置已填写'}
              </span>
            </div>
            <div
              className="dataset-binding-content"
              id={contentId}
              hidden={!expanded}
              onInvalidCapture={() => {
                flushSync(() => dispatchDisclosure({ type: 'open', index }));
              }}
            >
              {readOnly ? (
                <p className="field-hint">
                  数据集：{dataset?.name || '未选择'} · 归一化：{norm?.name || '未选择'} ·
                  混采权重：
                  {binding.weight}
                </p>
              ) : (
                <div className="form-grid">
                  <Field
                    label={`数据集 ${index + 1}`}
                    required
                    hint={
                      dataset?.path ||
                      (datasets.length
                        ? '选择当前机器的数据集资产。'
                        : '请先在「数据与模型」登记当前机器的数据集。')
                    }
                  >
                    <select
                      ref={index === bindings.length - 1 ? lastDatasetSelect : undefined}
                      required
                      value={binding.dataset_asset_id}
                      onChange={(event) => {
                        const selected = datasets.find((asset) => asset.id === event.target.value);
                        const matching = selected
                          ? norms.filter((asset) =>
                              normalizedAssetPath(asset.path).startsWith(
                                `${normalizedAssetPath(selected.path)}/`,
                              ),
                            )
                          : [];
                        const next = selectDatasetBinding(
                          binding,
                          {
                            dataset_asset_id: event.target.value,
                            norm_asset_id: matching.length === 1 ? matching[0].id : '',
                          },
                          processingChoices[0].recipe_index,
                          preserveProcessingOnDatasetChange,
                          sharedNormalization ? sharedNormId : undefined,
                        );
                        setProcessingModes((previous) => {
                          const modes = { ...previous };
                          const selectedMode = !binding.dataset_asset_id
                            ? previous[bindingKey(binding, index)]
                            : undefined;
                          const nextKey = bindingKey(next, index);
                          if (selectedMode) modes[nextKey] = selectedMode;
                          else delete modes[nextKey];
                          return modes;
                        });
                        onChange(bindings.map((value, row) => (row === index ? next : value)));
                      }}
                    >
                      <option value="">选择数据集</option>
                      {datasets.map((asset) => (
                        <option value={asset.id} key={asset.id}>
                          {asset.name} · {asset.version}
                        </option>
                      ))}
                    </select>
                  </Field>
                  {!sharedNormalization && (
                    <Field
                      label={`归一化参数 ${index + 1}`}
                      required
                      hint={
                        norm?.path ||
                        (norms.length
                          ? '选择此数据集对应的归一化参数。'
                          : '请先在「数据与模型」登记当前机器的归一化参数。')
                      }
                    >
                      <select
                        required
                        value={binding.norm_asset_id}
                        onChange={(event) => update(index, { norm_asset_id: event.target.value })}
                      >
                        <option value="">选择归一化参数</option>
                        {norms.map((asset) => (
                          <option value={asset.id} key={asset.id}>
                            {asset.name} · {asset.version}
                          </option>
                        ))}
                      </select>
                    </Field>
                  )}
                  <Field
                    label={`混采权重 ${index + 1}`}
                    required
                    hint={
                      '每条样本的重复权重；实际来源占比还取决于样本数。' +
                      (share === undefined
                        ? '请输入大于 0 的权重。'
                        : `当前权重占比（未考虑样本数）约 ${share.toFixed(2)}%`)
                    }
                  >
                    <input
                      type="number"
                      required
                      min={0.000000000001}
                      step="any"
                      value={Number.isFinite(binding.weight) ? binding.weight : ''}
                      onChange={(event) =>
                        update(index, {
                          weight: event.target.value === '' ? NaN : Number(event.target.value),
                        })
                      }
                    />
                  </Field>
                </div>
              )}
              {showProcessing && (
                <DatasetProcessing
                  key={bindingKey(binding, index)}
                  binding={binding}
                  normalizationOverride={normalizationOverride}
                  initialMode={processingModes[bindingKey(binding, index)]}
                  onModeChange={(mode) =>
                    setProcessingModes((previous) => ({
                      ...previous,
                      [bindingKey(binding, index)]: mode,
                    }))
                  }
                  dataset={dataset}
                  node={node}
                  imageLayoutEdits={imageLayoutEdits}
                  slots={slots}
                  entries={entries}
                  index={index}
                  stage={trainingStage}
                  managedParameters={managedParameters}
                  automatic={automatic}
                  readOnly={readOnly}
                  validationToken={validationTokens[index]}
                  onValidityChange={(value) => {
                    const key = bindingKey(binding, index);
                    setAutomaticValidity((previous) =>
                      previous[key]?.valid === value &&
                      previous[key]?.token === validationTokens[index]
                        ? previous
                        : { ...previous, [key]: { token: validationTokens[index], valid: value } },
                    );
                  }}
                  onChange={(next) =>
                    onChange(bindings.map((value, row) => (row === index ? next : value)))
                  }
                />
              )}
            </div>
          </div>
        );
      })}
      {!readOnly && (
        <div className="dataset-bindings-footer">
          <button
            type="button"
            className="button outline small"
            disabled={
              (!automatic && slots.length > 0 && !slots[0]?.multiple) || bindings.length >= 64
            }
            onClick={() => {
              pendingDatasetFocus.current = bindings.length;
              dispatchDisclosure({ type: 'append' });
              onChange(appendDatasetBinding(bindings, processingChoices[0].recipe_index));
            }}
          >
            <Plus size={14} />
            添加数据集
          </button>
        </div>
      )}
      {readOnly && onEditProcessing && (
        <button type="button" className="button outline small" onClick={onEditProcessing}>
          返回第 1 步修改数据处理
        </button>
      )}
    </section>
  );
}

export function DatasetBindingSummary({
  bindings,
  assets,
  profiles,
  normalizationOverride,
}: {
  bindings: DatasetBinding[];
  assets: Asset[];
  profiles?: Record<string, DatasetProfile> | null;
  normalizationOverride?: TrainingNormalizationMethod;
}) {
  return (
    <div className="dataset-binding-summary">
      {normalizationOverride && (
        <p className="field-hint">
          本次训练统一使用：{trainingNormalizationLabel(normalizationOverride)}。
          下方处理器参数保留原有声明，实际公式以本次训练选择为准。
        </p>
      )}
      {bindings.map((binding, index) => {
        const dataset = assets.find((asset) => asset.id === binding.dataset_asset_id);
        const norm = assets.find((asset) => asset.id === binding.norm_asset_id);
        const frozen = profiles?.[`dataset_${index}`];
        return (
          <div key={index}>
            <strong>
              数据集 {index + 1} · 权重 {binding.weight} ·{' '}
              {binding.processor
                ? binding.processor.name
                : `沿用当前配方数据项 ${binding.recipe_index + 1}`}
            </strong>
            {binding.processor && (
              <details open>
                <summary>数据处理参数</summary>
                {Object.entries(binding.processor.parameters).length ? (
                  <pre className="processor-parameters-json" aria-label="数据处理参数 JSON">
                    <code>{JSON.stringify(binding.processor.parameters, null, 2)}</code>
                  </pre>
                ) : (
                  <span>使用该处理器的默认参数</span>
                )}
              </details>
            )}
            <span>{dataset?.name || binding.dataset_asset_id}</span>
            {frozen && (
              <>
                <span>预检时的数据特征：{datasetProfileSummary(frozen)}</span>
                <span>任务描述：{frozen.task_description}</span>
              </>
            )}
            {dataset?.path && <code>{dataset.path}</code>}
            <span>归一化参数：{norm?.name || binding.norm_asset_id}</span>
            {norm?.path && <code>{norm.path}</code>}
          </div>
        );
      })}
    </div>
  );
}
