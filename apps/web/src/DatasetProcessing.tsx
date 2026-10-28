import { useEffect, useRef, useState } from 'react';
import { DataProcessorPicker, processorParameterErrors, selectBindingProcessor } from './DataProcessorPicker';
import { datasetProfileSummary, readDatasetProfile } from './DatasetProfileForm';
import { recommendDatasetProcessor } from './datasetRecommendations';
import {
  ComposedDatasetProcessing,
  composedChoice,
  trainingProfileSeed,
} from './ComposedDatasetProcessing';
import { trainingCompositionDraft } from './composedProcessing';
import type { ImageLayoutEdits } from './imageLayout';
import { Field } from './components';
import './training-data-processing.css';
import type {
  Asset,
  DataProcessorEntry,
  DatasetBinding,
  Node,
  RecipeDatasetSlot,
  RecipeLiteral,
} from './types';

export function DatasetProcessing(props: {
  binding: DatasetBinding;
  dataset?: Asset;
  node?: Node;
  imageLayoutEdits?: ImageLayoutEdits;
  entries: DataProcessorEntry[];
  slots: RecipeDatasetSlot[];
  index: number;
  stage: 'sft' | 'value' | 'offline_rl';
  managedParameters: Record<string, RecipeLiteral>;
  automatic?: boolean;
  readOnly?: boolean;
  validationToken?: string;
  onChange: (binding: DatasetBinding) => void;
  onValidityChange?: (valid: boolean) => void;
}) {
  const [mode, setMode] = useState<'recommended' | 'existing' | 'composed'>(() =>
    props.binding.processor?.name === 'ComposableDataConfig'
      ? 'composed'
      : props.binding.processor
        ? 'existing'
        : 'recommended',
  );
  const composed = props.readOnly
    ? props.binding.processor?.name === 'ComposableDataConfig'
    : mode === 'composed';
  const annotations = trainingProfileSeed(props.dataset).field_annotations;
  return (
    <section className="training-dataset-processing" aria-label={`数据处理方式 ${props.index + 1}`}>
      {!props.readOnly && (
        <Field
          label={`数据处理方式 ${props.index + 1}`}
          hint="每组数据独立配置，只保存到本次训练。"
        >
          <select
            value={mode}
            onChange={(event) => {
              const next = event.target.value as typeof mode;
              setMode(next);
              if (next === 'composed') {
                if (props.binding.processor?.name !== 'ComposableDataConfig') {
                  const { recipe_index: _recipe, processor: _processor, ...assets } = props.binding;
                  props.onChange({
                    ...assets,
                    processor: composedChoice(
                      trainingCompositionDraft(trainingProfileSeed(props.dataset)),
                    ),
                  });
                }
              } else {
                const { processor: _processor, recipe_index: _recipe, ...assets } = props.binding;
                props.onChange({ ...assets, recipe_index: props.slots[0]?.recipe_index || 0 });
              }
            }}
          >
            <option value="recommended">自动推荐处理器</option>
            <option value="existing">选择现有处理器</option>
            <option value="composed">自定义字段组合</option>
          </select>
        </Field>
      )}
      <div className={`training-processing-panels${annotations?.length ? ' has-source-tags' : ''}`}>
        {!!annotations?.length && (
          <details className="training-source-tags">
            <summary>查看数据集字段标签（{annotations.length} 项）</summary>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>原始字段</th>
                    <th>用途</th>
                    <th>表示</th>
                    <th>数值方式</th>
                    <th>形状</th>
                  </tr>
                </thead>
                <tbody>
                  {annotations.map((field) => (
                    <tr key={field.key}>
                      <td>
                        <code>{field.key}</code>
                      </td>
                      <td>{field.role === 'state' ? '状态' : '动作'}</td>
                      <td>
                        {field.representation === 'unknown'
                          ? '待确认'
                          : field.representation === 'gripper'
                            ? '夹爪开度'
                            : field.representation.toUpperCase()}
                      </td>
                      <td>
                        {field.value_mode === 'unknown'
                          ? '待确认'
                          : field.value_mode === 'absolute'
                            ? '绝对量'
                            : '增量'}
                      </td>
                      <td>{field.shape?.join(' × ') || '待读取'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="field-hint">
              这里展示资产保存的源字段标签；修改本次训练的处理方案不会改写这些标签。
            </p>
          </details>
        )}
        <div className="training-processing-controls">
          {composed ? (
            <ComposedDatasetProcessing {...props} />
          ) : (
            <LegacyDatasetProcessing key={mode} {...props} recommend={mode === 'recommended'} />
          )}
        </div>
      </div>
    </section>
  );
}

function LegacyDatasetProcessing({
  binding,
  dataset,
  entries,
  slots,
  index,
  stage,
  managedParameters,
  automatic: automaticMode = false,
  recommend = true,
  readOnly = false,
  validationToken,
  onChange,
  onValidityChange,
}: {
  binding: DatasetBinding;
  dataset?: Asset;
  entries: DataProcessorEntry[];
  slots: RecipeDatasetSlot[];
  index: number;
  stage: 'sft' | 'value' | 'offline_rl';
  managedParameters: Record<string, RecipeLiteral>;
  automatic?: boolean;
  recommend?: boolean;
  readOnly?: boolean;
  validationToken?: string;
  onChange: (binding: DatasetBinding) => void;
  onValidityChange?: (valid: boolean) => void;
}) {
  const [manual, setManual] = useState(!!binding.processor);
  const [confirmedScope, setConfirmedScope] = useState('');
  const automatic = useRef<string | undefined>(undefined);
  const change = useRef(onChange);
  change.current = onChange;
  const registered = readDatasetProfile(dataset?.metadata?.dataset_profile);
  const recipeSlot =
    slots.find((slot) => slot.recipe_index === binding.recipe_index) || slots[index] || slots[0];
  const modelFamily = automaticMode
    ? 'standard'
    : entries.find((entry) => entry.name === recipeSlot?.processing)?.dataset_contract
        ?.model_family;
  const recommendation = recommendDatasetProcessor(
    dataset,
    entries,
    stage,
    modelFamily,
    managedParameters,
  );
  const currentSignature = JSON.stringify(binding.processor);
  const scopeSignature = JSON.stringify({
    registered,
    stage,
    modelFamily,
    selected: binding.processor,
    sources: entries.map((entry) => [entry.name, entry.source, entry.dataset_contract]),
  });
  const next = recommendation.selected
    ? selectBindingProcessor(binding, recommendation.selected, managedParameters)
    : undefined;
  const nextSignature = JSON.stringify(next?.processor);
  useEffect(() => {
    if (readOnly || !recommend || manual || !next || recommendation.status !== 'unique') return;
    // A catalog arriving later must never replace a user-selected processor.
    if (binding.processor && currentSignature !== automatic.current) return;
    automatic.current = nextSignature;
    if (nextSignature !== currentSignature) change.current(next);
  }, [readOnly, recommend, manual, recommendation.status, nextSignature, currentSignature]);
  const autoSelected = !manual && !!binding.processor && automatic.current === currentSignature;
  const matched = autoSelected && recommendation.selected?.name === binding.processor?.name;
  const validity = useRef(onValidityChange);
  validity.current = onValidityChange;
  const currentCandidate = recommendation.candidates.find(
    ({ entry }) => entry.name === binding.processor?.name,
  );
  // Validate existing choices again after a remount. A local ref cannot prove
  // that an old automatic choice was subsequently confirmed by the user.
  const selectionValid =
    (!automaticMode || !!binding.processor) &&
    (!binding.processor ||
      !registered ||
      (!!currentCandidate && currentCandidate.missing.length === 0) ||
      manual ||
      confirmedScope === scopeSignature);
  useEffect(() => {
    validity.current?.(selectionValid);
  }, [selectionValid, validationToken]);
  const hasProfile = !!registered;
  const parameterErrors = binding.processor
    ? processorParameterErrors(
        binding.processor,
        entries.find((entry) => entry.name === binding.processor?.name),
        managedParameters,
      )
    : [];
  const picker = (
    <DataProcessorPicker
      binding={binding}
      slots={slots}
      entries={entries}
      index={index}
      managedParameters={managedParameters}
      allowRecipeDefault={!automaticMode}
      onChange={(value) => {
        setManual(true);
        setConfirmedScope('');
        automatic.current = undefined;
        onChange(value);
      }}
    />
  );
  if (readOnly)
    return (
      <div className="dataset-processing">
        <strong>
          {binding.processor?.name || (automaticMode ? '尚未选择数据处理器' : '沿用配方数据项')}
        </strong>
        {registered && <p>{datasetProfileSummary(registered)}</p>}
        {!selectionValid && (
          <p className="inline-note red training-validation-error" role="alert">
            {automaticMode && !binding.processor
              ? '自动组装需要选择本次训练的数据处理器，请返回第 1 步配置。'
              : '处理方案需要返回第 1 步补充或重新确认。'}
          </p>
        )}
        {parameterErrors.length > 0 && (
          <div className="inline-note red training-validation-error" role="alert">
            <div>
              <strong>处理器配置错误</strong>
              <ul className="training-validation-errors">
                {parameterErrors.map((error) => <li key={error}>{error}</li>)}
              </ul>
            </div>
          </div>
        )}
      </div>
    );
  return (
    <div className="dataset-processing">
      {hasProfile && (
        <div className="inline-note">
          <div>
            <strong>数据特征</strong>
            <p>{datasetProfileSummary(registered)}</p>
          </div>
        </div>
      )}
      {dataset && (
        <div
          className={`inline-note ${!selectionValid ? 'red training-validation-error' : recommendation.status === 'unique' ? '' : 'amber'}`}
          role={!selectionValid ? 'alert' : 'status'}
        >
          <div>
            <strong>{!selectionValid ? '数据处理方案未确认，无法预检' : matched ? `已自动选择：${binding.processor!.name}` : '处理器推荐'}</strong>
            <p>
              {autoSelected && !matched
                ? '数据特征或处理器目录已变化，请重新确认处理方式。'
                : recommendation.message}
            </p>
            {registered?.action_mode === 'unknown' && (
              <p className="field-hint">
                动作绝对值/增量编码未登记；字段匹配不证明动作单位或物理语义。
              </p>
            )}
            {recommendation.candidates.map(({ entry, missing }) => (
              <div key={entry.name}>
                <code>{entry.name}</code>
                {missing.length > 0 && (
                  <p className={selectionValid ? 'field-hint' : undefined}>
                    {selectionValid ? '待核对：' : '配置错误：'}{missing.slice(0, 8).join('、')}
                    {missing.length > 8 ? '…' : ''}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      {hasProfile && recommend ? (
        <details open={manual || recommendation.status !== 'unique' || !selectionValid}>
          <summary>高级：手动选择处理器与参数</summary>
          {picker}
          {!selectionValid && (
            <div className="inline-note red training-validation-error" role="alert">
              <div>
                <p>
                  当前选择尚未与登记描述完全匹配。请核对高级参数和动作编码，确认后由预检检查实际数据。
                </p>
                <button
                  type="button"
                  className="button outline small"
                  onClick={() => {
                    setManual(true);
                    setConfirmedScope(scopeSignature);
                  }}
                >
                  确认使用当前手动配置
                </button>
              </div>
            </div>
          )}
        </details>
      ) : (
        picker
      )}
    </div>
  );
}
