import { Database, Plus, Trash2 } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { Field } from './components';
import { normalizedAssetPath } from './datasetBindingHelpers';
import {
  DataProcessorCatalogStatus,
  DataProcessorPicker,
  processorParameterErrors,
  useDataProcessorCatalog,
} from './DataProcessorPicker';
import type {
  Asset,
  DatasetBinding,
  Node,
  Profile,
  RecipeDatasetSlot,
  RecipeLiteral,
} from './types';

export function DatasetBindings({
  bindings,
  slots,
  assets,
  node,
  profile,
  onValidityChange,
  managedParameters = {},
  onChange,
}: {
  bindings: DatasetBinding[];
  slots: RecipeDatasetSlot[];
  assets: Asset[];
  node?: Node;
  profile?: Profile;
  onValidityChange?: (valid: boolean) => void;
  managedParameters?: Record<string, RecipeLiteral>;
  onChange: (bindings: DatasetBinding[]) => void;
}) {
  const datasets = assets.filter((asset) => asset.kind === 'DATASET');
  const norms = assets.filter((asset) => asset.kind === 'NORM');
  const processingChoices = slots.length
    ? slots
    : [{ recipe_index: 0, weight: 1, processing: '配方默认处理' }];
  const catalogState = useDataProcessorCatalog(node, profile);
  const entries = catalogState.catalog?.entries || [];
  const validityRef = useRef(onValidityChange);
  validityRef.current = onValidityChange;
  const valid = bindings.every(
    (binding) =>
      !binding.processor ||
      processorParameterErrors(
        binding.processor,
        entries.find((entry) => entry.name === binding.processor!.name),
        managedParameters,
      ).length === 0,
  );
  useEffect(() => {
    validityRef.current?.(valid);
  }, [valid]);
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
        <button
          type="button"
          className="button outline small"
          disabled={(slots.length > 0 && !slots[0]?.multiple) || bindings.length >= 64}
          onClick={() =>
            onChange([
              ...bindings,
              {
                dataset_asset_id: '',
                norm_asset_id: '',
                recipe_index: processingChoices[0].recipe_index,
                weight: 1,
              },
            ])
          }
        >
          <Plus size={14} />
          添加数据集
        </button>
      </div>
      <p className="field-hint">
        本次训练只使用这里选择的已登记资产；配方中的目录仅作源码参考。每个数据集绑定自己的归一化参数，混采权重决定相对采样比例。
      </p>
      <DataProcessorCatalogStatus {...catalogState} />
      {!slots.length && (
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
        return (
          <div className="dataset-binding" key={index}>
            <div className="dataset-binding-title">
              <strong>数据集 {index + 1}</strong>
              <button
                type="button"
                className="icon-button"
                disabled={bindings.length <= 1}
                aria-label={`移除数据集 ${index + 1}`}
                title="移除数据集"
                onClick={() => onChange(bindings.filter((_, row) => row !== index))}
              >
                <Trash2 size={15} />
              </button>
            </div>
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
                    update(index, {
                      dataset_asset_id: event.target.value,
                      norm_asset_id: matching.length === 1 ? matching[0].id : '',
                    });
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
              <Field
                label={`混采权重 ${index + 1}`}
                required
                hint={
                  share === undefined
                    ? '请输入大于 0 的权重。'
                    : `当前采样比例约 ${share.toFixed(2)}%`
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
            <DataProcessorPicker
              binding={binding}
              slots={slots}
              entries={entries}
              index={index}
              managedParameters={managedParameters}
              onChange={(next) =>
                onChange(bindings.map((value, row) => (row === index ? next : value)))
              }
            />
          </div>
        );
      })}
    </section>
  );
}

export function DatasetBindingSummary({
  bindings,
  assets,
}: {
  bindings: DatasetBinding[];
  assets: Asset[];
}) {
  return (
    <div className="dataset-binding-summary">
      {bindings.map((binding, index) => {
        const dataset = assets.find((asset) => asset.id === binding.dataset_asset_id);
        const norm = assets.find((asset) => asset.id === binding.norm_asset_id);
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
                  Object.entries(binding.processor.parameters).map(([path, value]) => (
                    <code className="block" key={path}>
                      {path} = {JSON.stringify(value)}
                    </code>
                  ))
                ) : (
                  <span>使用该处理器的默认参数</span>
                )}
              </details>
            )}
            <span>{dataset?.name || binding.dataset_asset_id}</span>
            {dataset?.path && <code>{dataset.path}</code>}
            <span>归一化参数：{norm?.name || binding.norm_asset_id}</span>
            {norm?.path && <code>{norm.path}</code>}
          </div>
        );
      })}
    </div>
  );
}
