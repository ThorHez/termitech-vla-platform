import { useEffect, useRef, useState } from 'react';
import { Database, Plus, Trash2 } from 'lucide-react';
import { Field } from './components';
import {
  DataProcessorCatalogStatus,
  processorParameterErrors,
  useDataProcessorCatalog,
} from './DataProcessorPicker';
import { DatasetProcessing } from './DatasetProcessing';
import {
  appendNormDataset,
  normDatasetErrors,
  normProcessingBinding,
  normProcessingChoice,
  selectNormDataset,
} from './normStats';
import type { Asset, Node, NormDataset, Profile } from './types';
import type { ImageLayoutEdits } from './imageLayout';

export function NormDatasetEditor({
  rows,
  assets,
  node,
  profile,
  recipe,
  imageLayoutEdits,
  automatic = false,
  preserveProcessingOnDatasetChange = false,
  onChange,
  onValidityChange,
  readOnly = false,
}: {
  rows: NormDataset[];
  assets: readonly Asset[];
  node?: Node;
  profile?: Profile;
  recipe?: string;
  imageLayoutEdits?: ImageLayoutEdits;
  automatic?: boolean;
  preserveProcessingOnDatasetChange?: boolean;
  onChange: (rows: NormDataset[]) => void;
  onValidityChange?: (valid: boolean) => void;
  readOnly?: boolean;
}) {
  const validate = !readOnly || !!onValidityChange;
  const catalog = useDataProcessorCatalog(
    validate ? node : undefined,
    validate ? profile : undefined,
  );
  const entries = catalog.catalog?.entries || [];
  const datasets = assets.filter(
    (asset) =>
      asset.kind === 'DATASET' &&
      (!node || asset.node_id === node.id) &&
      !['ARCHIVED', 'MISSING', 'INVALID', 'UNAVAILABLE'].includes(
        asset.status || asset.state || '',
      ),
  );
  const [validity, setValidity] = useState<Record<string, boolean>>({});
  const scope = JSON.stringify([node?.id, profile?.id, profile?.code_dir, recipe, automatic]);
  const tokens = rows.map((row) =>
    JSON.stringify([
      scope,
      row,
      datasets.find((asset) => asset.id === row.dataset_asset_id)?.metadata,
      entries,
    ]),
  );
  const valid =
    !normDatasetErrors(rows, automatic).length &&
    rows.every(
      (row, index) =>
        !row.processor ||
        (validity[tokens[index]] === true &&
          !processorParameterErrors(
            row.processor,
            entries.find((entry) => entry.name === row.processor!.name),
            {},
            'norm',
          ).length),
    );
  const callback = useRef(onValidityChange);
  callback.current = onValidityChange;
  useEffect(() => {
    callback.current?.(valid);
  }, [valid, onValidityChange]);
  const change = (index: number, row: NormDataset) =>
    onChange(rows.map((current, position) => (position === index ? row : current)));
  if (readOnly && !onValidityChange) return <NormDatasetSummary rows={rows} assets={assets} />;
  return (
    <section className="dataset-bindings" aria-label="norm_stats 数据集与处理方式">
      <div className="dataset-bindings-heading">
        <div>
          <Database size={16} />
          <h3>待统计数据集</h3>
          <span>{rows.length} 个</span>
        </div>
      </div>
      <p className="inline-note">
        {rows.length > 1 ? '批量计算模式：' : ''}每个数据集独立完整统计，各自产出一份
        norm_stats.json；不进行混采或合并统计。
      </p>
      {!readOnly && <DataProcessorCatalogStatus {...catalog} />}
      {rows.map((row, index) => {
        const asset = datasets.find((item) => item.id === row.dataset_asset_id);
        const token = tokens[index];
        return (
          <article className="dataset-binding" key={`${scope}:${index}`}>
            <div className="dataset-binding-title">
              <strong>数据集 {index + 1}</strong>
              {!readOnly && (
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`移除统计数据集 ${index + 1}`}
                  disabled={rows.length <= 1}
                  onClick={() => onChange(rows.filter((_, position) => position !== index))}
                >
                  <Trash2 size={15} />
                </button>
              )}
            </div>
            {readOnly ? (
              <>
                <strong>{asset?.name || row.dataset_asset_id || '尚未选择'}</strong>
                {asset && <code className="block">{asset.path}</code>}
                {row.recipe_index !== undefined && (
                  <p className="field-hint">配方数据项序号：{row.recipe_index + 1}</p>
                )}
              </>
            ) : (
              <Field
                label={`待统计数据集 ${index + 1}`}
                required
                hint="只选择本机原始数据；输出保存到各自独立目录。"
              >
                <select
                  value={row.dataset_asset_id}
                  required
                  onChange={(event) =>
                    change(
                      index,
                      selectNormDataset(
                        row,
                        datasets.find((item) => item.id === event.target.value),
                        automatic,
                        preserveProcessingOnDatasetChange,
                      ),
                    )
                  }
                >
                  <option value="">选择数据集</option>
                  {row.dataset_asset_id && !asset && (
                    <option value={row.dataset_asset_id}>原数据集暂不可用，请重新选择</option>
                  )}
                  {datasets.map((item) => (
                    <option
                      key={item.id}
                      value={item.id}
                      disabled={rows.some(
                        (other, position) =>
                          position !== index && other.dataset_asset_id === item.id,
                      )}
                    >
                      {item.name} · {item.version}
                    </option>
                  ))}
                </select>
              </Field>
            )}
            {!readOnly && !automatic && row.recipe_index !== undefined && (
              <Field
                label={`配方数据项序号 ${index + 1}`}
                required
                hint="单数据集配方填写 1；多数据集配方按所需数据项的顺序填写，预检会核对。"
              >
                <input
                  type="number"
                  min={1}
                  max={64}
                  step={1}
                  value={Number.isFinite(row.recipe_index) ? row.recipe_index + 1 : ''}
                  onChange={(event) =>
                    change(index, {
                      dataset_asset_id: row.dataset_asset_id,
                      recipe_index:
                        event.target.value === '' ? NaN : Number(event.target.value) - 1,
                    })
                  }
                />
              </Field>
            )}
            <DatasetProcessing
              key={`${scope}:${index}:${row.dataset_asset_id}`}
              binding={normProcessingBinding(row)}
              dataset={asset}
              node={node}
              imageLayoutEdits={imageLayoutEdits}
              index={index}
              stage="sft"
              purpose="norm"
              automatic={automatic}
              managedParameters={{}}
              entries={entries}
              slots={
                automatic
                  ? []
                  : [
                      {
                        recipe_index: row.recipe_index ?? 0,
                        weight: 1,
                        processing: '所选代码配方的数据处理',
                      },
                    ]
              }
              validationToken={token}
              readOnly={readOnly}
              onChange={(binding) => change(index, normProcessingChoice(binding))}
              onValidityChange={(next) =>
                setValidity((current) =>
                  current[token] === next ? current : { ...current, [token]: next },
                )
              }
            />
          </article>
        );
      })}
      {!readOnly && (
        <button
          type="button"
          className="button outline small"
          disabled={rows.length >= 64}
          onClick={() => onChange(appendNormDataset(rows, automatic))}
        >
          <Plus size={14} />
          添加待统计数据集
        </button>
      )}
    </section>
  );
}

export function NormDatasetSummary({
  rows,
  assets,
}: {
  rows: readonly NormDataset[];
  assets: readonly Asset[];
}) {
  return (
    <section className="dataset-binding-summary" aria-label="norm_stats 数据集与处理方式">
      <h3>待统计数据集 · {rows.length} 个</h3>
      <p className="field-hint">
        每个数据集独立完整统计，各自产出一份 norm_stats.json；不进行混采或合并统计。
      </p>
      {rows.map((row, index) => {
        const dataset = assets.find((asset) => asset.id === row.dataset_asset_id);
        return (
          <div key={index}>
            <strong>
              数据集 {index + 1} · {dataset?.name || row.dataset_asset_id || '尚未选择'}
            </strong>
            {dataset?.path && <code>{dataset.path}</code>}
            {row.processor ? (
              <>
                <span>{row.processor.name}</span>
                <details>
                  <summary>数据处理参数</summary>
                  <pre className="processor-parameters-json" aria-label="数据处理参数 JSON">
                    <code>{JSON.stringify(row.processor.parameters, null, 2)}</code>
                  </pre>
                </details>
              </>
            ) : (
              <span>配方数据项序号：{row.recipe_index + 1}</span>
            )}
          </div>
        );
      })}
    </section>
  );
}
