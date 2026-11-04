import {
  useEffect,
  useId,
  useLayoutEffect,
  useReducer,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { flushSync } from 'react-dom';
import { ChevronDown, Database, Plus, Trash2 } from 'lucide-react';
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
import {
  createDatasetBindingDisclosure,
  datasetBindingDisclosureReducer,
} from './datasetBindingDisclosure';

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
  const disclosureId = useId();
  const [disclosure, dispatchDisclosure] = useReducer(
    datasetBindingDisclosureReducer,
    rows.length,
    createDatasetBindingDisclosure,
  );
  const lastDatasetSelect = useRef<HTMLSelectElement>(null);
  const pendingDatasetFocus = useRef<number | null>(null);
  useLayoutEffect(() => {
    const index = pendingDatasetFocus.current;
    const select = lastDatasetSelect.current;
    if (
      index === null ||
      index !== rows.length - 1 ||
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
  }, [rows.length, disclosure.expanded]);
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
  const disclosureScope = useRef(scope);
  useEffect(() => {
    if (disclosureScope.current !== scope) {
      disclosureScope.current = scope;
      pendingDatasetFocus.current = null;
      dispatchDisclosure({ type: 'sync', count: 0 });
    }
    dispatchDisclosure({ type: 'sync', count: rows.length });
  }, [scope, rows.length]);
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
        const parameterErrors = row.processor
          ? processorParameterErrors(
              row.processor,
              entries.find((entry) => entry.name === row.processor!.name),
              {},
              'norm',
            )
          : [];
        const processingIncomplete =
          normDatasetErrors(
            [{ ...row, dataset_asset_id: row.dataset_asset_id || 'pending' }],
            automatic,
          ).length > 0 ||
          (row.processor && validity[token] === false) ||
          parameterErrors.length > 0;
        const processingPending =
          !!row.processor && validity[token] === undefined && !processingIncomplete;
        const issues = [
          !asset ? (row.dataset_asset_id ? '原数据集暂不可用，请重新选择' : '请选择数据集') : '',
          row.dataset_asset_id &&
          rows.some(
            (other, position) =>
              position !== index && other.dataset_asset_id === row.dataset_asset_id,
          )
            ? '数据集已重复选择'
            : '',
          processingIncomplete ? '处理配置待完善，请展开检查' : '',
        ].filter(Boolean);
        const processingLabel = row.processor
          ? row.processor.name === 'ComposableDataConfig'
            ? '自定义字段组合'
            : row.processor.name
          : automatic
            ? '待配置'
            : `配方数据项 ${Number.isFinite(row.recipe_index) ? row.recipe_index! + 1 : '待填写'}`;
        return (
          <NormDatasetCard
            key={`${scope}:${index}`}
            index={index}
            dataset={asset}
            processingLabel={processingLabel}
            processingPending={processingPending}
            issues={issues}
            expanded={disclosure.expanded.includes(index)}
            contentId={`${disclosureId}-norm-dataset-${index}`}
            readOnly={readOnly}
            removable={rows.length > 1}
            onToggle={(position) => dispatchDisclosure({ type: 'toggle', index: position })}
            onRemove={(position) => {
              dispatchDisclosure({ type: 'remove', index: position });
              onChange(rows.filter((_, current) => current !== position));
            }}
            onInvalid={(position) =>
              flushSync(() => dispatchDisclosure({ type: 'open', index: position }))
            }
          >
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
                  ref={index === rows.length - 1 ? lastDatasetSelect : undefined}
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
          </NormDatasetCard>
        );
      })}
      {!readOnly && (
        <button
          type="button"
          className="button outline small"
          disabled={rows.length >= 64}
          onClick={() => {
            pendingDatasetFocus.current = rows.length;
            dispatchDisclosure({ type: 'append' });
            onChange(appendNormDataset(rows, automatic));
          }}
        >
          <Plus size={14} />
          添加待统计数据集
        </button>
      )}
    </section>
  );
}

export function NormDatasetCard({
  index,
  dataset,
  processingLabel,
  processingPending,
  issues,
  expanded,
  contentId,
  readOnly,
  removable,
  onToggle,
  onRemove,
  onInvalid,
  children,
}: {
  index: number;
  dataset?: Asset;
  processingLabel: string;
  processingPending: boolean;
  issues: readonly string[];
  expanded: boolean;
  contentId: string;
  readOnly: boolean;
  removable: boolean;
  onToggle: (index: number) => void;
  onRemove: (index: number) => void;
  onInvalid: (index: number) => void;
  children: ReactNode;
}) {
  return (
    <article className="dataset-binding">
      <div className="dataset-binding-title">
        {readOnly ? (
          <strong>数据集 {index + 1}</strong>
        ) : (
          <button
            type="button"
            className="dataset-binding-toggle"
            aria-expanded={expanded}
            aria-controls={contentId}
            onClick={() => onToggle(index)}
          >
            <ChevronDown size={15} className={expanded ? '' : 'collapsed'} />
            <strong>数据集 {index + 1}</strong>
            <span>{expanded ? '收起配置' : '展开配置'}</span>
          </button>
        )}
        {!readOnly && (
          <button
            type="button"
            className="icon-button"
            aria-label={`移除统计数据集 ${index + 1}`}
            disabled={!removable}
            onClick={() => onRemove(index)}
          >
            <Trash2 size={15} />
          </button>
        )}
      </div>
      {!readOnly && (
        <div className="dataset-binding-overview">
          <strong>{dataset ? `${dataset.name} · ${dataset.version}` : '尚未选择数据集'}</strong>
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
      )}
      <div
        className="dataset-binding-content"
        id={contentId}
        hidden={!readOnly && !expanded}
        onInvalidCapture={() => onInvalid(index)}
      >
        {children}
      </div>
    </article>
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
