import { composedChoice, trainingProfileSeed } from './ComposedDatasetProcessing';
import { composedProcessingErrors, trainingCompositionDraft } from './composedProcessing';
import {
  inferDatasetActionHorizon,
  withoutKai0StateTransform,
  withoutStateTokenLayout,
} from './trainingConfiguration';
import type { Asset, DataProcessorChoice, DatasetBinding, NormDataset, RunRequest } from './types';

export function normProcessorChoice(processor: DataProcessorChoice): DataProcessorChoice {
  processor = withoutKai0StateTransform(processor);
  const raw = processor.parameters.composition;
  return processor.name === 'ComposableDataConfig' &&
    raw &&
    typeof raw === 'object' &&
    !Array.isArray(raw)
    ? {
        ...processor,
        parameters: { ...processor.parameters, composition: { ...raw, cameras: [] } },
      }
    : processor;
}

export function newNormDataset(automatic: boolean, dataset?: Asset): NormDataset {
  return automatic
    ? {
        dataset_asset_id: dataset?.id || '',
        processor: composedChoice(
          trainingCompositionDraft({ ...trainingProfileSeed(dataset), cameras: [] }, 'norm'),
          'norm',
        ),
      }
    : { dataset_asset_id: dataset?.id || '', recipe_index: 0 };
}

export function normAdvancedRecipe(request: RunRequest, saved = false): boolean {
  return request.template_id === 'openpi-norm'
    ? !request.auto_config && (!!request.recipe?.trim() || saved)
    : saved;
}

export function selectNormDataset(
  row: NormDataset,
  dataset: Asset | undefined,
  automatic: boolean,
  preserveProcessing = false,
): NormDataset {
  const unfinishedComposition =
    row.processor?.name === 'ComposableDataConfig' &&
    composedProcessingErrors(row.processor.parameters.composition, 'norm').length > 0;
  if (
    automatic &&
    (!row.processor ||
      (!preserveProcessing &&
        (row.dataset_asset_id || unfinishedComposition) &&
        row.processor.name === 'ComposableDataConfig'))
  )
    return newNormDataset(true, dataset);
  return { ...row, dataset_asset_id: dataset?.id || '' };
}

export function inferNormActionHorizon(rows: readonly NormDataset[], assets: readonly Asset[]) {
  // Statistics are independent jobs. Keep each source's diagnostics without
  // importing the training mixture's cross-dataset compatibility requirement.
  const results = rows.map((row, index) => {
    const inferred = inferDatasetActionHorizon([normProcessingBinding(row)], assets);
    return {
      ...inferred,
      sources: inferred.sources.map((source) => ({ ...source, bindingIndex: index })),
      diagnostics: inferred.diagnostics.map((diagnostic) => ({
        ...diagnostic,
        bindingIndex: index,
      })),
    };
  });
  const horizons = [...new Set(results.flatMap((result) => result.recommendedHorizon ?? []))];
  const errors = results.flatMap((result) => result.errors);
  return {
    ...(horizons.length === 1 && !errors.length ? { recommendedHorizon: horizons[0] } : {}),
    sources: results.flatMap((result) => result.sources),
    diagnostics: results.flatMap((result) => result.diagnostics),
    errors,
    warnings: results.flatMap((result) => result.warnings),
  };
}

export function normDatasetsForRequest(
  request: Pick<RunRequest, 'norm_datasets' | 'inputs'>,
): NormDataset[] {
  return (
    request.norm_datasets ?? [{ dataset_asset_id: request.inputs.dataset || '', recipe_index: 0 }]
  );
}

// Reuse the existing processing editor without carrying training-only asset
// choices into the normalization request or its serialized draft.
export function normProcessingBinding(row: NormDataset): DatasetBinding {
  return { ...row, norm_asset_id: '', weight: 1 };
}

export function normProcessingChoice(binding: DatasetBinding): NormDataset {
  return binding.processor
    ? { dataset_asset_id: binding.dataset_asset_id, processor: binding.processor }
    : { dataset_asset_id: binding.dataset_asset_id, recipe_index: binding.recipe_index ?? 0 };
}

export function appendNormDataset(rows: readonly NormDataset[], automatic = false): NormDataset[] {
  const first = rows[0];
  const copied = first ? structuredClone(first) : newNormDataset(automatic);
  return [...rows, { ...copied, dataset_asset_id: '' }];
}

export function normDatasetErrors(rows: readonly NormDataset[], automatic = false): string[] {
  if (!rows.length || rows.length > 64) return ['请选择 1 到 64 个本机数据集。'];
  const ids = new Set<string>();
  const errors: string[] = [];
  for (const [index, row] of rows.entries()) {
    if (!row.dataset_asset_id) errors.push(`请选择数据集 ${index + 1}。`);
    else if (ids.has(row.dataset_asset_id)) errors.push(`数据集 ${index + 1} 已重复选择。`);
    ids.add(row.dataset_asset_id);
    if ((row.recipe_index !== undefined) === (row.processor !== undefined))
      errors.push(`数据集 ${index + 1} 必须选择一种处理方式。`);
    if (automatic && !row.processor)
      errors.push(`数据集 ${index + 1} 需配置独立处理器，自动构造不使用代码配方数据项。`);
    if (
      row.recipe_index !== undefined &&
      (!Number.isInteger(row.recipe_index) || row.recipe_index < 0 || row.recipe_index > 63)
    )
      errors.push(`数据集 ${index + 1} 的配方数据项序号必须为 1 到 64。`);
  }
  return errors;
}

export function buildNormPreviewRequest(request: RunRequest): RunRequest {
  return {
    name: request.name,
    node_id: request.node_id,
    profile_id: request.profile_id,
    template_id: 'openpi-norm',
    ...(request.auto_config
      ? {
          auto_config: {
            ...withoutStateTokenLayout(request.auto_config),
            finetune_mode: 'full' as const,
          },
        }
      : { recipe: request.recipe?.trim() }),
    inputs: {},
    norm_datasets: normDatasetsForRequest(request).map((row) =>
      row.processor
        ? { dataset_asset_id: row.dataset_asset_id, processor: normProcessorChoice(row.processor) }
        : { dataset_asset_id: row.dataset_asset_id, recipe_index: row.recipe_index },
    ),
    parameters: Object.fromEntries(
      Object.entries(request.parameters).filter(([key]) =>
        ['batch_size', 'num_workers'].includes(key),
      ),
    ),
    gpu_count: 0,
  };
}
