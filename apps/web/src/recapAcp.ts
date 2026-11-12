import {
  composedChoice,
  stageCompositionDraft,
  trainingProfileSeed,
} from './ComposedDatasetProcessing';
import { composedProcessingErrors, type ComposedProcessingSpec } from './composedProcessing';
import { DEFAULT_AUTO_CONFIG } from './trainingConfiguration';
import type { TrainingDraftSnapshot } from './trainingDraft';
import { isTrainingNormalizationMethod } from './trainingNormalization';
import type { Asset, Run } from './types';

const usableAsset = (asset?: Asset) =>
  ['REGISTERED', 'READY'].includes(asset?.state ?? asset?.status ?? '') &&
  asset?.metadata?.availability !== 'removed';

export function recapAcpSnapshot(
  run: Run,
  selected: Asset,
  assets: readonly Asset[],
): TrainingDraftSnapshot {
  const stage =
    run.training_stage ??
    run.code_expectation?.training_stage ??
    run.attempts?.at(-1)?.code_snapshot?.training_stage;
  if (
    run.template_id !== 'recap' ||
    run.state !== 'SUCCEEDED' ||
    (stage !== 'value' && stage !== 'stage_progress')
  )
    throw new Error('请从已完成的 Value 或 Kai0 作业选择 ACP 数据集。');
  const dataset = assets.find((asset) => asset.id === selected.id);
  if (
    !dataset ||
    dataset.kind !== 'DATASET' ||
    dataset.node_id !== run.node_id ||
    !usableAsset(dataset)
  )
    throw new Error('标注数据集当前不可用，或与来源训练不在同一台机器。');

  const attached = dataset.metadata?.recap_annotation_source;
  const rawLinkage = stage === 'value' ? attached : dataset.metadata;
  if (!rawLinkage || typeof rawLinkage !== 'object' || Array.isArray(rawLinkage))
    throw new Error('标注数据集缺少来源记录。');
  const linkage = rawLinkage as Record<string, unknown>;
  const attemptId = stage === 'value' ? linkage.attempt_id : dataset.origin_attempt_id;
  const attempt = run.attempts?.find((candidate) => candidate.id === attemptId);
  if (!attempt || (attempt.state ?? attempt.status) !== 'SUCCEEDED')
    throw new Error('标注数据集未关联本次训练中已成功的作业。');
  if (
    stage === 'value'
      ? linkage.run_id !== run.id ||
        !run.outputs?.some(
          (asset) =>
            asset.id === linkage.checkpoint_asset_id &&
            asset.kind === 'MODEL' &&
            asset.node_id === run.node_id &&
            asset.origin_attempt_id === attempt.id,
        )
      : !run.outputs?.some(
          (asset) => asset.id === dataset.id && asset.origin_attempt_id === attempt.id,
        )
  )
    throw new Error('标注数据集的来源与当前训练不一致。');

  const sourceDataset = linkage.source_dataset_asset_id;
  const sourceNorm = linkage.source_norm_asset_id;
  const sourceIndex = linkage.source_dataset_index;
  if (
    sourceIndex !== undefined &&
    (typeof sourceIndex !== 'number' ||
      !Number.isInteger(sourceIndex) ||
      sourceIndex < 0 ||
      sourceIndex > 63)
  )
    throw new Error('标注数据集的来源数据组编号无效。');
  const recordedBindings = run.dataset_bindings ?? run.code_expectation?.dataset_bindings ?? [];
  const matchingBindings = recordedBindings.filter(
    (binding, index) =>
      (sourceIndex === undefined || sourceIndex === index) &&
      binding.dataset_asset_id === sourceDataset &&
      binding.norm_asset_id === sourceNorm,
  );
  const inputs = run.inputs || {};
  const matchesLegacyInputs = Object.keys(inputs).some((role) => {
    const match = /^dataset(_\d+)?$/u.exec(role);
    if (!match) return false;
    if (sourceIndex !== undefined) {
      if (recordedBindings.length) return false;
      const suffix = match[1] || '';
      if (
        suffix
          ? suffix !== `_${sourceIndex}`
          : sourceIndex !== 0 || Object.hasOwn(inputs, 'dataset_0')
      )
        return false;
    }
    const data = inputs[role];
    const norm = inputs[`norm${match[1] || ''}`];
    return (
      (typeof data === 'string' ? data : data?.id) === sourceDataset &&
      (typeof norm === 'string' ? norm : norm?.id) === sourceNorm
    );
  });
  if (
    typeof sourceDataset !== 'string' ||
    sourceDataset === dataset.id ||
    typeof sourceNorm !== 'string' ||
    (!matchingBindings.length && !matchesLegacyInputs)
  )
    throw new Error('标注数据集的原始数据和归一化参数与来源训练不一致。');
  const norm = assets.find((asset) => asset.id === sourceNorm);
  if (!norm || norm.kind !== 'NORM' || norm.node_id !== run.node_id || !usableAsset(norm))
    throw new Error('来源训练的归一化参数在当前机器不可用，请手动配置 ACP。');

  const sourceProcessor = matchingBindings.length === 1 ? matchingBindings[0].processor : undefined;
  const rawComposition = sourceProcessor?.parameters.composition;
  const composition =
    sourceProcessor?.name === 'ComposableDataConfig' &&
    !composedProcessingErrors(rawComposition).length
      ? (rawComposition as unknown as ComposedProcessingSpec)
      : stageCompositionDraft(trainingProfileSeed(dataset), 'offline_rl');
  const normalization = run.normalization_method ?? run.code_expectation?.normalization_method;
  const baseName = (run.name.trim() || 'ReCap').slice(0, 154).replace(/[\uD800-\uDBFF]$/u, '');

  return {
    step: 0,
    advancedRecipe: false,
    editedParameters: [],
    editedStageParameters: [],
    entrypointEdited: false,
    request: {
      name: `${baseName} · ACP`,
      node_id: run.node_id,
      profile_id: run.profile_id,
      template_id: 'recap',
      training_stage: 'offline_rl',
      inputs: {},
      dataset_bindings: [
        {
          dataset_asset_id: dataset.id,
          norm_asset_id: norm.id,
          weight: 1,
          processor: composedChoice({ ...composition, aux_keys: ['is_positive'] }),
        },
      ],
      auto_config: { ...DEFAULT_AUTO_CONFIG },
      ...(isTrainingNormalizationMethod(normalization)
        ? { normalization_method: normalization }
        : {}),
      parameters: {},
      gpu_count: 1,
    },
  };
}
