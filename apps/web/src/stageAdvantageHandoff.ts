import { recapModelError, stageAnnotationSource } from './recap';
import { retryTrainingSnapshot } from './runRetry';
import type { TrainingDraftSnapshot } from './trainingDraft';
import type { Asset, Run } from './types';

export interface StageAdvantageHandoff {
  checkpoint: Asset;
  stage: 'stage_progress' | 'stage_annotation';
}

const available = (asset?: Asset) =>
  ['READY', 'REGISTERED'].includes(asset?.state ?? asset?.status ?? '') &&
  asset?.metadata?.availability !== 'removed';

export function stageAdvantageCheckpoints(run: Run) {
  const stage =
    run.training_stage ??
    run.code_expectation?.training_stage ??
    run.attempts?.at(-1)?.code_snapshot?.training_stage;
  if (run.template_id !== 'recap' || run.state !== 'SUCCEEDED' || stage !== 'stage_progress')
    return [];
  const attempts = new Set(
    run.attempts
      ?.filter((attempt) => (attempt.state ?? attempt.status) === 'SUCCEEDED')
      .map(({ id }) => id),
  );
  return (run.outputs || []).filter(
    (asset) =>
      asset.kind === 'MODEL' &&
      asset.node_id === run.node_id &&
      !!asset.origin_attempt_id &&
      attempts.has(asset.origin_attempt_id) &&
      available(asset) &&
      asset.metadata?.model_role === 'stage_progress_estimator' &&
      !recapModelError(asset, 'stage_progress'),
  );
}

const annotationKeys = ['n_step', 'positive_ratio', 'stage_count'];

export function stageAdvantageHandoffSnapshot(
  run: Run,
  { checkpoint, stage }: StageAdvantageHandoff,
  assets: readonly Asset[],
): TrainingDraftSnapshot {
  const model = assets.find((asset) => asset.id === checkpoint.id);
  if (
    !stageAdvantageCheckpoints(run).some(({ id }) => id === checkpoint.id) ||
    model?.kind !== 'MODEL' ||
    model.node_id !== run.node_id ||
    !available(model) ||
    model.metadata?.model_role !== 'stage_progress_estimator' ||
    recapModelError(model, 'stage_progress')
  )
    throw new Error('请从已完成的阶段进度作业选择当前机器可用的模型检查点。');
  if (stage === 'stage_annotation' && stageAnnotationSource(model)?.run_id !== run.id)
    throw new Error('所选检查点缺少本次阶段进度训练的有效标注来源记录。');
  const snapshot = retryTrainingSnapshot(run);
  const { request } = snapshot;
  const bindings = request.dataset_bindings || [];
  const [binding] = bindings;
  const dataset = assets.find(({ id }) => id === binding?.dataset_asset_id);
  const norm = assets.find(({ id }) => id === binding?.norm_asset_id);
  if (
    bindings.length !== 1 ||
    dataset?.kind !== 'DATASET' ||
    dataset.node_id !== run.node_id ||
    !available(dataset) ||
    norm?.kind !== 'NORM' ||
    norm.node_id !== run.node_id ||
    !available(norm)
  )
    throw new Error('原数据集或归一化参数在当前机器不可用，请手动配置。');
  const annotation = stage === 'stage_annotation';
  const previousConfig = request.stage_config ?? {};
  const stageConfig = Object.fromEntries(
    Object.entries(previousConfig).filter(([key]) => annotation === annotationKeys.includes(key)),
  );
  const name = request.name.slice(0, 148).replace(/[\uD800-\uDBFF]$/u, '');
  return {
    ...snapshot,
    editedStageParameters: Object.keys(stageConfig),
    request: {
      ...request,
      name: `${name} · ${annotation ? '优势标注' : '继续训练'}`,
      training_stage: stage,
      stage_config: Object.keys(stageConfig).length ? stageConfig : undefined,
      inputs: { model: model.id },
      ...(annotation
        ? {
            dataset_bindings: [{ ...binding, weight: 1 }],
            gpu_count: 1,
            requested_gpu_uuids: undefined,
            training_gpu_policy: undefined,
            parameters: Object.hasOwn(request.parameters, 'batch_size')
              ? { batch_size: request.parameters.batch_size }
              : {},
          }
        : {}),
    },
  };
}
