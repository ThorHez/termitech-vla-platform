import { modelAssetFormat } from './modelAssetFormat';
import type { Asset, RecipeDetail, RunRequest, TrainingStage, TrainingStageSpec } from './types';

export const RECAP_PYTORCH_ENTRYPOINT = 'scripts/acp/train_recap_pytorch.py';
export const STAGE_ADVANTAGE_MODE = 'Stage Advantage';

export function recapStageEntrypoint(stage: TrainingStageSpec | undefined, datasetCount: number) {
  if (stage?.id === 'offline_rl')
    return datasetCount > 1 ? 'scripts/train_multi_dataset.py' : 'scripts/train.py';
  return (
    (datasetCount > 1 ? stage?.default_multi_entrypoint : undefined) ||
    stage?.default_entrypoint ||
    RECAP_PYTORCH_ENTRYPOINT
  );
}

export const isStageAdvantageStage = (stage?: string | null) =>
  stage === 'stage_progress' || stage === 'stage_annotation';
export const isRecapAnnotationStage = (stage?: string | null) =>
  stage === 'value_annotation' || stage === 'stage_annotation';

export const isStageAdvantageDataset = (asset?: Asset) =>
  ['recap_stage_progress', 'recap_stage_annotation'].some(
    (source) => source === asset?.metadata?.registration_source,
  );
export const isValueAnnotationDataset = (asset?: Asset) =>
  asset?.metadata?.registration_source === 'recap_value_annotation';

export function stageAnnotationSource(asset?: Asset, stage: TrainingStage = 'stage_annotation') {
  const source =
    stage === 'value_annotation' ? asset?.value_annotation_source : asset?.stage_annotation_source;
  if (
    asset?.kind === 'MODEL' &&
    (asset.state ?? asset.status) === 'READY' &&
    asset.metadata?.availability !== 'removed' &&
    typeof source?.attempt_id === 'string' &&
    source.attempt_id &&
    source.attempt_id === asset.origin_attempt_id &&
    typeof source.run_id === 'string' &&
    source.run_id &&
    typeof source.run_name === 'string' &&
    source.run_name.trim() &&
    Number.isSafeInteger(source.step) &&
    source.step >= 0
  )
    return source;
  return undefined;
}

export function stageAnnotationCheckpointLabel(asset: Asset, stage?: TrainingStage) {
  const source = stageAnnotationSource(asset, stage);
  return source
    ? `${source.run_name} · 第 ${source.step.toLocaleString('zh-CN')} 步`
    : `${asset.name} · ${asset.version}`;
}

export function recapModelError(asset: Asset, stage?: TrainingStage | null): string | undefined {
  if (stage === 'value_annotation')
    return stageAnnotationSource(asset, stage)
      ? undefined
      : '请选择本机已成功完成的 Value 训练所产出的可用检查点。';

  const format = modelAssetFormat(asset);
  const role = asset.metadata?.model_role;
  if (stage === 'stage_annotation') {
    const external =
      asset.kind === 'MODEL' &&
      asset.origin_attempt_id == null &&
      ['READY', 'REGISTERED'].includes(asset.state ?? asset.status ?? '') &&
      asset.metadata?.availability !== 'removed' &&
      format === 'openpi-pytorch' &&
      (role == null || role === 'stage_progress_estimator');
    return stageAnnotationSource(asset, stage) || external
      ? undefined
      : '请选择本机已成功完成的阶段进度训练所产出的检查点，或在「数据与模型」登记的 PyTorch 阶段进度检查点。';
  }
  if (stage === 'offline_rl') {
    const state = asset.state ?? asset.status;
    if (
      (state && state !== 'READY' && state !== 'REGISTERED') ||
      asset.metadata?.availability === 'removed'
    )
      return '所选 JAX / Orbax 策略权重当前不可用，请重新选择。';
    if (role === 'stage_progress_estimator' || role === 'value_estimator')
      return '所选估计器权重与 ACP 不匹配，请选择 JAX / Orbax 策略权重。';
    if (format === 'openpi-pytorch')
      return 'ACP 已改用 JAX / Orbax，请为这份配置重新选择 JAX / Orbax 初始策略权重。';
    if (format !== 'openpi-orbax')
      return 'ACP 需要 JAX / Orbax 策略权重。请在「数据与模型」登记或补录实际的 JAX / Orbax 格式。';
    return undefined;
  }
  if (format === 'openpi-orbax')
    return 'ReCap 需要 PyTorch 权重，请选择 model.safetensors 或包含该文件的检查点目录。';
  if (isStageAdvantageStage(stage) && format !== 'openpi-pytorch')
    return 'Stage Advantage 仅显示已标记为 PyTorch 的权重。请在「数据与模型」补录权重格式或登记 PyTorch 权重。';
  if (
    stage &&
    ((role === 'stage_progress_estimator' && !isStageAdvantageStage(stage)) ||
      (role === 'value_estimator' && stage !== 'value'))
  )
    return '所选估计器权重与当前 ReCap 阶段不匹配，请选择相应的 PyTorch 检查点。';
  return undefined;
}

/** The controller chooses the ReCap entrypoint and owns shared FSDP/EMA defaults. */
export function withoutLegacyRecapSettings(request: RunRequest): RunRequest {
  if (
    request.template_id !== 'recap' ||
    (!request.entrypoint &&
      !Object.hasOwn(request.parameters, 'fsdp_devices') &&
      !Object.hasOwn(request.parameters, 'ema_decay'))
  )
    return request;
  const parameters = { ...request.parameters };
  delete parameters.fsdp_devices;
  delete parameters.ema_decay;
  return { ...request, parameters, entrypoint: undefined };
}

export function recapSettingsMigrationMessage(stage?: TrainingStage | null) {
  return stage === 'offline_rl'
    ? 'ACP 使用 JAX。平台会选择训练入口，并按 GPU 数量与配方配置 FSDP 和 EMA。'
    : '此配置使用 PyTorch ReCap 入口，已移除保存的入口及旧 FSDP 和 EMA 参数。';
}

export const trainingStageLabels: Record<TrainingStage, string> = {
  value: '价值函数训练',
  value_annotation: '优势值标注',
  stage_progress: '阶段进度训练',
  stage_annotation: '优势标注',
  offline_rl: '离线强化学习（策略提升）',
};
export const recapStageSupervision = {
  value: 'value_target',
  value_annotation: 'value_target',
  stage_progress: 'stage_progress_gt',
  stage_annotation: 'stage_progress_gt',
  offline_rl: 'is_positive',
} as const satisfies Record<TrainingStage, string>;

export const stageParameterLabels: Record<string, string> = {
  val_ratio: '验证集比例',
  eval_interval: '评估间隔（步）',
  eval_batches: '每次评估批次数',
  n_step: '优势窗口（帧）',
  positive_ratio: '各阶段正标签比例',
  stage_count: '阶段数',
  acp_dropout: 'ACP 条件丢弃概率',
};

export const stageParameterHints: Record<string, string> = {
  eval_interval: '启用验证后，每隔多少个训练步评估一次。',
  eval_batches: '每次评估最多读取的验证批次数。',
  n_step: '比较同一轨迹中相隔多少帧的进度，与动作预测长度独立。',
  positive_ratio: '每个阶段内按优势排序，取该比例作为正标签。',
  stage_count: '与数据标注所用的阶段数一致，用于分阶段计算优势分位数。',
};

export function recapRecipeConfiguration(detail: RecipeDetail, stage: TrainingStage) {
  const values: Record<string, number> = {};
  const fields = new Map(detail.fields.map((field) => [field.path, field]));
  const literal = (path: string) => {
    const field = fields.get(path);
    return field?.kind === 'literal' ? field.value : undefined;
  };
  let mixedDropout = false;
  if (stage === 'value' || stage === 'stage_progress') {
    for (const name of ['val_ratio', 'eval_interval', 'eval_batches']) {
      const value = literal(name);
      if (typeof value === 'number' && Number.isFinite(value)) values[name] = value;
    }
  } else if (stage === 'offline_rl') {
    const dropoutFields = detail.fields.filter((field) =>
      /^data(?:\.datasets\[\d+\])?\.acp_dropout$/u.test(field.path),
    );
    const dropout = dropoutFields.map((field) =>
      field.kind === 'literal' && typeof field.value === 'number' ? field.value : undefined,
    );
    if (dropout.length && dropout.every((value) => Number.isFinite(value) && value === dropout[0]))
      values.acp_dropout = dropout[0]!;
    else if (dropout.length) mixedDropout = true;
  }
  return {
    values,
    mixedDropout,
    valueHead: [
      'model.value_head_only',
      'model.freeze_backbone',
      'model.freeze_filter',
      'freeze_filter',
    ]
      .map((path) => {
        const field = fields.get(path);
        return field
          ? {
              path,
              value: field.kind === 'literal' ? JSON.stringify(field.value) : field.expression,
            }
          : undefined;
      })
      .filter((field): field is { path: string; value: string } => Boolean(field)),
  };
}
