import type { Asset, RecipeDetail, RunRequest, TrainingStage } from './types';

export const RECAP_PYTORCH_ENTRYPOINT = 'scripts/acp/train_recap_pytorch.py';

/** A registered directory can have no format metadata. The Agent checks its actual files. */
export function recapModelError(asset: Asset, stage?: TrainingStage | null): string | undefined {
  const format = asset.metadata?.format;
  if (['openpi-orbax', 'orbax', 'jax', 'openpi-jax'].some((value) => value === format))
    return 'ReCap 需要 PyTorch 权重，请选择 model.safetensors 或包含该文件的检查点目录。';
  const role = asset.metadata?.model_role;
  if (
    stage &&
    ((role === 'stage_progress_estimator' && stage !== 'stage_progress') ||
      (role === 'value_estimator' && stage !== 'value'))
  )
    return '所选估计器权重与当前 ReCap 阶段不匹配，请选择相应的 PyTorch 检查点。';
  return undefined;
}

/** Saved JAX settings are omitted when creating a new PyTorch ReCap job. */
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

export const trainingStageLabels: Record<TrainingStage, string> = {
  value: '价值函数训练',
  stage_progress: 'Kai0 阶段进度与优势标注',
  offline_rl: '离线强化学习（策略提升）',
};
export const recapStageSupervision = {
  value: 'value_target',
  stage_progress: 'stage_progress_gt',
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
  if (stage === 'value') {
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
