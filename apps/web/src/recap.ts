import type { RecipeDetail, TrainingStage } from './types';

export const trainingStageLabels: Record<TrainingStage, string> = {
  value: '价值函数训练',
  offline_rl: '离线强化学习（策略提升）',
};
export const stageParameterLabels: Record<string, string> = {
  val_ratio: '验证集比例',
  eval_interval: '评估间隔（步）',
  eval_batches: '每次评估批次数',
  acp_dropout: 'ACP 条件丢弃概率',
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
  } else {
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
