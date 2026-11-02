import type { RunRequest, TrainingNormalizationMethod } from './types';

export const DEFAULT_TRAINING_NORMALIZATION_METHOD: TrainingNormalizationMethod = 'quantile';
export const TRAINING_NORMALIZATION_LABELS: Record<TrainingNormalizationMethod, string> = {
  minmax: 'Min-Max（最小/最大值）',
  quantile: 'q01/q99（1%–99%分位数）',
  zscore: 'Z-score（均值/标准差）',
};

export function isTrainingNormalizationMethod(
  value: unknown,
): value is TrainingNormalizationMethod {
  return typeof value === 'string' && Object.hasOwn(TRAINING_NORMALIZATION_LABELS, value);
}

export function trainingNormalizationLabel(value?: TrainingNormalizationMethod | null) {
  return value ? TRAINING_NORMALIZATION_LABELS[value] : '沿用配方 / 旧配置';
}

export function withTrainingNormalizationMethod(
  request: RunRequest,
  method?: TrainingNormalizationMethod,
): RunRequest {
  const { normalization_method: _previous, ...rest } = request;
  return method === undefined ? rest : { ...rest, normalization_method: method };
}
