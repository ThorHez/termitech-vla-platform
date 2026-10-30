/** Starting heuristics for one machine; explicit user values always win. */
export const TRAINING_RESOURCE_FIELDS = ['fsdp_devices', 'num_workers'] as const;
export type TrainingResourceField = (typeof TRAINING_RESOURCE_FIELDS)[number];
export const RESOURCE_RECOMMENDATION_AUTO_PREFIX = 'resource_recommendation.auto.';

export interface TrainingResourceRecommendations {
  gpuCount?: number;
  cpuCount?: number;
  availableCpuCount?: number;
  fsdp_devices?: number;
  num_workers?: number;
}

export function isTrainingResourceField(key: string): key is TrainingResourceField {
  return TRAINING_RESOURCE_FIELDS.some((field) => field === key);
}

const positiveInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0;

export function recommendTrainingResources(
  gpuCount: number,
  metrics?: Record<string, unknown>,
): TrainingResourceRecommendations {
  const cpuCount = positiveInteger(metrics?.cpu_count) ? metrics.cpu_count : undefined;
  const availableCpuCount = positiveInteger(metrics?.available_cpu_count)
    ? metrics.available_cpu_count
    : undefined;
  if (!positiveInteger(gpuCount)) return { cpuCount, availableCpuCount };
  return {
    gpuCount,
    cpuCount,
    availableCpuCount,
    fsdp_devices: gpuCount,
    num_workers: Math.max(
      0,
      Math.min(
        2 * gpuCount,
        16,
        availableCpuCount === undefined ? Infinity : availableCpuCount - 1,
      ),
    ),
  };
}

export function resourceRecommendationMarker(key: TrainingResourceField): string {
  return RESOURCE_RECOMMENDATION_AUTO_PREFIX + key;
}

export function automaticResourceRecommendation(
  edited: ReadonlySet<string>,
  key: TrainingResourceField,
): boolean {
  return edited.has(resourceRecommendationMarker(key));
}

/** The field edit marker protects automatic values from late recipe-default reads. */
export function setResourceRecommendationMode(
  edited: Set<string>,
  key: TrainingResourceField,
  automatic: boolean,
): void {
  edited.add(key);
  if (automatic) edited.add(resourceRecommendationMarker(key));
  else edited.delete(resourceRecommendationMarker(key));
}

export function resourceRecommendationTracking(edited: Iterable<string>): Set<string> {
  const allowed = new Set<string>(
    TRAINING_RESOURCE_FIELDS.flatMap((key) => [key, resourceRecommendationMarker(key)]),
  );
  return new Set(Array.from(edited).filter((key) => allowed.has(key)));
}

export function retainedTrainingResources(
  parameters: Record<string, number>,
): Record<string, number> {
  return Object.fromEntries(
    TRAINING_RESOURCE_FIELDS.filter((key) => Object.hasOwn(parameters, key)).map((key) => [
      key,
      parameters[key],
    ]),
  );
}

export function applyResourceRecommendations(
  parameters: Record<string, number>,
  edited: ReadonlySet<string>,
  recommendations: TrainingResourceRecommendations,
): Record<string, number> {
  let result = parameters;
  for (const key of TRAINING_RESOURCE_FIELDS) {
    const value = recommendations[key];
    if (
      automaticResourceRecommendation(edited, key) &&
      value !== undefined &&
      !Object.is(parameters[key], value)
    ) {
      if (result === parameters) result = { ...parameters };
      result[key] = value;
    }
  }
  return result;
}

export function trainingResourceRecommendationHint(
  key: TrainingResourceField,
  recommendations: TrainingResourceRecommendations,
): string {
  if (key === 'fsdp_devices')
    return '推荐与申请的 GPU 数量一致，自动模式会随 GPU 数量更新；手动填写时须能整除申请的 GPU 数量，清空后沿用配置。';
  const cpu =
    recommendations.availableCpuCount === undefined
      ? `Agent 可见 CPU 数未知${recommendations.cpuCount === undefined ? '' : `（主机总计 ${recommendations.cpuCount} 个）`}，暂按 GPU 数量估算。`
      : `Agent 可见 CPU ${recommendations.availableCpuCount} 个，推荐上限按 CPU 数减 1 计算。`;
  return `每张 GPU 按 2 个加载进程、最多 16 个估算；${cpu}这是起始建议，可按读取速度和内存占用手动调整，0 表示主进程加载。`;
}
