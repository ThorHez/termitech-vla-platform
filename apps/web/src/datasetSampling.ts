import type { DatasetBinding } from './types';

export interface SampleCountReading {
  status: 'loading' | 'ready' | 'unavailable';
  count?: number;
}

export interface DatasetSamplingShare {
  sampleCount?: number;
  weightPercent?: number;
  samplePercent?: number;
  samplingPercent?: number;
  weightIssue?: string;
  sampleIssue?: string;
  samplingIssue?: string;
}

/** Ratios for standard per-frame sampling, without allocating or visiting samples. */
export function datasetSamplingShares(
  bindings: readonly DatasetBinding[],
  counts: Readonly<Record<string, SampleCountReading>>,
): DatasetSamplingShare[] {
  if (!bindings.length) return [];
  const weights = bindings.map((row) => row.weight);
  const weightsValid = weights.every((weight) => Number.isFinite(weight) && weight > 0);
  const weightIssue = weightsValid ? undefined : '请填写所有数据集的有效正权重';
  const weightScale = weightsValid ? Math.max(...weights) : 1;
  const scaledWeights = weights.map((weight) => weight / weightScale);
  const weightTotal = scaledWeights.reduce((sum, weight) => sum + weight, 0);
  const samples = bindings.map((binding) => {
    const reading = counts[binding.dataset_asset_id];
    return binding.dataset_asset_id &&
      reading?.status === 'ready' &&
      Number.isSafeInteger(reading.count) &&
      reading.count! >= 0
      ? reading.count
      : undefined;
  });
  const missing = samples.flatMap((count, index) => (count === undefined ? [index + 1] : []));
  const sampleScale = Math.max(0, ...samples.map((count) => count ?? 0));
  const sampleIssue = missing.length
    ? `缺少第 ${missing.join('、')} 组样本数`
    : sampleScale === 0
      ? '样本总数为 0'
      : undefined;
  const scaledSamples = samples.map((count) => (sampleScale > 0 ? (count ?? 0) / sampleScale : 0));
  const sampleTotal = scaledSamples.reduce((sum, count) => sum + count, 0);
  // Scaling both factors avoids overflow from large frame counts or weights.
  const masses = scaledSamples.map((count, index) => count * scaledWeights[index]);
  const massTotal = masses.reduce((sum, mass) => sum + mass, 0);
  const samplingIssue =
    sampleIssue || weightIssue || (!(massTotal > 0) ? '权重差异过大，无法可靠计算' : undefined);
  return bindings.map((_, index) => ({
    sampleCount: samples[index],
    weightPercent: !weightIssue ? (100 * scaledWeights[index]) / weightTotal : undefined,
    samplePercent: !sampleIssue ? (100 * scaledSamples[index]) / sampleTotal : undefined,
    samplingPercent: !samplingIssue ? (100 * masses[index]) / massTotal : undefined,
    weightIssue,
    sampleIssue,
    samplingIssue,
  }));
}

export function samplingPercentText(percent: number | undefined, issue?: string): string {
  return percent === undefined
    ? `暂无法计算${issue ? `（${issue}）` : ''}`
    : `约 ${percent.toFixed(2)}%`;
}
