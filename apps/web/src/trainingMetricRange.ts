import type { TrainingMetricPoint } from './trainingMetricData';

export type TrainingMetricRange =
  | { mode: 'all' }
  | { mode: 'recent'; steps: number }
  | { mode: 'custom'; start: number; end: number | null };

export interface TrainingMetricBounds {
  start: number;
  end: number | null;
}

export const ALL_TRAINING_STEPS: TrainingMetricRange = { mode: 'all' };

export function parseTrainingMetricRange(
  startText: string,
  endText: string,
): { range: TrainingMetricRange; error?: never } | { range?: never; error: string } {
  const parse = (value: string) => {
    const text = value.trim();
    if (!/^\d+$/.test(text)) return null;
    const number = Number(text);
    return Number.isSafeInteger(number) ? number : null;
  };
  const start = startText.trim() === '' ? 0 : parse(startText);
  const end = endText.trim() === '' ? null : parse(endText);
  if (start === null || (endText.trim() !== '' && end === null)) {
    return { error: '步数需要是大于或等于 0 的整数。' };
  }
  if (end !== null && end < start) {
    return { error: '结束步数不能小于起始步数。' };
  }
  return { range: { mode: 'custom', start, end } };
}

export function trainingMetricRangeBounds(
  range: TrainingMetricRange,
  latestStep: number | null,
): TrainingMetricBounds {
  const latest =
    latestStep !== null && Number.isFinite(latestStep) && latestStep >= 0 ? latestStep : null;
  if (range.mode === 'custom') return { start: range.start, end: range.end ?? latest };
  if (range.mode === 'recent') {
    return { start: latest === null ? 0 : Math.max(0, latest - range.steps), end: latest };
  }
  return { start: 0, end: latest };
}

export function trainingMetricPointsInRange(
  points: readonly TrainingMetricPoint[],
  bounds: TrainingMetricBounds,
): TrainingMetricPoint[] {
  return points.filter(
    (point) => point.step >= bounds.start && (bounds.end === null || point.step <= bounds.end),
  );
}
