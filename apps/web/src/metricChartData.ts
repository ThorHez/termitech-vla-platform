export interface MetricChartPoint {
  step: number;
  value: number | null;
  timestamp?: string;
}

export type MetricChartInput =
  | MetricChartPoint
  | { step: number; loss: number | null; timestamp?: string };

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

function paddedDomain(low: number, high: number) {
  if (low === 0 && high === 0) return { low: -1, high: 1 };
  const scale = Math.max(Math.abs(low), Math.abs(high));
  const span = low === high ? 1 : high / scale - low / scale;
  const padding = span * 0.08;
  const floor = Math.max(-Number.MAX_VALUE, (low / scale - padding) * scale);
  const ceiling = Math.min(Number.MAX_VALUE, (high / scale + padding) * scale);
  // Preserve even subnormal constant values when percentage padding rounds away.
  return {
    low: floor === ceiling ? low - Number.MIN_VALUE : floor,
    high: floor === ceiling ? high + Number.MIN_VALUE : ceiling,
  };
}

export function buildMetricChartModel(values: readonly MetricChartInput[]) {
  const points: MetricChartPoint[] = [];
  const segments: MetricChartPoint[][] = [];
  let segment: MetricChartPoint[] = [];
  let first = Infinity;
  let last = -Infinity;
  let low = Infinity;
  let high = -Infinity;
  let knownCount = 0;
  let latest: MetricChartPoint | undefined;
  for (const input of values) {
    if (!finite(input.step)) {
      segment = [];
      continue;
    }
    const raw = 'value' in input ? input.value : input.loss;
    const point = { step: input.step, value: finite(raw) ? raw : null, timestamp: input.timestamp };
    points.push(point);
    first = Math.min(first, point.step);
    last = Math.max(last, point.step);
    if (point.value === null) {
      segment = [];
      continue;
    }
    if (!segment.length) {
      segment = [];
      segments.push(segment);
    }
    segment.push(point);
    low = Math.min(low, point.value);
    high = Math.max(high, point.value);
    knownCount++;
    latest = point;
  }
  return {
    points,
    segments,
    first: points.length ? first : 0,
    last: points.length ? last : 0,
    domain: knownCount ? paddedDomain(low, high) : { low: -1, high: 1 },
    constant: knownCount > 0 && low === high,
    knownCount,
    latest,
  };
}

// Normalizing before subtracting avoids overflowing a range such as -MAX..MAX.
export function metricFraction(value: number, low: number, high: number) {
  if (low === high) return 0.5;
  const scale = Math.max(Math.abs(low), Math.abs(high), Number.MIN_VALUE);
  return Math.max(0, Math.min(1, (value / scale - low / scale) / (high / scale - low / scale)));
}

export function metricInterpolate(low: number, high: number, fraction: number) {
  const result = low * (1 - fraction) + high * fraction;
  return Number.isFinite(result) ? result : Math.sign(result) * Number.MAX_VALUE;
}

export function nearestMetricPoint(
  points: readonly MetricChartPoint[],
  fraction: number,
  first: number,
  last: number,
) {
  let nearest = -1;
  let distance = Infinity;
  for (let index = 0; index < points.length; index++) {
    const candidate = Math.abs(metricFraction(points[index].step, first, last) - fraction);
    if (candidate < distance) {
      distance = candidate;
      nearest = index;
    }
  }
  return nearest;
}

export function formatMetricValue(value: number | null | undefined, significantDigits = 6) {
  if (!finite(value)) return '—';
  if (value === 0) return '0';
  if (Math.abs(value) < 0.001 || Math.abs(value) >= 1e6) {
    const [mantissa, exponent] = value.toExponential(significantDigits - 1).split('e');
    const power = Number(exponent);
    return `${Number(mantissa)}e${power >= 0 ? '+' : ''}${power}`;
  }
  return Number(value.toPrecision(significantDigits)).toString();
}
