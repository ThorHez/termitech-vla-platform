export interface NormalizationChartFeature {
  key: string;
  dimension: number;
  mean: number[];
  std: number[];
  q01?: number[];
  q99?: number[];
  min?: number[];
  max?: number[];
}

export interface NormalizationChartRow {
  index: number;
  mean?: number;
  std?: number;
  q01?: number;
  q99?: number;
  min?: number;
  max?: number;
  range?: readonly [number, number];
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

export function normalizationChartRows(
  feature: NormalizationChartFeature,
  start: number,
  count: number,
): NormalizationChartRow[] {
  const dimension = Number.isSafeInteger(feature.dimension)
    ? Math.max(0, Math.min(4096, feature.dimension))
    : 0;
  const first = Number.isSafeInteger(start) && start >= 0 ? start : 0;
  const length = finite(count)
    ? Math.max(0, Math.min(32, Math.floor(count), dimension - first))
    : 0;
  return Array.from({ length }, (_, offset) => {
    const index = first + offset;
    const row: NormalizationChartRow = { index };
    for (const key of ['mean', 'std', 'q01', 'q99', 'min', 'max'] as const) {
      const value = feature[key]?.[index];
      if (finite(value) && (key !== 'std' || value >= 0)) row[key] = value;
    }
    if (row.q01 !== undefined && row.q99 !== undefined && row.q01 <= row.q99)
      row.range = [row.q01, row.q99];
    return row;
  });
}

export interface StatisticsChartDomain {
  min: number;
  max: number;
  scale: number;
  low: number;
  high: number;
  ticks: number[];
  empty: boolean;
}

const product = (value: number, scale: number) => {
  const result = value * scale;
  return Number.isFinite(result)
    ? result === 0
      ? 0
      : result
    : Math.sign(value) * Number.MAX_VALUE;
};

/** Scale before subtracting so domains spanning -MAX_VALUE..MAX_VALUE stay finite. */
export function chartDomain(
  values: readonly unknown[],
  includeZero = false,
): StatisticsChartDomain {
  const valid = values.filter(finite);
  const empty = valid.length === 0;
  if (includeZero && !empty) valid.push(0);
  let scale = valid.reduce((largest, value) => Math.max(largest, Math.abs(value)), 0) || 1;
  let low = empty ? 0 : Math.min(...valid.map((value) => value / scale));
  let high = empty ? 1 : Math.max(...valid.map((value) => value / scale));
  if (low === high) {
    if (low === 0) {
      scale = 1;
      low = includeZero ? 0 : -1;
      high = 1;
    } else {
      const padding = Math.abs(low) * 0.1;
      low -= padding;
      high += padding;
    }
  } else if (!empty) {
    const padding = (high - low) * 0.06;
    low = includeZero && low === 0 ? 0 : low - padding;
    high += padding;
  }
  const limit = Math.min(4, Number.MAX_VALUE / scale);
  low = Math.max(-limit, low);
  high = Math.min(limit, high);
  const ticks = () => [
    ...new Set(
      Array.from({ length: 5 }, (_, index) => product(low + ((high - low) * index) / 4, scale)),
    ),
  ];
  let valuesOnAxis = ticks();
  // At subnormal magnitudes fewer than two representable ticks can remain.
  if (valuesOnAxis.length < 2) {
    if (low > 0) low = 0;
    else if (high < 0) high = 0;
    else high = 1;
    valuesOnAxis = ticks();
  }
  return {
    min: product(low, scale),
    max: product(high, scale),
    scale,
    low,
    high,
    ticks: valuesOnAxis,
    empty,
  };
}

export function chartY(
  value: unknown,
  domain: StatisticsChartDomain,
  top: number,
  height: number,
): number | undefined {
  if (!finite(value) || !finite(top) || !finite(height) || height < 0 || !finite(top + height))
    return undefined;
  const unit = Math.max(domain.low, Math.min(domain.high, value / domain.scale));
  const position = (unit - domain.low) / (domain.high - domain.low);
  const y = top + (1 - position) * height;
  return finite(y) ? y : undefined;
}

export function chartNumber(value: unknown): string {
  if (!finite(value)) return '—';
  if (value === 0) return '0';
  const absolute = Math.abs(value);
  const text =
    absolute >= 100000 || absolute < 0.001 ? value.toExponential(3) : value.toPrecision(5);
  return text.replace(/(\.\d*?[1-9])0+(?=e|$)|\.0+(?=e|$)/g, '$1');
}
