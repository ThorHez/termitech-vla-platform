import type { JsonObject, RunEvent } from './types';

const metadataFields = new Set(['step', 'total_steps', 'timestamp']);

export interface TrainingMetricPoint {
  step: number;
  value: number | null;
  timestamp?: string;
}

export interface TrainingMetricSeries {
  name: string;
  points: TrainingMetricPoint[];
  sampleCount: number;
  latest: {
    value: number | null;
    step: number | null;
    timestamp?: string;
    source: 'event' | 'snapshot';
  };
}

export function finiteMetric(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function formatMetricValue(value: unknown): string {
  const number = finiteMetric(value);
  if (number === null) return '—';
  if (number === 0) return '0';
  if (Math.abs(number) < 0.0001 || Math.abs(number) >= 1000000) {
    const [mantissa, exponent] = number.toExponential(5).split('e');
    return `${Number(mantissa)}e${Number(exponent) >= 0 ? '+' : ''}${Number(exponent)}`;
  }
  return number.toLocaleString('zh-CN', { maximumSignificantDigits: 6 });
}

export function orderedMetricEvents(events: RunEvent[]): RunEvent[] {
  return events
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => event.type === 'METRIC')
    .sort((left, right) => left.event.seq - right.event.seq || left.index - right.index)
    .map(({ event }) => event);
}

export function metricNames(events: RunEvent[], latestMetrics: JsonObject = {}): string[] {
  const names = new Set<string>();
  for (const values of [
    latestMetrics,
    ...events.filter((event) => event.type === 'METRIC').map((event) => event.data),
  ]) {
    for (const [name, value] of Object.entries(values)) {
      if (!metadataFields.has(name) && finiteMetric(value) !== null) names.add(name);
    }
  }
  return [...names].sort((left, right) =>
    left === 'loss' ? -1 : right === 'loss' ? 1 : left.localeCompare(right, 'en'),
  );
}

export function trainingMetricSeries(
  events: RunEvent[],
  latestMetrics: JsonObject = {},
): TrainingMetricSeries[] {
  const ordered = orderedMetricEvents(events);
  return metricNames(ordered, latestMetrics).map((name) => {
    const points = new Map<number, TrainingMetricPoint>();
    let sampleCount = 0;
    let latest: TrainingMetricSeries['latest'] = {
      value: finiteMetric(latestMetrics[name]),
      step: null,
      source: 'snapshot',
    };
    for (const event of ordered) {
      // Different metrics can have different reporting cadence. An omitted key
      // is not an observation; an explicitly invalid value is a visible gap.
      if (!Object.prototype.hasOwnProperty.call(event.data, name)) continue;
      const value = finiteMetric(event.data[name]);
      const step = finiteMetric(event.data.step);
      if (value !== null) sampleCount += 1;
      latest = { value, step, timestamp: event.timestamp, source: 'event' };
      if (step !== null) points.set(step, { step, value, timestamp: event.timestamp });
    }
    const snapshotTimestamp =
      typeof latestMetrics.timestamp === 'string' ? Date.parse(latestMetrics.timestamp) : NaN;
    const observedTimestamp = latest.timestamp ? Date.parse(latest.timestamp) : NaN;
    const snapshotValue = finiteMetric(latestMetrics[name]);
    // The merged snapshot can lead a paged event feed. A newer changed value
    // is useful, but its shared timestamp/step cannot identify this metric's
    // actual observation. Equal values retain the known observation metadata.
    if (
      latest.source === 'event' &&
      Object.prototype.hasOwnProperty.call(latestMetrics, name) &&
      Number.isFinite(snapshotTimestamp) &&
      Number.isFinite(observedTimestamp) &&
      snapshotTimestamp > observedTimestamp &&
      snapshotValue !== latest.value
    ) {
      latest = { value: snapshotValue, step: null, source: 'snapshot' };
    }
    return {
      name,
      points: [...points.values()].sort((left, right) => left.step - right.step),
      sampleCount,
      latest,
    };
  });
}

export function metricProgress(
  events: RunEvent[],
  latestMetrics: JsonObject = {},
  totalSteps?: unknown,
) {
  const ordered = orderedMetricEvents(events);
  const lastStep = [...ordered].reverse().find((event) => finiteMetric(event.data.step) !== null);
  const lastTotal = [...ordered]
    .reverse()
    .find((event) => finiteMetric(event.data.total_steps) !== null);
  return {
    step: finiteMetric(latestMetrics.step) ?? finiteMetric(lastStep?.data.step),
    total:
      finiteMetric(latestMetrics.total_steps) ??
      finiteMetric(lastTotal?.data.total_steps) ??
      finiteMetric(totalSteps),
    reportCount: ordered.length,
  };
}
