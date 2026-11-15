export type HistoryWindow = '1h' | '6h' | '24h' | '7d' | '30d';

interface NetworkInterfaceRates {
  name: string;
  rx_bytes_per_second: number | null;
  tx_bytes_per_second: number | null;
}

export interface NetworkMetrics {
  default_interface: string | null;
  interfaces: NetworkInterfaceRates[];
}

export interface ResourceHistoryPoint {
  timestamp: string;
  sample_count: number;
  cpu_percent: number | null;
  memory_percent: number | null;
  disk_percent: number | null;
  network_interfaces?: NetworkInterfaceRates[];
  gpus: {
    uuid: string;
    name: string;
    utilization_pct: number | null;
    memory_percent: number | null;
  }[];
}

export interface ResourceHistoryResponse {
  node_id: string;
  window: HistoryWindow;
  start: string;
  end: string;
  sample_interval_seconds: number;
  bucket_seconds: number;
  retention_days: number;
  points: ResourceHistoryPoint[];
  network?: {
    default_interface: string | null;
    day_date: string;
    day_timezone: string;
    interfaces: {
      name: string;
      day_rx_bytes: number | null;
      day_tx_bytes: number | null;
      day_partial: boolean;
      observed_from?: string | null;
      observed_until?: string | null;
    }[];
  };
}

export const historyWindows: { value: HistoryWindow; label: string }[] = [
  { value: '1h', label: '近 1 小时' },
  { value: '6h', label: '近 6 小时' },
  { value: '24h', label: '近 24 小时' },
  { value: '7d', label: '近 7 天' },
  { value: '30d', label: '近 30 天' },
];

export function validPercent(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 100
    ? value
    : null;
}

function validCounter(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

function decimalNetworkValue(value: number | null, unit: 'GB' | 'Mbps') {
  if (value === null) return '—';
  if (value > 0 && value < 0.001) return `< 0.001 ${unit}`;
  return `${value.toLocaleString('zh-CN', { maximumFractionDigits: 3 })} ${unit}`;
}

export function formatNetworkMbps(value: number | null) {
  return decimalNetworkValue(value, 'Mbps');
}

export function networkMbps(bytesPerSecond: unknown): number | null {
  const valid = validCounter(bytesPerSecond);
  return valid === null ? null : valid / 125_000;
}

export function formatNetworkRate(bytesPerSecond: number | null | undefined) {
  return formatNetworkMbps(networkMbps(bytesPerSecond));
}

export function formatNetworkBytes(bytes: number | null | undefined) {
  const valid = validCounter(bytes);
  return decimalNetworkValue(valid === null ? null : valid / 1_000_000_000, 'GB');
}

export function networkAxisMaximum(series: ChartValue[][]) {
  const maximum = series.reduce(
    (peak, values) => values.reduce((current, point) => Math.max(current, point.value ?? 0), peak),
    0.004,
  );
  const step = 10 ** Math.floor(Math.log10(maximum / 4));
  return Math.ceil(maximum / (4 * step)) * 4 * step;
}

export function historicalGpus(points: ResourceHistoryPoint[]) {
  const byUuid = new Map<string, { uuid: string; name: string }>();
  for (const point of points) {
    for (const gpu of point.gpus) byUuid.set(gpu.uuid, { uuid: gpu.uuid, name: gpu.name });
  }
  return [...byUuid.values()];
}

export interface ChartValue {
  timestamp: string;
  value: number | null;
}

export function seriesValues(
  points: ResourceHistoryPoint[],
  read: (point: ResourceHistoryPoint) => unknown,
  parse = validPercent,
): ChartValue[] {
  return points.map((point) => ({
    timestamp: point.timestamp,
    value: point.sample_count > 0 ? parse(read(point)) : null,
  }));
}

// Keep absent observations as breaks. An offline interval must never be drawn
// as either zero usage or an uninterrupted line between the surrounding samples.
export function chartSegments(values: ChartValue[], bucketSeconds: number): ChartValue[][] {
  const segments: ChartValue[][] = [];
  let current: ChartValue[] = [];
  for (const value of values) {
    const timestamp = Date.parse(value.timestamp);
    if (value.value === null || !Number.isFinite(timestamp)) {
      current = [];
      continue;
    }
    const previous = current.at(-1);
    if (previous && timestamp - Date.parse(previous.timestamp) > bucketSeconds * 1500) current = [];
    if (!current.length) segments.push(current);
    current.push(value);
  }
  return segments;
}

export function seriesSummary(values: ChartValue[]) {
  const known = values.filter((point) => point.value !== null);
  const latest = known.at(-1);
  return {
    latest: latest?.value ?? null,
    timestamp: latest?.timestamp,
    average: known.length
      ? known.reduce((sum, point) => sum + (point.value as number), 0) / known.length
      : null,
  };
}

export function nearestPointIndex(points: ResourceHistoryPoint[], timestamp: number): number {
  if (!points.length) return -1;
  let closest = 0;
  for (let index = 1; index < points.length; index++) {
    if (
      Math.abs(Date.parse(points[index].timestamp) - timestamp) <
      Math.abs(Date.parse(points[closest].timestamp) - timestamp)
    )
      closest = index;
  }
  return closest;
}
