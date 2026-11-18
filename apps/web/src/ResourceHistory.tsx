import { useId, useState } from 'react';
import { Activity, RefreshCw } from 'lucide-react';
import { date, useApi } from './api';
import { Empty, ErrorNotice, Loading, Panel } from './components';
import {
  chartSegments,
  formatNetworkBytes,
  formatNetworkMbps,
  formatNetworkRate,
  formatTrafficBytes,
  historicalGpus,
  historyWindows,
  nearestPointIndex,
  networkAxisMaximum,
  networkMbps,
  seriesSummary,
  seriesValues,
} from './resourceHistoryData';
import type {
  ChartValue,
  HistoryWindow,
  NetworkProcessReport,
  ResourceHistoryResponse,
} from './resourceHistoryData';
import './resource-history.css';

const percent = (value: number | null) => (value === null ? '—' : `${value.toFixed(1)}%`);
const palette = {
  cpu: 'var(--chart-cpu, #0b8c87)',
  memory: 'var(--chart-memory, #5578d7)',
  disk: 'var(--chart-disk, #bc862f)',
  gpu: 'var(--chart-gpu, #0b8c87)',
  vram: 'var(--chart-ram, #8b62c4)',
  receive: 'var(--chart-memory, #5578d7)',
  transmit: 'var(--chart-cpu, #0b8c87)',
};

interface ChartSeries {
  label: string;
  color: string;
  values: ChartValue[];
}

function axisTime(timestamp: number, duration: number) {
  return new Date(timestamp).toLocaleString('zh-CN', {
    ...(duration >= 86400000 ? ({ month: '2-digit', day: '2-digit' } as const) : {}),
    hour: '2-digit',
    minute: '2-digit',
  });
}

function bucketTime(timestamp: string, bucketSeconds: number) {
  const format = (time: number) =>
    new Date(time).toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      ...(bucketSeconds < 60 ? ({ second: '2-digit' } as const) : {}),
    });
  const start = Date.parse(timestamp);
  return `${format(start)} – ${format(start + bucketSeconds * 1000)}`;
}

export function ResourceTrendChart({
  title,
  history,
  series,
  compact = false,
  unit = '%',
}: {
  title: string;
  history: ResourceHistoryResponse;
  series: ChartSeries[];
  compact?: boolean;
  unit?: '%' | 'Mbps';
}) {
  const labelId = useId();
  const [selected, setSelected] = useState<number | null>(null);
  const start = Date.parse(history.start);
  const duration = Math.max(1, Date.parse(history.end) - start);
  const canvasWidth = compact ? 560 : 800;
  const percentAxis = unit === '%';
  const maximum = percentAxis ? 100 : networkAxisMaximum(series.map((item) => item.values));
  const labelValue = percentAxis ? percent : formatNetworkMbps;
  const left = percentAxis ? 43 : 62,
    top = 15,
    width = canvasWidth - left - 27,
    height = 152;
  const x = (timestamp: string) =>
    left + Math.max(0, Math.min(1, (Date.parse(timestamp) - start) / duration)) * width;
  const y = (value: number) => top + height * (1 - value / maximum);
  const point = selected === null ? undefined : history.points[selected];
  const known = series.some((item) => item.values.some((value) => value.value !== null));
  return (
    <div className="resource-trend-chart">
      <div className="resource-chart-title">
        <h3>{title}</h3>
        <div className="resource-chart-legend">
          {series.map((item) => (
            <span key={item.label}>
              <i style={{ background: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      </div>
      {known ? (
        <>
          <svg
            viewBox={`0 0 ${canvasWidth} 205`}
            role="img"
            tabIndex={0}
            aria-labelledby={`${labelId}-title ${labelId}-desc`}
            onPointerMove={(event) => {
              const bounds = event.currentTarget.getBoundingClientRect();
              const fraction = Math.max(
                0,
                Math.min(
                  1,
                  (((event.clientX - bounds.left) / bounds.width) * canvasWidth - left) / width,
                ),
              );
              setSelected(nearestPointIndex(history.points, start + fraction * duration));
            }}
            onPointerLeave={() => setSelected(null)}
            onFocus={() => setSelected(history.points.length - 1)}
            onBlur={() => setSelected(null)}
            onKeyDown={(event) => {
              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
              event.preventDefault();
              setSelected((current) =>
                event.key === 'Home'
                  ? 0
                  : event.key === 'End'
                    ? history.points.length - 1
                    : Math.max(
                        0,
                        Math.min(
                          history.points.length - 1,
                          (current ?? history.points.length - 1) +
                            (event.key === 'ArrowLeft' ? -1 : 1),
                        ),
                      ),
              );
            }}
          >
            <title id={`${labelId}-title`}>
              {`${title}，${percentAxis ? '百分比' : 'Mbps'}趋势图`}
            </title>
            <desc id={`${labelId}-desc`}>
              {percentAxis ? '纵轴 0 至 100%。' : '纵轴单位为 Mbps。'}
              无有效指标的聚合时段保留断点。使用左右方向键逐点查看，Home 和 End 跳转首尾。
            </desc>
            {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
              <g key={fraction}>
                <line
                  x1={left}
                  x2={left + width}
                  y1={y(maximum * fraction)}
                  y2={y(maximum * fraction)}
                  className="resource-chart-grid"
                />
                <text x={left - 9} y={y(maximum * fraction) + 4} textAnchor="end">
                  {percentAxis
                    ? `${maximum * fraction}%`
                    : (maximum * fraction).toLocaleString('zh-CN', { maximumFractionDigits: 3 })}
                </text>
              </g>
            ))}
            {(compact ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map((fraction) => (
              <text
                key={fraction}
                x={left + width * fraction}
                y={194}
                textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'}
              >
                {axisTime(start + duration * fraction, duration)}
              </text>
            ))}
            {series.flatMap((item) =>
              chartSegments(item.values, history.bucket_seconds).map((segment, index) => (
                <g key={`${item.label}-${index}`}>
                  {segment.length > 1 ? (
                    <path
                      d={segment
                        .map(
                          (value, i) =>
                            `${i ? 'L' : 'M'}${x(value.timestamp).toFixed(2)},${y(value.value as number).toFixed(2)}`,
                        )
                        .join(' ')}
                      fill="none"
                      stroke={item.color}
                      strokeWidth="2"
                      strokeLinejoin="round"
                    />
                  ) : (
                    <circle
                      cx={x(segment[0].timestamp)}
                      cy={y(segment[0].value as number)}
                      r="3"
                      fill={item.color}
                    />
                  )}
                </g>
              )),
            )}
            {point && (
              <g>
                <line
                  x1={x(point.timestamp)}
                  x2={x(point.timestamp)}
                  y1={top}
                  y2={top + height}
                  className="resource-chart-cursor"
                />
                {series.map((item) => {
                  const value = item.values[selected as number]?.value;
                  return value == null ? null : (
                    <circle
                      key={item.label}
                      cx={x(point.timestamp)}
                      cy={y(value)}
                      r="4"
                      fill={item.color}
                      stroke="var(--bg-surface, white)"
                      strokeWidth="2"
                    />
                  );
                })}
              </g>
            )}
          </svg>
          <div className="resource-chart-tooltip" aria-live="polite">
            {point ? (
              <>
                <time dateTime={point.timestamp}>
                  {bucketTime(point.timestamp, history.bucket_seconds)}
                </time>
                {series.map((item) => (
                  <span key={item.label}>
                    <i style={{ background: item.color }} />
                    {item.label}{' '}
                    <strong>{labelValue(item.values[selected as number]?.value ?? null)}</strong>
                  </span>
                ))}
                {!point.sample_count && <span>此时段无采样</span>}
              </>
            ) : (
              <span className="resource-chart-help">移动指针或聚焦后使用方向键查看数值</span>
            )}
          </div>
        </>
      ) : (
        <div className="resource-chart-unavailable">该时段未采集到可用指标</div>
      )}
    </div>
  );
}

export function GpuHistoryCharts({
  history,
  gpuIds,
  selectGpus,
}: {
  history: ResourceHistoryResponse;
  gpuIds: string[] | null;
  selectGpus: (uuids: string[] | null) => void;
}) {
  const gpus = historicalGpus(history.points);
  const selected = gpus.filter((gpu) => gpuIds === null || gpuIds.includes(gpu.uuid));
  return (
    <>
      <div className="resource-history-gpu-heading">
        <h3>GPU 资源占用</h3>
        {!!gpus.length && (
          <div className="resource-history-gpu-actions">
            <span>
              已选 {selected.length} / {gpus.length} 张
            </span>
            <button type="button" className="text-link" onClick={() => selectGpus(null)}>
              全选
            </button>
            <button type="button" className="text-link" onClick={() => selectGpus([])}>
              清空
            </button>
          </div>
        )}
      </div>
      {!!gpus.length && (
        <fieldset className="resource-history-gpu-picker">
          <legend>选择 GPU（可多选）</legend>
          <div className="resource-history-gpu-options">
            {gpus.map((gpu) => {
              const checked = gpuIds === null || gpuIds.includes(gpu.uuid);
              return (
                <label key={gpu.uuid} className={checked ? 'selected' : ''}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => {
                      const current = gpuIds ?? gpus.map((item) => item.uuid);
                      selectGpus(
                        checked
                          ? current.filter((uuid) => uuid !== gpu.uuid)
                          : [...current, gpu.uuid],
                      );
                    }}
                  />
                  <span>
                    <strong>{gpu.name}</strong>
                    <code>{gpu.uuid}</code>
                  </span>
                </label>
              );
            })}
          </div>
        </fieldset>
      )}
      {selected.length ? (
        <div className={`resource-history-gpu-grid${selected.length === 1 ? ' single' : ''}`}>
          {selected.map((gpu) => (
            <ResourceTrendChart
              key={gpu.uuid}
              title={`${gpu.name} · ${gpu.uuid}`}
              history={history}
              compact={selected.length > 1}
              series={[
                {
                  label: 'GPU 利用率',
                  color: palette.gpu,
                  values: seriesValues(
                    history.points,
                    (point) => point.gpus.find((item) => item.uuid === gpu.uuid)?.utilization_pct,
                  ),
                },
                {
                  label: '显存占用',
                  color: palette.vram,
                  values: seriesValues(
                    history.points,
                    (point) => point.gpus.find((item) => item.uuid === gpu.uuid)?.memory_percent,
                  ),
                },
              ]}
            />
          ))}
        </div>
      ) : (
        <div className="resource-chart-unavailable">
          {gpus.length
            ? '未选择 GPU，请勾选要查看的显卡或点击全选。'
            : '此时段暂无 GPU 历史指标，未知值保留为空。'}
        </div>
      )}
    </>
  );
}

export function NetworkHistoryCharts({
  history,
  interfaceName,
  selectInterface,
}: {
  history: ResourceHistoryResponse;
  interfaceName: string | null;
  selectInterface: (name: string) => void;
}) {
  const network = history.network;
  const names = new Set(network?.interfaces.map((item) => item.name));
  for (const point of history.points) {
    for (const item of point.network_interfaces ?? []) names.add(item.name);
  }
  if (interfaceName !== null) names.add(interfaceName);
  const interfaces = [...names];
  const defaultInterface = network?.default_interface;
  const selected =
    interfaceName ??
    (defaultInterface && names.has(defaultInterface) ? defaultInterface : interfaces[0]);
  const today = network?.interfaces.find((item) => item.name === selected);
  const observedUntil = Date.parse(today?.observed_until ?? '');
  return (
    <section className="resource-network-history" aria-label="网卡流量">
      <div className="resource-network-heading">
        <h3>网卡流量</h3>
        {selected && (
          <label>
            网卡
            <select value={selected} onChange={(event) => selectInterface(event.target.value)}>
              {interfaces.map((name) => (
                <option value={name} key={name}>
                  {name === defaultInterface ? `${name}（默认出口）` : name}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {selected ? (
        <>
          <div className="resource-history-summaries resource-network-summaries">
            <div>
              <span>
                <i style={{ background: palette.receive }} />
                今日入站
              </span>
              <strong>{formatNetworkBytes(today?.day_rx_bytes)}</strong>
            </div>
            <div>
              <span>
                <i style={{ background: palette.transmit }} />
                今日出站
              </span>
              <strong>{formatNetworkBytes(today?.day_tx_bytes)}</strong>
            </div>
          </div>
          <p className="resource-network-note">
            {network
              ? `${network.day_date} ${network.day_timezone === 'Asia/Shanghai' ? '北京时间' : network.day_timezone}。 `
              : ''}
            {Number.isFinite(observedUntil)
              ? `统计截至 ${new Date(observedUntil).toLocaleTimeString('zh-CN', {
                  timeZone: 'Asia/Shanghai',
                  hour12: false,
                })}。 `
              : ''}
            {today?.day_partial ? '今日采样覆盖不完整，累计值仅包含已观测流量。 ' : ''}
            数值来自所选网卡，公网计费用量以云厂商记录为准。
          </p>
          <ResourceTrendChart
            key={selected}
            title="收发速率（Mbps）"
            history={history}
            unit="Mbps"
            series={[
              {
                label: '入站',
                color: palette.receive,
                values: seriesValues(
                  history.points,
                  (point) =>
                    point.network_interfaces?.find((item) => item.name === selected)
                      ?.rx_bytes_per_second,
                  networkMbps,
                ),
              },
              {
                label: '出站',
                color: palette.transmit,
                values: seriesValues(
                  history.points,
                  (point) =>
                    point.network_interfaces?.find((item) => item.name === selected)
                      ?.tx_bytes_per_second,
                  networkMbps,
                ),
              },
            ]}
          />
        </>
      ) : (
        <div className="resource-chart-unavailable">
          暂无网卡流量数据，等待支持流量采集的 Agent 上报。
        </div>
      )}
    </section>
  );
}

type NetworkProcessTrafficProps = {
  report?: NetworkProcessReport;
  lastSeen?: string | null;
  nodeStatus: string;
  receivedAt?: Date;
  maxAgeMs?: number;
};

const processTrafficMessages = {
  ready: '本次采样暂无可归属到进程的 TCP 流量。',
  warming_up: '正在建立采样基线，等待下一次上报。',
  unavailable: 'Agent 暂时无法采集进程流量。',
  stale: '暂无近期的进程流量数据。',
};

const processTrafficViews = [
  { id: 'today', label: '今日累计' },
  { id: 'realtime', label: '实时速率' },
] as const;

export function NetworkProcessTraffic({
  report,
  lastSeen,
  nodeStatus,
  receivedAt,
  maxAgeMs = 30_000,
}: NetworkProcessTrafficProps) {
  const [selection, setSelection] = useState<(typeof processTrafficViews)[number]['id']>();
  const today = report?.today;
  const cumulative = (selection ?? (report && !today ? 'realtime' : 'today')) === 'today';
  const active = nodeStatus === 'ONLINE' || nodeStatus === 'RECOVERING';
  const status =
    !active || (receivedAt && Date.now() - receivedAt.getTime() > maxAgeMs)
      ? 'stale'
      : report?.status;
  const rows = cumulative
    ? (today?.processes ?? []).map((row) => ({
        key: row.instance_id,
        name: row.name,
        pid: row.pid,
        rx: formatTrafficBytes(row.rx_bytes),
        tx: formatTrafficBytes(row.tx_bytes),
        total: formatTrafficBytes(row.rx_bytes + row.tx_bytes),
        exited: row.active === false,
      }))
    : (status === 'ready' ? (report?.processes ?? []) : []).map((row) => ({
        key: row.pid,
        name: row.name,
        pid: row.pid,
        rx: formatNetworkRate(row.rx_bytes_per_second),
        tx: formatNetworkRate(row.tx_bytes_per_second),
        total: null,
        exited: false,
      }));
  const statusMessage = status ? processTrafficMessages[status] : '这台机器尚未上报进程流量。';
  const emptyMessage = cumulative
    ? today
      ? '当天暂无可归属到进程的累计 TCP 流量。'
      : '这台机器尚未上报今日累计流量。'
    : statusMessage;

  return (
    <section
      className="resource-network-history resource-network-processes"
      aria-label="进程网络流量"
    >
      <div className="resource-network-heading">
        <h3>进程网络流量（TCP）</h3>
        <div className="resource-history-windows" role="group" aria-label="进程流量视图">
          {processTrafficViews.map(({ id, label }) => {
            const selected = id === (cumulative ? 'today' : 'realtime');
            return (
              <button
                type="button"
                key={id}
                aria-pressed={selected}
                className={selected ? 'selected' : ''}
                onClick={() => setSelection(id)}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>
      {cumulative && today && <p className="resource-process-timestamp">北京时间 {today.date}</p>}
      {!cumulative && status === 'ready' && (
        <p className="resource-process-timestamp">最近上报 {date(lastSeen)}</p>
      )}
      {rows.length ? (
        <div className="table-scroll">
          <table aria-label={cumulative ? '进程 TCP 累计流量排行榜' : '进程 TCP 收发速率排行榜'}>
            <thead>
              <tr>
                <th scope="col">进程</th>
                <th scope="col">PID</th>
                <th scope="col">{cumulative ? '累计入站' : '入站'}</th>
                <th scope="col">{cumulative ? '累计出站' : '出站'}</th>
                {cumulative && <th scope="col">合计</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key}>
                  <td>
                    <strong>{row.name}</strong>
                    {row.exited && <span className="resource-process-exited">已退出</span>}
                  </td>
                  <td className="mono">{row.pid}</td>
                  <td>{row.rx}</td>
                  <td>{row.tx}</td>
                  {cumulative && (
                    <td>
                      <strong>{row.total}</strong>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="resource-chart-unavailable">{emptyMessage}</div>
      )}
      <p className="resource-network-note">
        {cumulative ? '按当天累计收发量排序。已退出的进程仍保留当天记录。' : '按当前收发速率排序。'}
        统计整机可归属的 TCP 连接。UDP 和短连接可通过网卡总量观察。
        {cumulative &&
          today &&
          (today.partial || status === 'stale') &&
          '累计仅包含已观测流量，采集覆盖可能不完整。'}
        {!cumulative && status === 'ready' && report?.partial && '部分连接的归属无法确认。'}
      </p>
    </section>
  );
}

function HistoryContent({
  nodeId,
  window,
  refreshIntervalMs,
  status,
  gpuIds,
  selectGpus,
  interfaceName,
  selectInterface,
  networkProcesses,
}: {
  nodeId: string;
  window: HistoryWindow;
  refreshIntervalMs: number;
  status: string;
  gpuIds: string[] | null;
  selectGpus: (uuids: string[] | null) => void;
  interfaceName: string | null;
  selectInterface: (name: string) => void;
  networkProcesses: NetworkProcessTrafficProps;
}) {
  const history = useApi<ResourceHistoryResponse>(
    `/nodes/${encodeURIComponent(nodeId)}/resource-history?window=${window}`,
    refreshIntervalMs,
  );
  const data =
    history.data?.node_id === nodeId && history.data.window === window ? history.data : undefined;
  const systemSeries = data
    ? [
        {
          label: 'CPU',
          color: palette.cpu,
          values: seriesValues(data.points, (point) => point.cpu_percent),
        },
        {
          label: '内存',
          color: palette.memory,
          values: seriesValues(data.points, (point) => point.memory_percent),
        },
        {
          label: '产出目录磁盘',
          color: palette.disk,
          values: seriesValues(data.points, (point) => point.disk_percent),
        },
      ]
    : [];
  return (
    <div className="resource-history-content">
      <div className="resource-history-status">
        <span>
          <span className={`status-dot ${status === 'ONLINE' ? 'online' : ''}`} />
          {status === 'ONLINE'
            ? `每 ${refreshIntervalMs / 1000} 秒自动刷新`
            : '机器当前未在线，已有历史仍可查看'}
        </span>
        <button
          type="button"
          className="text-link"
          onClick={history.refresh}
          disabled={history.loading}
        >
          <RefreshCw size={13} className={history.loading ? 'spin' : ''} />
          刷新
        </button>
      </div>
      <ErrorNotice
        error={history.error}
        retry={history.refresh}
        title="资源历史读取失败"
        hint={
          data ? `当前展示上次读取的数据（${date(history.updatedAt?.toISOString())}）。` : undefined
        }
      />
      {!data ? (
        history.loading && <Loading label="正在读取资源历史…" />
      ) : (
        <>
          {data.points.length ? (
            <>
              <div className="resource-history-summaries">
                {systemSeries.map((item) => {
                  const summary = seriesSummary(item.values);
                  return (
                    <div key={item.label}>
                      <span>
                        <i style={{ background: item.color }} />
                        {item.label}
                      </span>
                      <strong>{percent(summary.latest)}</strong>
                      <small>
                        最近有效时段 {summary.timestamp ? date(summary.timestamp) : '—'} · 时段均值{' '}
                        {percent(summary.average)}
                      </small>
                    </div>
                  );
                })}
              </div>
              <ResourceTrendChart title="整机资源占用" history={data} series={systemSeries} />
            </>
          ) : (
            <Empty
              icon={Activity}
              title="此时段暂无资源历史"
              description="历史记录从功能启用后的 Agent 心跳开始积累；更早的数据无法补录。机器在线后可稍后刷新，或切换时间范围。"
            />
          )}
          <NetworkHistoryCharts
            history={data}
            interfaceName={interfaceName}
            selectInterface={selectInterface}
          />
        </>
      )}
      <NetworkProcessTraffic {...networkProcesses} />
      {!!data?.points.length && (
        <>
          <GpuHistoryCharts history={data} gpuIds={gpuIds} selectGpus={selectGpus} />
          <p className="resource-history-note">
            保存近 {data.retention_days} 天记录，最多每 {data.sample_interval_seconds} 秒采样一次。
            当前按{' '}
            {data.bucket_seconds < 60
              ? `${data.bucket_seconds} 秒`
              : `${data.bucket_seconds / 60} 分钟`}
            聚合均值；时段均值为图中有效聚合点的平均值。
            无有效指标的聚合时段显示断点；时段内短暂中断可能被聚合。磁盘对应产出目录所在文件系统。时间按浏览器本地时区显示。
          </p>
        </>
      )}
    </div>
  );
}

export function ResourceHistory({
  nodeId,
  status,
  historyRefreshMs = 30_000,
  processReport,
  lastSeen,
  processReportReceivedAt,
  processReportMaxAgeMs,
}: {
  nodeId: string;
  status: string;
  historyRefreshMs?: number;
  processReport?: NetworkProcessReport;
  lastSeen?: string | null;
  processReportReceivedAt?: Date;
  processReportMaxAgeMs?: number;
}) {
  const [window, setWindow] = useState<HistoryWindow>('24h');
  const [gpuSelection, setGpuSelection] = useState<{
    nodeId: string;
    uuids: string[] | null;
  } | null>(null);
  const [networkSelection, setNetworkSelection] = useState<{ nodeId: string; name: string } | null>(
    null,
  );
  return (
    <Panel
      className="resource-history"
      title="资源占用历史"
      subtitle="查看 CPU、内存、磁盘、网卡流量与各张 GPU 的使用趋势"
      action={
        <div className="resource-history-windows" role="group" aria-label="资源历史时间范围">
          {historyWindows.map((item) => (
            <button
              type="button"
              key={item.value}
              aria-pressed={window === item.value}
              className={window === item.value ? 'selected' : ''}
              onClick={() => setWindow(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
      }
    >
      <HistoryContent
        key={`${nodeId}-${window}`}
        nodeId={nodeId}
        window={window}
        refreshIntervalMs={historyRefreshMs}
        status={status}
        gpuIds={gpuSelection?.nodeId === nodeId ? gpuSelection.uuids : null}
        selectGpus={(uuids) => setGpuSelection({ nodeId, uuids })}
        interfaceName={networkSelection?.nodeId === nodeId ? networkSelection.name : null}
        selectInterface={(name) => setNetworkSelection({ nodeId, name })}
        networkProcesses={{
          report: processReport,
          lastSeen,
          nodeStatus: status,
          receivedAt: processReportReceivedAt,
          maxAgeMs: processReportMaxAgeMs,
        }}
      />
    </Panel>
  );
}
