import { useId, useState } from 'react';
import { Activity, RefreshCw } from 'lucide-react';
import { date, useApi } from './api';
import { Empty, ErrorNotice, Loading, Panel } from './components';
import {
  chartSegments,
  historicalGpus,
  historyWindows,
  nearestPointIndex,
  seriesSummary,
  seriesValues,
} from './resourceHistoryData';
import type { ChartValue, HistoryWindow, ResourceHistoryResponse } from './resourceHistoryData';
import './resource-history.css';

const percent = (value: number | null) => (value === null ? '—' : `${value.toFixed(1)}%`);
const palette = {
  cpu: 'var(--chart-cpu, #0b8c87)',
  memory: 'var(--chart-memory, #5578d7)',
  disk: 'var(--chart-disk, #bc862f)',
  gpu: 'var(--chart-gpu, #0b8c87)',
  vram: 'var(--chart-ram, #8b62c4)',
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
}: {
  title: string;
  history: ResourceHistoryResponse;
  series: ChartSeries[];
}) {
  const labelId = useId();
  const [selected, setSelected] = useState<number | null>(null);
  const start = Date.parse(history.start);
  const duration = Math.max(1, Date.parse(history.end) - start);
  const left = 43,
    top = 15,
    width = 730,
    height = 152;
  const x = (timestamp: string) =>
    left + Math.max(0, Math.min(1, (Date.parse(timestamp) - start) / duration)) * width;
  const y = (value: number) => top + height * (1 - value / 100);
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
            viewBox="0 0 800 205"
            role="img"
            tabIndex={0}
            aria-labelledby={`${labelId}-title ${labelId}-desc`}
            onPointerMove={(event) => {
              const bounds = event.currentTarget.getBoundingClientRect();
              const fraction = Math.max(
                0,
                Math.min(1, (((event.clientX - bounds.left) / bounds.width) * 800 - left) / width),
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
            <title id={`${labelId}-title`}>{`${title}，百分比趋势图`}</title>
            <desc id={`${labelId}-desc`}>
              纵轴 0 至 100%。无有效指标的聚合时段保留断点。使用左右方向键逐点查看，Home 和 End
              跳转首尾。
            </desc>
            {[0, 25, 50, 75, 100].map((value) => (
              <g key={value}>
                <line
                  x1={left}
                  x2={left + width}
                  y1={y(value)}
                  y2={y(value)}
                  className="resource-chart-grid"
                />
                <text x={left - 9} y={y(value) + 4} textAnchor="end">
                  {value}%
                </text>
              </g>
            ))}
            {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
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
                    <strong>{percent(item.values[selected as number]?.value ?? null)}</strong>
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

function HistoryContent({
  nodeId,
  window,
  status,
  gpuId,
  selectGpu,
}: {
  nodeId: string;
  window: HistoryWindow;
  status: string;
  gpuId: string;
  selectGpu: (uuid: string) => void;
}) {
  const history = useApi<ResourceHistoryResponse>(
    `/nodes/${encodeURIComponent(nodeId)}/resource-history?window=${window}`,
    30000,
  );
  const data =
    history.data?.node_id === nodeId && history.data.window === window ? history.data : undefined;
  const gpus = historicalGpus(data?.points || []);
  const selectedGpu = gpus.find((gpu) => gpu.uuid === gpuId) || gpus[0];
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
          {status === 'ONLINE' ? '每 30 秒自动刷新' : '机器当前未在线，已有历史仍可查看'}
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
      ) : !data.points.length ? (
        <Empty
          icon={Activity}
          title="此时段暂无资源历史"
          description="历史记录从功能启用后的 Agent 心跳开始积累；更早的数据无法补录。机器在线后可稍后刷新，或切换时间范围。"
        />
      ) : (
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
          <div className="resource-history-gpu-heading">
            <h3>GPU 资源占用</h3>
            {!!gpus.length && (
              <label>
                选择 GPU
                <select
                  aria-label="历史指标 GPU"
                  value={selectedGpu?.uuid || ''}
                  onChange={(event) => selectGpu(event.target.value)}
                >
                  {gpus.map((gpu) => (
                    <option key={gpu.uuid} value={gpu.uuid}>
                      {gpu.name} · {gpu.uuid}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          {selectedGpu ? (
            <ResourceTrendChart
              key={selectedGpu.uuid}
              title={`${selectedGpu.name} · ${selectedGpu.uuid}`}
              history={data}
              series={[
                {
                  label: 'GPU 利用率',
                  color: palette.gpu,
                  values: seriesValues(
                    data.points,
                    (point) =>
                      point.gpus.find((gpu) => gpu.uuid === selectedGpu.uuid)?.utilization_pct,
                  ),
                },
                {
                  label: '显存占用',
                  color: palette.vram,
                  values: seriesValues(
                    data.points,
                    (point) =>
                      point.gpus.find((gpu) => gpu.uuid === selectedGpu.uuid)?.memory_percent,
                  ),
                },
              ]}
            />
          ) : (
            <div className="resource-chart-unavailable">
              此时段暂无 GPU 历史指标，未知值保留为空。
            </div>
          )}
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

export function ResourceHistory({ nodeId, status }: { nodeId: string; status: string }) {
  const [window, setWindow] = useState<HistoryWindow>('24h');
  const [gpuId, setGpuId] = useState('');
  return (
    <Panel
      className="resource-history"
      title="资源占用历史"
      subtitle="查看 CPU、内存、磁盘与各张 GPU 的使用趋势"
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
        status={status}
        gpuId={gpuId}
        selectGpu={setGpuId}
      />
    </Panel>
  );
}
