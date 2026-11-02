import {
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from 'react';
import {
  chartDomain,
  chartNumber,
  chartY,
  normalizationChartRows,
  type NormalizationChartFeature,
} from './datasetStatisticsChartData';

export type { NormalizationChartFeature } from './datasetStatisticsChartData';

const WIDTH = 720;
const HEIGHT = 236;
const LEFT = 76;
const TOP = 20;
const BOTTOM = HEIGHT - 40;
const PLOT_HEIGHT = BOTTOM - TOP;
const MEAN = 'var(--chart-gpu, #008f83)';
const RANGE = 'var(--chart-memory, #568cce)';
const STD = 'var(--chart-cpu, #9c78bd)';

export function DatasetNormalizationCharts({
  feature,
  start,
  count,
}: {
  feature: NormalizationChartFeature;
  start: number;
  count: number;
}) {
  const id = useId();
  const wrapper = useRef<HTMLDivElement>(null);
  const firstPlot = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(WIDTH);
  const rows = normalizationChartRows(feature, start, count);
  const hasRows = rows.length > 0;
  useLayoutEffect(() => {
    const plot = firstPlot.current;
    if (!plot) return;
    const measure = () => {
      const measured = plot.getBoundingClientRect().width;
      if (Number.isFinite(measured) && measured > 0) {
        const next = Math.max(280, Math.min(WIDTH, measured));
        setWidth((previous) => (Math.abs(previous - next) < 0.5 ? previous : next));
      }
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(plot);
    return () => observer.disconnect();
  }, [hasRows]);
  const scope = `${feature.key}:${feature.dimension}:${start}:${count}`;
  const [selection, setSelection] = useState<{
    scope: string;
    index: number;
    mode: 'pointer' | 'keyboard';
  } | null>(null);
  const selected =
    selection?.scope === scope ? rows.find((row) => row.index === selection.index) : undefined;
  const meanDomain = chartDomain(rows.flatMap((row) => [row.mean, ...(row.range || [])]));
  const stdDomain = chartDomain(
    rows.map((row) => row.std),
    true,
  );
  const right = width - 18;
  const cell = rows.length ? (right - LEFT) / rows.length : 0;
  const tickStride = Math.max(1, Math.ceil(rows.length / 6));
  const x = (offset: number) => LEFT + (offset + 0.5) * cell;
  const validRanges = rows.filter((row) => row.range).length;
  const rangeNote =
    !feature.q01 && !feature.q99
      ? '该文件未提供 q01/q99，仅显示均值。'
      : !validRanges
        ? '当前维度缺少有效的 q01/q99，仅显示均值。'
        : validRanges < rows.length
          ? '部分维度缺少有效的 q01/q99，仅绘制已有区间。'
          : '区间表示第 1%–99% 分位数，圆点表示均值；各维度独立显示。';

  const choose = (offset: number, mode: 'pointer' | 'keyboard') => {
    const row = rows[offset];
    if (row) setSelection({ scope, index: row.index, mode });
  };
  const pointer = (event: PointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (!bounds.width || !bounds.height || !rows.length) return;
    const point = ((event.clientX - bounds.left) / bounds.width) * width;
    const vertical = ((event.clientY - bounds.top) / bounds.height) * HEIGHT;
    if (point < LEFT || point > right || vertical < TOP || vertical > BOTTOM) return;
    choose(Math.min(rows.length - 1, Math.floor((point - LEFT) / cell)), 'pointer');
  };
  const keyboard = (event: KeyboardEvent<SVGSVGElement>) => {
    const offset = selected ? rows.findIndex((row) => row.index === selected.index) : -1;
    if (event.key === 'Escape') {
      event.preventDefault();
      setSelection(null);
    } else if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const next =
        event.key === 'Home'
          ? 0
          : event.key === 'End'
            ? rows.length - 1
            : event.key === 'ArrowRight'
              ? Math.min(rows.length - 1, offset + 1)
              : Math.max(0, offset - 1);
      choose(next, 'keyboard');
    }
  };

  if (!rows.length)
    return <p className="dataset-normalization-chart-empty">当前范围没有可显示的维度。</p>;

  return (
    <div
      className="dataset-normalization-charts"
      ref={wrapper}
      aria-label={`${feature.key} 逐维统计图表`}
    >
      {(['mean', 'std'] as const).map((kind) => {
        const means = kind === 'mean';
        const title = means ? '均值与分位范围图' : '标准差图';
        const domain = means ? meanDomain : stdDomain;
        const selectedOffset = selected
          ? rows.findIndex((row) => row.index === selected.index)
          : -1;
        const baseline = chartY(0, domain, TOP, PLOT_HEIGHT)!;
        return (
          <figure
            className={`dataset-normalization-chart dataset-normalization-chart-${kind}`}
            key={kind}
          >
            <figcaption>
              {means ? '均值与 1%–99% 分位范围' : '标准差 · 各维度的离散程度'}
            </figcaption>
            <svg
              ref={means ? firstPlot : undefined}
              className="dataset-normalization-chart-plot"
              viewBox={`0 0 ${width} ${HEIGHT}`}
              role="img"
              tabIndex={0}
              aria-label={`${feature.key} ${title}`}
              aria-labelledby={`${id}-${kind}-title`}
              aria-describedby={`${id}-${kind}-description`}
              onPointerMove={pointer}
              onPointerDown={(event) => {
                event.currentTarget.focus({ preventScroll: true });
                pointer(event);
              }}
              onPointerLeave={(event) => {
                if (event.pointerType !== 'touch')
                  setSelection((previous) => (previous?.mode === 'pointer' ? null : previous));
              }}
              onPointerCancel={() => setSelection(null)}
              onKeyDown={keyboard}
              onFocus={() => {
                if (selected) setSelection({ scope, index: selected.index, mode: 'keyboard' });
                else choose(0, 'keyboard');
              }}
              onBlur={(event) => {
                if (!wrapper.current?.contains(event.relatedTarget as Node | null))
                  setSelection(null);
              }}
            >
              <title id={`${id}-${kind}-title`}>{`${feature.key} ${title}`}</title>
              <desc id={`${id}-${kind}-description`}>
                维度 {rows[0].index} 至 {rows[rows.length - 1].index}。
                {means ? rangeNote : '柱高表示标准差，零值在基线标点；缺失或负标准差不绘制。'}
                使用左右方向键、Home、End 选择维度，Escape 清除；也可指向或轻触图中维度查看数值。
              </desc>
              <g aria-hidden="true">
                {domain.ticks.map((tick, tickIndex) => {
                  const y = chartY(tick, domain, TOP, PLOT_HEIGHT)!;
                  return (
                    <g key={tickIndex}>
                      <line
                        className="dataset-normalization-chart-grid"
                        x1={LEFT}
                        x2={right}
                        y1={y}
                        y2={y}
                        stroke="var(--border-default, #dce5e9)"
                        strokeDasharray="3 4"
                      />
                      <text
                        className="dataset-normalization-chart-axis"
                        x={LEFT - 8}
                        y={y + 4}
                        textAnchor="end"
                        fontSize={12}
                        fill="var(--text-muted, #758897)"
                      >
                        {chartNumber(tick)}
                      </text>
                    </g>
                  );
                })}
                <line
                  x1={LEFT}
                  x2={right}
                  y1={BOTTOM}
                  y2={BOTTOM}
                  stroke="var(--border-default, #dce5e9)"
                />
                {selectedOffset >= 0 && (
                  <line
                    className="dataset-normalization-chart-selection"
                    x1={x(selectedOffset)}
                    x2={x(selectedOffset)}
                    y1={TOP}
                    y2={BOTTOM}
                    stroke="var(--text-muted, #758897)"
                    strokeDasharray="3 3"
                  />
                )}
                {rows.map((row, offset) => {
                  const center = x(offset);
                  const active = row.index === selected?.index;
                  const meanY = chartY(row.mean, meanDomain, TOP, PLOT_HEIGHT);
                  const stdY = chartY(row.std, stdDomain, TOP, PLOT_HEIGHT);
                  const rangeTop = row.range
                    ? chartY(row.range[1], meanDomain, TOP, PLOT_HEIGHT)!
                    : undefined;
                  const rangeBottom = row.range
                    ? chartY(row.range[0], meanDomain, TOP, PLOT_HEIGHT)!
                    : undefined;
                  const barWidth = Math.min(16, cell * 0.6);
                  return (
                    <g
                      key={row.index}
                      className={active ? 'is-selected' : undefined}
                      data-dimension={row.index}
                    >
                      {means && row.range && (
                        <g
                          className="dataset-normalization-chart-range"
                          stroke={RANGE}
                          strokeWidth={active ? 3 : 2}
                        >
                          <line x1={center} x2={center} y1={rangeTop} y2={rangeBottom} />
                          <line x1={center - 4} x2={center + 4} y1={rangeTop} y2={rangeTop} />
                          <line x1={center - 4} x2={center + 4} y1={rangeBottom} y2={rangeBottom} />
                        </g>
                      )}
                      {means && meanY !== undefined && (
                        <circle
                          className="dataset-normalization-chart-mean"
                          cx={center}
                          cy={meanY}
                          r={active ? 4.5 : 3.2}
                          fill={MEAN}
                        />
                      )}
                      {!means &&
                        stdY !== undefined &&
                        (row.std === 0 ? (
                          <circle
                            className="dataset-normalization-chart-std-zero"
                            cx={center}
                            cy={baseline}
                            r={active ? 3.5 : 2.5}
                            fill={STD}
                          />
                        ) : (
                          <rect
                            className="dataset-normalization-chart-std"
                            x={center - barWidth / 2}
                            y={Math.min(stdY, baseline)}
                            width={barWidth}
                            height={Math.abs(baseline - stdY)}
                            rx={2}
                            fill={STD}
                            opacity={active ? 1 : 0.76}
                          />
                        ))}
                      {((offset % tickStride === 0 &&
                        (offset === 0 || rows.length - 1 - offset >= Math.ceil(tickStride / 2))) ||
                        offset === rows.length - 1) && (
                        <text
                          className="dataset-normalization-chart-axis"
                          x={center}
                          y={BOTTOM + 18}
                          textAnchor="middle"
                          fontSize={12}
                          fill="var(--text-muted, #758897)"
                        >
                          {row.index}
                        </text>
                      )}
                    </g>
                  );
                })}
                <text
                  className="dataset-normalization-chart-axis"
                  x={right}
                  y={HEIGHT - 3}
                  textAnchor="end"
                  fontSize={12}
                  fill="var(--text-muted, #758897)"
                >
                  维度编号
                </text>
                {domain.empty && (
                  <text
                    className="dataset-normalization-chart-empty"
                    x={(LEFT + right) / 2}
                    y={(TOP + BOTTOM) / 2}
                    textAnchor="middle"
                    fontSize={12}
                    fill="var(--text-muted, #758897)"
                  >
                    当前维度没有可绘制的有限数值
                  </text>
                )}
              </g>
            </svg>
            <div className="dataset-normalization-chart-legend">
              {means ? (
                <>
                  <span style={{ color: MEAN }}>● 均值</span>
                  <span style={{ color: RANGE }}>┃ q01–q99</span>
                </>
              ) : (
                <span style={{ color: STD }}>■ 标准差（std）</span>
              )}
            </div>
            <p className="dataset-normalization-chart-note">
              {means ? rangeNote : '数值越大，离散程度越高；零标准差仍显示标点。'}
            </p>
          </figure>
        );
      })}
      <div
        className="dataset-normalization-chart-tooltip"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {selected ? (
          <>
            <strong>维度 {selected.index}</strong>
            <span>均值 mean：{selected.mean === undefined ? '—' : String(selected.mean)}</span>
            <span>标准差 std：{selected.std === undefined ? '—' : String(selected.std)}</span>
            <span>q01：{selected.q01 === undefined ? '—' : String(selected.q01)}</span>
            <span>q99：{selected.q99 === undefined ? '—' : String(selected.q99)}</span>
            {selected.min !== undefined && <span>min：{String(selected.min)}</span>}
            {selected.max !== undefined && <span>max：{String(selected.max)}</span>}
          </>
        ) : (
          '指向、轻触图中维度，或聚焦图表后使用左右键查看具体数值。'
        )}
      </div>
    </div>
  );
}
