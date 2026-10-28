import { useMemo, useState } from 'react';
import { Activity } from 'lucide-react';
import { date } from './api';
import { Empty, ErrorNotice, Loading, MetricChart } from './components';
import { formatMetricValue, metricProgress, trainingMetricSeries } from './trainingMetricData';
import type { JsonObject, RunEvent } from './types';
import './training-metrics.css';

export interface TrainingMetricsProps {
  events: RunEvent[];
  latestMetrics?: JsonObject;
  totalSteps?: unknown;
  loading?: boolean;
  error?: Error;
  retry?: () => void;
  historyLimited?: boolean;
}

export function TrainingMetrics({
  events,
  latestMetrics,
  totalSteps,
  loading,
  error,
  retry,
  historyLimited,
}: TrainingMetricsProps) {
  const series = useMemo(
    () => trainingMetricSeries(events, latestMetrics),
    [events, latestMetrics],
  );
  const progress = useMemo(
    () => metricProgress(events, latestMetrics, totalSteps),
    [events, latestMetrics, totalSteps],
  );
  // Null means the initial automatic selection. An empty array is an explicit
  // request to clear the display and must survive background metric refreshes.
  const [selection, setSelection] = useState<string[] | null>(null);
  const selected = selection === null ? series.slice(0, 1).map((metric) => metric.name) : selection;
  const visible = series.filter((metric) => selected.includes(metric.name));
  return (
    <div className="training-metric-view">
      <ErrorNotice error={error} retry={retry} title="训练指标读取失败" />
      <div className="training-metric-summary">
        <div>
          <span>当前步数</span>
          <strong>
            {formatMetricValue(progress.step)}
            <small> / {formatMetricValue(progress.total)}</small>
          </strong>
        </div>
        <div>
          <span>已载入指标上报</span>
          <strong>
            {progress.reportCount.toLocaleString('zh-CN')}
            <small> 条</small>
          </strong>
        </div>
        <div>
          <span>数值指标</span>
          <strong>
            {series.length}
            <small> 项</small>
          </strong>
        </div>
      </div>
      {historyLimited && (
        <p className="training-metric-history-note">
          仅展示当前载入的历史采样，采样数不代表本次训练的完整记录。
        </p>
      )}
      {!series.length ? (
        loading ? (
          <Loading label="正在读取训练指标…" />
        ) : (
          <Empty
            icon={Activity}
            title="等待训练上报数值指标"
            description="训练进程上报后，将自动展示可用指标；未上报的数值保留为空。"
          />
        )
      ) : (
        <>
          <fieldset className="training-metric-picker">
            <legend>选择展示指标（可多选）</legend>
            <div className="training-metric-picker-actions">
              <button
                type="button"
                className="text-link"
                onClick={() => setSelection(series.map((metric) => metric.name))}
                disabled={visible.length === series.length}
              >
                全选
              </button>
              <button
                type="button"
                className="text-link"
                onClick={() => setSelection([])}
                disabled={!visible.length}
              >
                清空
              </button>
            </div>
            <div className="training-metric-options">
              {series.map((metric) => (
                <label
                  key={metric.name}
                  className={selected.includes(metric.name) ? 'selected' : ''}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(metric.name)}
                    onChange={(event) => {
                      const checked = event.target.checked;
                      setSelection(
                        checked
                          ? [...selected, metric.name]
                          : selected.filter((name) => name !== metric.name),
                      );
                    }}
                  />
                  <span>{metric.name}</span>
                </label>
              ))}
            </div>
          </fieldset>
          {visible.length ? (
            <div className="training-metric-charts">
              {visible.map((metric) => (
                <section
                  className="training-metric-card"
                  key={metric.name}
                  aria-label={`${metric.name} 指标`}
                >
                  <div className="training-metric-card-heading">
                    <div>
                      <h3>{metric.name}</h3>
                      <p>
                        {metric.sampleCount.toLocaleString('zh-CN')} 次有效上报 ·{' '}
                        {metric.points
                          .filter((point) => point.value !== null)
                          .length.toLocaleString('zh-CN')}{' '}
                        个曲线采样点
                      </p>
                    </div>
                    <div className="training-metric-latest">
                      <span>最近值</span>
                      <strong>{formatMetricValue(metric.latest.value)}</strong>
                    </div>
                  </div>
                  <div className="training-metric-observation">
                    {metric.latest.source === 'snapshot' ? (
                      '来自最新指标快照；该指标的上报时间与步数未知'
                    ) : (
                      <>
                        <span>
                          该指标最近已载入上报 Step {formatMetricValue(metric.latest.step)}
                        </span>
                        <span>
                          {metric.latest.timestamp ? date(metric.latest.timestamp) : '上报时间未知'}
                        </span>
                        {metric.latest.value === null && <span>本次上报无有效数值</span>}
                      </>
                    )}
                  </div>
                  {metric.points.some((point) => point.value !== null) ? (
                    <MetricChart values={metric.points} metricName={metric.name} />
                  ) : (
                    <div className="training-metric-no-curve">
                      暂无带有效步数的历史采样，曲线等待后续上报。
                    </div>
                  )}
                </section>
              ))}
            </div>
          ) : (
            <div className="training-metric-no-curve">选择一个或多个指标查看趋势。</div>
          )}
          <p className="training-metric-caption">
            每项指标独立绘制纵轴；缺失值保留为空，显式无效值形成断点。同一步数采用该指标最近一次上报。
          </p>
        </>
      )}
    </div>
  );
}
