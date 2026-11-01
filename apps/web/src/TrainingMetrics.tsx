import { useEffect, useMemo, useState } from 'react';
import { Activity } from 'lucide-react';
import { date } from './api';
import { Empty, ErrorNotice, Loading, MetricChart } from './components';
import { formatMetricValue, metricProgress, trainingMetricSeries } from './trainingMetricData';
import {
  ALL_TRAINING_STEPS,
  parseTrainingMetricRange,
  trainingMetricPointsInRange,
  trainingMetricRangeBounds,
} from './trainingMetricRange';
import type { TrainingMetricRange } from './trainingMetricRange';
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
  historyComplete?: boolean;
  scopeKey?: string;
}

function initialMetricView(scopeKey?: string) {
  return {
    scopeKey,
    selection: null as string[] | null,
    range: ALL_TRAINING_STEPS,
    customOpen: false,
    customStart: '0',
    customEnd: '',
    rangeError: undefined as string | undefined,
  };
}

export function TrainingMetrics({
  events,
  latestMetrics,
  totalSteps,
  loading,
  error,
  retry,
  historyLimited,
  historyComplete = true,
  scopeKey,
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
  const [storedView, setStoredView] = useState(() => initialMetricView(scopeKey));
  // Derive a fresh view immediately when navigating between runs/attempts,
  // without briefly rendering the previous run's range or metric selection.
  const view = storedView.scopeKey === scopeKey ? storedView : initialMetricView(scopeKey);
  useEffect(() => {
    setStoredView((previous) =>
      previous.scopeKey === scopeKey ? previous : initialMetricView(scopeKey),
    );
  }, [scopeKey]);
  const updateView = (changes: Partial<ReturnType<typeof initialMetricView>>) =>
    setStoredView((previous) => ({
      ...(previous.scopeKey === scopeKey ? previous : initialMetricView(scopeKey)),
      ...changes,
    }));
  const selection = view.selection;
  const selected = selection === null ? series.slice(0, 1).map((metric) => metric.name) : selection;
  const visible = series.filter((metric) => selected.includes(metric.name));
  // History and the run snapshot refresh independently. Neither a stale
  // snapshot nor a delayed history page should move the visible end backward.
  const latestStep = series.reduce<number | null>((latest, metric) => {
    const last = metric.points.at(-1)?.step;
    return last === undefined ? latest : Math.max(latest ?? 0, last);
  }, progress.step);
  const bounds = trainingMetricRangeBounds(view.range, latestStep);
  const chooseRange = (range: TrainingMetricRange) =>
    updateView({ range, customOpen: false, rangeError: undefined });
  const resetRange = () =>
    updateView({
      range: ALL_TRAINING_STEPS,
      customOpen: false,
      customStart: '0',
      customEnd: '',
      rangeError: undefined,
    });
  return (
    <div className="training-metric-view">
      <ErrorNotice error={error} retry={retry} title="训练指标读取失败" />
      <div className="training-metric-summary">
        <div>
          <span>当前步数</span>
          <strong>
            {formatMetricValue(latestStep)}
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
      {!historyComplete && (
        <p className="training-metric-history-note" role="status">
          {error ? '全程指标尚未补齐' : '正在补齐全程指标'}，当前曲线仅包含已载入的上报。
        </p>
      )}
      {historyLimited && historyComplete && (
        <p className="training-metric-history-note">
          仅展示当前载入的历史采样，采样数不代表本次训练的完整记录。
        </p>
      )}
      <fieldset className="training-metric-range">
        <legend>显示步数</legend>
        <div className="training-metric-range-presets">
          <button
            type="button"
            aria-pressed={!view.customOpen && view.range.mode === 'all'}
            onClick={() => chooseRange(ALL_TRAINING_STEPS)}
          >
            从开始到现在
          </button>
          {[1000, 5000, 10000].map((steps) => (
            <button
              type="button"
              key={steps}
              aria-pressed={
                !view.customOpen && view.range.mode === 'recent' && view.range.steps === steps
              }
              onClick={() => chooseRange({ mode: 'recent', steps })}
            >
              最近 {steps.toLocaleString('zh-CN')} 步
            </button>
          ))}
          <button
            type="button"
            aria-pressed={view.customOpen}
            onClick={() => updateView({ customOpen: true })}
          >
            自定义
          </button>
        </div>
        {view.customOpen && (
          <div className="training-metric-range-custom">
            <label>
              <span>起始步数</span>
              <input
                type="number"
                min="0"
                step="1"
                value={view.customStart}
                placeholder="0"
                aria-invalid={Boolean(view.rangeError)}
                onChange={(event) =>
                  updateView({ customStart: event.target.value, rangeError: undefined })
                }
              />
            </label>
            <label>
              <span>结束步数</span>
              <input
                type="number"
                min="0"
                step="1"
                value={view.customEnd}
                placeholder="最新步数"
                aria-invalid={Boolean(view.rangeError)}
                onChange={(event) =>
                  updateView({ customEnd: event.target.value, rangeError: undefined })
                }
              />
            </label>
            <button
              type="button"
              className="button outline small"
              onClick={() => {
                const parsed = parseTrainingMetricRange(view.customStart, view.customEnd);
                if (parsed.error) updateView({ rangeError: parsed.error });
                else updateView({ range: parsed.range!, rangeError: undefined });
              }}
            >
              应用范围
            </button>
            <button type="button" className="text-link" onClick={resetRange}>
              重置范围
            </button>
            <p>起止步数均包含在范围内；结束留空时跟随最新步数，点击应用后生效。</p>
            {view.rangeError && (
              <p className="training-metric-range-error" role="alert">
                {view.rangeError}
              </p>
            )}
          </div>
        )}
        <p className="training-metric-range-summary">
          显示 Step {formatMetricValue(bounds.start)} –{' '}
          {bounds.end === null ? '最新' : formatMetricValue(bounds.end)}
          {view.range.mode === 'custom' && view.range.end === null && ' · 跟随最新步数'}
        </p>
      </fieldset>
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
                onClick={() => updateView({ selection: series.map((metric) => metric.name) })}
                disabled={visible.length === series.length}
              >
                全选
              </button>
              <button
                type="button"
                className="text-link"
                onClick={() => updateView({ selection: [] })}
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
                      updateView({
                        selection: checked
                          ? [...selected, metric.name]
                          : selected.filter((name) => name !== metric.name),
                      });
                    }}
                  />
                  <span>{metric.name}</span>
                </label>
              ))}
            </div>
          </fieldset>
          {visible.length ? (
            <div className="training-metric-charts">
              {visible.map((metric) => {
                const rangePoints = trainingMetricPointsInRange(metric.points, bounds);
                return (
                  <section
                    className="training-metric-card"
                    key={metric.name}
                    aria-label={`${metric.name} 指标`}
                  >
                    <div className="training-metric-card-heading">
                      <div>
                        <h3>{metric.name}</h3>
                        <p>
                          {historyComplete && !historyLimited ? '全程' : '已载入'}{' '}
                          {metric.sampleCount.toLocaleString('zh-CN')} 次有效上报 · 区间{' '}
                          {rangePoints
                            .filter((point) => point.value !== null)
                            .length.toLocaleString('zh-CN')}{' '}
                          个曲线采样点
                        </p>
                      </div>
                      <div className="training-metric-latest">
                        <span>{view.range.mode === 'all' ? '最近值' : '全程最近值'}</span>
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
                            {metric.latest.timestamp
                              ? date(metric.latest.timestamp)
                              : '上报时间未知'}
                          </span>
                          {metric.latest.value === null && <span>本次上报无有效数值</span>}
                        </>
                      )}
                    </div>
                    {rangePoints.some((point) => point.value !== null) ? (
                      <MetricChart
                        values={rangePoints}
                        metricName={metric.name}
                        stepDomain={
                          bounds.end !== null && bounds.end >= bounds.start
                            ? { start: bounds.start, end: bounds.end }
                            : undefined
                        }
                      />
                    ) : (
                      <div className="training-metric-no-curve">
                        {view.range.mode === 'all'
                          ? '暂无带有效步数的历史采样，曲线等待后续上报。'
                          : '所选步数范围内暂无该指标的有效上报。'}
                      </div>
                    )}
                  </section>
                );
              })}
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
