import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

const event = (seq, data, changes = {}) => ({
  seq,
  data,
  type: 'METRIC',
  timestamp: `2026-10-28T01:${String(seq).padStart(2, '0')}:00Z`,
  ...changes,
});

try {
  const { finiteMetric, formatMetricValue, metricNames, metricProgress, trainingMetricSeries } =
    await server.ssrLoadModule('/src/trainingMetricData.ts');
  const { TrainingMetrics } = await server.ssrLoadModule('/src/TrainingMetrics.tsx');
  const { parseTrainingMetricRange, trainingMetricRangeBounds, trainingMetricPointsInRange } =
    await server.ssrLoadModule('/src/trainingMetricRange.ts');

  await test('custom step ranges accept zero, inclusive single-step ranges and an open latest bound', () => {
    assert.deepEqual(parseTrainingMetricRange('0', '0'), {
      range: { mode: 'custom', start: 0, end: 0 },
    });
    assert.deepEqual(parseTrainingMetricRange(' 100 ', ' 100 '), {
      range: { mode: 'custom', start: 100, end: 100 },
    });
    assert.deepEqual(parseTrainingMetricRange('', ''), {
      range: { mode: 'custom', start: 0, end: null },
    });
    assert.deepEqual(parseTrainingMetricRange('100', ''), {
      range: { mode: 'custom', start: 100, end: null },
    });
    for (const value of ['-1', '0.5', '1e3', 'Infinity', 'NaN', '9007199254740992']) {
      assert.ok(parseTrainingMetricRange(value, '').error, value);
      assert.ok(parseTrainingMetricRange('0', value).error, value);
    }
    assert.match(parseTrainingMetricRange('500', '499').error, /结束步数不能小于起始步数/);
  });

  await test('all and recent ranges follow the current step; fixed custom ranges remain stable', () => {
    assert.deepEqual(trainingMetricRangeBounds({ mode: 'all' }, 20700), { start: 0, end: 20700 });
    assert.deepEqual(trainingMetricRangeBounds({ mode: 'all' }, 0), { start: 0, end: 0 });
    assert.deepEqual(trainingMetricRangeBounds({ mode: 'all' }, null), { start: 0, end: null });
    const recent = { mode: 'recent', steps: 1000 };
    assert.deepEqual(trainingMetricRangeBounds(recent, 20700), { start: 19700, end: 20700 });
    assert.deepEqual(trainingMetricRangeBounds(recent, 20800), { start: 19800, end: 20800 });
    assert.deepEqual(trainingMetricRangeBounds(recent, 100), { start: 0, end: 100 });
    const fixed = { mode: 'custom', start: 500, end: 1500 };
    assert.deepEqual(trainingMetricRangeBounds(fixed, 20700), { start: 500, end: 1500 });
    assert.deepEqual(trainingMetricRangeBounds(fixed, 20800), { start: 500, end: 1500 });
    const follow = { mode: 'custom', start: 500, end: null };
    assert.deepEqual(trainingMetricRangeBounds(follow, 20700), { start: 500, end: 20700 });
    assert.deepEqual(trainingMetricRangeBounds(follow, 20800), { start: 500, end: 20800 });
  });

  await test('range filtering retains sparse observations and explicit gaps without inventing boundary samples', () => {
    const points = [
      { step: 0, value: 0 },
      { step: 200, value: 0.5 },
      { step: 500, value: null },
      { step: 1000, value: 0.2 },
      { step: 20700, value: 0.1 },
    ];
    assert.deepEqual(
      trainingMetricPointsInRange(points, { start: 200, end: 1000 }),
      points.slice(1, 4),
    );
    assert.deepEqual(trainingMetricPointsInRange(points, { start: 0, end: 0 }), [points[0]]);
    assert.deepEqual(trainingMetricPointsInRange(points, { start: 201, end: 499 }), []);
    assert.deepEqual(
      trainingMetricPointsInRange(points, { start: 1000, end: null }),
      points.slice(3),
    );
    assert.deepEqual(
      points.map((point) => point.step),
      [0, 200, 500, 1000, 20700],
    );
  });

  await test('default controls expose full-history and custom ranges while incomplete history stays explicit', () => {
    const html = renderToStaticMarkup(
      React.createElement(TrainingMetrics, {
        events: [event(1, { step: 17400, loss: 0.004 })],
        latestMetrics: { step: 20700, loss: 0.0035 },
        historyComplete: false,
      }),
    );
    assert.match(html, /<legend>显示步数<\/legend>/);
    assert.match(html, /aria-pressed="true">从开始到现在/);
    assert.match(html, /最近 1,000 步/);
    assert.match(html, /最近 5,000 步/);
    assert.match(html, /最近 10,000 步/);
    assert.match(html, />自定义<\/button>/);
    assert.match(html, /正在补齐全程指标/);
    assert.match(html, /已载入 1 次有效上报/);
    assert.doesNotMatch(html, /全程 1 次有效上报/);
    assert.match(html, /显示 Step 0 – 20,700/);
  });

  await test('failed partial-history reads show retry and incomplete coverage rather than a completed whole run', () => {
    const html = renderToStaticMarkup(
      React.createElement(TrainingMetrics, {
        events: [event(1, { step: 17400, loss: 0.004 })],
        historyComplete: false,
        error: new Error('history unavailable'),
        retry: () => {},
      }),
    );
    assert.match(html, /训练指标读取失败/);
    assert.match(html, /全程指标尚未补齐/);
    assert.doesNotMatch(html, /正在补齐/);
    assert.match(html, /重试/);
  });

  await test('history leading a stale run snapshot is not clipped from the full-range chart', () => {
    const html = renderToStaticMarkup(
      React.createElement(TrainingMetrics, {
        events: [event(1, { step: 0, loss: 1 }), event(2, { step: 200, loss: 0.5 })],
        latestMetrics: { step: 100, loss: 0.8 },
      }),
    );
    assert.match(html, /显示 Step 0 – 200/);
    assert.match(html, /区间 2 个曲线采样点/);
  });

  await test('all reported finite numeric keys are discovered, with metadata and other event types excluded', () => {
    const names = metricNames(
      [
        event(1, {
          step: 3,
          total_steps: 100,
          timestamp: 17,
          loss: 0.5,
          learning_rate: 1e-8,
          'train/loss': 0.2,
          'eval/success_rate': 0,
          grad_norm: -3,
          flag: true,
          string: '1',
          nested: { score: 1 },
          unknown: null,
          invalid: Infinity,
        }),
        event(2, { 'gpu/throughput': 20 }, { type: 'RESOURCE' }),
      ],
      { novel_metric: 42 },
    );
    assert.deepEqual(names, [
      'loss',
      'eval/success_rate',
      'grad_norm',
      'learning_rate',
      'novel_metric',
      'train/loss',
    ]);
  });

  await test('sparse metrics retain their own last event metadata and never adopt merged snapshot time or step', () => {
    const events = [
      event(1, { step: 10, loss: 0.2, 'eval/success_rate': 0 }),
      event(2, { step: 20, loss: 0.1 }),
    ];
    const series = trainingMetricSeries(events, {
      step: 20,
      loss: 0.1,
      'eval/success_rate': 0,
      timestamp: events[1].timestamp,
    });
    const evaluation = series.find((metric) => metric.name === 'eval/success_rate');
    assert.equal(evaluation.sampleCount, 1);
    assert.deepEqual(evaluation.points, [{ step: 10, value: 0, timestamp: events[0].timestamp }]);
    assert.deepEqual(evaluation.latest, {
      step: 10,
      value: 0,
      timestamp: events[0].timestamp,
      source: 'event',
    });
    assert.deepEqual(
      metricProgress([...events, event(3, { step: 90 }, { type: 'LOG' })], {}, 100),
      { step: 20, total: 100, reportCount: 2 },
    );
  });

  await test('curves sort by finite steps, use last same-step observation and preserve explicit invalid gaps', () => {
    const events = [
      event(4, { step: 3, loss: 0.4 }),
      event(1, { step: 1, loss: 0 }),
      event(2, { step: 2, loss: 0.2 }),
      event(3, { step: 2, loss: null }),
      event(5, { step: 4, other: 1 }),
      event(6, { step: Infinity, loss: -0.01 }),
      event(7, { loss: null }),
    ];
    const loss = trainingMetricSeries(events, { loss: 0.99 })[0];
    assert.deepEqual(
      loss.points.map(({ step, value }) => ({ step, value })),
      [
        { step: 1, value: 0 },
        { step: 2, value: null },
        { step: 3, value: 0.4 },
      ],
    );
    assert.equal(loss.sampleCount, 4);
    assert.deepEqual(loss.latest, {
      value: null,
      step: null,
      timestamp: events[6].timestamp,
      source: 'event',
    });
    assert.deepEqual(
      events.map((row) => row.seq),
      [4, 1, 2, 3, 5, 6, 7],
      'input event order is never mutated',
    );
  });

  await test('demonstrably newer changed snapshot values lead delayed history without borrowing global observation metadata', () => {
    const observed = event(1, { step: 10, loss: 0.8 });
    const snapshot = { step: 100, loss: 0.3, timestamp: event(2, {}).timestamp };
    const metric = trainingMetricSeries([observed], snapshot)[0];
    assert.deepEqual(metric.latest, { value: 0.3, step: null, source: 'snapshot' });
    assert.deepEqual(metric.points, [{ step: 10, value: 0.8, timestamp: observed.timestamp }]);
    assert.equal(metric.sampleCount, 1);
    for (const value of [null, 'invalid', Infinity]) {
      const invalid = trainingMetricSeries([observed], { ...snapshot, loss: value })[0];
      assert.deepEqual(invalid.latest, { value: null, step: null, source: 'snapshot' });
    }
  });

  await test('older, equal or unorderable snapshots cannot replace an observed value', () => {
    const observed = event(2, { step: 20, loss: 0.5 });
    for (const timestamp of [event(1, {}).timestamp, observed.timestamp, 'invalid', undefined]) {
      const metric = trainingMetricSeries([observed], { loss: 0.2, timestamp })[0];
      assert.deepEqual(metric.latest, {
        value: 0.5,
        step: 20,
        timestamp: observed.timestamp,
        source: 'event',
      });
    }
    const noTime = trainingMetricSeries([{ ...observed, timestamp: undefined }], {
      loss: 0.2,
      timestamp: event(3, {}).timestamp,
    })[0];
    assert.deepEqual(noTime.latest, {
      value: 0.5,
      step: 20,
      timestamp: undefined,
      source: 'event',
    });
  });

  await test('an unrelated newer shared snapshot timestamp never refreshes an unchanged sparse metric', () => {
    const observed = event(1, { step: 10, 'eval/score': 0, loss: 0.8 });
    const latest = event(2, { step: 100, loss: 0.2 });
    const metrics = trainingMetricSeries([observed, latest], {
      step: 100,
      loss: 0.2,
      'eval/score': 0,
      timestamp: event(3, {}).timestamp,
    });
    assert.deepEqual(metrics.find((metric) => metric.name === 'eval/score').latest, {
      value: 0,
      step: 10,
      timestamp: observed.timestamp,
      source: 'event',
    });
    assert.deepEqual(metrics.find((metric) => metric.name === 'loss').latest, {
      value: 0.2,
      step: 100,
      timestamp: latest.timestamp,
      source: 'event',
    });
  });

  await test('snapshot-only metrics expose unknown time and step, including ordinary prototype-like keys', () => {
    const snapshot = JSON.parse(
      '{"step":20,"timestamp":"2026-10-28T09:30:00Z","learning_rate":0.00000002,"__proto__":3,"constructor":4}',
    );
    const series = trainingMetricSeries([], snapshot);
    assert.equal(series.length, 3);
    for (const metric of series) {
      assert.deepEqual(metric.points, []);
      assert.equal(metric.sampleCount, 0);
      assert.equal(metric.latest.source, 'snapshot');
      assert.equal(metric.latest.step, null);
      assert.equal(metric.latest.timestamp, undefined);
    }
    assert.equal(series.find((metric) => metric.name === '__proto__').latest.value, 3);
    assert.equal(series.find((metric) => metric.name === 'constructor').latest.value, 4);
    assert.equal({}.polluted, undefined);
  });

  await test('value formatting preserves real zero, negative and very small values without inventing missing numbers', () => {
    assert.equal(formatMetricValue(0), '0');
    assert.equal(formatMetricValue(1e-9), '1e-9');
    assert.equal(formatMetricValue(-2.5e-8), '-2.5e-8');
    assert.equal(formatMetricValue(12345678), '1.23457e+7');
    assert.equal(formatMetricValue(-0.0123456), '-0.0123456');
    for (const value of [null, undefined, NaN, Infinity, '0', false]) {
      assert.equal(finiteMetric(value), null);
      assert.equal(formatMetricValue(value), '—');
    }
  });

  await test('default loss chart coexists with accessible selection for every reported metric', () => {
    const html = renderToStaticMarkup(
      React.createElement(TrainingMetrics, {
        events: [event(1, { step: 0, loss: 0, learning_rate: 1e-9, 'eval/<success>': -1 })],
        totalSteps: 100,
        historyLimited: true,
      }),
    );
    assert.match(html, /<legend>选择展示指标（可多选）<\/legend>/);
    assert.equal((html.match(/type="checkbox"/g) || []).length, 3);
    assert.match(html, /type="checkbox"[^>]*checked=""[^>]*\/><span>loss<\/span>/);
    assert.match(html, /eval\/&lt;success&gt;/);
    assert.match(html, /aria-label="loss 指标"/);
    assert.doesNotMatch(html, /aria-label="learning_rate 指标"/);
    assert.match(html, /仅展示当前载入的历史采样/);
    assert.match(html, /0 次有效上报|1 次有效上报/);
  });

  await test('snapshot-only default metric clearly separates value from unknown historical metadata', () => {
    const html = renderToStaticMarkup(
      React.createElement(TrainingMetrics, {
        events: [],
        latestMetrics: { learning_rate: 1e-9, step: 300, timestamp: '2026-10-28T12:00:00Z' },
      }),
    );
    assert.match(html, /1e-9/);
    assert.match(html, /该指标的上报时间与步数未知/);
    assert.match(html, /0 次有效上报/);
    assert.match(html, /暂无带有效步数的历史采样/);
    assert.doesNotMatch(html, /12:00/);
  });

  await test('metadata-only input waits for real metrics without a fabricated loss series', () => {
    const html = renderToStaticMarkup(
      React.createElement(TrainingMetrics, {
        events: [event(1, { step: 20, total_steps: 100 })],
      }),
    );
    assert.match(html, /等待训练上报数值指标/);
    assert.doesNotMatch(html, /type="checkbox"|loss|<svg[^>]*aria-label="训练/);
  });
} finally {
  await server.close();
}
