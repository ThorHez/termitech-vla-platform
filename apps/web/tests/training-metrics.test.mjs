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
