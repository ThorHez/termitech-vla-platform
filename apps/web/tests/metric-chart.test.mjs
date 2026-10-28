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

try {
  const { MetricChart } = await server.ssrLoadModule('/src/components.tsx');
  const {
    buildMetricChartModel,
    metricFraction,
    metricInterpolate,
    formatMetricValue,
    nearestMetricPoint,
  } = await server.ssrLoadModule('/src/metricChartData.ts');
  const render = (props) => renderToStaticMarkup(React.createElement(MetricChart, props));

  await test('generic metrics render zero, negative and tiny values with their own accessible name', () => {
    for (const [metricName, value] of [
      ['accuracy', 0],
      ['advantage', -0.7],
      ['learning_rate', 1e-8],
    ]) {
      const html = render({ metricName, values: [{ step: 0, value }] });
      assert.match(html, new RegExp(`aria-label="${metricName} 训练指标曲线，共 1 个有效采样点`));
      assert.match(html, /<circle /);
      assert.doesNotMatch(html, /等待训练上报|NaN|Infinity/);
      assert.match(html, /tabindex="0"/);
      assert.match(html, /Home 和 End/);
    }
    assert.equal(formatMetricValue(1e-8), '1e-8');
    assert.equal(formatMetricValue(-0), '0');
    assert.equal(formatMetricValue(null), '—');
  });

  await test('empty and invalid data use the selected metric name; legacy loss input remains accepted', () => {
    assert.match(render({ metricName: 'grad_norm', values: [] }), /等待训练上报 grad_norm 指标/);
    assert.match(
      render({ metricName: 'learning_rate', values: [{ step: 1, value: NaN }] }),
      /等待训练上报 learning_rate 指标/,
    );
    assert.match(
      render({ values: [{ step: 1, loss: 0.2 }] }),
      /loss 训练指标曲线，共 1 个有效采样点/,
    );
  });

  await test('explicit missing values and malformed steps split paths instead of bridging gaps', () => {
    const input = [
      { step: 0, value: 0 },
      { step: 1, value: -1 },
      { step: 2, value: null },
      { step: 3, value: 3 },
      { step: 4, value: Infinity },
      { step: 5, value: 5 },
      { step: NaN, value: 6 },
      { step: 7, value: 7 },
      { step: 8, value: 8 },
    ];
    const model = buildMetricChartModel(input);
    assert.deepEqual(
      model.segments.map((segment) => segment.map((point) => point.step)),
      [[0, 1], [3], [5], [7, 8]],
    );
    assert.equal(model.points[2].value, null);
    const html = render({ metricName: 'custom', values: input });
    assert.equal((html.match(/<polyline /g) || []).length, 2);
    assert.equal((html.match(/<circle /g) || []).length, 2);
    assert.doesNotMatch(html, /NaN|Infinity/);
  });

  await test('all finite numeric domains including extreme and subnormal values produce finite geometry', () => {
    for (const series of [
      [0, 0],
      [-1, -1],
      [Number.MIN_VALUE, Number.MIN_VALUE],
      [1e-300, 2e-300],
      [-Number.MAX_VALUE, Number.MAX_VALUE],
      [Number.MAX_VALUE, Number.MAX_VALUE],
      [-Number.MAX_VALUE, -Number.MAX_VALUE],
      [1e308, 1.0000000000000002e308],
    ]) {
      const input = series.map((value, step) => ({ step, value }));
      const { domain } = buildMetricChartModel(input);
      assert.ok(
        Number.isFinite(domain.low) && Number.isFinite(domain.high) && domain.low < domain.high,
        JSON.stringify(series),
      );
      for (const value of series)
        assert.ok(Number.isFinite(metricFraction(value, domain.low, domain.high)));
      for (const fraction of [0, 1 / 3, 2 / 3, 1])
        assert.ok(Number.isFinite(metricInterpolate(domain.low, domain.high, fraction)));
      assert.doesNotMatch(render({ metricName: 'finite', values: input }), /NaN|Infinity/);
    }
    const extremeSteps = [
      { step: -Number.MAX_VALUE, value: 1 },
      { step: Number.MAX_VALUE, value: 2 },
    ];
    assert.doesNotMatch(render({ values: extremeSteps }), /NaN|Infinity/);
  });

  await test('each chart has its own gradient and constant series show a point', () => {
    const html = renderToStaticMarkup(
      React.createElement(
        'div',
        null,
        React.createElement(MetricChart, {
          metricName: 'loss',
          values: [
            { step: 1, value: 0 },
            { step: 2, value: 0 },
          ],
        }),
        React.createElement(MetricChart, {
          metricName: 'lr',
          values: [
            { step: 1, value: 1e-8 },
            { step: 2, value: 2e-8 },
          ],
        }),
      ),
    );
    const ids = [...html.matchAll(/<linearGradient id="([^"]+)"/g)].map((match) => match[1]);
    assert.equal(ids.length, 2);
    assert.equal(new Set(ids).size, 2);
    for (const id of ids) assert.ok(html.includes(`url(#${id})`));
    assert.match(html, /<circle /);
  });

  await test('long histories avoid spread argument limits and pointer selection follows step distance', () => {
    const input = Array.from({ length: 150000 }, (_, step) => ({ step, value: Math.sin(step) }));
    const model = buildMetricChartModel(input);
    assert.equal(model.knownCount, input.length);
    assert.ok(model.domain.low < -1 && model.domain.high > 1);
    const points = [
      { step: 0, value: 0 },
      { step: 2, value: null },
      { step: 100, value: 1 },
    ];
    assert.equal(nearestMetricPoint(points, 0.01, 0, 100), 0);
    assert.equal(nearestMetricPoint(points, 0.03, 0, 100), 1);
    assert.equal(nearestMetricPoint(points, 0.98, 0, 100), 2);
    assert.equal(nearestMetricPoint([], 0.5, 0, 1), -1);
  });
} finally {
  await server.close();
}
