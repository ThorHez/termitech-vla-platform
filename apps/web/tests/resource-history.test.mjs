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

const stamp = (seconds) => new Date(Date.UTC(2026, 9, 27) + seconds * 1000).toISOString();
const point = (seconds, value, changes = {}) => ({
  timestamp: stamp(seconds),
  sample_count: 1,
  cpu_percent: value,
  memory_percent: null,
  disk_percent: null,
  gpus: [],
  ...changes,
});

try {
  const {
    validPercent,
    seriesValues,
    chartSegments,
    seriesSummary,
    historicalGpus,
    nearestPointIndex,
  } = await server.ssrLoadModule('/src/resourceHistoryData.ts');
  const { ResourceTrendChart, ResourceHistory } = await server.ssrLoadModule(
    '/src/ResourceHistory.tsx',
  );

  await test('zero is an observed percentage; absent, malformed and out-of-range values stay unknown', () => {
    assert.equal(validPercent(0), 0);
    assert.equal(validPercent(100), 100);
    for (const value of [null, undefined, '', '80', NaN, Infinity, -1, 101])
      assert.equal(validPercent(value), null);
    const values = seriesValues(
      [point(0, 0), point(30, null), point(60, 60, { sample_count: 0 })],
      (row) => row.cpu_percent,
    );
    assert.deepEqual(
      values.map((row) => row.value),
      [0, null, null],
    );
    assert.deepEqual(seriesSummary(values), { latest: 0, timestamp: stamp(0), average: 0 });
  });

  await test('missing buckets and sparse offline intervals split the chart without discarding valid zeros', () => {
    const values = seriesValues(
      [
        point(0, 20),
        point(30, 0),
        point(60, null, { sample_count: 0 }),
        point(90, 45),
        point(600, 75),
        point(630, 80),
      ],
      (row) => row.cpu_percent,
    );
    assert.deepEqual(
      chartSegments(values, 30).map((segment) => segment.map((row) => row.value)),
      [[20, 0], [45], [75, 80]],
    );
    assert.deepEqual(seriesSummary(values), { latest: 80, timestamp: stamp(630), average: 44 });
  });

  await test('GPU history follows stable UUIDs through removal and reordering', () => {
    const gpu = (uuid, usage) => ({
      uuid,
      name: 'same GPU model',
      utilization_pct: usage,
      memory_percent: null,
    });
    const points = [
      point(0, null, { gpus: [gpu('A', 0), gpu('B', 70)] }),
      point(30, null, { gpus: [gpu('B', 50)] }),
      point(60, null, { gpus: [gpu('B', 25), gpu('A', 30)] }),
    ];
    assert.deepEqual(
      historicalGpus(points).map((row) => row.uuid),
      ['A', 'B'],
    );
    assert.deepEqual(
      seriesValues(
        points,
        (row) => row.gpus.find((item) => item.uuid === 'A')?.utilization_pct,
      ).map((row) => row.value),
      [0, null, 30],
    );
    assert.equal(nearestPointIndex(points, Date.parse(stamp(40))), 1);
    assert.equal(nearestPointIndex([], Date.now()), -1);
  });

  await test('SVG renders isolated samples, real time labels and keyboard instructions', () => {
    const points = [point(0, 0), point(30, null, { sample_count: 0 }), point(60, 50)];
    const history = {
      node_id: 'one',
      window: '1h',
      start: stamp(0),
      end: stamp(3600),
      bucket_seconds: 30,
      sample_interval_seconds: 30,
      retention_days: 30,
      points,
    };
    const html = renderToStaticMarkup(
      React.createElement(ResourceTrendChart, {
        title: '整机资源',
        history,
        series: [
          {
            label: 'CPU',
            color: '#0b8c87',
            values: seriesValues(points, (row) => row.cpu_percent),
          },
        ],
      }),
    );
    assert.match(html, /tabindex="0"/);
    assert.match(html, /左右方向键逐点查看/);
    assert.match(html, />100%<\/text>/);
    assert.equal((html.match(/<circle/g) || []).length, 2);
    assert.doesNotMatch(html, /<path/);
    assert.ok(
      html.includes(
        new Date(stamp(0)).toLocaleString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
      ),
    );
    assert.match(html, /整机资源，百分比趋势图<\/title>/);
    const emptyHtml = renderToStaticMarkup(
      React.createElement(ResourceTrendChart, {
        title: '缺失',
        history,
        series: [{ label: '内存', color: '#000', values: seriesValues(points, () => null) }],
      }),
    );
    assert.match(emptyHtml, /该时段未采集到可用指标/);
    assert.doesNotMatch(emptyHtml, /<svg/);
  });

  await test('history panel offers bounded ranges to read-only machine viewers', () => {
    const html = renderToStaticMarkup(
      React.createElement(ResourceHistory, { nodeId: 'one', status: 'OFFLINE' }),
    );
    assert.match(html, /资源占用历史/);
    assert.equal((html.match(/aria-pressed=/g) || []).length, 5);
    assert.match(html, /aria-pressed="true"[^>]*>近 24 小时/);
    assert.match(html, /机器当前未在线，已有历史仍可查看/);
  });
} finally {
  await server.close();
}
