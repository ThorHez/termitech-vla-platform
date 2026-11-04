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
  const { ResourceTrendChart, ResourceHistory, GpuHistoryCharts } = await server.ssrLoadModule(
    '/src/ResourceHistory.tsx',
  );
  const gpu = (uuid, usage, memory = null) => ({
    uuid,
    name: 'same GPU model',
    utilization_pct: usage,
    memory_percent: memory,
  });
  const gpuHistory = (points) => ({
    node_id: 'one',
    window: '1h',
    start: stamp(0),
    end: stamp(3600),
    bucket_seconds: 30,
    sample_interval_seconds: 30,
    retention_days: 30,
    points,
  });
  const renderGpus = (history, gpuIds = null) =>
    renderToStaticMarkup(
      React.createElement(GpuHistoryCharts, { history, gpuIds, selectGpus: () => {} }),
    );
  const pickerLabels = (html) =>
    [...html.matchAll(/<label\b[^>]*>([\s\S]*?)<\/label>/g)].map((match) => ({
      uuid: match[1].match(/<code>([^<]+)<\/code>/)?.[1],
      checked: /<input\b[^>]*\bchecked=""/.test(match[1]),
    }));

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

  await test('GPU history defaults to all cards with distinct labels and separate compact charts', () => {
    const history = gpuHistory([
      point(0, null, { gpus: [gpu('GPU-A', 10), gpu('GPU-B', 20), gpu('GPU-C', 30)] }),
      point(30, null, { gpus: [gpu('GPU-C', 40), gpu('GPU-A', 50), gpu('GPU-B', 60)] }),
    ]);
    const html = renderGpus(history);
    assert.deepEqual(pickerLabels(html), [
      { uuid: 'GPU-A', checked: true },
      { uuid: 'GPU-B', checked: true },
      { uuid: 'GPU-C', checked: true },
    ]);
    assert.match(html, /<legend>选择 GPU（可多选）<\/legend>/);
    assert.match(html, /已选 3 \/ 3 张/);
    assert.match(html, /<button type="button"[^>]*>全选<\/button>/);
    assert.match(html, /<button type="button"[^>]*>清空<\/button>/);
    assert.equal((html.match(/viewBox="0 0 560 205"/g) || []).length, 3);
    assert.equal((html.match(/>100%<\/text>/g) || []).length, 3);
    assert.equal((html.match(/>0%<\/text>/g) || []).length, 3);
    assert.equal((html.match(/tabindex="0"/g) || []).length, 3);
    for (const uuid of ['GPU-A', 'GPU-B', 'GPU-C'])
      assert.ok(html.includes(`<h3>same GPU model · ${uuid}</h3>`));
    const chartLabels = [...html.matchAll(/aria-labelledby="([^"]+)"/g)].map((match) =>
      match[1].split(' '),
    );
    assert.equal(new Set(chartLabels.flat()).size, 6);
    for (const ids of chartLabels) for (const id of ids) assert.ok(html.includes(`id="${id}"`));
    assert.ok(
      html.includes(
        new Date(stamp(0)).toLocaleString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
      ),
    );
  });

  await test('selecting one GPU shows only its chart at full width', () => {
    const history = gpuHistory([point(0, null, { gpus: [gpu('GPU-A', 10), gpu('GPU-B', 20)] })]);
    const html = renderGpus(history, ['GPU-B']);
    assert.deepEqual(pickerLabels(html), [
      { uuid: 'GPU-A', checked: false },
      { uuid: 'GPU-B', checked: true },
    ]);
    assert.match(html, /已选 1 \/ 2 张/);
    assert.match(html, /class="resource-history-gpu-grid single"/);
    assert.equal((html.match(/<svg\b/g) || []).length, 1);
    assert.match(html, /viewBox="0 0 800 205"/);
    assert.match(html, /<h3>same GPU model · GPU-B<\/h3>/);
    assert.doesNotMatch(html, /<h3>same GPU model · GPU-A<\/h3>/);
  });

  await test('clearing GPU selection stays empty instead of silently selecting the first card', () => {
    const html = renderGpus(
      gpuHistory([point(0, null, { gpus: [gpu('GPU-A', 10), gpu('GPU-B', 20)] })]),
      [],
    );
    assert.deepEqual(pickerLabels(html), [
      { uuid: 'GPU-A', checked: false },
      { uuid: 'GPU-B', checked: false },
    ]);
    assert.match(html, /已选 0 \/ 2 张/);
    assert.match(html, /未选择 GPU，请勾选要查看的显卡或点击全选。/);
    assert.doesNotMatch(html, /<svg\b/);
  });

  await test('GPU selections absent in a time window never fall back to an unrelated card', () => {
    const history = gpuHistory([point(0, null, { gpus: [gpu('GPU-A', 10), gpu('GPU-B', 20)] })]);
    const missing = ['GPU-outside-window'];
    const emptyHtml = renderGpus(history, missing);
    assert.doesNotMatch(emptyHtml, /<svg\b/);
    assert.match(emptyHtml, /已选 0 \/ 2 张/);
    assert.deepEqual(missing, ['GPU-outside-window']);
    const selected = ['GPU-outside-window', 'GPU-B'];
    const partialHtml = renderGpus(history, selected);
    assert.equal((partialHtml.match(/<svg\b/g) || []).length, 1);
    assert.match(partialHtml, /<h3>same GPU model · GPU-B<\/h3>/);
    assert.doesNotMatch(partialHtml, /<h3>same GPU model · GPU-A<\/h3>/);
    assert.deepEqual(selected, ['GPU-outside-window', 'GPU-B']);
    const restoredHtml = renderGpus(
      gpuHistory([
        point(0, null, {
          gpus: [gpu('GPU-B', 20), gpu('GPU-outside-window', 30), gpu('GPU-A', 10)],
        }),
      ]),
      selected,
    );
    assert.equal((restoredHtml.match(/<svg\b/g) || []).length, 2);
    assert.match(restoredHtml, /<h3>same GPU model · GPU-outside-window<\/h3>/);
    assert.match(restoredHtml, /<h3>same GPU model · GPU-B<\/h3>/);
  });

  await test('each GPU chart follows its UUID through reordered cards and missing samples', () => {
    const history = gpuHistory([
      point(0, null, { gpus: [gpu('GPU-A', 0, 100), gpu('GPU-B', 75, 25)] }),
      point(30, null, { gpus: [gpu('GPU-B', 50, 50)] }),
      point(60, null, { gpus: [gpu('GPU-B', 25, 75), gpu('GPU-A', 100, 0)] }),
    ]);
    const charts = renderGpus(history).split('<div class="resource-trend-chart">').slice(1);
    assert.equal(charts.length, 2);
    assert.match(charts[0], /<h3>same GPU model · GPU-A<\/h3>/);
    assert.equal((charts[0].match(/<circle\b/g) || []).length, 4);
    assert.equal((charts[0].match(/cy="167"/g) || []).length, 2);
    assert.equal((charts[0].match(/cy="15"/g) || []).length, 2);
    assert.doesNotMatch(charts[0], /<path\b/);
    assert.match(charts[1], /<h3>same GPU model · GPU-B<\/h3>/);
    const paths = [...charts[1].matchAll(/<path\b[^>]*d="([^"]+)"/g)];
    assert.equal(paths.length, 2);
    const yValues = (path) =>
      [...path.matchAll(/[ML][^,]+,([^ ]+)/g)].map((match) => Number(match[1]));
    assert.deepEqual(yValues(paths[0][1]), [53, 91, 129]);
    assert.deepEqual(yValues(paths[1][1]), [129, 91, 53]);
    const noHistory = renderGpus(gpuHistory([point(0, null)]));
    assert.match(noHistory, /此时段暂无 GPU 历史指标，未知值保留为空。/);
    assert.doesNotMatch(noHistory, /<svg\b|<input\b/);
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
