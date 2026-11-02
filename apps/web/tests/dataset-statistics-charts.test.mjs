import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});
const feature = (dimension = 4, changes = {}) => ({
  key: 'state',
  dimension,
  mean: Array.from({ length: dimension }, (_, index) => index - 2),
  std: Array.from({ length: dimension }, (_, index) => index / 10),
  q01: Array.from({ length: dimension }, (_, index) => index - 4),
  q99: Array.from({ length: dimension }, (_, index) => index + 1),
  ...changes,
});

try {
  const {
    normalizationChartRows: rows,
    chartDomain,
    chartY,
    chartNumber,
  } = await server.ssrLoadModule('/src/datasetStatisticsChartData.ts');
  const { DatasetNormalizationCharts } = await server.ssrLoadModule(
    '/src/DatasetNormalizationCharts.tsx',
  );
  const render = (data, start = 0, count = 32) =>
    renderToStaticMarkup(
      React.createElement(DatasetNormalizationCharts, { feature: data, start, count }),
    );

  await test('visible dimension windows retain absolute indexes and clamp to 32 without mutating source arrays', () => {
    const input = feature(65),
      before = structuredClone(input);
    assert.deepEqual(
      rows(input, 32, 100).map((row) => row.index),
      Array.from({ length: 32 }, (_, i) => 32 + i),
    );
    assert.deepEqual(
      rows(input, 64, 32).map((row) => row.index),
      [64],
    );
    assert.equal(rows(input, 65, 32).length, 0);
    assert.equal(rows(input, 0, 0).length, 0);
    assert.equal(rows(input, 0, Infinity).length, 0);
    assert.deepEqual(input, before);
    assert.notStrictEqual(rows(input, 0, 2)[0].range, input.q01);
  });

  await test('zero means, zero deviation and zero-width quantile ranges stay valid', () => {
    const result = rows(feature(1, { mean: [0], std: [0], q01: [0], q99: [0] }), 0, 1);
    assert.deepEqual(result, [{ index: 0, mean: 0, std: 0, q01: 0, q99: 0, range: [0, 0] }]);
  });

  await test('missing and nonfinite values remain gaps and negative deviations do not become positive bars', () => {
    const result = rows(
      feature(3, {
        mean: [NaN, Infinity],
        std: [-1, 0, NaN],
        q01: [NaN, -1, 5],
        q99: [1, Infinity, -5],
      }),
      0,
      3,
    );
    assert.deepEqual(result, [
      { index: 0, q99: 1 },
      { index: 1, std: 0, q01: -1 },
      { index: 2, q01: 5, q99: -5 },
    ]);
    assert.ok(result.every((row) => row.range === undefined));
    assert.equal(rows(feature(1, { dimension: NaN }), 0, 32).length, 0);
  });

  await test('only complete ordered quantile pairs become intervals and min/max never substitute for missing quantiles', () => {
    const result = rows(
      feature(2, { q01: undefined, q99: undefined, min: [-100, -100], max: [100, 100] }),
      0,
      2,
    );
    assert.ok(result.every((row) => row.range === undefined));
    assert.equal(result[0].min, -100);
    assert.equal(result[0].max, 100);
    assert.deepEqual(rows(feature(1, { q01: [-1], q99: [1] }), 0, 1)[0].range, [-1, 1]);
  });

  await test('signed domains spanning both largest doubles map monotonically without subtraction overflow', () => {
    const max = Number.MAX_VALUE;
    const domain = chartDomain([-max, -max / 2, 0, max / 2, max]);
    assert.ok(
      [domain.min, domain.max, domain.scale, domain.low, domain.high, ...domain.ticks].every(
        Number.isFinite,
      ),
    );
    assert.ok(domain.min < 0 && domain.max > 0);
    const y = [-max, -max / 2, 0, max / 2, max].map((value) => chartY(value, domain, 20, 176));
    assert.ok(y.every((value) => Number.isFinite(value) && value >= 20 && value <= 196));
    assert.ok(y.every((value, index) => index === 0 || value < y[index - 1]));
  });

  await test('constant positive, negative, zero and subnormal values all receive representable finite axes', () => {
    for (const value of [
      0,
      4,
      -4,
      Number.MAX_VALUE,
      -Number.MAX_VALUE,
      Number.MIN_VALUE,
      -Number.MIN_VALUE,
      1e-300,
      -1e-300,
    ]) {
      const domain = chartDomain([value, value]);
      assert.ok(domain.low < domain.high, String(value));
      assert.ok(domain.ticks.length >= 2, String(value));
      assert.ok(domain.ticks.every(Number.isFinite), String(value));
      assert.ok(Number.isFinite(chartY(value, domain, 20, 176)), String(value));
      assert.equal(domain.empty, false);
    }
  });

  await test('deviation axes preserve a true zero baseline, including all-zero data', () => {
    for (const values of [[0, 0], [2, 2], [Number.MAX_VALUE]]) {
      const domain = chartDomain(values, true);
      assert.equal(domain.min, 0);
      assert.equal(chartY(0, domain, 20, 176), 196);
      assert.ok(values.every((value) => Number.isFinite(chartY(value, domain, 20, 176))));
    }
    assert.equal(chartDomain([NaN, undefined, Infinity], true).empty, true);
    assert.equal(chartY(NaN, chartDomain([1]), 20, 176), undefined);
  });

  await test('mixed magnitude domains stay finite across many exponent combinations', () => {
    for (let exponent = -323; exponent <= 308; exponent += 17) {
      const value = 10 ** exponent;
      const domain = chartDomain([-value, 0, value, value / 7]);
      for (const point of [-value, 0, value]) {
        const position = chartY(point, domain, 20, 176);
        assert.ok(
          Number.isFinite(position) && position >= 20 && position <= 196,
          `value=${point}, domain=${JSON.stringify(domain)}`,
        );
      }
    }
  });

  await test('axis labels format finite extremes and retain zero without producing Infinity text', () => {
    assert.equal(chartNumber(0), '0');
    assert.equal(chartNumber(-0), '0');
    assert.equal(chartNumber(undefined), '—');
    assert.equal(chartNumber(NaN), '—');
    assert.equal(chartNumber(Infinity), '—');
    for (const value of [Number.MIN_VALUE, Number.MAX_VALUE, -Number.MAX_VALUE, 1e-20, 1e200]) {
      const text = chartNumber(value);
      assert.doesNotMatch(text, /Infinity|NaN|undefined/);
      assert.ok(text.includes('e'));
      assert.ok(text.length <= 12, text);
    }
  });

  await test('both charts expose distinct accessible titles, descriptions and keyboard/touch instructions', () => {
    const html = render(feature());
    assert.equal([...html.matchAll(/<svg\b[^>]*role="img"[^>]*tabindex="0"/g)].length, 2);
    assert.match(html, /aria-label="state 均值与分位范围图"/);
    assert.match(html, /aria-label="state 标准差图"/);
    assert.equal([...html.matchAll(/<title\b/g)].length, 2);
    assert.equal([...html.matchAll(/<desc\b/g)].length, 2);
    assert.match(html, /左右方向键、Home、End/);
    assert.match(html, /Escape 清除/);
    assert.match(html, /轻触图中维度/);
    assert.match(html, /role="status" aria-live="polite"/);
    for (const [, id] of html.matchAll(/aria-(?:labelledby|describedby)="([^"]+)"/g))
      assert.ok(html.includes(`id="${id}"`));
  });

  await test('large dimension pages draw no more than 32 slots per chart with sparse absolute x ticks', () => {
    const html = render(feature(100), 32, 999);
    const indexes = [...html.matchAll(/data-dimension="(\d+)"/g)].map((match) => Number(match[1]));
    assert.equal(indexes.length, 64);
    assert.equal(Math.min(...indexes), 32);
    assert.equal(Math.max(...indexes), 63);
    assert.match(html, /维度 32 至 63/);
    const text = [...html.matchAll(/<text\b[^>]*y="214"[^>]*>(\d+)<\/text>/g)];
    assert.ok(text.length <= 16, 'each chart has at most eight dimension ticks');
    assert.ok(text.some((match) => match[1] === '32'));
    assert.ok(text.some((match) => match[1] === '63'));
  });

  await test('missing quantiles are stated explicitly and produce no range strokes despite available min/max', () => {
    const html = render(
      feature(3, { q01: undefined, q99: undefined, min: [-2, -2, -2], max: [2, 2, 2] }),
    );
    assert.match(html, /该文件未提供 q01\/q99，仅显示均值/);
    assert.doesNotMatch(html, /class="dataset-normalization-chart-range"/);
    assert.equal([...html.matchAll(/class="dataset-normalization-chart-mean"/g)].length, 3);
    const partial = render(feature(3, { q01: [-2], q99: [2] }));
    assert.match(partial, /部分维度缺少有效的 q01\/q99/);
    assert.equal([...partial.matchAll(/class="dataset-normalization-chart-range"/g)].length, 1);
  });

  await test('true zero deviations use baseline markers and invalid deviations have no plotted glyph', () => {
    const html = render(feature(3, { std: [0, -1, NaN] }));
    assert.equal([...html.matchAll(/class="dataset-normalization-chart-std-zero"/g)].length, 1);
    assert.doesNotMatch(html, /class="dataset-normalization-chart-std"/);
    assert.match(html, /class="dataset-normalization-chart-std-zero"[^>]*cy="196"/);
  });

  await test('extreme, constant and partial arrays never emit invalid SVG numeric coordinates', () => {
    for (const value of [
      feature(3, {
        mean: [-Number.MAX_VALUE, 0, Number.MAX_VALUE],
        std: [0, Number.MAX_VALUE, Number.MIN_VALUE],
        q01: undefined,
        q99: undefined,
      }),
      feature(4, { mean: [7, 7, 7, 7], std: [0, 0, 0, 0] }),
      feature(3, { mean: [NaN, Infinity], std: [-1, NaN], q01: undefined, q99: undefined }),
    ]) {
      const html = render(value);
      assert.doesNotMatch(html, /(?:="|\s)(?:NaN|Infinity|-Infinity)(?:"|\s)/);
      for (const [, attribute, coordinate] of html.matchAll(
        /\b(x|y|cx|cy|x1|x2|y1|y2|width|height|r|rx)="([^"]*)"/g,
      )) {
        assert.ok(Number.isFinite(Number(coordinate)), `${attribute}=${coordinate}`);
        if (['width', 'height', 'r', 'rx'].includes(attribute)) assert.ok(Number(coordinate) >= 0);
      }
    }
  });

  await test('empty windows do not invent a dimension or render a misleading plot', () => {
    const html = render(feature(2), 2, 32);
    assert.match(html, /当前范围没有可显示的维度/);
    assert.doesNotMatch(html, /<svg/);
  });
} finally {
  await server.close();
}
