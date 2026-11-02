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

try {
  const { DatasetStatistics, DatasetStatisticsView, statisticNumber } = await server.ssrLoadModule(
    '/src/DatasetStatistics.tsx',
  );
  const { emptyDatasetProfile } = await server.ssrLoadModule('/src/DatasetProfileForm.tsx');
  const asset = {
    id: 'dataset',
    node_id: 'source',
    kind: 'DATASET',
    name: '叠衣数据',
    path: '/datasets/cloth',
    version: 'v1',
    state: 'REGISTERED',
    norm_asset_id: 'norm',
  };
  const feature = {
    key: 'state',
    dimension: 2,
    mean: [0.25, 0.5],
    std: [1, 2],
  };
  const report = (changes = {}) => ({
    asset_id: asset.id,
    node_id: asset.node_id,
    status: 'READY',
    dataset: { status: 'READY', total_frames: 1200, total_episodes: 40, fps: 30, camera_count: 1 },
    normalization: { status: 'READY', asset_id: 'norm', features: [feature] },
    ...changes,
  });
  const render = (value = report(), selectedAsset = asset) =>
    renderToStaticMarkup(
      React.createElement(DatasetStatisticsView, { asset: selectedAsset, report: value }),
    );
  const values = (html) => {
    const table = html.match(
      /<table[^>]*aria-label="归一化逐维统计"[^>]*>([\s\S]*?)<\/table>/,
    )?.[1];
    assert.ok(table, 'per-dimension values table should be present');
    return [...table.match(/<tbody>([\s\S]*?)<\/tbody>/)[1].matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map(
      (row) => [...row[1].matchAll(/<td>([^<]*)<\/td>/g)].map((cell) => cell[1]),
    );
  };

  await test('statistics format finite numbers while rejecting nonnumbers and nonfinite values', () => {
    for (const value of [undefined, null, '', '12', NaN, Infinity, -Infinity, {}, []])
      assert.equal(statisticNumber(value), '—');
    assert.equal(statisticNumber(0), '0');
    assert.equal(statisticNumber(-0), '0');
    assert.equal(statisticNumber(-1.23456789), '-1.23457');
    assert.equal(statisticNumber(0.000000123456789), '1.23457e-7');
  });

  await test('partial metadata retains real zeros and leaves missing or invalid counts unknown', () => {
    const html = render(
      report({
        status: 'PARTIAL',
        dataset: { status: 'READY', total_frames: 0, total_episodes: -1, camera_count: 1.5 },
        normalization: { status: 'UNBOUND', features: [] },
      }),
      { ...asset, norm_asset_id: undefined },
    );
    assert.match(html, /总帧数<\/dt><dd>0<\/dd>/);
    assert.match(html, /轨迹数<\/dt><dd>—<\/dd>/);
    assert.match(html, /元信息频率<\/dt><dd>—<\/dd>/);
    assert.match(html, /图像字段数<\/dt><dd>—<\/dd>/);
    assert.match(html, /未绑定 norm_stats，暂时只展示数据集元信息/);
    assert.doesNotMatch(html, /aria-label="归一化逐维统计"/);
  });

  await test('optional quantiles render unknown cells without inventing values or dropping valid means', () => {
    assert.deepEqual(values(render()), [
      ['0', '0.25', '1', '—', '—'],
      ['1', '0.5', '2', '—', '—'],
    ]);
    const withBounds = { ...feature, q01: [-1, -2], q99: [1, 2], min: [-3, -4], max: [3, 4] };
    const html = render(
      report({ normalization: { status: 'READY', asset_id: 'norm', features: [withBounds] } }),
    );
    assert.deepEqual(values(html)[0], ['0', '0.25', '1', '-1', '1', '-3', '3']);
    assert.match(html, /最小值 min/);
    assert.match(html, /最大值 max/);
  });

  await test('a failed normalization read preserves metadata and explains the unavailable values', () => {
    const html = render(
      report({
        status: 'PARTIAL',
        normalization: {
          status: 'FAILED',
          asset_id: 'norm',
          error: { code: 'BAD_NORM', message: '归一化文件不是有效 JSON' },
        },
      }),
    );
    assert.match(html, /总帧数<\/dt><dd>1,200<\/dd>/);
    assert.match(html, /归一化文件不是有效 JSON/);
    assert.doesNotMatch(html, /aria-label="归一化逐维统计"/);
  });

  await test('a changed normalization binding hides the old report and its feature selector', () => {
    const html = render(report(), { ...asset, norm_asset_id: 'replacement-norm' });
    assert.match(html, /norm_stats 绑定已变化，请刷新统计/);
    assert.match(html, /总帧数<\/dt><dd>1,200<\/dd>/);
    assert.doesNotMatch(html, /aria-label="归一化逐维统计"|aria-label="统计字段"|<td>0\.25<\/td>/);
    assert.doesNotMatch(
      render(report({ normalization: { status: 'READY', features: [feature] } })),
      /aria-label="归一化逐维统计"|aria-label="统计字段"/,
      'a report without the bound normalization ID cannot authorize displaying its values',
    );
  });

  await test('metadata FPS conflicts remain visible while the registered training rate stays authoritative', () => {
    const profile = {
      ...emptyDatasetProfile(),
      robot_type: 'umi',
      task_description: 'fold cloth',
      fps: 20,
      cameras: [{ name: 'top', key: 'observation.images.top', kind: 'rgb' }],
    };
    const html = render(report(), { ...asset, metadata: { dataset_profile: profile } });
    assert.match(html, /登记频率为 20 Hz，元信息为 30 Hz；训练沿用登记配置/);
    assert.match(html, /元信息频率<\/dt><dd>30 Hz<\/dd>/);
    assert.doesNotMatch(
      render(report({ dataset: { status: 'READY', fps: 20 } }), {
        ...asset,
        metadata: { dataset_profile: profile },
      }),
      /登记频率为/,
    );
  });

  await test('large normalization vectors show at most 32 dimensions with accurate paging boundaries', () => {
    const wide = {
      ...feature,
      dimension: 65,
      mean: Array.from({ length: 65 }, (_, index) => index + 100),
      std: Array(65).fill(1),
    };
    const html = render(
      report({ normalization: { status: 'READY', asset_id: 'norm', features: [wide] } }),
    );
    const rows = values(html);
    assert.equal(rows.length, 32);
    assert.equal(rows[0][0], '0');
    assert.equal(rows.at(-1)[0], '31');
    assert.match(html, /第 1–32 维 \/ 共 65 维/);
    assert.match(html, /<button[^>]*disabled=""[^>]*>上一页<\/button>/);
    assert.match(html, /<button(?![^>]*disabled)[^>]*>下一页<\/button>/);
    const small = render();
    assert.equal(values(small).length, 2);
    assert.doesNotMatch(small, /下一页|上一页/);
  });

  await test('offline datasets explain why live statistics cannot load and disable refresh', () => {
    const html = renderToStaticMarkup(
      React.createElement(DatasetStatistics, { asset, node: { id: 'source', status: 'OFFLINE' } }),
    );
    assert.match(html, /机器离线，恢复连接后可读取统计/);
    assert.match(html, /<button[^>]*disabled=""/);
    assert.doesNotMatch(html, /正在读取已有统计|aria-label="归一化逐维统计"/);
  });
} finally {
  await server.close();
}
