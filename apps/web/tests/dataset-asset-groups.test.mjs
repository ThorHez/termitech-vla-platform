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
const asset = (id, kind = 'DATASET', changes = {}) => ({
  id,
  name: id,
  kind,
  node_id: 'node-1',
  path: `/data/${id}`,
  version: 'v1',
  state: 'REGISTERED',
  ...changes,
});
const entry = (asset) => ({ kind: 'asset', asset });
const norm = asset('normalization', 'NORM', { path: '/stats/norm_stats.json' });
const data = asset('dataset', 'DATASET', { norm_asset_id: norm.id });
const second = asset('dataset-2', 'DATASET', { norm_asset_id: norm.id });

try {
  const { datasetCatalogAssets: project, groupDatasetNormAssets: nest } =
    await server.ssrLoadModule('/src/datasetAssetGroups.ts');
  const { paginateAssetCatalog } = await server.ssrLoadModule('/src/assetCatalogPagination.ts');
  const { reconcileBatchSyncSelection, toggleBatchSyncVisible } = await server.ssrLoadModule(
    '/src/batchAssetSyncData.ts',
  );
  const { AssetGroupRows } = await server.ssrLoadModule('/src/AssetGroupRows.tsx');
  const catalog = (assets, filter = () => true, showArchived = false) => {
    const projection = project(assets, showArchived);
    return nest(projection.assets.filter(filter).map(entry), projection.boundNorms);
  };

  await test('each dataset owns a stable parent containing exactly its two real assets', () => {
    const rows = catalog([data, norm]);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].kind, 'dataset-group');
    assert.equal(rows[0].name, data.name);
    assert.equal(rows[0].nodeId, data.node_id);
    assert.deepEqual(rows[0].datasets, [data]);
    assert.strictEqual(rows[0].norm, norm);
    assert.deepEqual(rows[0].assets, [data, norm]);
    assert.equal(catalog([norm, data])[0].key, rows[0].key);
    const rebound = { ...data, norm_asset_id: 'new-norm' };
    assert.equal(catalog([rebound, asset('new-norm', 'NORM')])[0].key, rows[0].key);
  });

  await test('shared normalization produces independent named dataset parents in dataset order', () => {
    const rows = catalog([second, norm, data]);
    assert.equal(rows.length, 2);
    assert.deepEqual(
      rows.map((row) => row.name),
      [second.name, data.name],
    );
    assert.notEqual(rows[0].key, rows[1].key);
    assert.deepEqual(rows[0].assets, [second, norm]);
    assert.deepEqual(rows[1].assets, [data, norm]);
    assert.deepEqual(
      rows.map((row) => row.datasets.length),
      [1, 1],
    );
  });

  await test('parent positions belong to datasets, not an earlier norm or another dataset sharing it', () => {
    const before = asset('before', 'MODEL');
    const between = asset('between');
    const rows = catalog([before, norm, between, second, data]);
    assert.deepEqual(
      rows.map((row) => (row.kind === 'asset' ? row.asset.id : row.name)),
      ['before', 'between', second.name, data.name],
    );
  });

  await test('binding targets must be explicit same-machine norms; provenance and names never suffice', () => {
    for (const candidate of [
      { ...data, norm_asset_id: null },
      { ...data, norm_asset_id: 'missing' },
      { ...data, node_id: 'other' },
      { ...data, norm_asset_id: 'model' },
    ]) {
      const rows = catalog([
        candidate,
        { ...norm, metadata: { source_dataset_asset_id: candidate.id } },
        asset('model', 'MODEL'),
      ]);
      assert.ok(rows.every((row) => row.kind === 'asset'));
      assert.ok(rows.some((row) => row.asset.id === norm.id));
    }
  });

  await test('dataset and query filters retain children under matching parents without standalone norm resurrection', () => {
    const raw = [data, second, norm, asset('standalone', 'NORM')];
    assert.deepEqual(
      catalog(raw, (item) => item.kind === 'NORM').map((row) => row.asset.id),
      ['standalone'],
    );
    const rows = catalog(raw, (item) => item.id === data.id);
    assert.deepEqual(rows[0].assets, [data, norm]);
    assert.equal(rows.length, 1);
    assert.ok(!rows[0].assets.includes(second));
  });

  await test('archive visibility keeps valid owners and standalone fallback without reintroducing archived rows', () => {
    const archived = { ...data, status: 'ARCHIVED' };
    assert.deepEqual(
      catalog([archived, norm]).map((row) => row.asset.id),
      [norm.id],
    );
    assert.deepEqual(catalog([archived, norm], undefined, true)[0].assets, [archived, norm]);
    assert.deepEqual(catalog([data, { ...norm, status: 'ARCHIVED' }]), [entry(data)]);
  });

  await test('model group wrappers and caller objects remain unchanged', () => {
    const model = asset('model', 'MODEL');
    const modelGroup = {
      kind: 'model-group',
      key: 'model-key',
      name: '模型组',
      nodeId: model.node_id,
      assets: [model],
      latest: model,
    };
    const source = [entry(data), modelGroup];
    const before = structuredClone(source);
    Object.freeze(source[0]);
    Object.freeze(source);
    const rows = nest(source, new Map([[data.id, norm]]));
    assert.deepEqual(source, before);
    assert.strictEqual(rows[1], modelGroup);
    assert.strictEqual(rows[0].assets[0], data);
    assert.strictEqual(rows[0].assets[1], norm);
    assert.deepEqual(nest([], new Map()), []);
  });

  await test('shared norm children are selectable once globally and survive filtering and pagination', () => {
    const fillers = Array.from({ length: 19 }, (_, i) => asset(`filler-${i}`));
    const raw = [data, norm, ...fillers, second];
    const rows = catalog(raw);
    const first = paginateAssetCatalog(rows, 1);
    const next = paginateAssetCatalog(rows, 2);
    assert.equal(first.total, 21);
    assert.deepEqual(next.items[0].assets, [second, norm]);
    assert.deepEqual(next.assetIds, [second.id, norm.id]);
    const selected = toggleBatchSyncVisible(first.assetIds, next.assetIds, true);
    assert.equal(selected.filter((id) => id === norm.id).length, 1);
    assert.deepEqual(
      reconcileBatchSyncSelection([norm.id, data.id, second.id], project(raw).selectionAssets, [
        { id: 'node-1', status: 'ONLINE' },
      ]),
      [norm.id, data.id, second.id],
    );
    assert.deepEqual(toggleBatchSyncVisible([norm.id, data.id], [data.id], false), [norm.id]);
  });

  const render = (group, selectedIds = []) =>
    renderToStaticMarkup(
      React.createElement(
        'table',
        null,
        React.createElement(AssetGroupRows, {
          group,
          node: { name: 'source' },
          selectedIds,
          eligibleIds: group.assets.map((item) => item.id),
          canSelect: true,
          onSelectionChange() {},
          viewDataset() {},
          renderAsset: (item) =>
            React.createElement(
              'tr',
              { key: item.id },
              React.createElement(
                'td',
                null,
                React.createElement(
                  'button',
                  { 'aria-label': `查看资产：${item.name}` },
                  item.name,
                ),
                React.createElement('code', null, item.path),
              ),
            ),
        }),
      ),
    );

  await test('dataset parent exposes disclosure and dataset identity while both real child assets remain available', () => {
    const group = catalog([data, { ...norm, status: 'UNAVAILABLE' }])[0];
    const html = render(group);
    const heading = html.match(/<tbody[^>]*>([\s\S]*?)<\/tbody>/)[1];
    assert.match(heading, /aria-label="展开数据集 dataset"/);
    assert.match(heading, /aria-expanded="false"/);
    assert.match(heading, /查看数据集统计 dataset/);
    assert.ok(heading.includes(data.path));
    assert.ok(heading.includes(data.version));
    assert.match(heading, />数据集<|>数据集<!--/);
    assert.match(heading, /已登记/);
    assert.doesNotMatch(heading, /UNAVAILABLE|数据集集合|可用 1/);
    assert.match(html, /asset-dataset-group-items[^>]*hidden=""/);
    for (const item of [data, norm]) {
      assert.ok(html.includes(`aria-label="查看资产：${item.name}"`));
      assert.ok(html.includes(item.path));
    }
  });

  await test('parent checkbox represents only the actual dataset and never silently selects its norm child', () => {
    const group = catalog([data, norm])[0];
    assert.doesNotMatch(render(group, [norm.id]).match(/<input\b[^>]*>/)[0], /checked/);
    assert.match(render(group, [data.id]).match(/<input\b[^>]*>/)[0], /checked/);
    assert.match(render(group), /归一化参数可展开单独选择/);
  });
} finally {
  await server.close();
}
