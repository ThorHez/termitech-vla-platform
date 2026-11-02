import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});
const asset = (id, kind = 'DATASET') => ({
  id,
  name: id,
  kind,
  node_id: 'node',
  path: `/data/${id}`,
  version: 'v1',
});
const single = (id) => ({ kind: 'asset', asset: asset(id) });
const entries = (count) => Array.from({ length: count }, (_, index) => single(`asset-${index}`));
const modelGroup = () => {
  const assets = Array.from({ length: 50 }, (_, index) => asset(`checkpoint-${index}`, 'MODEL'));
  return {
    kind: 'model-group',
    key: 'model-group',
    name: '模型组',
    nodeId: 'node',
    assets,
    latest: assets[49],
  };
};
const datasetGroup = () => {
  const datasets = [asset('data-a'), asset('data-b')];
  const norm = asset('shared-norm', 'NORM');
  return {
    kind: 'dataset-group',
    key: 'dataset-group',
    name: '数据集合',
    nodeId: 'node',
    datasets,
    norm,
    assets: [...datasets, norm],
  };
};

try {
  const { ASSET_CATALOG_PAGE_SIZE, paginateAssetCatalog: paginate } = await server.ssrLoadModule(
    '/src/assetCatalogPagination.ts',
  );

  await test('twenty top-level entries fit one page while the twenty-first begins the next', () => {
    assert.equal(ASSET_CATALOG_PAGE_SIZE, 20);
    const first = paginate(entries(20), 1);
    assert.equal(first.items.length, 20);
    assert.deepEqual(
      {
        page: first.page,
        total: first.total,
        totalPages: first.totalPages,
        start: first.start,
        end: first.end,
        pageSize: first.pageSize,
      },
      { page: 1, total: 20, totalPages: 1, start: 1, end: 20, pageSize: 20 },
    );
    const all = entries(21);
    const second = paginate(all, 2);
    assert.deepEqual(second.items, [all[20]]);
    assert.deepEqual(second.assetIds, ['asset-20']);
    assert.equal(second.totalPages, 2);
    assert.equal(second.start, 21);
    assert.equal(second.end, 21);
  });

  await test('model and shared-normalization groups each occupy one slot and are never split', () => {
    const model = modelGroup(),
      dataset = datasetGroup();
    const all = [...entries(19), model, dataset];
    const first = paginate(all, 1),
      second = paginate(all, 2);
    assert.equal(first.items.length, 20);
    assert.strictEqual(first.items[19], model);
    assert.equal(first.items[19].assets.length, 50);
    assert.deepEqual(first.assetIds, [
      ...entries(19).map((entry) => entry.asset.id),
      ...model.assets.map((asset) => asset.id),
    ]);
    assert.deepEqual(second.items, [dataset]);
    assert.deepEqual(second.assetIds, ['data-a', 'data-b', 'shared-norm']);
    assert.equal(first.total, 21, 'totals count display groups rather than flattened assets');
    assert.equal(second.start, 21);
    assert.ok(!first.assetIds.includes('shared-norm'));
    assert.ok(!second.assetIds.includes('checkpoint-0'));
  });

  await test('page asset IDs contain exactly current-page members and preserve their appearance order', () => {
    const all = entries(43);
    assert.deepEqual(
      paginate(all, 1).assetIds,
      all.slice(0, 20).map((entry) => entry.asset.id),
    );
    assert.deepEqual(
      paginate(all, 2).assetIds,
      all.slice(20, 40).map((entry) => entry.asset.id),
    );
    const last = paginate(all, 3);
    assert.deepEqual(last.assetIds, ['asset-40', 'asset-41', 'asset-42']);
    assert.equal(last.start, 41);
    assert.equal(last.end, 43);
    assert.equal(last.total, 43);
  });

  await test('flattening de-duplicates repeated member IDs without dropping or mutating group entries', () => {
    const dataset = datasetGroup();
    const source = [
      single('data-a'),
      dataset,
      { kind: 'asset', asset: dataset.norm },
      single('last'),
    ];
    const page = paginate(source, 1);
    assert.deepEqual(page.assetIds, ['data-a', 'data-b', 'shared-norm', 'last']);
    assert.equal(page.items.length, 4);
    assert.strictEqual(page.items[1], dataset);
  });

  await test('invalid page numbers return page one while overshooting finite integers clamp to the last page', () => {
    const all = entries(41);
    for (const requested of [0, -1, -100, 1.5, NaN, Infinity, -Infinity, '2', null, undefined]) {
      const page = paginate(all, requested);
      assert.equal(page.page, 1, String(requested));
      assert.equal(page.start, 1);
      assert.equal(page.end, 20);
    }
    for (const requested of [4, 100, Number.MAX_SAFE_INTEGER, Number.MAX_VALUE]) {
      const page = paginate(all, requested);
      assert.equal(page.page, 3, String(requested));
      assert.deepEqual(page.assetIds, ['asset-40']);
      assert.equal(page.start, 41);
      assert.equal(page.end, 41);
    }
  });

  await test('a shrunken filtered catalog clamps a former later page without leaving a blank page', () => {
    const all = entries(65);
    assert.equal(paginate(all, 4).page, 4);
    const shorter = paginate(all.slice(0, 21), 4);
    assert.equal(shorter.page, 2);
    assert.equal(shorter.totalPages, 2);
    assert.deepEqual(shorter.assetIds, ['asset-20']);
    const singlePage = paginate(all.slice(0, 2), 4);
    assert.equal(singlePage.page, 1);
    assert.equal(singlePage.items.length, 2);
  });

  await test('empty results have one empty page and a zero-to-zero display range', () => {
    assert.deepEqual(paginate([], 7), {
      items: [],
      page: 1,
      pageSize: 20,
      total: 0,
      totalPages: 1,
      start: 0,
      end: 0,
      assetIds: [],
    });
  });

  await test('pagination preserves caller arrays, asset objects, groups and ordering', () => {
    const group = datasetGroup();
    const source = [...entries(20), group];
    const before = structuredClone(source);
    source.forEach(Object.freeze);
    Object.freeze(source);
    const page = paginate(source, 2);
    assert.deepEqual(source, before);
    assert.notStrictEqual(page.items, source);
    assert.strictEqual(page.items[0], group);
    assert.strictEqual(page.items[0].assets, group.assets);
  });
} finally {
  await server.close();
}
