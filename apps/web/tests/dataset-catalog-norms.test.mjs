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
const asset = (id, changes = {}) => ({
  id,
  name: id,
  node_id: 'source',
  kind: 'DATASET',
  path: `/data/${id}`,
  version: 'v1',
  status: 'REGISTERED',
  ...changes,
});
const norm = asset('norm', { kind: 'NORM', path: '/data/norm_stats.json' });
const dataset = asset('dataset', { norm_asset_id: norm.id });
const ids = (assets) => assets.map((item) => item.id);

try {
  const { datasetCatalogAssets: project, groupDatasetNormAssets } = await server.ssrLoadModule(
    '/src/datasetAssetGroups.ts',
  );
  const { catalogAssetGroups } = await server.ssrLoadModule('/src/modelAssetGroups.ts');
  const { paginateAssetCatalog } = await server.ssrLoadModule('/src/assetCatalogPagination.ts');
  const { reconcileBatchSyncSelection, toggleBatchSyncVisible, batchSyncRows } =
    await server.ssrLoadModule('/src/batchAssetSyncData.ts');

  await test('explicit same-machine bindings live under a dataset parent instead of peer catalog rows', () => {
    const raw = [dataset, norm, asset('independent', { kind: 'NORM' })];
    const before = structuredClone(raw);
    const projection = project(raw);
    assert.deepEqual(ids(projection.assets), ['dataset', 'independent']);
    assert.strictEqual(projection.boundNorms.get(dataset.id), norm);
    assert.deepEqual([...projection.embeddedNormIds], ['norm']);
    assert.deepEqual(
      groupDatasetNormAssets(catalogAssetGroups(projection.assets, raw), projection.boundNorms).map(
        (entry) => entry.kind,
      ),
      ['dataset-group', 'asset'],
    );
    assert.deepEqual(raw, before, 'catalog projection never changes full API or selector assets');
    assert.strictEqual(projection.assets[0], dataset);
    assert.equal(
      batchSyncRows([dataset], raw)[0].includeNormStats,
      true,
      'dataset sync still includes its bound norm',
    );
  });

  await test('only current valid bindings hide norms, never output provenance or invalid references', () => {
    for (const owner of [
      asset('dataset'),
      asset('dataset', { norm_asset_id: 'missing' }),
      asset('dataset', { norm_asset_id: norm.id, node_id: 'other' }),
    ]) {
      const sourceNorm = { ...norm, metadata: { source_dataset_asset_id: owner.id } };
      assert.deepEqual(ids(project([owner, sourceNorm]).assets), [owner.id, norm.id]);
    }
    const model = asset('model', { kind: 'MODEL' });
    const projection = project([asset('dataset', { norm_asset_id: model.id }), model, norm]);
    assert.deepEqual(ids(projection.assets), ['dataset', 'model', 'norm']);
    assert.equal(projection.boundNorms.size, 0);
  });

  await test('a shared norm remains available under every distinct dataset parent', () => {
    const second = asset('second', { norm_asset_id: norm.id });
    const raw = [dataset, norm, second];
    const projection = project(raw);
    assert.deepEqual(ids(projection.assets), ['dataset', 'second']);
    assert.strictEqual(projection.boundNorms.get(dataset.id), norm);
    assert.strictEqual(projection.boundNorms.get(second.id), norm);
    assert.equal(projection.embeddedNormIds.size, 1);
    assert.deepEqual(
      groupDatasetNormAssets(catalogAssetGroups(projection.assets, raw), projection.boundNorms).map(
        (entry) => entry.datasets[0].id,
      ),
      ['dataset', 'second'],
    );
  });

  await test('archive visibility decides adoption before other filters and never strands a standalone norm', () => {
    const archived = { ...dataset, status: 'ARCHIVED' };
    assert.deepEqual(ids(project([archived, norm]).assets), ['norm']);
    assert.deepEqual(ids(project([archived, norm], true).assets), ['dataset']);
    const visibleOwner = asset('visible', { norm_asset_id: norm.id });
    assert.deepEqual(ids(project([archived, norm, visibleOwner]).assets), ['visible']);
    const legacyArchived = { ...dataset, status: undefined, state: 'ARCHIVED' };
    assert.deepEqual(ids(project([legacyArchived, norm]).assets), ['norm']);
    for (const status of ['UNAVAILABLE', 'ARCHIVED']) {
      const projection = project([dataset, { ...norm, status }], true);
      assert.deepEqual(ids(projection.assets), ['dataset']);
      assert.equal(projection.boundNorms.get(dataset.id).status, status);
    }
  });

  await test('kind and search filters consume the projection without resurrecting bound norm rows', () => {
    const standalone = asset('independent', { kind: 'NORM' });
    const projection = project([dataset, norm, standalone]);
    assert.deepEqual(ids(projection.assets.filter((item) => item.kind === 'NORM')), [
      'independent',
    ]);
    for (const query of [norm.name, norm.path]) {
      const matches = projection.assets.filter((item) => {
        const bound = projection.boundNorms.get(item.id);
        return `${item.name} ${item.path} ${bound?.name || ''} ${bound?.path || ''}`.includes(
          query,
        );
      });
      assert.deepEqual(ids(matches), ['dataset']);
    }
  });

  await test('explicit norm selections remain across pages, kinds, machines and dataset binding changes', () => {
    const otherPage = asset('other-page', { kind: 'MODEL', node_id: 'other' });
    const nodes = [
      { id: 'source', status: 'ONLINE' },
      { id: 'other', status: 'ONLINE' },
    ];
    const raw = [dataset, norm, otherPage];
    const selected = reconcileBatchSyncSelection(
      ['norm', 'dataset', 'other-page'],
      project(raw).selectionAssets,
      nodes,
    );
    assert.deepEqual(selected, ['norm', 'dataset', 'other-page']);
    assert.deepEqual(toggleBatchSyncVisible(selected, ['dataset'], false), ['norm', 'other-page']);
    assert.deepEqual(toggleBatchSyncVisible(['other-page'], ['dataset'], true), [
      'other-page',
      'dataset',
    ]);
    const unbound = [{ ...dataset, norm_asset_id: null }, norm, otherPage];
    assert.deepEqual(
      reconcileBatchSyncSelection(['norm', 'other-page'], project(unbound).selectionAssets, nodes),
      ['norm', 'other-page'],
    );
  });

  await test('pagination counts dataset parents and model groups while exposing selectable real children', () => {
    const checkpoints = [1, 2].map((step) =>
      asset(`checkpoint-${step}`, {
        kind: 'MODEL',
        origin_attempt_id: 'attempt',
        metadata: { step },
      }),
    );
    const raw = [dataset, norm, asset('independent', { kind: 'NORM' }), ...checkpoints];
    const projection = project(raw);
    const entries = groupDatasetNormAssets(
      catalogAssetGroups(projection.assets, raw),
      projection.boundNorms,
    );
    assert.deepEqual(
      entries.map((entry) => entry.kind),
      ['dataset-group', 'asset', 'model-group'],
    );
    const page = paginateAssetCatalog(entries, 1);
    assert.equal(page.total, 3);
    assert.deepEqual(page.assetIds, [
      'dataset',
      'norm',
      'independent',
      'checkpoint-2',
      'checkpoint-1',
    ]);
  });
} finally {
  await server.close();
}
