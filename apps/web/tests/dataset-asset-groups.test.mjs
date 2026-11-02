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
const entries = (assets) => assets.map(entry);
const fixtures = () => {
  const norm = asset('normalization', 'NORM');
  const data = asset('dataset', 'DATASET', { norm_asset_id: norm.id });
  return { norm, data };
};

try {
  const { groupDatasetNormAssets: group } = await server.ssrLoadModule(
    '/src/datasetAssetGroups.ts',
  );

  await test('one explicit same-node binding produces a dataset-first group with stable identity', () => {
    const { data, norm } = fixtures();
    const rows = group(entries([data, norm]));
    assert.equal(rows.length, 1);
    assert.equal(rows[0].kind, 'dataset-group');
    assert.equal(rows[0].name, data.name);
    assert.equal(rows[0].nodeId, data.node_id);
    assert.deepEqual(rows[0].datasets, [data]);
    assert.strictEqual(rows[0].norm, norm);
    assert.deepEqual(rows[0].assets, [data, norm]);
    assert.ok(rows[0].key.startsWith('dataset-norm:'));
    assert.equal(group(entries([norm, data]))[0].key, rows[0].key);
  });

  await test('shared norms combine all matching datasets exactly once in filtered dataset order', () => {
    const { data, norm } = fixtures();
    const second = asset('dataset-2', 'DATASET', { norm_asset_id: norm.id });
    const third = asset('dataset-3', 'DATASET', { norm_asset_id: norm.id });
    const rows = group(entries([second, norm, data, third]));
    assert.equal(rows.length, 1);
    assert.equal(rows[0].name, `${norm.name} · 3 个数据集`);
    assert.deepEqual(rows[0].datasets, [second, data, third]);
    assert.deepEqual(rows[0].assets, [second, data, third, norm]);
    assert.equal(rows[0].assets.filter((item) => item.id === norm.id).length, 1);
  });

  await test('groups occupy their earliest member position even when the norm appeared first', () => {
    const { data, norm } = fixtures();
    const before = asset('before', 'MODEL');
    const between = asset('between', 'DATASET');
    const after = asset('after', 'NORM');
    const rows = group(entries([before, norm, between, data, after]));
    assert.deepEqual(
      rows.map((item) => (item.kind === 'asset' ? item.asset.id : 'group')),
      ['before', 'group', 'between', 'after'],
    );
    assert.deepEqual(rows[1].assets, [data, norm]);
  });

  await test('distinct bindings remain separate and retain independent first-member ordering', () => {
    const left = fixtures();
    const rightNorm = asset('right-norm', 'NORM');
    const rightData = asset('right-data', 'DATASET', { norm_asset_id: rightNorm.id });
    const rows = group(entries([left.norm, rightData, rightNorm, left.data]));
    assert.deepEqual(
      rows.map((row) => row.norm.id),
      [left.norm.id, rightNorm.id],
    );
    assert.notEqual(rows[0].key, rows[1].key);
  });

  await test('missing, empty and unknown bindings never group by name, metadata or matching paths', () => {
    const norm = asset('norm', 'NORM', { name: 'matching', path: '/data/shared/norm_stats.json' });
    const data = asset('data', 'DATASET', {
      name: 'matching',
      path: '/data/shared',
      metadata: { norm_asset_id: norm.id },
    });
    const assets = [
      data,
      norm,
      asset('null', 'DATASET', { norm_asset_id: null }),
      asset('empty', 'DATASET', { norm_asset_id: '' }),
      asset('missing', 'DATASET', { norm_asset_id: 'gone' }),
    ];
    const source = entries(assets);
    assert.deepEqual(group(source), source);
  });

  await test('wrong-kind and cross-machine targets stay as independent assets', () => {
    const wrong = asset('wrong', 'MODEL');
    const remote = asset('remote', 'NORM', { node_id: 'node-2' });
    const assets = [
      asset('wrong-data', 'DATASET', { norm_asset_id: wrong.id }),
      wrong,
      asset('remote-data', 'DATASET', { norm_asset_id: remote.id }),
      remote,
    ];
    const source = entries(assets);
    assert.deepEqual(group(source), source);
  });

  await test('kind and search filtering do not pull hidden norms or datasets back into results', () => {
    const { data, norm } = fixtures();
    const second = asset('other-data', 'DATASET', { norm_asset_id: norm.id });
    assert.deepEqual(group(entries([data])), entries([data]));
    assert.deepEqual(group(entries([norm])), entries([norm]));
    const rows = group(entries([data, norm]));
    assert.deepEqual(rows[0].assets, [data, norm]);
    assert.ok(!rows[0].assets.some((asset) => asset.id === second.id));
  });

  await test('archive visibility belongs to the caller and never reintroduces excluded members', () => {
    const { data, norm } = fixtures();
    const archivedNorm = { ...norm, state: 'ARCHIVED' };
    const all = entries([data, archivedNorm]);
    assert.deepEqual(group(all)[0].assets, [data, archivedNorm]);
    const live = all.filter((item) => item.asset.state !== 'ARCHIVED');
    assert.deepEqual(group(live), entries([data]));
    const archivedData = { ...data, state: 'ARCHIVED' };
    const normOnly = entries([archivedData, norm]).filter(
      (item) => item.asset.state !== 'ARCHIVED',
    );
    assert.deepEqual(group(normOnly), entries([norm]));
  });

  await test('shared norm grouping only contains visible dataset subset and leaves the other dataset excluded', () => {
    const { data, norm } = fixtures();
    const another = asset('another', 'DATASET', { norm_asset_id: norm.id });
    const archived = asset('archived', 'DATASET', { norm_asset_id: norm.id, state: 'ARCHIVED' });
    const visible = [data, another, archived, norm].filter((asset) => asset.state !== 'ARCHIVED');
    const rows = group(entries(visible));
    assert.deepEqual(rows[0].datasets, [data, another]);
    assert.deepEqual(rows[0].assets, [data, another, norm]);
  });

  await test('model groups are preserved by identity and are never flattened or consumed as norm targets', () => {
    const { data, norm } = fixtures();
    const model = asset('model', 'MODEL');
    const modelGroup = {
      kind: 'model-group',
      key: 'model-key',
      name: '模型组',
      nodeId: model.node_id,
      assets: [model],
      latest: model,
    };
    const rows = group([entry(norm), modelGroup, entry(data)]);
    assert.strictEqual(rows[1], modelGroup);
    assert.equal(rows[0].kind, 'dataset-group');
    const wrongData = asset('wrong-data', 'DATASET', { norm_asset_id: model.id });
    assert.deepEqual(group([entry(wrongData), modelGroup]), [entry(wrongData), modelGroup]);
  });

  await test('group construction leaves input wrappers, asset objects and arrays unchanged', () => {
    const { data, norm } = fixtures();
    const standalone = asset('alone');
    const source = entries([norm, standalone, data]);
    const before = structuredClone(source);
    source.forEach((item) => {
      Object.freeze(item.asset);
      Object.freeze(item);
    });
    Object.freeze(source);
    const rows = group(source);
    assert.deepEqual(source, before);
    assert.strictEqual(rows[0].norm, norm);
    assert.strictEqual(rows[0].datasets[0], data);
    assert.strictEqual(rows[1], source[1]);
    assert.notStrictEqual(rows, source);
  });

  await test('empty and fully unbound catalogs preserve the original entry sequence', () => {
    assert.deepEqual(group([]), []);
    const source = entries([asset('data'), asset('norm', 'NORM'), asset('model', 'MODEL')]);
    const rows = group(source);
    assert.deepEqual(rows, source);
    assert.ok(rows.every((item, index) => item === source[index]));
  });
} finally {
  await server.close();
}
