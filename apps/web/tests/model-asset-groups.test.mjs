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
const model = (id, step = 100, changes = {}) => ({
  id,
  kind: 'MODEL',
  node_id: 'train-node',
  name: `recipe / ${step}`,
  version: String(step),
  path: `/outputs/attempt/checkpoints/recipe/platform/${step}`,
  origin_attempt_id: 'attempt-1',
  state: 'READY',
  metadata: { format: 'openpi-orbax', recipe: 'recipe', step },
  created_at: '2026-11-02T00:00:00Z',
  ...changes,
});
const run = (id, attempts = ['attempt-1'], changes = {}) => ({
  id,
  name: `训练任务 ${id}`,
  node_id: 'train-node',
  attempts: attempts.map((id) => ({ id })),
  ...changes,
});
const copy = (id, original, changes = {}) =>
  model(id, undefined, {
    node_id: 'target-node',
    name: original.name,
    origin_attempt_id: null,
    metadata: {
      format: 'openpi-orbax',
      asset_sync: {
        byte_verified: true,
        source_asset_id: original.id,
        source_node_id: original.node_id,
      },
    },
    ...changes,
  });

try {
  const { catalogAssetGroups: group, checkpointStep } = await server.ssrLoadModule(
    '/src/modelAssetGroups.ts',
  );

  await test('same execution checkpoints collapse without needing loaded run history', () => {
    const assets = [model('low', 9000), model('high', 10000)];
    const rows = group(assets, assets);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].kind, 'model-group');
    assert.equal(rows[0].name, 'recipe');
    assert.equal(rows[0].nodeId, 'train-node');
    assert.equal(rows[0].latest.id, 'high');
    assert.equal(rows[0].latestStep, 10000);
    assert.deepEqual(
      rows[0].assets.map((asset) => asset.id),
      ['high', 'low'],
    );
  });

  await test('one run can combine checkpoint outputs across retries while retaining the real run name', () => {
    const assets = [model('a', 100), model('b', 200, { origin_attempt_id: 'attempt-2' })];
    const rows = group(assets, assets, [run('run-1', ['attempt-1', 'attempt-2'])]);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].name, '训练任务 run-1');
    assert.equal(rows[0].runId, 'run-1');
    assert.deepEqual(
      rows[0].assets.map((asset) => asset.id),
      ['b', 'a'],
    );
  });

  await test('identical names, recipes and model versions do not merge different training runs', () => {
    const assets = [
      model('a', 100),
      model('b', 200),
      model('c', 100, { origin_attempt_id: 'attempt-2' }),
      model('d', 200, { origin_attempt_id: 'attempt-2' }),
    ];
    const rows = group(assets, assets, [run('run-1'), run('run-2', ['attempt-2'])]);
    assert.equal(rows.length, 2);
    assert.notEqual(rows[0].key, rows[1].key);
    assert.deepEqual(
      rows.map((row) => row.assets.map((asset) => asset.id)),
      [
        ['b', 'a'],
        ['d', 'c'],
      ],
    );
  });

  await test('current node is always part of group identity even when provenance is identical', () => {
    const assets = [
      model('a', 100),
      model('b', 200),
      model('c', 100, { node_id: 'other-node' }),
      model('d', 200, { node_id: 'other-node' }),
    ];
    const rows = group(assets, assets);
    assert.equal(rows.length, 2);
    assert.deepEqual(
      rows.map((row) => row.nodeId),
      ['train-node', 'other-node'],
    );
    assert.notEqual(rows[0].key, rows[1].key);
  });

  await test('assets without training provenance remain independent, including matching name suffixes', () => {
    const assets = [
      model('a', 100, { origin_attempt_id: null }),
      model('b', 200, { origin_attempt_id: null }),
      model('dataset', 100, { kind: 'DATASET' }),
      model('norm', 200, { kind: 'NORM' }),
    ];
    const rows = group(assets, assets);
    assert.deepEqual(
      rows,
      assets.map((asset) => ({ kind: 'asset', asset })),
    );
  });

  await test('filtered subsets never bring hidden checkpoints back into the visible results', () => {
    const assets = [model('a', 100), model('b', 200), model('c', 300)];
    assert.deepEqual(group([assets[1]], assets), [{ kind: 'asset', asset: assets[1] }]);
    const rows = group(assets.slice(0, 2), assets);
    assert.deepEqual(
      rows[0].assets.map((asset) => asset.id),
      ['b', 'a'],
    );
    assert.equal(rows[0].latestStep, 200);
  });

  await test('group insertion retains the first matching position and leaves unrelated rows in order', () => {
    const standalone = model('base', 500, { origin_attempt_id: null });
    const dataset = model('data', 0, { kind: 'DATASET' });
    const assets = [standalone, model('a', 100), dataset, model('b', 200)];
    const before = structuredClone(assets);
    const rows = group(assets, assets);
    assert.deepEqual(
      rows.map((row) => (row.kind === 'asset' ? row.asset.id : 'group')),
      ['base', 'group', 'data'],
    );
    assert.deepEqual(assets, before);
    assert.notStrictEqual(rows[1].assets, assets);
  });

  await test('known steps sort numerically with unknowns last, never using ordinary version numbers', () => {
    const assets = [
      model('two', 2),
      model('unknown', 0, { metadata: {}, version: '99999999' }),
      model('ten', '10'),
      model('nine', 9),
      model('zero', 0),
    ];
    const rows = group(assets, assets);
    assert.deepEqual(
      rows[0].assets.map((asset) => asset.id),
      ['ten', 'nine', 'two', 'zero', 'unknown'],
    );
    assert.equal(rows[0].latestStep, 10);
    assert.equal(checkpointStep(assets[1]), undefined);
  });

  await test('invalid and malformed sample metadata cannot become valid checkpoint steps', () => {
    for (const step of [
      -1,
      1.2,
      NaN,
      Infinity,
      true,
      null,
      {},
      [],
      '1e3',
      '-2',
      '2.5',
      ' 2 ',
      '9007199254740992',
    ])
      assert.equal(
        checkpointStep(model('bad', 0, { metadata: { step } })),
        undefined,
        String(step),
      );
    for (const metadata of [null, [], 'bad', 4])
      assert.equal(checkpointStep(model('bad', 0, { metadata })), undefined);
    assert.equal(checkpointStep(model('zero', '000')), 0);
    assert.equal(checkpointStep(model('safe', Number.MAX_SAFE_INTEGER)), Number.MAX_SAFE_INTEGER);
  });

  await test('archived membership is controlled only by the caller filtered assets', () => {
    const assets = [model('ready', 100), model('old', 50, { state: 'ARCHIVED' })];
    assert.equal(group(assets, assets)[0].assets.length, 2);
    const visible = assets.filter((asset) => asset.state !== 'ARCHIVED');
    assert.deepEqual(group(visible, assets), [{ kind: 'asset', asset: assets[0] }]);
  });

  await test('verified copied models trace their source execution without merging source and target nodes', () => {
    const originals = [model('a', 100), model('b', 200)];
    const copies = originals.map((asset) => copy(`copy-${asset.id}`, asset));
    const all = [...copies, ...originals];
    const rows = group(all, all, [run('run-1')]);
    assert.equal(rows.length, 2);
    assert.deepEqual(
      rows.map((row) => row.nodeId),
      ['target-node', 'train-node'],
    );
    assert.deepEqual(
      rows[0].assets.map((asset) => asset.id),
      ['copy-b', 'copy-a'],
    );
    assert.equal(rows[0].latestStep, 200);
    assert.equal(rows[0].name, '训练任务 run-1');
  });

  await test('copy chains can resolve across multiple nodes, but missing or circular sources remain separate', () => {
    const original = model('a', 100);
    const first = copy('first', original);
    const second = copy('second', first, { node_id: 'third-node' });
    const other = model('b', 200);
    const otherCopy = copy('other-second', other, { node_id: 'third-node' });
    const all = [original, first, second, other, otherCopy];
    const rows = group([second, otherCopy], all);
    assert.equal(rows[0].kind, 'model-group');
    const missing = copy('missing', model('missing-source'));
    const cycleA = copy('cycle-a', { id: 'cycle-b', node_id: 'target-node', name: 'cycle' });
    const cycleB = copy('cycle-b', cycleA);
    const bad = [missing, cycleA, cycleB];
    assert.deepEqual(
      group(bad, bad),
      bad.map((asset) => ({ kind: 'asset', asset })),
    );
  });

  await test('copy proofs require verification and an exact source-node match', () => {
    const originals = [model('a', 100), model('b', 200)];
    const invalid = [copy('wrong-node', originals[0]), copy('unverified', originals[1])];
    invalid[0].metadata.asset_sync.source_node_id = 'wrong-node';
    invalid[1].metadata.asset_sync.byte_verified = false;
    assert.deepEqual(
      group(invalid, [...originals, ...invalid]),
      invalid.map((asset) => ({ kind: 'asset', asset })),
    );
  });

  await test('dedicated checkpoint transfers use their explicit original run and source machine', () => {
    const transferred = (id, sourceNode = 'train-node') =>
      model(id, 0, {
        origin_attempt_id: null,
        node_id: 'target-node',
        version: 'manifest-digest',
        metadata: {
          format: 'openpi-transferred',
          source_run_id: 'run-1',
          source_node_id: sourceNode,
        },
      });
    const assets = [transferred('a'), transferred('b')];
    const rows = group(assets, assets, [run('run-1')]);
    assert.equal(rows[0].kind, 'model-group');
    assert.equal(rows[0].runId, 'run-1');
    assert.equal(rows[0].latestStep, undefined);
    const mismatch = [transferred('a', 'other-source'), transferred('b', 'other-source')];
    assert.ok(group(mismatch, mismatch, [run('run-1')]).every((row) => row.kind === 'asset'));
    const plain = assets.map((asset) => ({
      ...asset,
      metadata: { ...asset.metadata, format: 'user-model' },
    }));
    assert.ok(group(plain, plain, [run('run-1')]).every((row) => row.kind === 'asset'));
  });

  await test('same-step retries prefer more recently registered checkpoints without changing input order', () => {
    const assets = [model('old', 100), model('new', 100, { created_at: '2026-11-02T01:00:00Z' })];
    const rows = group(assets, assets);
    assert.equal(rows[0].latest.id, 'new');
    assert.deepEqual(
      assets.map((asset) => asset.id),
      ['old', 'new'],
    );
  });
} finally {
  await server.close();
}
