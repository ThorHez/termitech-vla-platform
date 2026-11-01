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
const nodes = [
  { id: 'source', status: 'ONLINE' },
  { id: 'other', status: 'ONLINE' },
  { id: 'target', status: 'ONLINE' },
  { id: 'offline', status: 'OFFLINE' },
];
const asset = (id, changes = {}) => ({
  id,
  name: id,
  node_id: 'source',
  kind: 'DATASET',
  path: `/data/${id}`,
  version: 'v1',
  state: 'REGISTERED',
  ...changes,
});
const assets = [
  asset('a', { norm_asset_id: 'norm' }),
  asset('b', { node_id: 'other', kind: 'MODEL' }),
  asset('norm', { kind: 'NORM', path: '/data/norm_stats.json' }),
];
try {
  const h = await server.ssrLoadModule('/src/batchAssetSyncData.ts');
  const { BatchAssetSync } = await server.ssrLoadModule('/src/BatchAssetSync.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const input = (changes = {}) => ({
    rows: h.batchSyncRows(assets),
    assets,
    nodes,
    targetNodeId: 'target',
    directory: '/data/shared',
    ...changes,
  });
  const render = (auth) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: auth },
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(BatchAssetSync, { assetIds: ['a'], close() {}, created() {} }),
        ),
      ),
    );

  await test('datasets, models, and norms can be selected only while source assets and machines are available', () => {
    for (const kind of ['DATASET', 'MODEL', 'NORM'])
      assert.equal(h.batchSyncSourceError(asset('a', { kind }), nodes), undefined);
    for (const candidate of [
      undefined,
      asset('a', { state: 'ARCHIVED' }),
      asset('a', { state: 'UNAVAILABLE' }),
      asset('a', { status: 'UNAVAILABLE' }),
      asset('a', { kind: 'OTHER' }),
      asset('a', { node_id: 'offline' }),
      asset('a', { node_id: 'missing' }),
      asset('a', { path: '/' }),
      asset('a', { path: '/data/../private' }),
      asset('a', { path: 'relative' }),
    ])
      assert.ok(h.batchSyncSourceError(candidate, nodes));
  });
  await test('selection reconciliation removes invalid assets and duplicate IDs while retaining order', () => {
    assert.deepEqual(
      h.reconcileBatchSyncSelection(
        ['b', 'a', 'b', 'removed', 'bad'],
        [...assets, asset('bad', { state: 'ARCHIVED' })],
        nodes,
      ),
      ['b', 'a'],
    );
    assert.deepEqual(
      h.reconcileBatchSyncSelection(
        ['a', 'b'],
        assets,
        nodes.map((node) => ({ ...node, status: node.id === 'source' ? 'OFFLINE' : node.status })),
      ),
      ['b'],
    );
  });
  await test('select-all only changes current filtered eligible IDs and preserves hidden selections', () => {
    assert.deepEqual(h.toggleBatchSyncVisible(['a'], ['b', 'norm'], true), ['a', 'b', 'norm']);
    assert.deepEqual(h.toggleBatchSyncVisible(['a', 'b', 'norm'], ['b', 'norm'], false), ['a']);
    assert.deepEqual(h.toggleBatchSyncVisible(['a'], ['a'], true), ['a']);
  });
  await test('target defaults reuse directory basenames and file asset names without silently renaming collisions', () => {
    const rows = h.batchSyncRows([
      asset('a', { path: '/data/day-1/base/' }),
      asset('b', { path: '/data/day-2/base/' }),
      assets[2],
    ]);
    assert.deepEqual(
      rows.map((row) => row.targetName),
      ['base', 'base', 'norm'],
    );
    assert.ok(rows.every((row) => !row.includeNormStats && !row.expectedNormAssetId));
    assert.match(
      h.batchSyncError(
        input({
          rows,
          assets: [
            asset('a', { path: '/data/day-1/base/' }),
            asset('b', { path: '/data/day-2/base/' }),
            assets[2],
          ],
        }),
      ),
      /重名/,
    );
    rows[1].targetName = 'base-day-2';
    assert.equal(h.batchSyncError(input({ rows })), undefined);
  });
  await test('available bound norms are selected by default using the complete catalog, while missing and incompatible bindings stay off', () => {
    const rows = h.batchSyncRows([assets[0], assets[1], assets[2]], assets);
    assert.deepEqual(
      rows.map(({ includeNormStats, expectedNormAssetId }) => ({
        includeNormStats,
        expectedNormAssetId,
      })),
      [
        { includeNormStats: true, expectedNormAssetId: 'norm' },
        { includeNormStats: false, expectedNormAssetId: undefined },
        { includeNormStats: false, expectedNormAssetId: undefined },
      ],
    );
    assert.equal(h.batchSyncRows([assets[0]], assets)[0].includeNormStats, true);
    assert.equal(h.batchSyncRows([assets[0]])[0].includeNormStats, false);
    for (const available of [
      [],
      [asset('norm', { kind: 'NORM', state: 'ARCHIVED' })],
      [asset('norm', { kind: 'NORM', state: 'UNAVAILABLE' })],
      [asset('norm', { kind: 'MODEL' })],
      [asset('norm', { kind: 'NORM', node_id: 'other' })],
    ]) {
      const row = h.batchSyncRows([assets[0]], available)[0];
      assert.equal(row.includeNormStats, false);
      assert.equal(row.expectedNormAssetId, undefined);
    }
    const noBinding = h.batchSyncRows([asset('unbound')], assets)[0];
    assert.equal(noBinding.includeNormStats, false);
  });
  await test('a user can cancel the default bundled norm without retaining a norm expectation in the actual request', () => {
    const rows = h.batchSyncRows([assets[0]], assets);
    const original = h.batchSyncRequest(rows, 'target', '/data/shared');
    rows[0] = { ...rows[0], includeNormStats: false, expectedNormAssetId: undefined };
    const cancelled = h.batchSyncRequest(rows, 'target', '/data/shared');
    assert.equal(cancelled.entries[0].include_norm_stats, false);
    assert.equal(Object.hasOwn(cancelled.entries[0], 'expected_norm_asset_id'), false);
    assert.notEqual(JSON.stringify(original), JSON.stringify(cancelled));
    assert.equal(h.batchSyncError(input({ rows })), undefined);
  });
  await test('only model rows can explicitly exclude root train_state and default requests retain their prior shape', () => {
    const rows = h.batchSyncRows(assets);
    assert.ok(rows.every((row) => row.excludeTrainState === false));
    const complete = h.batchSyncRequest(rows, 'target', '/data/shared', assets);
    assert.ok(complete.entries.every((entry) => !Object.hasOwn(entry, 'exclude_train_state')));
    rows[1].excludeTrainState = true;
    assert.equal(h.batchSyncError(input({ rows })), undefined);
    const excluded = h.batchSyncRequest(rows, 'target', '/data/shared', assets);
    assert.equal(excluded.entries[1].exclude_train_state, true);
    assert.equal(Object.hasOwn(excluded.entries[0], 'exclude_train_state'), false);
    assert.equal(Object.hasOwn(excluded.entries[2], 'exclude_train_state'), false);
    assert.equal(
      Object.hasOwn(
        h.batchSyncRequest(rows, 'target', '/data/shared').entries[1],
        'exclude_train_state',
      ),
      false,
    );
    const identity = h.batchSyncIdentity(undefined, complete, () => 'complete');
    const changed = h.batchSyncIdentity(identity, excluded, () => 'excluded');
    assert.equal(changed.key, 'excluded');
    assert.equal(
      h.batchSyncIdentity(changed, excluded, () => 'unexpected'),
      changed,
    );
    for (const index of [0, 2]) {
      const forged = rows.map((row, position) => ({
        ...row,
        excludeTrainState: position === index,
      }));
      assert.match(h.batchSyncError(input({ rows: forged })), /只有模型/);
      assert.equal(
        Object.hasOwn(
          h.batchSyncRequest(forged, 'target', '/data/shared', assets).entries[index],
          'exclude_train_state',
        ),
        false,
      );
    }
  });
  await test('batch receipts must preserve each frozen model exclusion policy, including omitted false for older tasks', () => {
    const rows = h.batchSyncRows(assets);
    rows[1].excludeTrainState = true;
    const request = h.batchSyncRequest(rows, 'target', '/data/shared', assets);
    const receipt = {
      idempotency_key: 'key',
      transfers: request.entries.map((entry, index) => ({
        id: 'sync-' + index,
        ...entry,
        target_node_id: 'target',
        target_directory: '/data/shared',
      })),
    };
    assert.equal(h.batchSyncReceiptMatches(receipt, 'key', request), true);
    assert.equal(
      h.batchSyncReceiptMatches(
        {
          ...receipt,
          transfers: receipt.transfers.map((item) => ({ ...item, exclude_train_state: false })),
        },
        'key',
        request,
      ),
      false,
    );
    assert.equal(
      h.batchSyncReceiptMatches(
        {
          ...receipt,
          transfers: receipt.transfers.map((item) => ({ ...item, exclude_train_state: true })),
        },
        'key',
        request,
      ),
      false,
    );
    rows[1].excludeTrainState = false;
    const complete = h.batchSyncRequest(rows, 'target', '/data/shared', assets);
    const legacy = {
      ...receipt,
      transfers: receipt.transfers.map(({ exclude_train_state, ...item }) => item),
    };
    assert.equal(h.batchSyncReceiptMatches(legacy, 'key', complete), true);
  });
  await test('one through twenty items are allowed; empty, repeated sources and twenty-one items are blocked', () => {
    assert.equal(h.batchSyncError(input()), undefined);
    assert.match(h.batchSyncError(input({ rows: [] })), /选择/);
    assert.match(
      h.batchSyncError(input({ rows: h.batchSyncRows([assets[0], assets[0]]) })),
      /重复源/,
    );
    const many = Array.from({ length: 21 }, (_, i) => asset('item-' + i));
    assert.match(h.batchSyncError(input({ rows: h.batchSyncRows(many), assets: many })), /20/);
    assert.equal(
      h.batchSyncError(input({ rows: h.batchSyncRows(many.slice(0, 20)), assets: many })),
      undefined,
    );
  });
  await test('all sources must differ from the common online target and invalid path names block before submit', () => {
    for (const changes of [
      { targetNodeId: '' },
      { targetNodeId: 'source' },
      { targetNodeId: 'other' },
      { targetNodeId: 'offline' },
      { directory: 'relative' },
      { directory: '/data/../private' },
      { rows: h.batchSyncRows([assets[0]]).map((row) => ({ ...row, targetName: '../bad' })) },
    ])
      assert.ok(h.batchSyncError(input(changes)));
  });
  await test('normalization is per dataset and explicit; changed bindings or incompatible assets block', () => {
    const rows = h.batchSyncRows(assets);
    rows[0] = { ...rows[0], includeNormStats: true, expectedNormAssetId: 'norm' };
    assert.equal(h.batchSyncError(input({ rows })), undefined);
    assert.match(
      h.batchSyncError(
        input({
          rows,
          assets: assets
            .map((asset) => (asset.id === 'a' ? { ...asset, norm_asset_id: 'new-norm' } : asset))
            .concat(asset('new-norm', { kind: 'NORM' })),
        }),
      ),
      /已变化/,
    );
    assert.match(
      h.batchSyncError(
        input({
          rows,
          assets: assets.map((asset) =>
            asset.id === 'norm' ? { ...asset, state: 'UNAVAILABLE' } : asset,
          ),
        }),
      ),
      /不可用/,
    );
    rows[1].includeNormStats = true;
    assert.match(h.batchSyncError(input({ rows })), /只有数据集/);
  });
  await test('request preserves each source choice and exact normalization binding without mixing source paths', () => {
    const rows = h.batchSyncRows(assets);
    rows[0] = {
      ...rows[0],
      targetName: ' cloth-copy ',
      includeNormStats: true,
      expectedNormAssetId: 'norm',
    };
    const request = h.batchSyncRequest(rows, 'target', ' /data/shared/// ');
    assert.equal(request.target_directory, '/data/shared');
    assert.deepEqual(request.entries[0], {
      source_asset_id: 'a',
      target_name: 'cloth-copy',
      include_norm_stats: true,
      expected_norm_asset_id: 'norm',
    });
    assert.deepEqual(request.entries[1], {
      source_asset_id: 'b',
      target_name: 'b',
      include_norm_stats: false,
    });
    assert.ok(request.entries.every((row) => !Object.hasOwn(row, 'source_path')));
  });
  await test('unchanged network retries retain identity while target, names and norm choices create new identity', () => {
    const rows = h.batchSyncRows(assets);
    const request = h.batchSyncRequest(rows, 'target', '/data/shared');
    const identity = h.batchSyncIdentity(undefined, request, () => 'original');
    assert.equal(
      h.batchSyncIdentity(identity, request, () => 'unexpected'),
      identity,
    );
    for (const changed of [
      { ...request, target_node_id: 'other' },
      { ...request, target_directory: '/other' },
      h.batchSyncRequest(
        rows.map((row) => ({ ...row, targetName: row.targetName + '-copy' })),
        'target',
        '/data/shared',
      ),
    ])
      assert.equal(h.batchSyncIdentity(identity, changed, () => 'changed').key, 'changed');
  });
  await test('receipt validates complete source and destination matching before accepting task links', () => {
    const request = h.batchSyncRequest(h.batchSyncRows(assets), 'target', '/data/shared');
    const receipt = {
      idempotency_key: 'key',
      transfers: request.entries.map((entry, index) => ({
        id: 'sync-' + index,
        ...entry,
        target_node_id: 'target',
        target_directory: '/data/shared',
      })),
    };
    assert.equal(h.batchSyncReceiptMatches(receipt, 'key', request), true);
    assert.equal(
      h.batchSyncReceiptMatches(
        {
          ...receipt,
          transfers: receipt.transfers.map((item) => ({
            ...item,
            target_directory: '/data//shared/',
          })),
        },
        'key',
        request,
      ),
      true,
    );
    assert.equal(
      h.batchSyncReceiptMatches(
        {
          ...receipt,
          transfers: receipt.transfers.map((item) => ({ ...item, target_directory: '/other' })),
        },
        'key',
        request,
      ),
      false,
    );
    assert.equal(
      h.batchSyncReceiptMatches(
        {
          ...receipt,
          transfers: receipt.transfers.map((item) => ({ ...item, include_norm_stats: true })),
        },
        'key',
        request,
      ),
      false,
    );
    for (const changed of [
      { ...receipt, idempotency_key: 'foreign' },
      { ...receipt, transfers: receipt.transfers.slice(1) },
      { ...receipt, transfers: receipt.transfers.map((item) => ({ ...item, id: 'duplicate' })) },
      {
        ...receipt,
        transfers: receipt.transfers.map((item) => ({ ...item, target_node_id: 'other' })),
      },
    ])
      assert.equal(h.batchSyncReceiptMatches(changed, 'key', request), false);
  });
  await test('read-only and legacy sessions cannot mount a batch create form', () => {
    for (const auth of [{ user: { id: 'owner' }, can: () => false }, { can: () => true }]) {
      const html = render(auth);
      assert.match(html, /个人账户/);
      assert.doesNotMatch(html, /开始批量同步|目标父目录/);
    }
  });
  await test('personal writers see explicit background job and no-overwrite copy boundaries', () => {
    const html = render({ user: { id: 'owner' }, can: () => true });
    assert.match(html, /目标父目录/);
    assert.match(html, /不会被覆盖/);
    assert.match(html, /重名时请明确修改/);
    assert.match(html, /一次最多 20 项/);
    assert.match(html, /整批校验通过后才创建后台任务/);
  });
} finally {
  await server.close();
}
