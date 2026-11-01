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

const dataset = {
  id: 'dataset',
  kind: 'DATASET',
  node_id: 'source',
  name: 'cloth',
  path: '/data/cloth',
  state: 'REGISTERED',
  version: 'v1',
  norm_asset_id: 'norm',
};
const norm = {
  id: 'norm',
  kind: 'NORM',
  node_id: 'source',
  name: 'cloth raw14',
  path: '/data/norm_stats.json',
  state: 'REGISTERED',
  version: 'v1',
};
const assets = [
  dataset,
  norm,
  { ...norm, id: 'other-node', node_id: 'other' },
  { ...norm, id: 'archived', state: 'ARCHIVED' },
  { ...norm, id: 'gone', state: 'UNAVAILABLE' },
  { ...norm, id: 'model', kind: 'MODEL' },
];

try {
  const {
    datasetNormCandidates,
    datasetNormSelectionError,
    datasetNormTransferError,
    defaultDatasetNormSync,
    boundDatasetNorm,
  } = await server.ssrLoadModule('/src/datasetNormAssets.ts');
  const { DatasetNormField, DatasetNormSummary, DatasetNormBindingDialog } =
    await server.ssrLoadModule('/src/DatasetNormBinding.tsx');
  const { AssetSyncNormOption, AssetNormTransferInfo } =
    await server.ssrLoadModule('/src/AssetTransfers.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { assetTransferCreateRequest } = await server.ssrLoadModule('/src/assetTransfer.ts');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const html = (component, props) => renderToStaticMarkup(React.createElement(component, props));

  await test('norm binding candidates are available NORM assets on the exact same node only', () => {
    assert.deepEqual(
      datasetNormCandidates('source', assets).map((asset) => asset.id),
      ['norm'],
    );
    assert.equal(boundDatasetNorm(dataset, assets), norm);
    assert.equal(boundDatasetNorm({ ...dataset, norm_asset_id: null }, assets), undefined);
  });
  await test('norm binding is optional and cannot guess by name or accept missing, wrong-kind, cross-node assets', () => {
    for (const value of [undefined, null, ''])
      assert.equal(datasetNormSelectionError('source', value, assets), undefined);
    assert.equal(datasetNormSelectionError('source', 'norm', assets), undefined);
    for (const value of ['other-node', 'model', 'archived', 'gone', 'does-not-exist'])
      assert.ok(datasetNormSelectionError('source', value, assets));
    assert.equal(
      datasetNormCandidates('source', [{ ...norm, name: 'unrelated name' }])[0].id,
      'norm',
    );
  });
  await test('binding selector defaults to no binding and shows explicit valid choices with paths', () => {
    const unbound = html(DatasetNormField, { nodeId: 'source', assets, value: '', onChange() {} });
    assert.match(unbound, /value="" selected=""/);
    assert.match(unbound, /不绑定/);
    assert.doesNotMatch(unbound, /value="other-node"|value="archived"|value="gone"|value="model"/);
    const selected = html(DatasetNormField, {
      nodeId: 'source',
      assets,
      value: 'norm',
      onChange() {},
    });
    assert.match(selected, /value="norm" selected=""/);
    assert.match(selected, /\/data\/norm_stats\.json/);
  });
  await test('unavailable saved binding stays visible and can be replaced or cleared', () => {
    const summary = html(DatasetNormSummary, {
      asset: { ...dataset, norm_asset_id: 'gone' },
      assets,
    });
    assert.match(summary, /cloth raw14/);
    assert.match(summary, /不可用/);
    const missing = html(DatasetNormSummary, {
      asset: { ...dataset, norm_asset_id: 'missing-id' },
      assets,
    });
    assert.match(missing, /missing-id/);
    const field = html(DatasetNormField, {
      nodeId: 'source',
      assets,
      value: 'gone',
      onChange() {},
    });
    assert.match(field, /value="gone" disabled="" selected=""/);
    assert.match(field, /value=""/);
    assert.match(field, /value="norm"/);
  });
  await test('usable same-machine bindings default to bundled sync while unbound and unavailable cases stay unchecked', () => {
    assert.equal(defaultDatasetNormSync(dataset, assets), true);
    for (const normId of [null, '', 'does-not-exist', 'other-node', 'model', 'archived', 'gone']) {
      assert.equal(defaultDatasetNormSync({ ...dataset, norm_asset_id: normId }, assets), false);
    }
    assert.equal(defaultDatasetNormSync({ ...dataset, kind: 'MODEL' }, assets), false);
    assert.equal(defaultDatasetNormSync({ ...dataset, kind: 'NORM' }, assets), false);
    assert.equal(defaultDatasetNormSync(dataset, []), false);
    const initial = html(AssetSyncNormOption, {
      asset: dataset,
      assets,
      checked: defaultDatasetNormSync(dataset, assets),
      onChange() {},
    });
    assert.match(initial, /checked=""/);
    const request = assetTransferCreateRequest(
      dataset,
      'target',
      '/target',
      'cloth',
      defaultDatasetNormSync(dataset, assets),
    );
    assert.equal(request.include_norm_stats, true);
    assert.equal(request.expected_norm_asset_id, 'norm');
  });
  await test('users can cancel bundled norm stats and the bound source validates independently of data-only copies', () => {
    const unbound = { ...dataset, norm_asset_id: null };
    assert.equal(datasetNormTransferError(unbound, assets, false), undefined);
    assert.equal(datasetNormTransferError(dataset, assets, true), undefined);
    assert.match(datasetNormTransferError(unbound, assets, true), /尚未绑定/);
    assert.match(datasetNormTransferError(norm, assets, true), /只有数据集/);
    assert.match(
      datasetNormTransferError({ ...dataset, norm_asset_id: 'gone' }, assets, true),
      /不可用/,
    );
    const option = html(AssetSyncNormOption, {
      asset: dataset,
      assets,
      checked: false,
      onChange() {},
    });
    assert.match(option, /一并同步已绑定的 norm_stats/);
    assert.match(option, /cloth raw14/);
    assert.doesNotMatch(option, /checked=""/);
    const chosen = html(AssetSyncNormOption, {
      asset: dataset,
      assets,
      checked: true,
      onChange() {},
    });
    assert.match(chosen, /checked=""/);
    assert.match(chosen, /独立的 norm-stats 目录/);
    assert.match(chosen, /自动绑定到目标数据集/);
  });
  await test('bundled create requests freeze the displayed binding and change their idempotency signature when it changes', () => {
    const single = assetTransferCreateRequest(dataset, 'target', '/target///', ' cloth ', false);
    assert.equal(single.include_norm_stats, false);
    assert.equal(Object.hasOwn(single, 'expected_norm_asset_id'), false);
    const bundled = assetTransferCreateRequest(dataset, 'target', '/target///', ' cloth ', true);
    assert.equal(bundled.expected_norm_asset_id, 'norm');
    assert.equal(bundled.target_directory, '/target');
    assert.equal(bundled.target_name, 'cloth');
    assert.notEqual(JSON.stringify(single), JSON.stringify(bundled));
    assert.notEqual(
      JSON.stringify(bundled),
      JSON.stringify(
        assetTransferCreateRequest(
          { ...dataset, norm_asset_id: 'norm-v2' },
          'target',
          '/target///',
          ' cloth ',
          true,
        ),
      ),
    );
  });
  await test('missing or invalid bindings disable opting in while allowing an invalid checked choice to be cleared', () => {
    const noBinding = html(AssetSyncNormOption, {
      asset: { ...dataset, norm_asset_id: null },
      assets,
      checked: false,
      onChange() {},
    });
    assert.match(noBinding, /disabled=""/);
    assert.match(noBinding, /尚未绑定 norm_stats/);
    const stale = html(AssetSyncNormOption, {
      asset: { ...dataset, norm_asset_id: 'gone' },
      assets,
      checked: false,
      onChange() {},
    });
    assert.match(stale, /disabled=""/);
    assert.match(stale, /可仅同步数据集/);
    const clearable = html(AssetSyncNormOption, {
      asset: { ...dataset, norm_asset_id: 'gone' },
      assets,
      checked: true,
      onChange() {},
    });
    assert.doesNotMatch(clearable, /disabled=""/);
    assert.equal(
      html(AssetSyncNormOption, { asset: norm, assets, checked: false, onChange() {} }),
      '',
    );
  });
  await test('norm transfer receipt shows its own source and target but cannot claim the whole bundle is bound early', () => {
    const item = {
      id: 'sync',
      include_norm_stats: true,
      target_directory: '/target',
      state: 'TRANSFERRING',
      norm_sync: {
        source_asset_id: 'norm',
        name: 'cloth raw14',
        source_path: '/data/norm_stats.json',
        target_name: 'norm-stats-sync',
        state: 'SUCCEEDED',
        bytes_total: 100,
        bytes_transferred: 100,
        progress_pct: 100,
      },
    };
    const pending = html(AssetNormTransferInfo, { item });
    assert.match(pending, /附带 norm_stats/);
    assert.match(pending, /等待整体登记/);
    assert.match(pending, /\/target\/norm-stats-sync/);
    assert.doesNotMatch(pending, /已同步并绑定/);
    const complete = html(AssetNormTransferInfo, {
      item: {
        ...item,
        state: 'SUCCEEDED',
        target_asset_id: 'target-data',
        norm_sync: {
          ...item.norm_sync,
          target_asset_id: 'target-norm',
          asset_path: '/target/norm-stats-sync/norm_stats.json',
        },
      },
    });
    assert.match(complete, /已同步并绑定/);
    assert.match(complete, /\/target\/norm-stats-sync\/norm_stats\.json/);
    assert.equal(html(AssetNormTransferInfo, { item: { ...item, include_norm_stats: false } }), '');
  });
  await test('read-only accounts cannot submit changes to dataset norm bindings', () => {
    const output = renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: () => false } },
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(DatasetNormBindingDialog, { asset: dataset, close() {} }),
        ),
      ),
    );
    assert.match(output, /无法修改绑定/);
    assert.doesNotMatch(output, /type="submit"|保存绑定|保存为不绑定/);
  });
  await test('a stopped parent cannot leave norm stats looking queued, active or awaiting automatic registration', () => {
    for (const state of ['FAILED', 'CANCELLED', 'INTERRUPTED']) {
      const initial = {
        state,
        include_norm_stats: true,
        target_directory: '/target',
        norm_sync: { ...norm, source_asset_id: norm.id, source_path: norm.path, state: 'QUEUED' },
      };
      const skipped = html(AssetNormTransferInfo, { item: initial });
      assert.match(skipped, /尚未执行/);
      assert.doesNotMatch(skipped, /等待同步/);
      const interrupted = html(AssetNormTransferInfo, {
        item: { ...initial, norm_sync: { ...initial.norm_sync, state: 'TRANSFERRING' } },
      });
      assert.match(interrupted, /已随整体任务停止/);
      assert.doesNotMatch(interrupted, /正在同步文件/);
      const verified = html(AssetNormTransferInfo, {
        item: { ...initial, norm_sync: { ...initial.norm_sync, state: 'SUCCEEDED' } },
      });
      assert.match(verified, /文件已校验，整体任务未完成/);
      assert.doesNotMatch(verified, /等待整体登记|已同步并绑定/);
    }
  });
} finally {
  await server.close();
}
