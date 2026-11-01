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

const asset = {
  id: 'asset-1',
  node_id: 'source',
  name: 'fold-cloth',
  path: '/data/cloth/',
  kind: 'DATASET',
  version: 'v1',
  state: 'REGISTERED',
};
const sourceNode = { id: 'source', name: 'T001', hostname: 'source', status: 'ONLINE' };
const targetNode = { id: 'target', name: 'T002', hostname: 'target', status: 'ONLINE' };
const transfer = (changes = {}) => ({
  id: 'sync-1',
  name: 'fold-cloth 同步',
  source_asset_id: 'asset-1',
  source_kind: 'DATASET',
  source_node_id: 'source',
  source_path: '/data/cloth',
  target_node_id: 'target',
  target_directory: '/data/shared',
  target_name: 'cloth-v1',
  target_path: '/data/shared/cloth-v1',
  state: 'TRANSFERRING',
  bytes_total: 1024,
  bytes_transferred: 512,
  files_total: 10,
  files_transferred: 5,
  speed_bytes_per_second: 128,
  progress_pct: 50,
  attempt: 1,
  created_at: '2026-10-31T01:00:00Z',
  updated_at: '2026-10-31T01:00:10Z',
  ...changes,
});

try {
  const helpers = await server.ssrLoadModule('/src/assetTransfer.ts');
  const {
    AssetTransferCard,
    AssetSyncDialog,
    AssetTransferProgressDialog,
    AssetSyncBackgroundNotice,
    AssetTransferTaskDetail,
    AssetSyncModelOption,
  } = await server.ssrLoadModule('/src/AssetTransfers.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const render = (element, writable = true) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: (permission) => permission === 'assets.write' && writable } },
        React.createElement(DataProvider, { enabled: false }, element),
      ),
    );
  const renderCard = (changes = {}, writable = true) =>
    render(
      React.createElement(AssetTransferCard, {
        item: transfer(changes),
        changed() {},
        viewTarget() {},
      }),
      writable,
    );

  await test('target paths use one new child directory, including Unicode and spaces', () => {
    assert.equal(helpers.assetTargetPath('/data/shared///', ' cloth v1 '), '/data/shared/cloth v1');
    assert.equal(helpers.assetTargetPath('/', '叠衣服 数据'), '/叠衣服 数据');
    assert.equal(helpers.assetTargetPath('/data/.cache', 'v1.2'), '/data/.cache/v1.2');
  });
  await test('parent and child validation reject traversal and control characters', () => {
    for (const name of [
      '',
      '.',
      '..',
      'nested/path',
      'nested\\path',
      'line\nfeed',
      'zero\0byte',
      'control\u0085name',
      'x'.repeat(161),
      '衣'.repeat(86),
    ]) {
      assert.ok(helpers.assetTargetNameError(name), name);
      assert.equal(helpers.assetTargetPath('/data', name), undefined);
    }
    for (const parent of [
      '',
      'relative/path',
      '/data/../other',
      '/data/./child',
      '/line\nfeed',
      '/data\\child',
    ]) {
      assert.ok(helpers.assetTargetDirectoryError(parent), parent);
      assert.equal(helpers.assetTargetPath(parent, 'cloth'), undefined);
    }
    assert.equal(helpers.assetTargetNameError('衣'.repeat(85)), undefined);
  });
  await test('directory defaults preserve dataset names while file assets use a containing folder', () => {
    assert.equal(helpers.defaultAssetTargetName(asset), 'cloth');
    assert.equal(
      helpers.defaultAssetTargetName({
        ...asset,
        path: '/data/norm_stats.json',
        name: 'cloth-norm',
      }),
      'cloth-norm',
    );
    assert.equal(
      helpers.defaultAssetTargetName({ ...asset, path: '/data/../../', name: 'safe-name' }),
      'safe-name',
    );
    assert.equal(helpers.defaultAssetTargetName({ name: '../bad', path: '/' }), 'synced-asset');
  });
  await test('only registered supported assets between two online distinct nodes can start', () => {
    const input = { asset, sourceNode, targetNode, directory: '/data', targetName: 'cloth' };
    assert.equal(helpers.assetSyncBlocker(input), undefined);
    for (const kind of ['DATASET', 'NORM', 'MODEL']) {
      assert.equal(helpers.assetSyncBlocker({ ...input, asset: { ...asset, kind } }), undefined);
    }
    for (const changes of [
      { asset: { ...asset, state: 'ARCHIVED' } },
      { asset: { ...asset, state: 'UNAVAILABLE' } },
      { asset: { ...asset, path: '/' } },
      { asset: { ...asset, kind: 'OTHER' } },
      { sourceNode: undefined },
      { sourceNode: { ...sourceNode, status: 'OFFLINE' } },
      { sourceNode: { ...sourceNode, id: 'different-source' } },
      { targetNode: undefined },
      { targetNode: sourceNode },
      { targetNode: { ...targetNode, status: 'OFFLINE' } },
    ])
      assert.ok(helpers.assetSyncBlocker({ ...input, ...changes }));
  });
  await test('only a deliberate model choice adds exclude_train_state without changing the default request shape', () => {
    const model = { ...asset, kind: 'MODEL' };
    const complete = helpers.assetTransferCreateRequest(model, 'target', '/data', 'model');
    assert.equal(Object.hasOwn(complete, 'exclude_train_state'), false);
    assert.deepEqual(
      helpers.assetTransferCreateRequest(model, 'target', '/data', 'model', false, false),
      complete,
    );
    const excluded = helpers.assetTransferCreateRequest(
      model,
      'target',
      '/data',
      'model',
      false,
      true,
    );
    assert.equal(excluded.exclude_train_state, true);
    assert.notEqual(JSON.stringify(excluded), JSON.stringify(complete));
    for (const kind of ['DATASET', 'NORM'])
      assert.equal(
        Object.hasOwn(
          helpers.assetTransferCreateRequest(
            { ...asset, kind },
            'target',
            '/data',
            'copy',
            false,
            true,
          ),
          'exclude_train_state',
        ),
        false,
      );
  });
  await test('model exclusion option is cancellable, disabled while pending, and hidden for data and norms', () => {
    const model = { ...asset, kind: 'MODEL' };
    const complete = render(
      React.createElement(AssetSyncModelOption, { asset: model, checked: false, onChange() {} }),
    );
    assert.match(complete, /不同步 train_state\//);
    assert.match(complete, /根部的 train_state\//);
    assert.match(complete, /params 与其他文件保留/);
    assert.match(complete, /恢复完整训练状态/);
    assert.doesNotMatch(complete, /checked=""/);
    const pending = render(
      React.createElement(AssetSyncModelOption, {
        asset: model,
        checked: true,
        disabled: true,
        onChange() {},
      }),
    );
    assert.match(pending, /checked=""/);
    assert.match(pending, /disabled=""/);
    for (const kind of ['DATASET', 'NORM'])
      assert.equal(
        render(
          React.createElement(AssetSyncModelOption, {
            asset: { ...asset, kind },
            checked: true,
            onChange() {},
          }),
        ),
        '',
      );
  });
  await test('model sync receipts display frozen train-state policy without claiming source optimizer files exist', () => {
    const excluded = renderCard({ source_kind: 'MODEL', exclude_train_state: true });
    assert.match(excluded, /<section[^>]*aria-label="模型同步范围"/);
    assert.match(excluded, /排除 train_state\//);
    assert.match(excluded, /不参与同步/);
    assert.match(excluded, /若保存了优化器/);
    assert.match(excluded, /params 与其他文件/);
    assert.match(excluded, /正常同步（如存在）/);
    assert.match(excluded, /本任务配置已固定，重试沿用/);
    assert.ok(excluded.indexOf('模型同步范围') < excluded.indexOf('asset-transfer-progress'));
    for (const exclude_train_state of [false, undefined]) {
      const complete = renderCard({ source_kind: 'MODEL', exclude_train_state });
      assert.match(complete, /aria-label="模型同步范围"/);
      assert.match(complete, /完整复制/);
      assert.match(complete, /参与同步（如存在）/);
      assert.doesNotMatch(complete, /排除 train_state\/|不参与同步/);
    }
    for (const source_kind of ['DATASET', 'NORM'])
      assert.doesNotMatch(
        renderCard({ source_kind, exclude_train_state: true }),
        /模型同步范围|排除 train_state\//,
      );
    assert.doesNotMatch(excluded, /type="checkbox"/);
    const queued = renderCard({ source_kind: 'MODEL', exclude_train_state: true, state: 'QUEUED' });
    assert.match(queued, /不参与同步/);
    assert.match(queued, /准备中/);
    assert.doesNotMatch(queued, /同步完成，已登记|文件完整性校验通过/);
  });
  await test('queued and preparing records do not pretend transfer has begun', () => {
    for (const state of ['QUEUED', 'PREPARING', 'SCANNING']) {
      const html = renderCard({ state, progress_pct: 99 });
      assert.match(html, /准备中/);
      assert.match(html, /文件总大小与数量待统计/);
      assert.doesNotMatch(html, />99\.0%/);
      assert.match(html, /取消同步/);
    }
  });
  await test('progress and speed show actual transfer while completion remains below 100', () => {
    const html = renderCard({ progress_pct: 120 });
    assert.match(html, /99\.0%/);
    assert.match(html, /128 B\/s/);
    assert.match(html, /512 B/);
    assert.match(html, /源资产/);
    assert.match(html, /\/data\/shared\/cloth-v1/);
    assert.doesNotMatch(html, /同步完成，已登记|重试同步/);
  });
  await test('cancelling stays active until stopped and cancelled records allow safe retry', () => {
    assert.equal(helpers.assetTransferTerminal('CANCELLING'), false);
    const stopping = renderCard({ state: 'CANCELLING' });
    assert.match(stopping, /正在停止同步/);
    assert.match(stopping, /停止确认前保留当前任务/);
    assert.match(stopping, /disabled=""[^>]*>.*?取消同步/s);
    assert.doesNotMatch(stopping, /重试同步/);
    const cancelled = renderCard({ state: 'CANCELLED' });
    assert.match(cancelled, /保留记录与本任务部分文件供重试/);
    assert.match(cancelled, /重试同步/);
    assert.doesNotMatch(cancelled, /取消同步<\/button>/);
  });
  await test('failed and interrupted records retain errors and retry controls', () => {
    for (const state of ['FAILED', 'INTERRUPTED']) {
      const html = renderCard({
        state,
        error: { code: 'SSH_UNAVAILABLE', message: '机器连接不可用' },
      });
      assert.match(html, /SSH_UNAVAILABLE/);
      assert.match(html, /机器连接不可用/);
      assert.match(html, /重试同步/);
      assert.doesNotMatch(html, /查看目标资产/);
    }
  });
  await test('success identifies the actual registered file path without claiming training readiness', () => {
    const html = renderCard({
      state: 'SUCCEEDED',
      source_kind: 'NORM',
      target_asset_id: 'asset-target',
      asset_path: '/data/shared/cloth-v1/norm_stats.json',
    });
    assert.match(html, /同步完成，已登记/);
    assert.match(html, /100%/);
    assert.match(html, /\/data\/shared\/cloth-v1\/norm_stats\.json/);
    assert.match(html, /训练使用前仍需预检/);
    assert.match(html, /查看目标资产/);
    assert.doesNotMatch(html, /重试同步|取消同步<\/button>|使用此权重部署/);
    assert.doesNotMatch(renderCard({ state: 'SUCCEEDED' }), /同步完成，已登记|查看目标资产/);
  });
  await test('read-only users see receipts but no transfer write actions or create form', () => {
    assert.doesNotMatch(renderCard({ state: 'FAILED' }, false), /重试同步|取消同步<\/button>/);
    assert.match(
      renderCard({ state: 'SUCCEEDED', target_asset_id: 'asset-target' }, false),
      /查看目标资产/,
    );
    const dialog = render(
      React.createElement(AssetSyncDialog, { asset, close() {}, created() {} }),
      false,
    );
    assert.match(dialog, /权限不足/);
    assert.doesNotMatch(dialog, /开始同步/);
  });
  await test('copy dialog clearly separates parent directory from the child and does not promise overwrite', () => {
    const html = render(React.createElement(AssetSyncDialog, { asset, close() {}, created() {} }));
    assert.match(html, /目标机器/);
    assert.match(html, /目标父目录/);
    assert.match(html, /目标子目录名/);
    assert.match(html, /目标已有目录不会被覆盖/);
    assert.match(html, /训练仍需预检/);
    assert.match(html, /恢复机器连接/);
    assert.match(html, /disabled=""/);
  });
  await test('created task receipt immediately shows preparation without waiting for the task list', () => {
    const html = render(
      React.createElement(AssetTransferProgressDialog, {
        initialTransfer: transfer({ state: 'QUEUED', progress_pct: 0, bytes_total: 0 }),
        close() {},
        changed() {},
      }),
    );
    assert.match(html, /数据同步进度/);
    assert.match(html, /fold-cloth 同步/);
    assert.match(html, /准备中/);
    assert.match(html, /文件总大小与数量待统计/);
    assert.match(html, /关闭窗口不会取消同步/);
    assert.doesNotMatch(html, /开始同步|>0\.0%/);
  });
  await test('progress dialog preserves the actual transfer receipt before its first detail response', () => {
    const html = render(
      React.createElement(AssetTransferProgressDialog, {
        initialTransfer: transfer(),
        close() {},
        changed() {},
      }),
    );
    assert.match(html, /50\.0%/);
    assert.match(html, /512 B/);
    assert.match(html, /128 B\/s/);
    assert.match(html, /每 2 秒自动更新/);
    assert.match(html, /取消同步/);
  });
  await test('task list provides a progress entry and read-only progress remains non-mutating', () => {
    const html = render(
      React.createElement(AssetTransferCard, {
        item: transfer(),
        changed() {},
        viewProgress() {},
      }),
    );
    assert.match(html, /查看进度/);
    const readOnly = render(
      React.createElement(AssetTransferProgressDialog, {
        initialTransfer: transfer({ state: 'FAILED' }),
        close() {},
        changed() {},
      }),
      false,
    );
    assert.match(readOnly, /数据同步进度/);
    assert.doesNotMatch(readOnly, /重试同步|取消同步<\/button>/);
  });
  await test('background submission receipt offers a durable task link without opening a modal', () => {
    const html = render(React.createElement(AssetSyncBackgroundNotice, { transfer: transfer() }));
    assert.match(html, /数据同步已加入后台任务/);
    assert.match(html, /href="#\/runs\/sync\/sync-1"/);
    assert.match(html, /查看任务/);
    assert.match(html, /关闭页面/);
    assert.doesNotMatch(html, /role="dialog"|开始同步|取消同步|重试同步/);
  });
  await test('sync task detail can mount from its URL id without a creation receipt or task list', () => {
    const html = render(React.createElement(AssetTransferTaskDetail, { id: 'persisted-task' }));
    assert.match(html, /返回训练任务/);
    assert.match(html, /数据同步任务/);
    assert.match(html, /正在读取数据同步任务/);
    assert.doesNotMatch(html, /目标父目录|开始同步|取消同步|重试同步/);
  });
  await test('background task links keep the transfer identity inside one URL segment', () => {
    assert.equal(helpers.assetTransferTaskPath('sync-1'), '/runs/sync/sync-1');
    assert.equal(helpers.assetTransferTaskPath('id /?#'), '/runs/sync/id%20%2F%3F%23');
  });
} finally {
  await server.close();
}
