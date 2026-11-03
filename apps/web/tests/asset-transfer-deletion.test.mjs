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
const originalFetch = globalThis.fetch;

try {
  const { canDeleteAssetTransferRecord, deleteAssetTransferRecord } = await server.ssrLoadModule(
    '/src/assetTransferDeletion.ts',
  );
  const { setSessionCsrf, ApiError } = await server.ssrLoadModule('/src/api.ts');
  const { BackgroundTaskTable } = await server.ssrLoadModule('/src/BackgroundTasks.tsx');
  const { AssetTransferCard } = await server.ssrLoadModule('/src/AssetTransfers.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const { buildBackgroundTasks } = await server.ssrLoadModule('/src/backgroundTaskEntries.ts');
  const transfer = (state) => ({
    id: state,
    name: `${state} 同步`,
    state,
    source_asset_id: 'source-asset',
    source_kind: 'DATASET',
    source_node_id: 'source',
    source_path: '/data/source',
    target_node_id: 'target',
    target_directory: '/data',
    target_name: 'copy',
    target_path: '/data/copy',
    bytes_total: 100,
    bytes_transferred: 50,
    files_total: 1,
    files_transferred: 0,
    created_at: '2026-11-01T00:00:00Z',
    updated_at: '2026-11-01T00:00:00Z',
  });
  const render = (element, permissions = ['assets.transfer']) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: {
            user: { id: 'test-admin', role: 'admin' },
            can: (permission) => permissions.includes(permission),
          },
        },
        React.createElement(DataProvider, { enabled: false }, element),
      ),
    );

  await test('only ended asset transfers permit deleting their task records', () => {
    for (const state of ['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED'])
      assert.equal(canDeleteAssetTransferRecord(transfer(state)), true, state);
    for (const state of [
      'QUEUED',
      'PREPARING',
      'SCANNING',
      'TRANSFERRING',
      'VERIFYING',
      'CANCELLING',
      'UNKNOWN',
      '',
      'OTHER',
    ])
      assert.equal(canDeleteAssetTransferRecord(transfer(state)), false, state);
  });

  await test('merged task rows offer deletion for ended sync records while running training stays protected', () => {
    const tasks = buildBackgroundTasks(
      [
        {
          id: 'training',
          name: '运行中的训练',
          state: 'RUNNING',
          node_id: 'source',
          template_id: 'openpi-sft',
          created_at: '2026-11-01T00:00:00Z',
        },
      ],
      ['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED', 'TRANSFERRING', 'CANCELLING'].map(
        transfer,
      ),
    );
    const element = React.createElement(BackgroundTaskTable, { tasks });
    const writable = render(element, ['assets.transfer', 'training.write']);
    assert.equal((writable.match(/aria-label="删除同步记录 /g) || []).length, 4);
    assert.match(writable, /aria-label="删除同步记录 FAILED 同步"/);
    assert.doesNotMatch(writable, /aria-label="删除同步记录 (TRANSFERRING|CANCELLING) 同步"/);
    assert.doesNotMatch(writable, /aria-label="删除训练记录 运行中的训练"/);
    assert.doesNotMatch(render(element, ['training.write']), /aria-label="删除同步记录 /);
    assert.doesNotMatch(render(element, []), /aria-label="删除同步记录 /);
  });

  await test('sync cards share the same deletion eligibility and asset permission', () => {
    const card = (state, permissions) =>
      render(
        React.createElement(AssetTransferCard, {
          item: transfer(state),
          changed() {},
        }),
        permissions,
      );
    assert.match(card('FAILED'), /aria-label="删除同步记录 FAILED 同步"/);
    assert.doesNotMatch(card('TRANSFERRING'), /aria-label="删除同步记录 /);
    assert.doesNotMatch(card('FAILED', ['training.write']), /aria-label="删除同步记录 /);
  });

  await test('deletion sends authenticated DELETE to the encoded task and accepts empty 204', async () => {
    setSessionCsrf('test-csrf');
    globalThis.fetch = async (path, options) => {
      assert.equal(path, '/api/v1/asset-transfers/sync%2F%E4%B8%80');
      assert.equal(options.method, 'DELETE');
      assert.equal(options.credentials, 'same-origin');
      assert.equal(options.headers.get('X-CSRF-Token'), 'test-csrf');
      assert.equal(options.body, undefined);
      return new Response(null, { status: 204 });
    };
    await deleteAssetTransferRecord('sync/一');
    setSessionCsrf();
  });

  await test('a nonterminal transfer conflict remains a visible API error', async () => {
    globalThis.fetch = async () =>
      new Response(
        JSON.stringify({
          error: { code: 'ASSET_TRANSFER_NOT_TERMINAL', message: '请先停止同步并等待任务结束' },
        }),
        { status: 409 },
      );
    await assert.rejects(deleteAssetTransferRecord('active-sync'), (error) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.code, 'ASSET_TRANSFER_NOT_TERMINAL');
      assert.equal(error.status, 409);
      assert.equal(error.message, '请先停止同步并等待任务结束');
      return true;
    });
  });
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
