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
  const { canManageResource, resourceOwnerLabel } = await server.ssrLoadModule(
    '/src/resourcePermissions.tsx',
  );
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const { RunDeleteAction } = await server.ssrLoadModule('/src/RunDeleteAction.tsx');
  const { AssetTransferCard } = await server.ssrLoadModule('/src/AssetTransfers.tsx');
  const { TransferCard } = await server.ssrLoadModule('/src/CheckpointTransfers.tsx');
  const { DeploymentDetail, DeploymentRecordView } = await server.ssrLoadModule(
    '/src/PolicyDeployments.tsx',
  );
  const transport = await server.ssrLoadModule('/src/api.ts');
  const { reservationSelectionLimit } = await server.ssrLoadModule('/src/GpuReservations.tsx');
  const { RunQueueReason } = await server.ssrLoadModule('/src/RunDetail.tsx');
  const user = (role = 'operator', id = 'alice') => ({ id, role, username: id, display_name: id });
  const render = (element, account = user()) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: {
            user: account,
            can: (permission) =>
              account?.role === 'admin' ||
              (account?.role === 'operator' &&
                [
                  'training.write',
                  'assets.transfer',
                  'recipes.query',
                  'assets.query',
                  'dataset-templates.write',
                ].includes(permission)),
          },
        },
        React.createElement(DataProvider, { enabled: false }, element),
      ),
    );
  const base = {
    id: 'resource',
    owner_user_id: 'alice',
    name: 'Shared readable record',
    state: 'FAILED',
    node_id: 'machine',
    created_at: '2026-11-03T00:00:00Z',
    updated_at: '2026-11-03T00:00:00Z',
  };
  const transfer = {
    ...base,
    source_node_id: 'source',
    target_node_id: 'target',
    source_path: '/data/source',
    target_directory: '/data',
    target_name: 'copy',
    bytes_total: 100,
    bytes_transferred: 50,
    files_total: 1,
    files_transferred: 0,
    logs: 'shared task log',
  };

  await test('resource ownership leaves legacy and foreign records under administrator control', () => {
    assert.equal(canManageResource(undefined, 'alice'), false);
    for (const owner of [undefined, null, '', 'bob'])
      assert.equal(canManageResource(user(), owner), false);
    assert.equal(canManageResource(user(), 'alice'), true);
    assert.equal(canManageResource(user('viewer'), 'alice'), false);
    for (const owner of [null, undefined, 'bob', 'alice'])
      assert.equal(canManageResource(user('admin'), owner), true);
    assert.equal(resourceOwnerLabel(null), '历史记录（未分配创建者）');
    assert.equal(
      resourceOwnerLabel('bob', [{ id: 'bob', username: 'b', display_name: '小王' }]),
      '小王',
    );
  });

  await test('account-revoked queue reasons explain the paused start without exposing raw internal codes', () => {
    for (const reason of [
      'ACCOUNT_EXECUTION_REVOKED',
      'ACCOUNT_DISABLED',
      'ACCOUNT_PERMISSION_REVOKED',
    ]) {
      const html = renderToStaticMarkup(React.createElement(RunQueueReason, { reason }));
      assert.match(html, /排队原因：执行账号/);
      assert.match(html, /暂停启动/);
      assert.match(html, /联系管理员处理/);
      assert.doesNotMatch(html, /ACCOUNT_/);
    }
    assert.equal(renderToStaticMarkup(React.createElement(RunQueueReason, {})), '');
    const unknown = renderToStaticMarkup(
      React.createElement(RunQueueReason, { reason: '<new reason>' }),
    );
    assert.match(unknown, /&lt;new reason&gt;/);
  });

  await test('training deletion is shown only to the creator or administrator including historical records', () => {
    const element = (owner_user_id) =>
      React.createElement(RunDeleteAction, { run: { ...base, owner_user_id } });
    assert.match(render(element('alice')), /删除记录/);
    for (const owner of ['bob', null, undefined]) assert.equal(render(element(owner)), '');
    assert.match(render(element(null), user('admin')), /删除记录/);
    assert.equal(render(element('alice'), user('viewer')), '');
  });

  await test('shared synchronization logs remain readable while retry, cancel and deletion belong to the creator', () => {
    for (const Component of [AssetTransferCard, TransferCard]) {
      const element = (owner, state = 'FAILED') =>
        React.createElement(Component, {
          item: { ...transfer, owner_user_id: owner, state },
          changed() {},
        });
      assert.match(render(element('alice')), /重试同步/);
      assert.match(render(element('alice', 'TRANSFERRING')), /取消同步/);
      for (const owner of ['bob', null]) {
        const html = render(element(owner));
        assert.match(html, /shared task log/);
        assert.doesNotMatch(html, /重试同步|取消同步|删除同步记录/);
      }
      assert.match(render(element(null), user('admin')), /重试同步/);
      assert.doesNotMatch(render(element('alice'), user('viewer')), /重试同步|取消同步/);
    }
  });

  await test('only owned or administrator-controlled checkpoint records can be attached to a deployment', () => {
    const element = (owner_user_id) =>
      React.createElement(TransferCard, {
        item: { ...transfer, owner_user_id, state: 'SUCCEEDED' },
        changed() {},
        useTransfer() {},
      });
    assert.match(render(element('alice')), /使用此权重部署/);
    for (const owner of ['bob', null]) {
      const html = render(element(owner));
      assert.match(html, /shared task log/);
      assert.match(html, /此同步记录仅供查看/);
      assert.doesNotMatch(html, /使用此权重部署/);
    }
    assert.match(render(element('bob'), user('admin')), /使用此权重部署/);
    assert.doesNotMatch(render(element('alice'), user('viewer')), /使用此权重部署/);
  });

  await test('deployment controls and continuing preparation are owner scoped but shared detail stays visible', () => {
    const runtime = {
      ...base,
      state: 'STOPPED',
      checkpoint_path: '/model',
      profile_id: 'profile',
      config_name: 'demo',
      port: 8000,
      checks: [],
      logs: 'shared service log',
    };
    const runtimeElement = (owner) =>
      React.createElement(DeploymentDetail, {
        id: runtime.id,
        initialItem: { ...runtime, owner_user_id: owner },
        changed() {},
        tab: 'test',
      });
    assert.match(render(runtimeElement('alice')), /运行单次推理测试/);
    for (const owner of ['bob', null])
      assert.doesNotMatch(render(runtimeElement(owner)), /运行单次推理测试|启动服务/);
    assert.match(render(runtimeElement(null), user('admin')), /运行单次推理测试/);
    const prepared = (owner) =>
      React.createElement(DeploymentRecordView, {
        record: {
          ...base,
          owner_user_id: owner,
          state: 'DRAFT',
          deployment_id: null,
          deployment: null,
          transfers: [],
          transfer_ids: [],
        },
        changed() {},
        continueConfig() {},
      });
    assert.match(render(prepared('alice')), /继续配置部署/);
    assert.doesNotMatch(render(prepared('bob')), />继续配置部署<|aria-label="删除部署记录/);
    assert.match(render(prepared('bob')), /Shared readable record/);
  });

  await test('successful stale responses cannot continue a recipe writeback under a replacement account', async () => {
    let release;
    const calls = [];
    globalThis.fetch = async (path, options) => {
      calls.push({ path, options });
      return new Promise((resolve) => {
        release = resolve;
      });
    };
    transport.setSessionCsrf('csrf-A', 'alice');
    const client = transport.createOperationClient();
    const saving = (async () => {
      const recipe = await client.post('/recipes', { name: 'A recipe' });
      return client.post(`/recipe-revisions/${recipe.id}/writeback`, { config_name: 'A_config' });
    })();
    transport.setSessionCsrf('csrf-B', 'bob');
    release(new Response('{"id":"revision-A"}', { status: 201 }));
    await assert.rejects(saving, (error) => error.code === 'SESSION_CHANGED');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].options.headers.get('X-Operation-User'), 'alice');
    await assert.rejects(client.post('/recipes', {}), (error) => error.code === 'SESSION_CHANGED');
    assert.equal(calls.length, 1, 'an old operation cannot even start another request');
  });

  await test('aborted operations stop before submitting and per-account assertions follow the active identity', async () => {
    const calls = [];
    globalThis.fetch = async (path, options) => {
      calls.push({ path, options });
      return new Response('{}');
    };
    transport.setSessionCsrf('csrf-B', 'bob');
    const client = transport.createOperationClient();
    const abort = new AbortController();
    abort.abort();
    await assert.rejects(
      client.post('/recipes', {}, { signal: abort.signal }),
      (error) => error.name === 'AbortError',
    );
    assert.equal(calls.length, 0);
    await transport.post('/runs', {});
    assert.equal(calls[0].options.headers.get('X-Operation-User'), 'bob');
  });

  await test('GPU selection respects the cross-machine cap and permits shrinking an over-limit legacy selection', () => {
    const policy = { max_per_user: 2, ttl_seconds: 28800, reserved_count: 2, remaining_count: 0 };
    assert.equal(reservationSelectionLimit(policy, 1, 1), undefined);
    assert.match(reservationSelectionLimit(policy, 1, 2), /合计 3 张/);
    assert.equal(reservationSelectionLimit({ ...policy, reserved_count: 4 }, 4, 2), undefined);
    assert.equal(
      reservationSelectionLimit(undefined, 0, 3),
      undefined,
      'server still validates when policy has not loaded',
    );
  });
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
