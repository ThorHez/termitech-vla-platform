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
  const { canDeleteDeploymentRecord, deleteDeploymentRecord } = await server.ssrLoadModule(
    '/src/deploymentRecordDeletion.ts',
  );
  const { setSessionCsrf, ApiError } = await server.ssrLoadModule('/src/api.ts');
  const { DeploymentRecordView, DeploymentWorkspaceList } = await server.ssrLoadModule(
    '/src/PolicyDeployments.tsx',
  );
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const { buildDeploymentWorkspaceEntries } = await server.ssrLoadModule(
    '/src/deploymentWorkspace.ts',
  );
  const record = (changes = {}) => ({
    id: 'record',
    name: '部署准备',
    state: 'READY_TO_DEPLOY',
    node_id: 'machine',
    profile_id: 'openpi',
    source_run_id: 'source',
    deployment_id: null,
    deployment: null,
    selected_transfer_id: null,
    transfer_ids: [],
    transfers: [],
    created_at: '2026-11-01T00:00:00Z',
    updated_at: '2026-11-01T00:00:00Z',
    ...changes,
  });
  const runtime = (changes = {}) => ({
    id: 'runtime',
    name: '服务',
    node_id: 'machine',
    profile_id: 'openpi',
    state: 'STOPPED',
    pending_command: null,
    pid: null,
    endpoint: null,
    checkpoint_path: '/machine/checkpoint',
    config_name: 'generated',
    port: 8000,
    checks: [],
    logs: '',
    created_at: '2026-11-01T00:00:00Z',
    updated_at: '2026-11-01T00:00:00Z',
    ...changes,
  });
  const render = (element, writable = true) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: {
            user: { id: 'test-admin', role: 'admin' },
            can: (permission) => writable && permission === 'training.write',
          },
        },
        React.createElement(DataProvider, { enabled: false }, element),
      ),
    );

  await test('preparation deletion accepts terminal synchronization histories and an empty draft', () => {
    assert.equal(canDeleteDeploymentRecord(record({ state: 'DRAFT' }), null, []), true);
    for (const state of ['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED'])
      assert.equal(canDeleteDeploymentRecord(record(), null, [{ state }]), true, state);
    for (const state of [
      'QUEUED',
      'PREPARING',
      'EXPORTING',
      'TRANSFERRING',
      'VERIFYING',
      'CANCELLING',
      'UNKNOWN',
    ])
      assert.equal(
        canDeleteDeploymentRecord(record(), null, [{ state: 'SUCCEEDED' }, { state }]),
        false,
        state,
      );
  });

  await test('runtime deletion requires a stopped or failed service with no known process or pending operation', () => {
    for (const state of ['STOPPED', 'FAILED'])
      assert.equal(canDeleteDeploymentRecord(record(), runtime({ state }), []), true, state);
    for (const state of ['PENDING', 'STARTING', 'RUNNING', 'STOPPING', 'UNKNOWN', 'OTHER'])
      assert.equal(canDeleteDeploymentRecord(record(), runtime({ state }), []), false, state);
    for (const changes of [
      { pid: 1234 },
      { endpoint: 'ws://localhost:8000' },
      { pending_command: 'POLICY_DEPLOY_START' },
    ])
      assert.equal(canDeleteDeploymentRecord(record(), runtime(changes), []), false);
    assert.equal(
      canDeleteDeploymentRecord(record(), runtime(), [{ state: 'TRANSFERRING' }]),
      false,
    );
  });

  await test('summary actions fail closed for active and unknown states', () => {
    for (const state of [
      'DRAFT',
      'READY_TO_DEPLOY',
      'STOPPED',
      'FAILED',
      'CANCELLED',
      'INTERRUPTED',
    ])
      assert.equal(canDeleteDeploymentRecord(record({ state })), true, state);
    for (const state of ['PREPARING', 'STARTING', 'RUNNING', 'STOPPING', 'UNKNOWN', 'OTHER'])
      assert.equal(canDeleteDeploymentRecord(record({ state })), false, state);
  });

  await test('list and detail provide permitted deletion without exposing it for running services or readers', () => {
    const rows = buildDeploymentWorkspaceEntries([
      record(),
      record({ id: 'active', state: 'RUNNING' }),
    ]);
    const list = (writable) =>
      render(React.createElement(DeploymentWorkspaceList, { entries: rows, nodes: [] }), writable);
    assert.equal((list(true).match(/aria-label="删除部署记录 /g) || []).length, 1);
    assert.doesNotMatch(list(false), /aria-label="删除部署记录 /);
    const detail = (value) =>
      render(
        React.createElement(DeploymentRecordView, {
          record: value,
          changed() {},
          continueConfig() {},
        }),
      );
    assert.match(detail(record()), /aria-label="删除部署记录 部署准备"/);
    assert.doesNotMatch(
      detail(record({ transfers: [{ state: 'TRANSFERRING' }] })),
      /aria-label="删除部署记录 /,
    );
    assert.match(
      detail(record({ deployment_id: 'runtime', deployment: runtime() })),
      /aria-label="删除部署记录 部署准备"/,
    );
    assert.doesNotMatch(
      detail(record({ deployment_id: 'runtime', deployment: runtime({ state: 'RUNNING' }) })),
      /aria-label="删除部署记录 /,
    );
  });

  await test('deletion sends authenticated DELETE to the encoded record and accepts empty 204', async () => {
    setSessionCsrf('test-csrf');
    globalThis.fetch = async (path, options) => {
      assert.equal(path, '/api/v1/deployment-records/record%2F%E4%B8%80');
      assert.equal(options.method, 'DELETE');
      assert.equal(options.credentials, 'same-origin');
      assert.equal(options.headers.get('X-CSRF-Token'), 'test-csrf');
      return new Response(null, { status: 204 });
    };
    await deleteDeploymentRecord('record/一');
    setSessionCsrf();
  });

  await test('active-resource and shared-transfer rejections remain visible errors', async () => {
    for (const conflict of [
      { code: 'DEPLOYMENT_RECORD_ACTIVE', message: '请先停止服务并等待同步任务结束' },
      { code: 'DEPLOYMENT_RECORD_IN_USE', message: '请先删除使用此同步权重的其他部署记录' },
    ]) {
      globalThis.fetch = async () =>
        new Response(JSON.stringify({ error: conflict }), { status: 409 });
      await assert.rejects(deleteDeploymentRecord('record'), (error) => {
        assert.ok(error instanceof ApiError);
        assert.equal(error.code, conflict.code);
        assert.equal(error.status, 409);
        assert.equal(error.message, conflict.message);
        return true;
      });
    }
  });
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
