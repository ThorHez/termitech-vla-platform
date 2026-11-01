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
  const { canDeleteRunRecord, deleteRunRecord } = await server.ssrLoadModule('/src/runDeletion.ts');
  const { setSessionCsrf, ApiError } = await server.ssrLoadModule('/src/api.ts');
  const { BackgroundTaskTable } = await server.ssrLoadModule('/src/BackgroundTasks.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { buildBackgroundTasks } = await server.ssrLoadModule('/src/backgroundTaskEntries.ts');
  const run = (state) => ({
    id: state,
    name: `${state} 训练`,
    state,
    node_id: 'machine',
    template_id: 'smoke',
    created_at: '2026-11-01T00:00:00Z',
  });

  await test('only confirmed terminal training states permit deleting records', () => {
    for (const state of ['SUCCEEDED', 'FAILED', 'CANCELLED'])
      assert.equal(canDeleteRunRecord(run(state)), true, state);
    for (const state of ['QUEUED', 'STARTING', 'RUNNING', 'CANCELLING', 'UNKNOWN', '', 'OTHER'])
      assert.equal(canDeleteRunRecord(run(state)), false, state);
  });

  await test('training list exposes deletion only for finished training and permitted accounts', () => {
    const tasks = buildBackgroundTasks(
      ['SUCCEEDED', 'FAILED', 'CANCELLED', 'QUEUED', 'RUNNING', 'UNKNOWN'].map(run),
      [
        {
          id: 'sync',
          name: '数据同步',
          state: 'SUCCEEDED',
          source_node_id: 'source',
          target_node_id: 'target',
          created_at: '2026-11-01T00:00:00Z',
        },
      ],
    );
    const render = (writable) =>
      renderToStaticMarkup(
        React.createElement(
          AuthContext.Provider,
          {
            value: { can: (permission) => writable && permission === 'training.write' },
          },
          React.createElement(BackgroundTaskTable, { tasks }),
        ),
      );
    assert.equal((render(true).match(/aria-label="删除训练记录 /g) || []).length, 3);
    assert.doesNotMatch(render(false), /aria-label="删除训练记录 /);
    for (const state of ['QUEUED', 'RUNNING', 'UNKNOWN'])
      assert.doesNotMatch(render(true), new RegExp(`aria-label="删除训练记录 ${state} 训练"`));
  });

  await test('deletion sends authenticated DELETE for the encoded record and accepts empty 204', async () => {
    setSessionCsrf('test-csrf');
    globalThis.fetch = async (path, options) => {
      assert.equal(path, '/api/v1/runs/run%2F%E4%B8%80');
      assert.equal(options.method, 'DELETE');
      assert.equal(options.credentials, 'same-origin');
      assert.equal(options.headers.get('X-CSRF-Token'), 'test-csrf');
      assert.equal(options.body, undefined);
      return new Response(null, { status: 204 });
    };
    await deleteRunRecord('run/一');
    setSessionCsrf();
  });

  await test('server conflicts remain visible errors instead of reporting deletion success', async () => {
    for (const conflict of [
      { code: 'RUN_NOT_TERMINAL', message: '请先取消训练并等待结束。' },
      {
        code: 'RUN_HAS_DEPLOYMENTS',
        message: '该训练任务有关联部署记录，请先删除部署记录后再删除训练任务',
      },
    ]) {
      globalThis.fetch = async () =>
        new Response(JSON.stringify({ error: conflict }), { status: 409 });
      await assert.rejects(deleteRunRecord('protected-run'), (error) => {
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
