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
  const { fetchBackgroundTaskPage, legacyBackgroundTaskPage, initialBackgroundTaskQuery } =
    await server.ssrLoadModule('/src/backgroundTaskPagination.ts');
  const { ApiError } = await server.ssrLoadModule('/src/api.ts');
  const { BackgroundTaskPagination } = await server.ssrLoadModule('/src/BackgroundTasks.tsx');
  const runs = Array.from({ length: 21 }, (_, index) => ({
    id: `run-${index}`,
    name: `训练 ${index}`,
    state: index === 20 ? 'RUNNING' : 'FAILED',
    node_id: 'source',
    template_id: 'openpi-sft',
    created_at: `2026-11-01T00:${String(index).padStart(2, '0')}:00Z`,
  }));
  const transfers = ['FAILED', 'INTERRUPTED'].map((state, index) => ({
    id: `sync-${index}`,
    name: `叠衣同步 ${index}`,
    state,
    source_asset_id: 'asset',
    source_node_id: 'source',
    target_node_id: 'target',
    source_path: '/data/source',
    target_directory: '/data',
    target_path: '/data/copy',
    created_at: `2026-11-01T01:0${index}:00Z`,
  }));
  const nodes = [{ id: 'target', name: 'Thor', hostname: 'target-host' }];
  const jsonResponse = (value, status = 200) => new Response(JSON.stringify(value), { status });

  await test('supported services use only their server page without a legacy load or limit marker', async () => {
    const paths = [];
    const page = {
      items: [],
      total: 1450,
      total_all: 2000,
      page: 63,
      page_size: 20,
      total_pages: 73,
    };
    globalThis.fetch = async (path) => {
      paths.push(path);
      return jsonResponse(page);
    };
    assert.deepEqual(
      await fetchBackgroundTaskPage({ ...initialBackgroundTaskQuery, page: 63 }),
      page,
    );
    assert.equal(paths.length, 1);
    assert.match(paths[0], /^\/api\/v1\/background-tasks\?/);
  });

  await test('only a missing pagination endpoint loads both legacy sources and keeps fixed twenty-row pages', async () => {
    const paths = [];
    const controller = new AbortController();
    globalThis.fetch = async (path, options) => {
      paths.push(path);
      assert.equal(options.signal, controller.signal);
      if (path.startsWith('/api/v1/background-tasks?'))
        return jsonResponse({ detail: 'Not Found' }, 404);
      if (path === '/api/v1/runs') return jsonResponse(runs);
      if (path === '/api/v1/asset-transfers?limit=200') return jsonResponse(transfers);
      throw new Error(`Unexpected path: ${path}`);
    };
    const result = await fetchBackgroundTaskPage(
      initialBackgroundTaskQuery,
      nodes,
      controller.signal,
    );
    assert.deepEqual(paths.slice(1).sort(), ['/api/v1/asset-transfers?limit=200', '/api/v1/runs']);
    assert.equal(result.items.length, 20);
    assert.equal(result.total, 23);
    assert.equal(result.total_all, 23);
    assert.equal(result.total_pages, 2);
    assert.equal(result.history_limited, true);
    assert.deepEqual(
      result.items.slice(0, 3).map((task) => task.key),
      ['data-sync:sync-1', 'data-sync:sync-0', 'training:run-20'],
    );
    assert.equal(result.items[2].item.state, 'RUNNING');
    const last = legacyBackgroundTaskPage(
      { ...initialBackgroundTaskQuery, page: 2 },
      runs,
      transfers,
      nodes,
    );
    assert.equal(last.items.length, 3);
    assert.equal(last.page, 2);
  });

  await test('legacy mode filters the merged records and clamps pages after deletion or no matches', () => {
    const filtered = legacyBackgroundTaskPage(
      {
        page: 8,
        type: 'data-sync',
        status: 'FAILED',
        query: 'THOR',
      },
      runs,
      transfers,
      nodes,
    );
    assert.equal(filtered.total, 2);
    assert.equal(filtered.total_all, 23);
    assert.equal(filtered.page, 1);
    assert.equal(
      filtered.items.every((task) => task.kind === 'data-sync'),
      true,
    );
    const active = legacyBackgroundTaskPage(
      { ...initialBackgroundTaskQuery, status: 'ACTIVE' },
      runs,
      transfers,
    );
    assert.deepEqual(
      active.items.map((task) => task.key),
      ['training:run-20'],
    );
    const afterDeletion = legacyBackgroundTaskPage(
      { ...initialBackgroundTaskQuery, page: 2 },
      runs.slice(0, 18),
      transfers,
    );
    assert.equal(afterDeletion.page, 1);
    assert.equal(afterDeletion.items.length, 20);
    const empty = legacyBackgroundTaskPage(
      { ...initialBackgroundTaskQuery, page: 4, query: 'no-match' },
      runs,
      transfers,
    );
    assert.equal(empty.page, 1);
    assert.equal(empty.total, 0);
    assert.equal(empty.total_pages, 1);
    assert.deepEqual(empty.items, []);
  });

  await test('permission, server and network failures remain errors without any legacy requests', async () => {
    for (const status of [403, 500]) {
      const paths = [];
      globalThis.fetch = async (path) => {
        paths.push(path);
        return jsonResponse({ error: { code: `HTTP_${status}`, message: 'read failed' } }, status);
      };
      await assert.rejects(
        fetchBackgroundTaskPage(initialBackgroundTaskQuery),
        (error) => error instanceof ApiError && error.status === status,
      );
      assert.equal(paths.length, 1);
    }
    let calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      throw new TypeError('network failure');
    };
    await assert.rejects(
      fetchBackgroundTaskPage(initialBackgroundTaskQuery),
      (error) => error instanceof ApiError && error.code === 'NETWORK_ERROR',
    );
    assert.equal(calls, 1);
  });

  await test('legacy-source errors cannot be reported as a successful partial history', async () => {
    const paths = [];
    globalThis.fetch = async (path) => {
      paths.push(path);
      if (path.startsWith('/api/v1/background-tasks?')) return jsonResponse({}, 404);
      if (path === '/api/v1/runs')
        return jsonResponse(
          { error: { code: 'READ_FAILED', message: 'training source unavailable' } },
          500,
        );
      return jsonResponse(transfers);
    };
    await assert.rejects(
      fetchBackgroundTaskPage(initialBackgroundTaskQuery),
      (error) =>
        error instanceof ApiError &&
        error.status === 500 &&
        error.message === 'training source unavailable',
    );
    assert.equal(paths.length, 3);
  });

  await test('fallback pagination explicitly labels its total as loaded records', () => {
    const result = legacyBackgroundTaskPage(initialBackgroundTaskQuery, runs, transfers);
    const html = renderToStaticMarkup(
      React.createElement(BackgroundTaskPagination, { result, changePage() {} }),
    );
    assert.match(html, /显示 1–20 条，共 23 条（已加载）/);
    const normal = renderToStaticMarkup(
      React.createElement(BackgroundTaskPagination, {
        result: { ...result, history_limited: undefined },
        changePage() {},
      }),
    );
    assert.doesNotMatch(normal, /已加载/);
  });
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
