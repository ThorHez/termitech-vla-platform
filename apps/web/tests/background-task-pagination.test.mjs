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

try {
  const {
    BACKGROUND_TASK_PAGE_SIZE,
    backgroundTaskPagePath,
    backgroundTaskPageRange,
    backgroundTaskQueryReducer,
    initialBackgroundTaskQuery,
  } = await server.ssrLoadModule('/src/backgroundTaskPagination.ts');
  const { BackgroundTaskPagination, BackgroundTasksPage } = await server.ssrLoadModule(
    '/src/BackgroundTasks.tsx',
  );
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const response = (changes = {}) => ({
    items: [],
    total: 65,
    total_all: 1450,
    page: 2,
    page_size: 20,
    total_pages: 4,
    ...changes,
  });
  const renderPagination = (value, loading = false) =>
    renderToStaticMarkup(
      React.createElement(BackgroundTaskPagination, { result: value, loading, changePage() {} }),
    );
  const disabledButtons = (html) => (html.match(/<button[^>]* disabled=""/g) || []).length;

  await test('requests use the paged server endpoint with a fixed page size and encoded filters', () => {
    assert.equal(BACKGROUND_TASK_PAGE_SIZE, 20);
    const path = backgroundTaskPagePath({
      page: 63,
      type: 'data-sync',
      status: 'FAILED',
      query: ' Thor & 叠衣 ',
    });
    const request = new URL(path, 'http://localhost');
    assert.equal(request.pathname, '/background-tasks');
    assert.equal(request.searchParams.get('page'), '63');
    assert.equal(request.searchParams.get('page_size'), '20');
    assert.equal(request.searchParams.get('type'), 'data-sync');
    assert.equal(request.searchParams.get('status'), 'FAILED');
    assert.equal(request.searchParams.get('q'), 'Thor & 叠衣');
  });

  await test('changing status, type or search always returns to page one without losing other filters', () => {
    const current = { page: 8, type: 'training', status: 'FAILED', query: 'cloth' };
    assert.deepEqual(backgroundTaskQueryReducer(current, { kind: 'status', value: 'ACTIVE' }), {
      ...current,
      page: 1,
      status: 'ACTIVE',
    });
    assert.deepEqual(backgroundTaskQueryReducer(current, { kind: 'type', value: 'data-sync' }), {
      ...current,
      page: 1,
      type: 'data-sync',
    });
    assert.deepEqual(backgroundTaskQueryReducer(current, { kind: 'query', value: 'Thor' }), {
      ...current,
      page: 1,
      query: 'Thor',
    });
    assert.deepEqual(initialBackgroundTaskQuery, {
      page: 1,
      type: 'ALL',
      status: 'ALL',
      query: '',
    });
  });

  await test('polling preserves the selected page while deletion follows the server clamp', () => {
    const current = { ...initialBackgroundTaskQuery, page: 3 };
    assert.equal(
      backgroundTaskQueryReducer(current, { kind: 'receipt', request: current, page: 3 }),
      current,
    );
    const afterDeletion = backgroundTaskQueryReducer(current, {
      kind: 'receipt',
      request: current,
      page: 2,
    });
    assert.equal(afterDeletion.page, 2);
    assert.equal(
      new URL(backgroundTaskPagePath(afterDeletion), 'http://localhost').searchParams.get('page'),
      '2',
    );
    const empty = backgroundTaskQueryReducer(afterDeletion, {
      kind: 'receipt',
      request: afterDeletion,
      page: 1,
    });
    assert.equal(empty.page, 1);
  });

  await test('late receipts for an old filter or old page cannot move the current selection', () => {
    const old = { ...initialBackgroundTaskQuery, page: 6 };
    const current = backgroundTaskQueryReducer(old, { kind: 'query', value: 'Thor' });
    assert.equal(
      backgroundTaskQueryReducer(current, { kind: 'receipt', request: old, page: 5 }),
      current,
    );
    const next = backgroundTaskQueryReducer(current, { kind: 'page', page: 2 });
    assert.equal(
      backgroundTaskQueryReducer(next, { kind: 'receipt', request: current, page: 1 }),
      next,
    );
    for (const page of [0, -1, 1.5, NaN, Infinity])
      assert.equal(backgroundTaskQueryReducer(next, { kind: 'page', page }), next);
  });

  await test('ranges use the server total for empty, full and final pages beyond old list caps', () => {
    assert.deepEqual(backgroundTaskPageRange(response({ total: 0, page: 1, total_pages: 1 })), {
      start: 0,
      end: 0,
    });
    assert.deepEqual(backgroundTaskPageRange(response()), { start: 21, end: 40 });
    assert.deepEqual(backgroundTaskPageRange(response({ total: 41, page: 3 })), {
      start: 41,
      end: 41,
    });
    assert.deepEqual(backgroundTaskPageRange(response({ total: 1450, page: 63 })), {
      start: 1241,
      end: 1260,
    });
  });

  await test('pagination exposes accurate range and boundaries, and disables navigation while loading', () => {
    const middle = renderPagination(response());
    assert.match(middle, /显示 21–40 条，共 65 条/);
    assert.match(middle, /第 2 \/ 4 页/);
    assert.equal(disabledButtons(middle), 0);
    assert.equal(disabledButtons(renderPagination(response({ page: 1 }))), 1);
    assert.equal(disabledButtons(renderPagination(response({ page: 4 }))), 1);
    assert.equal(disabledButtons(renderPagination(response(), true)), 2);
    assert.equal(disabledButtons(renderPagination(undefined, true)), 2);
    const empty = renderPagination(response({ total: 0, page: 1, total_pages: 1 }));
    assert.match(empty, /显示 0–0 条，共 0 条/);
    assert.equal(disabledButtons(empty), 2);
  });

  await test('page keeps creation and search controls while replacing capped-history copy with fixed paging', () => {
    const html = renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: () => true } },
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(BackgroundTasksPage, { createRun() {} }),
        ),
      ),
    );
    assert.match(html, /新建训练/);
    assert.match(html, /aria-label="搜索任务"[^>]*maxLength="200"/i);
    assert.match(html, /每页固定显示 20 条/);
    assert.match(html, /每 3 秒更新当前页/);
    assert.match(html, /aria-label="任务分页"/);
    assert.doesNotMatch(html, /最近 200 条/);
  });
} finally {
  await server.close();
}
