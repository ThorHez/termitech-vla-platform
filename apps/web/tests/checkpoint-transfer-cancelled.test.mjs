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
  const { TransferGroups, TransferCard } = await server.ssrLoadModule(
    '/src/CheckpointTransfers.tsx',
  );
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const receipt = (id, state) => ({
    id,
    name: id,
    state,
    source_node_id: 'source',
    target_node_id: 'target',
    target_directory: '/target/models',
    target_path: null,
    bytes_total: 1000,
    bytes_transferred: 26,
    progress_pct: 2.6,
    files_total: 10,
    files_transferred: 0,
    attempt: 3,
    logs: '保留的同步日志',
    created_at: '2026-10-27T10:00:00Z',
    updated_at: '2026-10-27T10:01:24Z',
  });
  const render = (element) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: { can: (permission) => permission === 'training.write' },
        },
        React.createElement(DataProvider, { enabled: false }, element),
      ),
    );
  const group = (items) =>
    render(React.createElement(TransferGroups, { items, changed() {}, useTransfer() {} }));

  await test('only cancelled records move into a default-collapsed history with retry and logs retained', () => {
    const html = group([
      receipt('cancelled', 'CANCELLED'),
      receipt('active', 'TRANSFERRING'),
      { ...receipt('failed', 'FAILED'), error: { code: 'SSH_FAILED', message: '真实失败原因' } },
      receipt('success', 'SUCCEEDED'),
    ]);
    const history = html.indexOf('<details class="checkpoint-transfer-history"');
    assert.ok(history > 0);
    for (const id of ['active', 'failed', 'success'])
      assert.ok(html.indexOf(`aria-label="权重同步 ${id}"`) < history, id);
    assert.ok(html.indexOf('aria-label="权重同步 cancelled"') > history);
    assert.match(html.slice(history), /已取消记录（1）/);
    assert.match(html.slice(history), /已停止，保留记录和部分文件供重试/);
    assert.doesNotMatch(html, /<details class="checkpoint-transfer-history"[^>]*\bopen/);
    assert.match(html.slice(history), /重试同步/);
    assert.match(html.slice(history), /保留的同步日志/);
    assert.ok(html.indexOf('真实失败原因') < history);
  });

  await test('a polled cancellation leaves the normal list and appears in a closed history', () => {
    const before = group([receipt('changed-task', 'TRANSFERRING')]);
    assert.doesNotMatch(before, /checkpoint-transfer-history/);
    const after = group([receipt('changed-task', 'CANCELLED')]);
    assert.ok(
      after.indexOf('checkpoint-transfer-history') <
        after.indexOf('aria-label="权重同步 changed-task"'),
    );
    assert.doesNotMatch(after, /<details class="checkpoint-transfer-history"[^>]*\bopen/);
    assert.match(after, /停止时进度[^]*2\.6%/);
    assert.match(after, /已取消，传输已停止/);
    assert.doesNotMatch(after, />取消同步</);
  });

  await test('a compact cancelled card collapses details while other compact cards and real errors remain visible', () => {
    const cancelled = render(
      React.createElement(TransferCard, {
        item: receipt('cancelled', 'CANCELLED'),
        changed() {},
        compact: true,
      }),
    );
    assert.match(cancelled, /^<details class="checkpoint-transfer-cancelled-compact"/);
    assert.doesNotMatch(cancelled, /^<details[^>]*\bopen/);
    assert.match(cancelled, /展开查看停止时进度、日志或重试/);
    assert.match(cancelled, /重试同步/);
    for (const state of ['TRANSFERRING', 'FAILED', 'SUCCEEDED']) {
      const item = {
        ...receipt(state, state),
        error: state === 'FAILED' ? { code: 'SSH_FAILED', message: '真实失败原因' } : null,
      };
      const html = render(React.createElement(TransferCard, { item, changed() {}, compact: true }));
      assert.match(html, /^<article/);
      assert.doesNotMatch(html, /checkpoint-transfer-cancelled-compact/);
      if (state === 'FAILED') assert.match(html, /role="alert"[^]*真实失败原因/);
    }
  });

  await test('a cancelled transfer with no byte total does not invent zero progress or file counts', () => {
    const item = {
      ...receipt('never-started', 'CANCELLED'),
      bytes_total: 0,
      bytes_transferred: 0,
      progress_pct: 0,
    };
    const html = render(React.createElement(TransferCard, { item, changed() {} }));
    assert.match(html, /已取消，传输已停止/);
    assert.match(html, /传输尚未开始/);
    assert.doesNotMatch(html, /role="progressbar"|0\.0%|0 B|个文件/);
  });
} finally {
  await server.close();
}
