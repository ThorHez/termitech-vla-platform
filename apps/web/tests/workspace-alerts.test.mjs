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
  const { AlertList, WorkspaceAlerts } = await server.ssrLoadModule('/src/WorkspaceAlerts.tsx');
  const { deduplicateDiskAlerts } = await server.ssrLoadModule('/src/workspaceAlertData.ts');
  const diskAlert = (i, details) => ({
    id: `disk:${i}`,
    code: 'DISK_CAPACITY_HIGH',
    severity: 'warning',
    node_id: `machine-${i}`,
    run_id: null,
    message: `机器 ${i} 的磁盘 /data${i} 已使用 94%`,
    details,
  });

  await test('the workspace previews the first four combined alerts in controller order', () => {
    const regular = Array.from({ length: 6 }, (_, i) => ({
      id: `training:${i}`,
      code: 'TRAINING_FAILED',
      message: `训练 ${i} 失败`,
      run_id: `training-${i}`,
    }));
    const alerts = [diskAlert(0), regular[0], diskAlert(1), regular[1], diskAlert(2), regular[2]];
    const html = renderToStaticMarkup(
      React.createElement(WorkspaceAlerts, { alerts, enabled: true, refresh() {} }),
    );
    assert.equal(html.match(/class="alert-item"/g).length, 4);
    assert.ok(html.indexOf('机器 0 的磁盘') < html.indexOf('训练 0 失败'));
    assert.ok(html.indexOf('训练 0 失败') < html.indexOf('机器 1 的磁盘'));
    assert.match(html, /href="#\/nodes\/machine-1"/);
    assert.match(html, /href="#\/runs\/training-0"/);
    assert.doesNotMatch(html, /机器 2 的磁盘|训练 2 失败/);
    assert.match(html, /class="count-pill">6<\/span>/);
    assert.match(html, /aria-controls="workspace-alert-list" aria-expanded="false">查看全部/);
  });

  await test('there is no expand control when all alerts already fit in the preview', () => {
    const html = renderToStaticMarkup(
      React.createElement(WorkspaceAlerts, {
        alerts: Array.from({ length: 4 }, (_, i) => diskAlert(i)),
        enabled: true,
        refresh() {},
      }),
    );
    assert.equal(html.match(/class="alert-item"/g).length, 4);
    assert.doesNotMatch(html, /查看全部/);
  });

  await test('disk dedup accepts equivalent canonical mounts on the same node only', () => {
    const old = { ...diskAlert(0, { mountpoint: '/data' }), timestamp: '2026-11-14T02:00:00Z' };
    const recent = {
      ...old,
      id: 'recent',
      details: { mountpoint: '//data/./' },
      timestamp: '2026-11-14T02:01:00Z',
    };
    const stale = { ...old, id: 'stale', timestamp: '2026-11-14T02:00:30Z' };
    const otherNode = { ...recent, id: 'other', node_id: 'other-node' };
    const invalid = { ...recent, id: 'invalid', details: { mountpoint: '/data/../data' } };
    const legacy = { ...old, id: 'legacy', details: undefined };
    const regular = { code: 'TRAINING_FAILED' };
    assert.deepEqual(
      deduplicateDiskAlerts([old, regular, recent, stale, otherNode, invalid, legacy]),
      [recent, regular, otherNode, invalid, legacy],
    );
  });

  await test('disk warnings show the mounted filesystem, usage, available capacity and total', () => {
    const alert = diskAlert(0, {
      mountpoint: '/data/<jobs>',
      usage_percent: 99.95,
      free_bytes: 0,
      total_bytes: 4 * 1024 ** 4,
    });
    const html = renderToStaticMarkup(
      React.createElement(AlertList, {
        alerts: [alert],
        nodes: [{ id: alert.node_id, name: '研发机器' }],
        showNodeLink: false,
      }),
    );
    assert.match(html, /研发机器 的磁盘 <code>\/data\/&lt;jobs&gt;<\/code> 已使用 99\.95%/);
    assert.match(html, /可用 0 B \/ 共 4 TiB/);
    assert.doesNotMatch(html, /查看机器|展开其余/);
  });
} finally {
  await server.close();
}
