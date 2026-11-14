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
  const { AlertList } = await server.ssrLoadModule('/src/WorkspaceAlerts.tsx');
  const diskAlert = (i, details) => ({
    id: `disk:${i}`,
    code: 'DISK_CAPACITY_HIGH',
    severity: 'warning',
    node_id: `machine-${i}`,
    run_id: null,
    message: `机器 ${i} 的磁盘 /data${i} 已使用 94%`,
    details,
  });

  await test('all disk warnings appear before the regular alert preview', () => {
    const regular = Array.from({ length: 6 }, (_, i) => ({
      id: `training:${i}`,
      code: 'TRAINING_FAILED',
      message: `训练 ${i} 失败`,
      run_id: `training-${i}`,
    }));
    const disks = Array.from({ length: 6 }, (_, i) => diskAlert(i));
    const html = renderToStaticMarkup(
      React.createElement(AlertList, { alerts: [...regular, ...disks] }),
    );
    assert.equal(html.match(/磁盘空间告警/g).length, 6);
    assert.match(html, /机器 5 的磁盘 \/data5 已使用 94%/);
    assert.ok(html.indexOf('机器 5 的磁盘') < html.indexOf('训练 0 失败'));
    assert.match(html, /href="#\/nodes\/machine-5"/);
    assert.match(html, /href="#\/runs\/training-0"/);
    assert.doesNotMatch(html, /训练 4 失败/);
    assert.match(html, /展开其余 2 条异常/);
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
