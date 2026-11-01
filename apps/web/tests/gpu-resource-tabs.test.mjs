import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
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
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { GpuResourceTabs, GpuReservationDeviceTable } = await server.ssrLoadModule('/src/NodeGpuResources.tsx');
  const rows = [4, 5, 6, 7].map((index) => ({
    node_id: 'machine', gpu_uuid: `uuid-${index}`, user_id: 'alice', username: 'alice',
    display_name: '小李', created_at: '2026-11-01T02:06:28Z',
  }));
  const node = {
    id: 'machine', name: 'T001host', hostname: 'server', status: 'ONLINE',
    gpus: Array.from({ length: 8 }, (_, index) => ({
      uuid: `uuid-${index}`, name: 'NVIDIA A100', memory_total_mb: 81920,
      memory_used_mb: 0, memory_free_mb: 81920, utilization_pct: 0, processes: [],
      user_reservation: rows.find((row) => row.gpu_uuid === `uuid-${index}`) || null,
    })),
  };
  const render = (Component, props) => renderToStaticMarkup(
    React.createElement(AuthContext.Provider, {
      value: { user: { id: 'alice' }, can: (permission) => permission === 'training.write' },
    }, React.createElement(Component, props)),
  );
  const tabs = (tab) => render(GpuResourceTabs, {
    node, tab, onTabChange() {}, onManage() {}, reservations: rows,
    children: React.createElement('p', null, 'GPU_STATUS_CHILDREN'),
  });
  const bodyRows = (html) => [...(html.match(/<tbody>([\s\S]*?)<\/tbody>/)?.[1] || '').matchAll(/<tr(?:\s|>)/g)].length;

  await test('GPU status is the compact default and reservations stay behind an accessible tab', () => {
    const html = tabs('status');
    assert.equal([...html.matchAll(/role="tabpanel"/g)].length, 1);
    const tabButtons = [...html.matchAll(/<button\b[^>]*role="tab"[^>]*>[\s\S]*?<\/button>/g)].map((match) => match[0]);
    assert.equal(tabButtons.length, 2);
    assert.match(tabButtons.find((button) => button.includes('GPU 状态')), /aria-selected="true"/);
    assert.match(tabButtons.find((button) => button.includes('预留信息')), /aria-selected="false"/);
    assert.match(html, /GPU_STATUS_CHILDREN/);
    assert.equal(bodyRows(html), 0);
    assert.doesNotMatch(html, /小李|uuid-4/);
  });

  await test('reservation tab renders one panel containing every card rather than just reserved cards', () => {
    const html = tabs('reservations');
    assert.equal([...html.matchAll(/role="tabpanel"/g)].length, 1);
    assert.equal(bodyRows(html), 8);
    for (let index = 0; index < 8; index += 1) {
      assert.match(html, new RegExp(`GPU\\s+${index}`));
      assert.match(html, new RegExp(`uuid-${index}`));
    }
    assert.match(html, /小李/);
    assert.doesNotMatch(html, /GPU_STATUS_CHILDREN/);
    const selected = [...html.matchAll(/<button\b[^>]*role="tab"[^>]*>[\s\S]*?<\/button>/g)].map((match) => match[0]);
    assert.match(selected.find((button) => button.includes('预留信息')), /aria-selected="true"/);
  });

  await test('missing reserved devices remain visible once, alongside all current inventory', () => {
    const html = render(GpuReservationDeviceTable, {
      node,
      reservations: [...rows, { ...rows[0], gpu_uuid: 'missing-device' }],
    });
    assert.equal(bodyRows(html), 9);
    assert.match(html, /missing-device/);
    assert.match(html, /未上报/);
  });

  await test('an authoritative empty response clears old owner labels without hiding public cards', () => {
    const fallback = render(GpuReservationDeviceTable, { node });
    assert.match(fallback, /小李/);
    const cleared = render(GpuReservationDeviceTable, { node, reservations: [] });
    assert.equal(bodyRows(cleared), 8);
    assert.doesNotMatch(cleared, /小李|alice/);
  });

  await test('repeated and overlapping polling snapshots cannot multiply the GPU rows', () => {
    for (const reservations of [rows, rows.map((row) => ({ ...row })), [...rows, ...rows], [...rows, { ...rows[0], node_id: 'other-machine', display_name: '其他机器用户' }]]) {
      const html = render(GpuReservationDeviceTable, { node, reservations });
      assert.equal(bodyRows(html), 8);
      assert.equal([...html.matchAll(/小李/g)].length, 4);
      assert.doesNotMatch(html, /其他机器用户/);
    }
    const missing = { ...rows[0], gpu_uuid: 'missing-device' };
    assert.equal(bodyRows(render(GpuReservationDeviceTable, { node, reservations: [...rows, missing, missing] })), 9);
  });

  await test('detail polling components have distinct sibling identities and no flat record mount remains', async () => {
    const source = await readFile(new URL('../src/pages.tsx', import.meta.url), 'utf8');
    assert.doesNotMatch(source, /<NodeGpuReservationRecords\b/);
    assert.doesNotMatch(source, /<GpuReservationRecordsPanel\b/);
    assert.doesNotMatch(source, /<GpuReservationSummary\b/);
    assert.match(source, /<ResourceHistory\s+key=\{`history-\$\{id\}`\}/);
    assert.match(source, /<NodeGpuResources\s+key=\{`gpu-\$\{id\}`\}/);
  });
} finally {
  await server.close();
}
