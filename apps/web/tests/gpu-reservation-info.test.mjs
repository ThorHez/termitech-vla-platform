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
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const info = await server.ssrLoadModule('/src/GpuReservations.tsx');
  const rows = [4, 5, 6, 7].map((index) => ({
    node_id: 'machine',
    gpu_uuid: `GPU-${index}`,
    user_id: 'alice',
    username: 'alice',
    display_name: '小李',
    created_at: '2026-11-01T02:06:28Z',
    expires_at: '2026-11-01T10:06:28Z',
  }));
  const node = {
    id: 'machine',
    name: 'T001host',
    hostname: 'server',
    status: 'ONLINE',
    gpus: Array.from({ length: 8 }, (_, index) => ({
      uuid: `GPU-${index}`,
      name: 'A100',
      user_reservation: rows.find((row) => row.gpu_uuid === `GPU-${index}`) || null,
    })),
  };
  const render = (component, props, userId = 'alice', permissions = ['training.write']) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { user: { id: userId }, can: (permission) => permissions.includes(permission) } },
        React.createElement(component, props),
      ),
    );

  await test('save receipt explicitly identifies persisted cards and distinguishes release from reservation', () => {
    const saved = render(info.GpuReservationSaveReceipt, { node, rows });
    assert.match(saved, /role="status"/);
    assert.match(saved, /预留已保存，当前为你预留 4 张 GPU/);
    assert.match(saved, /GPU 4、GPU 5、GPU 6、GPU 7/);
    assert.match(saved, /小李/);
    assert.match(saved, /预留到期自动释放/);
    assert.match(saved, /到期/);
    const released = render(info.GpuReservationSaveReceipt, { node, rows: [] });
    assert.match(released, /已释放你的全部预留/);
    assert.doesNotMatch(released, /GPU 4|当前为你预留/);
  });
  await test('run details report only reservations on this execution GPU set', () => {
    const html = render(info.RunGpuReservationNotice, { node, gpuUuids: ['GPU-4', 'GPU-5'] });
    assert.match(html, /用户预留：小李/);
    assert.match(html, /GPU 4、GPU 5/);
    assert.doesNotMatch(html, /GPU 6|GPU 7/);
    assert.match(html, /查看预留记录/);
  });
} finally {
  await server.close();
}
