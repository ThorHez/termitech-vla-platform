import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const { gpuReservationRecords } = await server.ssrLoadModule('/src/gpuReservationRecords.ts');
  const row = (gpu_uuid, extra = {}) => ({
    node_id: 'node-a',
    gpu_uuid,
    user_id: 'alice',
    username: 'alice',
    display_name: '小李',
    created_at: '2026-11-01T02:06:28Z',
    ...extra,
  });
  const node = (id = 'node-a', uuids = ['GPU-99', 'GPU-3', 'GPU-0']) => ({
    id,
    name: id === 'node-a' ? '一号训练机' : '二号训练机',
    hostname: id,
    status: 'ONLINE',
    gpus: uuids.map((uuid) => ({ uuid, name: 'A100' })),
  });

  await test('saved batches show exact inventory card numbers and do not mutate API data', () => {
    const nodes = [node()];
    const rows = [row('GPU-0'), row('GPU-99')];
    const before = JSON.stringify({ nodes, rows });
    assert.deepEqual(gpuReservationRecords(nodes, rows), [
      {
        nodeId: 'node-a',
        nodeName: '一号训练机',
        userId: 'alice',
        username: 'alice',
        displayName: '小李',
        gpuUuids: ['GPU-99', 'GPU-0'],
        gpuLabels: ['GPU 0', 'GPU 2'],
        createdAt: '2026-11-01T02:06:28Z',
      },
    ]);
    assert.equal(JSON.stringify({ nodes, rows }), before);
  });

  await test('users, machines and separately created reservation batches remain distinguishable', () => {
    const nodes = [node('node-b'), node()];
    const rows = [
      row('GPU-99', { node_id: 'node-b' }),
      row('GPU-99'),
      row('GPU-3', { user_id: 'bob', username: 'bob', display_name: '小王' }),
      row('GPU-0', { created_at: '2026-11-01T02:10:00Z' }),
    ];
    const records = gpuReservationRecords(nodes, rows);
    assert.equal(records.length, 4);
    assert.deepEqual(
      records.map((r) => [r.nodeId, r.userId, r.gpuLabels, r.createdAt]),
      [
        ['node-a', 'alice', ['GPU 2'], '2026-11-01T02:10:00Z'],
        ['node-a', 'alice', ['GPU 0'], '2026-11-01T02:06:28Z'],
        ['node-a', 'bob', ['GPU 1'], '2026-11-01T02:06:28Z'],
        ['node-b', 'alice', ['GPU 0'], '2026-11-01T02:06:28Z'],
      ],
    );
    assert.deepEqual(gpuReservationRecords(nodes, [...rows].reverse()), records);
  });

  await test('authoritative records retain missing cards and machines for viewing or release', () => {
    const records = gpuReservationRecords(
      [node()],
      [
        row('missing-card'),
        row('removed-node-card', { node_id: 'missing-node', display_name: '' }),
      ],
    );
    const missingCard = records.find((r) => r.nodeId === 'node-a');
    assert.deepEqual(missingCard.gpuLabels, ['未上报 GPU（missing-card）']);
    const missingNode = records.find((r) => r.nodeId === 'missing-node');
    assert.equal(missingNode.nodeName, 'missing-node');
    assert.equal(missingNode.displayName, 'alice');
    assert.deepEqual(missingNode.gpuUuids, ['removed-node-card']);
    assert.deepEqual(gpuReservationRecords([node()], []), []);
  });

  await test('inventory fallback deduplicates each machine/device while keeping the same UUID on other machines', () => {
    const first = node();
    first.gpus[0].user_reservation = row('GPU-99');
    first.gpus[1].user_reservation = row('GPU-99');
    const second = node('node-b');
    second.gpus[0].user_reservation = row('GPU-99', { node_id: 'node-b' });
    const records = gpuReservationRecords([first, second]);
    assert.equal(records.length, 2);
    assert.deepEqual(
      records.map((r) => r.gpuUuids),
      [['GPU-99'], ['GPU-99']],
    );
  });

  await test('overlapping snapshots count a device once and sort timestamps by their actual instants', () => {
    const records = gpuReservationRecords(
      [node()],
      [
        row('GPU-99', { created_at: '2026-11-01T10:00:00+08:00' }),
        row('GPU-99', { created_at: '2026-11-01T02:06:28Z' }),
        row('GPU-3', { created_at: '2026-11-01T09:59:00+08:00' }),
        row('GPU-0', { created_at: '2026-11-01T02:01:00Z' }),
      ],
    );
    assert.equal(
      records.reduce((count, r) => count + r.gpuUuids.length, 0),
      3,
    );
    assert.deepEqual(
      records.map((r) => r.createdAt),
      ['2026-11-01T02:06:28Z', '2026-11-01T02:01:00Z', '2026-11-01T09:59:00+08:00'],
    );
  });
} finally {
  await server.close();
}
