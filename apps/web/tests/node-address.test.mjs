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
  const { nodeAddressText, matchesNodeQuery } = await server.ssrLoadModule('/src/nodeAddress.ts');
  const node = {
    id: 'machine-a',
    name: '一号训练机',
    hostname: 'gpu-server',
    status: 'OFFLINE',
    connection_address: { host: '192.0.2.15', port: 2201, source: 'ssh' },
  };
  await test('connection text preserves SSH ports and brackets IPv6 targets', () => {
    assert.equal(nodeAddressText(node), '192.0.2.15:2201');
    assert.equal(
      nodeAddressText({
        ...node,
        connection_address: { host: '2001:db8::15', port: 2201, source: 'ssh' },
      }),
      '[2001:db8::15]:2201',
    );
    assert.equal(
      nodeAddressText({
        ...node,
        connection_address: { host: '2001:db8::15', port: null, source: 'hostname' },
      }),
      '2001:db8::15',
    );
    assert.equal(nodeAddressText({ ...node, connection_address: null }), '');
  });
  await test('machine search finds IP, endpoint, machine name and hostname while offline', () => {
    for (const query of ['192.0.2.15', '.15:2201', '一号', ' GPU-SERVER ', '']) {
      assert.equal(matchesNodeQuery(node, query), true, query);
    }
    assert.equal(matchesNodeQuery(node, '192.0.2.99'), false);
    assert.equal(matchesNodeQuery({ ...node, connection_address: null }, '192.0.2.15'), false);
  });
} finally {
  await server.close();
}
