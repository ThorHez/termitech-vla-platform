import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const {
    buildDeploymentWorkspaceEntries,
    parseDeploymentWorkspaceRoute,
    deploymentRecordPath,
    deploymentPreparationPath,
  } = await server.ssrLoadModule('/src/deploymentWorkspace.ts');
  const created = '2026-10-27T12:00:00Z';
  const record = (id, changes = {}) => ({
    id,
    name: id,
    deployment_id: null,
    selected_transfer_id: null,
    transfer_ids: [],
    node_id: 'same-node',
    profile_id: 'same-profile',
    source_run_id: 'same-run',
    state: 'DRAFT',
    created_at: created,
    updated_at: created,
    ...changes,
  });

  await test('only persistent parent records are projected, regardless of runtime and child history', () => {
    const parent = record('parent', {
      deployment_id: 'runtime',
      selected_transfer_id: 'selected',
      transfer_ids: ['cancelled', 'selected'],
    });
    assert.deepEqual(buildDeploymentWorkspaceEntries([parent]), [
      { key: 'record:parent', record: parent },
    ]);
    assert.deepEqual(buildDeploymentWorkspaceEntries([]), []);
  });

  await test('same-run and same-machine parents remain separate explicit records', () => {
    const entries = buildDeploymentWorkspaceEntries([record('parent-a'), record('parent-b')]);
    assert.deepEqual(
      entries.map((entry) => entry.key),
      ['record:parent-a', 'record:parent-b'],
    );
  });

  await test('a parent without runtime remains a normal record and cancelled children create no top-level entries', () => {
    const parent = record('no-runtime', { state: 'CANCELLED', transfer_ids: ['cancelled-child'] });
    assert.deepEqual(
      buildDeploymentWorkspaceEntries([parent]).map((entry) => entry.record.id),
      ['no-runtime'],
    );
  });

  await test('repeated parent receipts deduplicate exact IDs without changing their first accepted snapshot', () => {
    const first = record('same-parent');
    assert.deepEqual(
      buildDeploymentWorkspaceEntries([first, record('same-parent'), record('another')]).map(
        (entry) => entry.record,
      ),
      [first, record('another')],
    );
  });

  await test('parent sorting is chronological, stable, and leaves input records unchanged', () => {
    const records = Object.freeze([
      Object.freeze(record('older', { created_at: '2026-10-27T12:00:00+08:00' })),
      Object.freeze(record('tie-first', { created_at: '2026-10-27T05:00:00Z' })),
      Object.freeze(record('tie-second', { created_at: '2026-10-27T05:00:00Z' })),
      Object.freeze(record('newest', { created_at: '2026-10-27T06:00:00Z' })),
      Object.freeze(record('invalid-date', { created_at: 'invalid' })),
    ]);
    assert.deepEqual(
      buildDeploymentWorkspaceEntries(records).map((entry) => entry.record.id),
      ['newest', 'tie-first', 'tie-second', 'older', 'invalid-date'],
    );
    assert.deepEqual(
      records.map((entry) => entry.id),
      ['older', 'tie-first', 'tie-second', 'newest', 'invalid-date'],
    );
  });

  await test('list, UUID and fixture record routes round-trip without a hash fragment', () => {
    assert.deepEqual(parseDeploymentWorkspaceRoute('/deployments'), { kind: 'list' });
    for (const id of ['095f390c-af65-4cf1-b630-655b8ecb09d2', 'fixture_1', 'record-a']) {
      assert.deepEqual(parseDeploymentWorkspaceRoute(deploymentRecordPath(id)), {
        kind: 'deployment',
        id,
      });
      assert.deepEqual(parseDeploymentWorkspaceRoute(deploymentPreparationPath(id)), {
        kind: 'preparation',
        id,
      });
      assert.ok(deploymentRecordPath(id).startsWith('/deployments/'));
      assert.ok(!deploymentPreparationPath(id).includes('#'));
    }
    assert.deepEqual(parseDeploymentWorkspaceRoute('/deployments/preparations/preparations'), {
      kind: 'preparation',
      id: 'preparations',
    });
    assert.deepEqual(parseDeploymentWorkspaceRoute('/deployments/%66ixture_1'), {
      kind: 'deployment',
      id: 'fixture_1',
    });
  });

  await test('path builders encode IDs while invalid decoded segments fail closed', () => {
    assert.equal(deploymentRecordPath('id/with space'), '/deployments/id%2Fwith%20space');
    assert.equal(
      deploymentPreparationPath('id/with space'),
      '/deployments/preparations/id%2Fwith%20space',
    );
    for (const path of [
      '/runs/record-a',
      '/deployments/',
      '/deployments/preparations',
      '/deployments//record-a',
      '/deployments/record-a/extra',
      '/deployments/preparations/record-a/extra',
      '/deployments/%',
      '/deployments/%E0%A4%A',
      '/deployments/%2F',
      '/deployments/%00',
      '/deployments/%2e%2e',
      '/deployments/preparations/%2F',
      '/deployments/record-a?tab=logs',
      '/deployments/record-a#logs',
      deploymentRecordPath('id/with space'),
      deploymentPreparationPath('id/with space'),
      '/deployments/' + 'a'.repeat(129),
      '',
      null,
      undefined,
    ])
      assert.deepEqual(parseDeploymentWorkspaceRoute(path), { kind: 'not-found' }, String(path));
  });
} finally {
  await server.close();
}
