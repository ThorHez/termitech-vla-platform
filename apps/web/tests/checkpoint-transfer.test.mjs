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
    transferMatches,
    transferTerminal,
    transferBytes,
    hasTransferCapability,
    checkpointSyncAllowed,
    normalizeTransferPath,
    transferForId,
  } = await server.ssrLoadModule('/src/checkpointTransfer.ts');
  const binding = {
    runId: 'run-a',
    checkpoint: '/source/checkpoint',
    nodeId: 'target',
    profileId: 'runtime',
    directory: '/target/models',
  };
  const receipt = {
    source_run_id: 'run-a',
    source_checkpoint_path: '/source/checkpoint',
    target_node_id: 'target',
    target_profile_id: 'runtime',
    target_directory: '/target/models',
  };
  await test('a completed receipt cannot authorize changed source, runtime, target or destination', () => {
    assert.equal(transferMatches(receipt, binding), true);
    for (const key of ['runId', 'checkpoint', 'nodeId', 'profileId', 'directory']) {
      assert.equal(
        transferMatches(receipt, { ...binding, [key]: binding[key] + '-changed' }),
        false,
        key,
      );
    }
    assert.equal(transferMatches(undefined, binding), false);
    assert.equal(
      transferMatches(receipt, {
        ...binding,
        checkpoint: '/source//checkpoint/',
        directory: ' /target/models/ ',
      }),
      true,
    );
  });
  await test('cancellation remains active until confirmed and failure or interruption can finish', () => {
    for (const state of [
      'QUEUED',
      'PREPARING',
      'EXPORTING',
      'TRANSFERRING',
      'VERIFYING',
      'CANCELLING',
    ])
      assert.equal(transferTerminal(state), false);
    for (const state of ['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED'])
      assert.equal(transferTerminal(state), true);
    assert.equal(transferBytes(NaN), '0 B');
    assert.equal(transferBytes(1024 ** 3), '1 GiB');
  });
  await test('the create receipt remains visible before polling and stale polling cannot select another task', () => {
    const created = { ...receipt, id: 'created', state: 'QUEUED' };
    const stale = { ...receipt, id: 'previous', state: 'SUCCEEDED' };
    const polled = { ...created, state: 'TRANSFERRING', progress_pct: 17 };
    assert.equal(transferForId(created.id, undefined, created), created);
    assert.equal(transferForId(created.id, stale, created), created);
    assert.equal(transferForId(created.id, polled, created), polled);
    assert.equal(transferForId('', stale, created), undefined);
    assert.equal(transferForId('new-unrelated-id', stale, created), undefined);
    assert.equal(
      transferMatches(transferForId(created.id, stale, created), {
        ...binding,
        checkpoint: '/different/checkpoint',
      }),
      false,
    );
  });
  await test('request paths and receipt matching share normalization without conflating root and empty input', () => {
    assert.equal(normalizeTransferPath(' /source//checkpoint/// '), '/source/checkpoint');
    assert.equal(normalizeTransferPath(' /target/models/// '), '/target/models');
    assert.equal(normalizeTransferPath('///'), '/');
    assert.equal(normalizeTransferPath(' '), '');
    assert.equal(
      transferMatches(
        {
          ...receipt,
          source_checkpoint_path: ' /source//checkpoint/ ',
          target_directory: '/target//models///',
        },
        binding,
      ),
      true,
    );
    assert.equal(
      transferMatches({ ...receipt, target_directory: '/' }, { ...binding, directory: '' }),
      false,
    );
  });
  await test('old or missing target capabilities never authorize portable model deployment', () => {
    assert.equal(hasTransferCapability(undefined, 'POLICY_DEPLOY_PORTABLE_V1'), false);
    assert.equal(
      hasTransferCapability(
        { metrics: { capabilities: ['POLICY_DEPLOY_START'] } },
        'POLICY_DEPLOY_PORTABLE_V1',
      ),
      false,
    );
    assert.equal(
      hasTransferCapability(
        { metrics: { capabilities: ['POLICY_DEPLOY_PORTABLE_V1'] } },
        'POLICY_DEPLOY_PORTABLE_V1',
      ),
      true,
    );
  });
  await test('model import and synchronization require no GPU selection or idle GPU', () => {
    const sourceNode = {
      id: 'source',
      status: 'ONLINE',
      metrics: { capabilities: ['POLICY_EXPORT'] },
    };
    const targetNode = {
      id: 'target',
      status: 'ONLINE',
      metrics: { capabilities: ['POLICY_DEPLOY_PORTABLE_V1'] },
    };
    const request = {
      sourceRun: { node_id: sourceNode.id },
      sourceNode,
      targetNode,
      profileSelected: true,
      directory: '/target/models',
      checkpoint: '/source/checkpoint',
      syncing: false,
      busy: false,
    };
    for (const gpus of [
      undefined,
      [],
      [
        {
          uuid: 'busy',
          reserved_by: 'training',
          processes: [{ pid: 10, kind: 'compute' }],
          memory_free_mb: 0,
        },
      ],
    ]) {
      assert.equal(
        checkpointSyncAllowed({
          ...request,
          sourceNode: { ...sourceNode, gpus },
          targetNode: { ...targetNode, gpus },
        }),
        true,
      );
    }
    assert.equal(checkpointSyncAllowed({ ...request, busy: true }), false);
    assert.equal(checkpointSyncAllowed({ ...request, syncing: true }), false);
    assert.equal(
      checkpointSyncAllowed({ ...request, targetNode: { ...targetNode, status: 'OFFLINE' } }),
      false,
    );
    assert.equal(checkpointSyncAllowed({ ...request, profileSelected: false }), false);
  });
} finally {
  await server.close();
}
