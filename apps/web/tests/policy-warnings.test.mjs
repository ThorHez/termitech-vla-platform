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
  const { checkpointSourceWarnings, checkpointSourceWarning } = await server.ssrLoadModule(
    '/src/policyWarningInfo.ts',
  );
  const { PolicyWarnings } = await server.ssrLoadModule('/src/PolicyWarnings.tsx');
  const { TransferCard } = await server.ssrLoadModule('/src/CheckpointTransfers.tsx');
  const { DeploymentDetail } = await server.ssrLoadModule('/src/PolicyDeployments.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const { sourceDeploymentBlocker } = await server.ssrLoadModule('/src/policyDeployment.ts');
  const { checkpointSyncAllowed, transferMatches, transferForId } = await server.ssrLoadModule(
    '/src/checkpointTransfer.ts',
  );
  const run = {
    id: 'cancelled-no-attempt',
    node_id: 'source',
    template_id: 'openpi-sft',
    state: 'CANCELLED',
    code_expectation: {
      recipe: 'auto_pi05',
      confirmed_config_sha256: 'a'.repeat(64),
      automatic_preflight: {
        plan: {
          datasets: [],
          generated_config: {
            schema_version: 1,
            language: 'python',
            filename: 'training_config.py',
            source: 'fixture = True',
            sha256: 'a'.repeat(64),
          },
        },
      },
    },
  };
  const renderInWorkspace = (element) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: { can: (permission) => permission === 'training.write' },
        },
        React.createElement(DataProvider, { enabled: false }, element),
      ),
    );

  await test('missing attempts or outputs warn before creation without blocking a confirmed training configuration', () => {
    assert.deepEqual(checkpointSourceWarnings(run, '/source/unrecorded/59999'), [
      checkpointSourceWarning,
    ]);
    assert.equal(sourceDeploymentBlocker(run), undefined);
    assert.deepEqual(checkpointSourceWarnings(undefined, '/source/checkpoint'), []);
    assert.deepEqual(checkpointSourceWarnings(run, '  '), []);
    assert.equal(
      checkpointSyncAllowed({
        sourceRun: run,
        sourceNode: {
          id: 'source',
          status: 'ONLINE',
          metrics: { capabilities: ['POLICY_EXPORT'] },
        },
        targetNode: {
          id: 'target',
          status: 'ONLINE',
          metrics: { capabilities: ['POLICY_DEPLOY_PORTABLE_V1'] },
        },
        profileSelected: true,
        checkpoint: '/source/unrecorded/59999',
        directory: '/target/models',
        blocked: sourceDeploymentBlocker(run),
        syncing: false,
        busy: false,
      }),
      true,
    );
  });

  await test('recorded directories match exact paths and descendants at directory boundaries', () => {
    const attempts = { ...run, attempts: [{ output_dir: '/source/run-1' }] };
    for (const path of ['/source/run-1', '/source/run-1/100', ' /source//run-1/100/// '])
      assert.deepEqual(checkpointSourceWarnings(attempts, path), [], path);
    for (const path of ['/source/run-10/100', '/source/run-1-other', '/source/other'])
      assert.deepEqual(checkpointSourceWarnings(attempts, path), [checkpointSourceWarning], path);
    const outputs = {
      ...run,
      outputs: [{ kind: 'MODEL', node_id: 'source', path: '/source/model-1' }],
    };
    assert.deepEqual(checkpointSourceWarnings(outputs, '/source/model-1/params'), []);
    assert.deepEqual(checkpointSourceWarnings(outputs, '/source/model-10'), [
      checkpointSourceWarning,
    ]);
    for (const asset of [
      { kind: 'MODEL', node_id: 'other', path: '/source/model-1' },
      { kind: 'DATASET', node_id: 'source', path: '/source/model-1' },
      { kind: 'MODEL', node_id: 'source', path: '/source/model-1', state: 'ARCHIVED' },
    ])
      assert.deepEqual(checkpointSourceWarnings({ ...run, outputs: [asset] }, '/source/model-1'), [
        checkpointSourceWarning,
      ]);
  });

  await test('warnings render amber status without an error, a checkbox, or a confirmation action', () => {
    const html = renderToStaticMarkup(
      React.createElement(PolicyWarnings, { warnings: [checkpointSourceWarning] }),
    );
    assert.match(html, /inline-note amber policy-warning/);
    assert.match(html, /role="status"/);
    assert.match(html, /权重目录不属于所选训练的已记录产物/);
    assert.match(html, /POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH/);
    assert.doesNotMatch(html, /role="alert"|error-notice|type="checkbox"|<button/);
    assert.equal(renderToStaticMarkup(React.createElement(PolicyWarnings, {})), '');
  });

  await test('a warned create receipt stays visible before polling and a successful transfer still offers deployment', () => {
    const receipt = {
      id: 'transfer-warning',
      name: '有提醒的同步',
      source_node_id: 'source',
      source_run_id: run.id,
      source_checkpoint_path: '/source/unrecorded/59999',
      target_node_id: 'target',
      target_profile_id: 'runtime',
      target_directory: '/target/models',
      target_path: '/target/models/transfer-warning',
      state: 'SUCCEEDED',
      bytes_transferred: 0,
      bytes_total: 0,
      files_total: 0,
      files_transferred: 0,
      warnings: [checkpointSourceWarning],
      created_at: '2026-10-27T12:00:00Z',
      updated_at: '2026-10-27T12:00:00Z',
    };
    const initial = transferForId(receipt.id, undefined, receipt);
    assert.equal(initial, receipt);
    assert.equal(
      transferMatches(initial, {
        runId: run.id,
        checkpoint: receipt.source_checkpoint_path,
        nodeId: 'target',
        profileId: 'runtime',
        directory: '/target/models',
      }),
      true,
    );
    const html = renderInWorkspace(
      React.createElement(TransferCard, { item: initial, changed() {}, useTransfer() {} }),
    );
    assert.match(html, /inline-note amber policy-warning/);
    assert.match(html, /POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH/);
    assert.match(html, /同步校验完成/);
    assert.match(html, /<button[^>]*>[^]*?使用此权重部署/);
    assert.doesNotMatch(html, /disabled=""|error-notice/);
  });

  await test('the deployment detail displays warnings from its create receipt while the first GET is pending', () => {
    const html = renderInWorkspace(
      React.createElement(DeploymentDetail, {
        id: 'deployment-warning',
        changed() {},
        initialItem: {
          id: 'deployment-warning',
          name: '有提醒的部署',
          node_id: 'target',
          profile_id: 'runtime',
          state: 'STOPPED',
          port: 8000,
          config_name: 'auto_pi05',
          checkpoint_path: '/source/unrecorded/59999',
          warnings: [checkpointSourceWarning],
          checks: [],
          logs: '',
          created_at: '2026-10-27T12:00:00Z',
          updated_at: '2026-10-27T12:00:00Z',
        },
      }),
    );
    assert.match(html, /有提醒的部署/);
    assert.match(html, /inline-note amber policy-warning/);
    assert.match(html, /POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH/);
    assert.doesNotMatch(html, /error-notice|不能用无关训练/);
  });
} finally {
  await server.close();
}
