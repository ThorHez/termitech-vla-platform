import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
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
  const {
    ProfileAddition,
    profileAdditionSupported,
    profileAdditionRequest,
    checkedProfileAdditionOperation,
  } = await server.ssrLoadModule('/src/ProfileAddition.tsx');
  const node = {
    id: 'node-a',
    name: '训练机器',
    status: 'ONLINE',
    metrics: { capabilities: ['PROFILE_REBIND', 'PROFILE_ADD'] },
  };
  const render = (value = node, permissions = ['machines.manage']) =>
    renderToStaticMarkup(
      React.createElement(AuthContext.Provider, {
        value: { user: { id: 'user-a' }, can: (permission) => permissions.includes(permission) },
      }, React.createElement(ProfileAddition, { node: value })),
    );
  const request = {
    profile_id: 'openpi-experiment',
    name: '实验环境',
    code_dir: '/data/experiment',
    python: '/data/experiment/.venv/bin/python',
  };
  const operation = {
    id: 'addition-1',
    node_id: node.id,
    profile_id: request.profile_id,
    action: 'add',
    requested: { name: request.name, code_dir: request.code_dir, python: request.python },
    status: 'READY',
    profile: {
      id: request.profile_id,
      node_id: node.id,
      name: request.name,
      code_dir: request.code_dir,
      python: request.python,
      adapter: 'openpi-sft',
      execution_mode: 'native',
    },
    runtime_check: { profile_id: request.profile_id, status: 'READY' },
    error: null,
  };

  await test('additional environment control requires machine management and explicit Agent support', () => {
    assert.match(render(), /添加训练环境/);
    assert.doesNotMatch(render(), /disabled/);
    for (const permissions of [[], ['platform.read'], ['training.write']]) {
      assert.equal(render(node, permissions), '');
    }
    for (const metrics of [undefined, {}, { capabilities: ['PROFILE_REBIND'] }, { capabilities: null }]) {
      const legacy = { ...node, metrics };
      assert.equal(profileAdditionSupported(legacy), false);
      assert.match(render(legacy), /disabled=""/);
      assert.match(render(legacy), /请更新 Agent 后新增训练环境/);
    }
    assert.match(render({ ...node, status: 'OFFLINE' }), /disabled=""/);
    assert.match(render({ ...node, status: 'OFFLINE' }), /机器在线后才能新增训练环境/);
  });

  await test('request paths are normalized before receipt comparison without changing retry identity', () => {
    const submitted = profileAdditionRequest(
      request.profile_id,
      '  实验环境  ',
      ' /data//experiment/./ ',
      ' /data/experiment/.venv//bin/python ',
    );
    assert.deepEqual(submitted, request);
    assert.equal(checkedProfileAdditionOperation(operation, node, submitted), operation);
  });

  await test('a queued addition remains pending until the Agent returns a saved environment and READY runtime', () => {
    const queued = { ...operation, status: 'PENDING', profile: null, runtime_check: null };
    assert.equal(checkedProfileAdditionOperation(queued, node, request).status, 'PENDING');
    for (const invalid of [
      { ...operation, profile: null },
      { ...operation, runtime_check: null },
      { ...operation, runtime_check: { profile_id: request.profile_id, status: 'CHECKING' } },
      { ...operation, runtime_check: { profile_id: 'another-profile', status: 'READY' } },
    ]) {
      assert.throws(() => checkedProfileAdditionOperation(invalid, node, request), /回执.*不一致/);
    }
  });

  await test('receipts from other machines, rebinds, profile IDs or request contents cannot report success', () => {
    for (const changes of [
      { node_id: 'node-b' },
      { profile_id: 'another-profile' },
      { action: 'rebind' },
      { id: 'another-operation' },
      { requested: { ...operation.requested, name: '公共环境' } },
      { requested: { ...operation.requested, code_dir: '/data/shared' } },
      { requested: { ...operation.requested, python: '/data/shared/bin/python' } },
      { profile: { ...operation.profile, node_id: 'node-b' } },
      { profile: { ...operation.profile, name: '公共环境' } },
      { profile: { ...operation.profile, code_dir: '/data/shared' } },
      { profile: { ...operation.profile, python: '/data/shared/bin/python' } },
      { profile: { ...operation.profile, adapter: 'smoke' } },
    ]) {
      assert.throws(
        () => checkedProfileAdditionOperation({ ...operation, ...changes }, node, request, operation.id),
        /回执.*不一致/,
      );
    }
    assert.equal(checkedProfileAdditionOperation(operation, node, request, operation.id), operation);
  });
} finally {
  await server.close();
}
