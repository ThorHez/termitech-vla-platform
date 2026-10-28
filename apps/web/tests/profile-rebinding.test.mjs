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
  const { ApiError } = await server.ssrLoadModule('/src/api.ts');
  const {
    ProfileRebinding,
    ProfileBindingModeNotice,
    runtimeBindingSupported,
    runtimeBindingError,
    runtimeBindingUnchanged,
  } = await server.ssrLoadModule('/src/ProfileRebinding.tsx');
  const node = { id: 'any-machine', name: '任意连接', status: 'ONLINE' };
  const profile = {
    id: 'any-profile',
    node_id: node.id,
    name: '现有环境',
    adapter: 'openpi-sft',
    execution_mode: 'native',
    code_dir: '/project/openpi',
    python: '/project/openpi/.venv/bin/python',
  };
  const render = (props = {}, permissions = ['machines.manage']) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: (permission) => permissions.includes(permission) } },
        React.createElement(ProfileRebinding, { node, profile, ...props }),
      ),
    );
  const button = (html) => html.match(/<button\b[^>]*>/)?.[0] || '';

  await test('an explicitly metadata-only connection has a disabled entry and visible Chinese explanation', () => {
    const metadata = {
      ...node,
      metrics: {
        connection_type: 'local_ssh_metadata_gateway',
        capabilities: ['PATH_SUGGEST', 'DATASET_INSPECT', 'DATA_PROCESSORS'],
      },
    };
    const html = render({ node: metadata });
    assert.equal(runtimeBindingSupported(metadata), false);
    assert.match(button(html), /disabled=""/);
    assert.match(html, /role="status"[^>]*>当前 SSH 数据连接未开放环境重绑，原连接绑定保持不变。/);
    assert.doesNotMatch(html, /role="dialog"|This local test|READ_ONLY_REAL_AGENT/);
  });

  await test('explicit rebind capability enables any machine without hardcoded identity', () => {
    for (const connection_type of ['local_ssh_metadata_gateway', 'native_agent']) {
      const capable = {
        ...node,
        id: 'different-machine',
        metrics: { connection_type, capabilities: ['PROFILE_REBIND'] },
      };
      assert.equal(runtimeBindingSupported(capable), true);
      assert.doesNotMatch(
        button(render({ node: capable, profile: { ...profile, node_id: capable.id } })),
        /disabled/,
      );
    }
  });

  await test('legacy Agents without the capability field keep their existing entry', () => {
    for (const metrics of [undefined, {}, { runtime_checks: [] }]) {
      const legacy = { ...node, metrics };
      assert.equal(runtimeBindingSupported(legacy), true);
      assert.doesNotMatch(button(render({ node: legacy })), /disabled/);
    }
    for (const capabilities of [[], ['DATASET_INSPECT'], null, 'PROFILE_REBIND']) {
      const explicit = { ...node, metrics: { capabilities } };
      assert.equal(runtimeBindingSupported(explicit), false);
      assert.match(button(render({ node: explicit })), /disabled/);
    }
  });

  await test('machine management permission and native OpenPI scope still gate all controls', () => {
    for (const permissions of [[], ['platform.read'], ['training.write', 'assets.write']]) {
      assert.equal(render({}, permissions), '');
    }
    assert.equal(render({ profile: { ...profile, adapter: 'smoke' } }), '');
    assert.equal(render({ profile: { ...profile, execution_mode: 'docker' } }), '');
  });

  await test('offline and in-use runtimes remain disabled even when the connection advertises rebind', () => {
    const capable = { ...node, metrics: { capabilities: ['PROFILE_REBIND'] } };
    const offline = render({ node: { ...capable, status: 'OFFLINE' } });
    assert.match(button(offline), /disabled=""/);
    assert.match(offline, /机器在线后才能重新绑定/);
    const busy = render({ node: capable, inUse: true });
    assert.match(button(busy), /disabled=""/);
    assert.match(busy, /该环境有未结束的训练任务，结束后才能重新绑定/);
  });

  await test('SSH binding scope describes validation and persistence in the current console connection', () => {
    const ssh = { ...node, metrics: { connection_type: 'local_ssh_metadata_gateway' } };
    const html = renderToStaticMarkup(React.createElement(ProfileBindingModeNotice, { node: ssh }));
    assert.match(html, /通过 SSH 校验/);
    assert.match(html, /绑定保存到当前控制台的数据连接/);
    assert.match(html, /当前控制台连接的项目与 Python 路径/);
    assert.doesNotMatch(html, /Agent|远端.*配置已/);
    const native = renderToStaticMarkup(React.createElement(ProfileBindingModeNotice, { node }));
    assert.match(native, /Agent 校验通过后保存，新训练使用新的绑定/);
    assert.doesNotMatch(native, /当前控制台的数据连接/);
  });

  await test('readonly admission failures become friendly messages while other errors retain actionable detail', () => {
    const denied = new ApiError(
      'READ_ONLY_REAL_AGENT',
      'This local test allows metadata reads only',
      403,
    );
    const ssh = { ...node, metrics: { connection_type: 'local_ssh_metadata_gateway' } };
    assert.equal(
      runtimeBindingError(denied, ssh).message,
      '当前 SSH 数据连接未开放环境重绑，原连接绑定保持不变。',
    );
    assert.equal(
      runtimeBindingError(denied, node).message,
      '当前连接不支持重新绑定项目与环境，原绑定保持不变。',
    );
    const conflict = new ApiError('PROFILE_REBIND_CONFLICT', '环境已变化，请刷新后重试', 409);
    assert.equal(runtimeBindingError(conflict, ssh), conflict);
  });
  await test('an SSH connection may validate its existing paths until its own profile is READY', () => {
    const ssh = {
      ...node,
      metrics: { connection_type: 'local_ssh_metadata_gateway', capabilities: ['PROFILE_REBIND'] },
    };
    for (const runtime_checks of [
      undefined,
      [],
      null,
      [{ profile_id: profile.id, status: 'CHECKING' }],
      [{ profile_id: profile.id, status: 'FAILED' }],
      [{ profile_id: 'different-profile', status: 'READY' }],
    ]) {
      const unknown = { ...ssh, metrics: { ...ssh.metrics, runtime_checks } };
      assert.equal(
        runtimeBindingUnchanged(unknown, profile, profile.code_dir, profile.python),
        false,
      );
    }
    const ready = {
      ...ssh,
      metrics: { ...ssh.metrics, runtime_checks: [{ profile_id: profile.id, status: 'READY' }] },
    };
    assert.equal(runtimeBindingUnchanged(ready, profile, profile.code_dir, profile.python), true);
    assert.equal(runtimeBindingUnchanged(ready, profile, '/other-project', profile.python), false);
    assert.equal(runtimeBindingUnchanged(ready, profile, profile.code_dir, '/other/python'), false);
  });

  await test('a successful local receipt immediately disables another same-path validation without mutating baseline', () => {
    const ssh = { ...node, metrics: { connection_type: 'local_ssh_metadata_gateway' } };
    const before = structuredClone(profile);
    assert.equal(
      runtimeBindingUnchanged(ssh, profile, profile.code_dir, profile.python, { ...profile }),
      true,
    );
    assert.equal(
      runtimeBindingUnchanged(ssh, profile, profile.code_dir, profile.python, {
        ...profile,
        node_id: 'other-node',
      }),
      false,
    );
    assert.equal(
      runtimeBindingUnchanged(ssh, profile, profile.code_dir, profile.python, {
        ...profile,
        python: '/unconfirmed/python',
      }),
      false,
    );
    assert.equal(runtimeBindingUnchanged(ssh, profile, profile.code_dir, profile.python), false);
    assert.deepEqual(profile, before);
  });

  await test('native Agents keep the existing unchanged-path guard independently of runtime check state', () => {
    assert.equal(
      runtimeBindingUnchanged(node, profile, profile.code_dir + '/', profile.python),
      true,
    );
    assert.equal(
      runtimeBindingUnchanged(
        { ...node, metrics: { runtime_checks: [] } },
        profile,
        profile.code_dir,
        profile.python,
      ),
      true,
    );
    assert.equal(runtimeBindingUnchanged(node, profile, '/other-project', profile.python), false);
  });
} finally {
  await server.close();
}
