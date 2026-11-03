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
  const {
    DEFAULT_AGENT_SERVICE,
    agentServiceError,
    checkedProbeConfiguration,
    deploymentService,
    sshAuthenticationRequest,
    sshInstallCheckRequest,
  } = await server.ssrLoadModule('/src/onboardingService.ts');
  const probe = (changes = {}) => ({
    connection_id: 'connection', host: '115.190.212.169', port: 22, username: 'root',
    fingerprint: 'verified-host', hostname: 'T001host', expires_at: '2099-01-01T00:00:00Z',
    can_install: true, checks: [], existing_agent: null, ...changes,
  });
  const canonicalConfig = {
    name: 'Production', roots: ['/data3'], output_root: '/data3/checkpoints',
    state_dir: '/data2/production/state', profiles: [{
      id: 'production-profile', name: 'Production runtime', adapter: 'openpi-sft',
      code_dir: '/data3/openpi', python: '/data3/openpi/.venv/bin/python', execution_mode: 'native',
    }],
  };
  const testService = 'termitech-vla-agent-mac-test';

  await test('service names follow the backend namespace contract', () => {
    for (const value of [DEFAULT_AGENT_SERVICE, testService, 'A_1', 'a'.repeat(100), ' custom '])
      assert.equal(agentServiceError(value), undefined, value);
    for (const value of ['', ' ', 'a'.repeat(101), 'other.service', 'a/b', '../agent',
      'a b', 'a;id', 'a\nb', '中文', 'a$(id)'])
      assert.match(agentServiceError(value), /Agent 服务名/, value);
  });

  await test('auth-only, direct probe, and saved-connection probe carry the selected service', () => {
    for (const name of [DEFAULT_AGENT_SERVICE, testService]) {
      const auth = sshAuthenticationRequest(' 115.190.212.169 ', '22', ' root ', ' secret ', ` ${name} `);
      const check = sshInstallCheckRequest(' /data2/termitech-vla-agent-for-test ', ` ${name} `);
      assert.deepEqual(auth, {
        host: '115.190.212.169', port: 22, username: 'root', password: ' secret ', service_name: name,
      });
      assert.deepEqual(check, { install_dir: '/data2/termitech-vla-agent-for-test', service_name: name });
      assert.equal({ ...auth, ...check }.service_name, name);
      assert.equal('password' in check, false);
    }
  });

  await test('an existing selected service retains its own profiles and storage', () => {
    const result = checkedProbeConfiguration(probe({
      service_name: DEFAULT_AGENT_SERVICE, existing_agent: canonicalConfig,
    }), DEFAULT_AGENT_SERVICE, '/data2/production', 'fallback');
    assert.deepEqual(result, canonicalConfig);
    result.profiles[0].python = '/different/python';
    result.roots.push('/different/root');
    assert.equal(canonicalConfig.profiles[0].python, '/data3/openpi/.venv/bin/python');
    assert.deepEqual(canonicalConfig.roots, ['/data3']);
  });

  await test('a fresh custom service starts with its own storage and blank runtime', () => {
    // A previous probe may have populated the form from the canonical Agent.
    // Applying the fresh probe replaces every unedited auto-filled value.
    const previous = checkedProbeConfiguration(probe({
      service_name: DEFAULT_AGENT_SERVICE, existing_agent: canonicalConfig,
    }), DEFAULT_AGENT_SERVICE, '/data2/production', 'fallback');
    const fresh = checkedProbeConfiguration(probe({ service_name: testService }), testService,
      '/data2/termitech-vla-agent-for-test/', 'fallback');
    assert.notEqual(fresh.name, previous.name);
    assert.deepEqual(fresh.roots, ['/data2']);
    assert.equal(fresh.output_root, '/data2/termitech-vla-agent-for-test/outputs');
    assert.equal(fresh.state_dir, '/data2/termitech-vla-agent-for-test/state');
    assert.equal(fresh.profiles.length, 1);
    assert.equal(fresh.profiles[0].python, '');
    assert.equal(fresh.profiles[0].code_dir, '');
    assert.notEqual(fresh.profiles[0].id, previous.profiles[0].id);
  });

  await test('custom probes reject a canonical or legacy response before importing config', () => {
    for (const service_name of [undefined, DEFAULT_AGENT_SERVICE, 'another-test-service'])
      assert.throws(() => checkedProbeConfiguration(probe({ service_name, existing_agent: canonicalConfig }),
        testService, '/data2/termitech-vla-agent-for-test', 'fallback'), /服务名不匹配/);
    assert.equal(checkedProbeConfiguration(probe({ existing_agent: canonicalConfig }), DEFAULT_AGENT_SERVICE,
      '/data2/production', 'fallback').name, canonicalConfig.name);
  });

  await test('editing historical deployments preserves custom service names with a legacy default', () => {
    assert.equal(deploymentService({ service_name: testService }), testService);
    assert.equal(deploymentService({}), DEFAULT_AGENT_SERVICE);
  });

  await test('the onboarding form exposes the canonical default and independent-instance guidance', async () => {
    const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
    const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
    const { NodeOnboarding } = await server.ssrLoadModule('/src/NodeOnboarding.tsx');
    const html = renderToStaticMarkup(React.createElement(AuthContext.Provider,
      { value: { can: () => true } }, React.createElement(DataProvider, { enabled: false },
        React.createElement(NodeOnboarding, { close() {} }))));
    assert.match(html, /高级：Agent 服务设置/);
    assert.match(html, /Agent 服务名/);
    assert.match(html, /value="termitech-vla-agent"/);
    assert.match(html, /独立测试 Agent 请使用不同服务名和安装目录/);
    assert.match(html, /termitech-vla-agent-mac-test/);
  });
} finally {
  await server.close();
}
