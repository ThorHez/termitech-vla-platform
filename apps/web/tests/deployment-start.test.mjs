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
  const { deploymentStartBlockedReason, deploymentDetailReceipt } = await server.ssrLoadModule(
    '/src/policyDeployment.ts',
  );
  const { DeploymentStartActions, DeploymentDetail, deploymentDetailTabs } =
    await server.ssrLoadModule('/src/PolicyDeployments.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const node = { id: 'target', status: 'ONLINE' };
  const stopped = {
    id: 'existing-deployment',
    name: '复用已有部署',
    state: 'STOPPED',
    node_id: 'target',
    profile_id: 'runtime',
    checkpoint_path: '/target/already-synced/checkpoint',
    transfer_id: 'existing-transfer',
    config_name: 'auto_pi05',
    port: 8000,
    checks: [],
    logs: '',
    created_at: '2026-10-27T12:00:00.000001+00:00',
    updated_at: '2026-10-27T12:00:00.000100+00:00',
  };
  const actions = (item, overrides = {}) =>
    renderToStaticMarkup(
      React.createElement(DeploymentStartActions, {
        item,
        node,
        writable: true,
        busy: '',
        onStart() {},
        ...overrides,
      }),
    );

  await test('start is available only for stopped or explicitly failed online records without an operation', () => {
    for (const state of ['STOPPED', 'FAILED']) {
      const item = { ...stopped, state };
      assert.equal(deploymentStartBlockedReason(item, node, true), undefined);
      assert.match(actions(item), new RegExp(state === 'STOPPED' ? '启动服务' : '重试启动'));
      assert.doesNotMatch(actions(item), /disabled=""/);
    }
    for (const state of ['UNKNOWN', 'PENDING', 'STARTING', 'RUNNING', 'STOPPING']) {
      const item = { ...stopped, state };
      assert.ok(deploymentStartBlockedReason(item, node, true));
      assert.doesNotMatch(actions(item), /<button/);
    }
    assert.ok(deploymentStartBlockedReason(stopped, node, false));
    assert.doesNotMatch(actions(stopped, { writable: false }), /<button/);
  });

  await test('offline, unobserved, busy and pending states explain why starting is unavailable', () => {
    for (const [item, overrides, expected] of [
      [stopped, { node: { ...node, status: 'OFFLINE' } }, /目标机器当前离线/],
      [stopped, { node: undefined }, /正在读取目标机器状态/],
      [stopped, { busy: 'refresh' }, /已有部署操作正在处理/],
      [{ ...stopped, pending_command: 'POLICY_DEPLOY_STOP' }, {}, /已有部署操作正在处理/],
    ]) {
      const html = actions(item, overrides);
      assert.match(html, /<button[^>]*disabled=""/);
      assert.match(html, expected);
    }
    const submitting = actions(stopped, { busy: 'start' });
    assert.match(submitting, /正在启动/);
    assert.match(submitting, /正在提交启动请求/);
    assert.match(submitting, /role="status"/);
  });

  await test('the accepted start receipt prevents an older stopped poll from reopening the start button', () => {
    const accepted = {
      ...stopped,
      state: 'PENDING',
      pending_command: 'POLICY_DEPLOY_START',
      updated_at: '2026-10-27T12:00:00.000200+00:00',
    };
    const item = deploymentDetailReceipt(stopped.id, stopped, accepted, stopped);
    assert.equal(item, accepted);
    assert.ok(deploymentStartBlockedReason(item, node, true));
    const html = actions(item);
    assert.match(html, /等待 Agent 启动服务/);
    assert.doesNotMatch(html, /<button/);
    assert.equal(deploymentDetailReceipt(stopped.id, undefined, accepted), accepted);
    assert.equal(
      deploymentDetailReceipt(
        stopped.id,
        { ...stopped, updated_at: accepted.updated_at },
        accepted,
      ),
      accepted,
    );
  });

  await test('newer runtime receipts supersede accepted starts at microsecond precision and preserve this record binding', () => {
    const accepted = {
      ...stopped,
      state: 'PENDING',
      pending_command: 'POLICY_DEPLOY_START',
      updated_at: '2026-10-27T12:00:00.000200+00:00',
    };
    const running = {
      ...accepted,
      state: 'RUNNING',
      pending_command: null,
      updated_at: '2026-10-27T12:00:00.000300+00:00',
    };
    assert.equal(deploymentDetailReceipt(stopped.id, running, accepted), running);
    assert.equal(deploymentDetailReceipt(stopped.id, stopped, accepted, running), running);
    assert.equal(deploymentDetailReceipt(stopped.id, stopped, undefined, accepted), accepted);
    assert.equal(
      deploymentDetailReceipt(stopped.id, { ...accepted }, accepted).pending_command,
      'POLICY_DEPLOY_START',
    );
    assert.equal(
      deploymentDetailReceipt(stopped.id, { ...running, id: 'another-record' }, accepted),
      accepted,
    );
    assert.equal(
      deploymentDetailReceipt(stopped.id, stopped, { ...accepted, id: 'another-record' }),
      stopped,
    );
    assert.equal(deploymentDetailReceipt('unknown', stopped, accepted, stopped), undefined);
    assert.equal(deploymentDetailReceipt(stopped.id, undefined, undefined, stopped), stopped);
    assert.equal(running.transfer_id, 'existing-transfer');
    assert.equal(running.checkpoint_path, stopped.checkpoint_path);
  });

  await test('real start actions stay outside the isolated data-return preview', () => {
    for (const tab of deploymentDetailTabs) {
      const html = renderToStaticMarkup(
        React.createElement(
          AuthContext.Provider,
          {
            value: { can: (permission) => permission === 'training.write' },
          },
          React.createElement(
            DataProvider,
            { enabled: false },
            React.createElement(DeploymentDetail, {
              id: stopped.id,
              initialItem: stopped,
              changed() {},
              tab: tab.id,
            }),
          ),
        ),
      );
      if (tab.id === 'workflow') {
        assert.doesNotMatch(html, /class="policy-deployment-start-actions"/);
        assert.match(html, /data-return-mock/);
      } else {
        const start = html.indexOf('启动服务');
        assert.ok(start > 0 && start < html.indexOf('aria-label="部署详情"'), tab.id);
      }
      assert.match(html, /href="#\/deployments\/existing-deployment\?tab=/);
    }
  });
} finally {
  await server.close();
}
