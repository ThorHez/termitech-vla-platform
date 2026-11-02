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
  const { AgentUpgradeAction, NodeManagement, nodeManagementAvailability } =
    await server.ssrLoadModule('/src/NodeManagement.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const {
    agentReleaseLabel,
    latestNodeUpgrade,
    nodeOperationComplete,
    nodeOperationFailed,
    nodeOperationLabel,
    nodeOperationSucceeded,
  } = await server.ssrLoadModule('/src/agentUpgradeState.ts');
  const { DeploymentLogs, deploymentStageLabel } =
    await server.ssrLoadModule('/src/DeploymentLogs.tsx');
  const info = {
    node_id: 'machine-a',
    node_status: 'ONLINE',
    managed: true,
    can_upgrade: true,
    can_disconnect: true,
    can_uninstall: true,
    reason: '',
    current_version: '0.1.0+old',
    target_version: '0.1.0+new',
  };
  const renderAction = (props = {}) =>
    renderToStaticMarkup(
      React.createElement(AgentUpgradeAction, {
        info,
        allowed: true,
        reason: '',
        select() {},
        ...props,
      }),
    );
  const job = (state, changes = {}) => ({
    id: 'upgrade-a',
    node_id: info.node_id,
    operation: 'upgrade',
    state,
    ...changes,
  });

  await test('upgrade entry shows concrete current and target versions with a usable action', () => {
    const html = renderAction();
    assert.match(html, /当前版本：0\.1\.0\+old/);
    assert.match(html, /目标版本：0\.1\.0\+new/);
    assert.match(html, /升级 Agent<\/button>/);
    assert.doesNotMatch(html, /disabled/);
    assert.doesNotMatch(html, /查看最近升级/);
    assert.match(html, /独立监控的训练可继续运行/);
    assert.match(html, /日志和指标在本地保存，恢复后补传/);
  });
  await test('isolated training permits a proven safe upgrade while destructive actions remain blocked', () => {
    assert.deepEqual(nodeManagementAvailability(info, 2, false), {
      canUpgrade: true,
      canDisconnect: false,
      canUninstall: false,
    });
    assert.deepEqual(nodeManagementAvailability(info, 0, true), {
      canUpgrade: true,
      canDisconnect: false,
      canUninstall: false,
    });
    assert.deepEqual(nodeManagementAvailability(info, 0, false), {
      canUpgrade: true,
      canDisconnect: true,
      canUninstall: true,
    });
  });
  await test('legacy or unknown supervisor proof remains blocked by backend management evidence', () => {
    const legacy = {
      ...info,
      can_upgrade: false,
      upgrade_reason: '活动训练尚未使用独立监控，请结束后升级。',
    };
    assert.equal(nodeManagementAvailability(legacy, 1, false).canUpgrade, false);
    assert.equal(nodeManagementAvailability(undefined, 0, false).canUpgrade, false);
    const html = renderAction({ info: legacy, allowed: false });
    assert.match(html, /disabled=""/);
    assert.match(html, /活动训练尚未使用独立监控/);
  });
  await test('busy and unavailable upgrade states show a disabled action and the reason', () => {
    const busy = renderAction({
      allowed: false,
      reason: '此机器有 1 个活动训练，请结束训练后再操作。',
    });
    assert.match(busy, /disabled=""/);
    assert.match(busy, /活动训练/);
    const unavailable = renderAction({
      allowed: false,
      info: { ...info, upgrade_reason: '请先构建 Agent 安装包' },
    });
    assert.match(unavailable, /disabled=""/);
    assert.match(unavailable, /请先构建 Agent 安装包/);
  });
  await test('unknown versions stay explicit and an existing upgrade exposes its result', () => {
    const html = renderAction({
      info: { ...info, current_version: null, target_version: null },
      view() {},
    });
    assert.match(html, /当前版本：未上报/);
    assert.match(html, /目标版本：中台当前安装包/);
    assert.match(html, /查看最近升级/);
  });
  await test('same package versions remain distinguishable by the immutable build identity', () => {
    const before = 'a'.repeat(64),
      after = 'b'.repeat(64);
    const html = renderAction({
      info: {
        ...info,
        current_version: '0.1.0',
        target_version: '0.1.0',
        current_build_id: before,
        target_build_id: after,
      },
    });
    assert.match(html, /当前版本：0\.1\.0（构建 a{12}）/);
    assert.match(html, /目标版本：0\.1\.0（构建 b{12}）/);
    assert.equal(agentReleaseLabel(null, null), '未上报');
    assert.equal(agentReleaseLabel(null, null, '中台当前安装包'), '中台当前安装包');
  });
  await test('upgrade logs use upgrade labels while original installation labels stay compatible', () => {
    assert.equal(deploymentStageLabel('ONLINE'), '接入完成');
    assert.equal(deploymentStageLabel('INSTALLING'), '安装 Agent');
    assert.equal(deploymentStageLabel('INSTALLING', 'upgrade'), '升级 Agent');
    const html = renderToStaticMarkup(
      React.createElement(DeploymentLogs, {
        job: {
          ...job('SUCCEEDED'),
          stage: 'ONLINE',
          logs: [
            {
              timestamp: '2026-11-01T10:00:00Z',
              stage: 'INSTALL_PACKAGE',
              message: 'verified new build',
            },
          ],
        },
        active: false,
      }),
    );
    assert.match(html, /升级完成/);
    assert.match(html, /更新 Agent 程序/);
    assert.match(html, /下载完整日志/);
    assert.doesNotMatch(html, /接入完成/);
  });
  await test('reopening management only resumes upgrade receipts bound to that machine', () => {
    const current = job('WAITING_AGENT');
    assert.equal(latestNodeUpgrade(current, info.node_id), current);
    assert.equal(
      latestNodeUpgrade(job('RUNNING', { node_id: 'machine-b' }), info.node_id),
      undefined,
    );
    assert.equal(
      latestNodeUpgrade(job('RUNNING', { operation: 'uninstall' }), info.node_id),
      undefined,
    );
    assert.equal(latestNodeUpgrade(null, info.node_id), undefined);
  });
  await test('waiting for heartbeat remains active and only SUCCEEDED claims a completed upgrade', () => {
    for (const state of ['QUEUED', 'RUNNING', 'WAITING_AGENT']) {
      assert.equal(nodeOperationComplete(job(state)), false);
      assert.equal(nodeOperationSucceeded(job(state)), false);
      assert.equal(nodeOperationLabel(job(state)), '正在升级 Agent');
    }
    for (const state of ['FAILED', 'INTERRUPTED']) {
      assert.equal(nodeOperationComplete(job(state)), true);
      assert.equal(nodeOperationSucceeded(job(state)), false);
      assert.equal(nodeOperationFailed(job(state)), true);
      assert.equal(nodeOperationLabel(job(state)), 'Agent 升级未完成');
    }
    assert.equal(nodeOperationComplete(job('SUCCEEDED')), true);
    assert.equal(nodeOperationSucceeded(job('SUCCEEDED')), true);
    assert.equal(nodeOperationLabel(job('SUCCEEDED')), 'Agent 升级完成');
  });
  await test('users without machine management permission cannot enter the management workflow', () => {
    const html = renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: { can: () => false },
        },
        React.createElement(NodeManagement, {
          node: { id: info.node_id, name: '训练机器' },
          close() {},
          reconnect() {},
        }),
      ),
    );
    assert.match(html, /权限不足/);
    assert.doesNotMatch(html, /升级 Agent|卸载 Agent/);
  });
} finally {
  await server.close();
}
