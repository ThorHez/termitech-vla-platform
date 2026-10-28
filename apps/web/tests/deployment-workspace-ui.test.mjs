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
  const {
    default: PolicyDeployments,
    DeploymentWorkspaceList,
    DeploymentDetail,
    DeploymentRecordView,
    deploymentDetailTabs,
  } = await server.ssrLoadModule('/src/PolicyDeployments.tsx');
  const { buildDeploymentWorkspaceEntries } = await server.ssrLoadModule(
    '/src/deploymentWorkspace.ts',
  );
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const render = (element) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: { can: (permission) => permission === 'training.write' },
        },
        React.createElement(DataProvider, { enabled: false }, element),
      ),
    );
  const runtime = {
    id: 'runtime-095f',
    record_id: 'parent-095f',
    name: '原有服务名称',
    node_id: 'target',
    profile_id: 'runtime',
    state: 'STOPPED',
    port: 8000,
    config_name: 'auto_pi05',
    checkpoint_path: '/target/synced-model',
    source_run_id: 'same-run',
    transfer_id: 'transfer-bd4e',
    checks: [],
    logs: 'SERVICE_LOG_MARKER',
    test_result: { marker: 'TEST_RESULT_MARKER' },
    created_at: '2026-10-27T12:00:00Z',
    updated_at: '2026-10-27T12:00:00Z',
  };
  const transfer = (id, state, changes = {}) => ({
    id,
    record_id: 'parent-095f',
    name: id,
    source_node_id: 'source',
    source_run_id: 'same-run',
    target_node_id: 'target',
    target_directory: '/target/models',
    state,
    bytes_total: 1024,
    bytes_transferred: 1024,
    progress_pct: 100,
    files_total: 1,
    files_transferred: 1,
    logs: id + '_LOG',
    created_at: '2026-10-27T11:00:00Z',
    updated_at: '2026-10-27T11:00:00Z',
    ...changes,
  });
  const children = [
    transfer('transfer-bd4e', 'SUCCEEDED', { record_id: 'legacy-primary-owner' }),
    transfer('cancelled-a0', 'CANCELLED'),
  ];
  const record = {
    id: 'parent-095f',
    name: '持久部署记录',
    node_id: 'target',
    profile_id: 'runtime',
    source_run_id: 'same-run',
    state: 'STOPPED',
    deployment_id: runtime.id,
    selected_transfer_id: 'transfer-bd4e',
    transfer_ids: children.map((item) => item.id),
    created_at: '2026-10-27T11:00:00Z',
    updated_at: '2026-10-27T12:00:00Z',
    deployment: runtime,
    transfers: children,
  };
  const detail = (tab) =>
    render(
      React.createElement(DeploymentRecordView, { record, tab, changed() {}, continueConfig() {} }),
    );

  await test('home renders only persistent parent rows without any child preparation or cancelled-history group', () => {
    const second = {
      ...record,
      id: 'another-parent',
      name: '未启动的父记录',
      deployment_id: null,
      deployment: null,
      state: 'CANCELLED',
    };
    const html = render(
      React.createElement(DeploymentWorkspaceList, {
        entries: buildDeploymentWorkspaceEntries([record, second]),
        nodes: [],
      }),
    );
    assert.match(html, /href="#\/deployments\/parent-095f"/);
    assert.match(html, /href="#\/deployments\/another-parent"/);
    assert.match(html, /未启动的父记录/);
    assert.doesNotMatch(
      html,
      /cancelled-a0|transfer-bd4e|runtime-095f|preparations\/|checkpoint-transfer-card|role="progressbar"|已取消的部署准备|SERVICE_LOG_MARKER|robot-mock/,
    );
    assert.equal((html.match(/data-record-kind="record"/g) || []).length, 2);
  });

  await test('home loads parent records without mounting service details or global transfer history', () => {
    const html = render(React.createElement(PolicyDeployments));
    assert.match(html, /正在读取部署记录/);
    assert.match(html, /新建部署/);
    assert.doesNotMatch(
      html,
      /policy-deployment-detail|部署服务日志|robot-mock|运行单次推理测试|部署准备记录读取失败|已取消记录/,
    );
  });

  await test('a runtime parent defaults to overview and all tab links retain the canonical parent id', () => {
    const html = detail();
    for (const tab of deploymentDetailTabs)
      assert.ok(html.includes(`href="#/deployments/parent-095f?tab=${tab.id}"`), tab.id);
    assert.match(html, /role="tab" aria-selected="true"[^>]*>概览/);
    assert.match(html, /持久部署记录/);
    assert.match(html, /服务 runtime-095f/);
    assert.doesNotMatch(
      html,
      /href="#\/deployments\/runtime-095f|SERVICE_LOG_MARKER|TEST_RESULT_MARKER|运行单次推理测试|robot-mock|checkpoint-transfer-card/,
    );
  });

  await test('sync trusts explicit parent children, marks the exact selected transfer and collapses cancelled siblings inside the tab', () => {
    const html = detail('sync');
    const history = html.indexOf('checkpoint-transfer-history');
    assert.ok(history > 0);
    assert.match(html, /当前服务采用的权重/);
    assert.ok(html.indexOf('aria-label="权重同步 transfer-bd4e"') < history);
    assert.ok(html.indexOf('aria-label="权重同步 cancelled-a0"') > history);
    assert.doesNotMatch(html, /<details class="checkpoint-transfer-history"[^>]*\bopen/);
    assert.match(html, /已取消记录（1）/);
    assert.doesNotMatch(html, /使用此权重部署|运行单次推理测试|robot-mock/);
    // The selected legacy transfer has another primary owner but is an explicit backend reference.
    assert.match(html, /transfer-bd4e_LOG/);
    const noSelected = detail('overview');
    assert.doesNotMatch(noSelected, /cancelled-a0|transfer-bd4e_LOG/);
  });

  await test('runtime log, test, client and unified workflow content stay in their selected tab', () => {
    const logs = detail('logs');
    assert.match(logs, /aria-label="部署服务日志"[^>]*>SERVICE_LOG_MARKER/);
    assert.doesNotMatch(logs, /TEST_RESULT_MARKER|运行单次推理测试|robot-mock|cancelled-a0_LOG/);
    const localTest = detail('test');
    assert.match(localTest, /运行单次推理测试/);
    assert.match(localTest, /TEST_RESULT_MARKER/);
    assert.doesNotMatch(localTest, /SERVICE_LOG_MARKER|robot-mock/);
    assert.match(detail('client'), /openpi_client/);
    const workflow = detail('workflow');
    assert.match(workflow, /data-return-mock/);
    assert.match(workflow, /role="tab" aria-selected="true"[^>]*>真机运行与回流</);
    assert.match(workflow, /aria-label="真机运行与回流工作区"/);
    assert.equal((workflow.match(/功能预览/g) || []).length, 1);
    assert.match(workflow, /尚未连接采集设备/);
    assert.match(workflow, /不写入资产或提交训练/);
    assert.match(detail('mock'), /data-return-mock/);
    const dataReturn = detail('data-return');
    assert.match(dataReturn, /data-return-mock/);
    assert.doesNotMatch(dataReturn, /SERVICE_LOG_MARKER|robot-mock|运行单次推理测试/);
  });

  await test('a parent without runtime keeps preparation details and an isolated data-return preview', () => {
    const pending = {
      ...record,
      deployment_id: null,
      deployment: null,
      state: 'CANCELLED',
      selected_transfer_id: 'cancelled-a0',
      transfers: [children[1]],
      transfer_ids: ['cancelled-a0'],
    };
    const renderParent = (tab) =>
      render(
        React.createElement(DeploymentRecordView, {
          record: pending,
          tab,
          changed() {},
          continueConfig() {},
        }),
      );
    const html = renderParent('overview');
    assert.match(html, /继续配置部署/);
    assert.match(html, /尚未启动服务/);
    assert.match(html, /\?tab=overview/);
    assert.match(html, /\?tab=sync/);
    assert.match(html, /\?tab=logs/);
    assert.match(html, /\?tab=workflow/);
    assert.doesNotMatch(
      html,
      /\?tab=test|\?tab=client|\?tab=mock|停止服务|运行单次推理测试|>启动服务<|robot-mock|cancelled-a0_LOG|checkpoint-transfer-card/,
    );
    assert.doesNotMatch(html, /已停止/);
    const sync = renderParent('sync');
    assert.match(sync, /已取消记录（1）/);
    assert.match(sync, /重试同步/);
    assert.doesNotMatch(sync, /<details class="checkpoint-transfer-history"[^>]*\bopen/);
    const logs = renderParent('logs');
    assert.match(logs, /准备日志/);
    assert.match(logs, /cancelled-a0_LOG/);
    assert.doesNotMatch(logs, /aria-label="部署服务日志"/);
    assert.doesNotMatch(renderParent('test'), /运行单次推理测试/);
    assert.match(renderParent('data-return'), /data-return-mock/);
  });

  await test('missing runtime bindings and invalid deep links cannot substitute another runtime', () => {
    const mismatch = render(
      React.createElement(DeploymentRecordView, {
        record: { ...record, deployment: { ...runtime, id: 'another-runtime' } },
        changed() {},
        continueConfig() {},
      }),
    );
    assert.match(mismatch, /服务记录不匹配/);
    assert.doesNotMatch(mismatch, /停止服务|启动服务|robot-mock/);
    const invalid = render(
      React.createElement(PolicyDeployments, { path: '/deployments/preparations' }),
    );
    assert.match(invalid, /无法识别此部署记录地址/);
    const legacy = render(
      React.createElement(PolicyDeployments, { path: '/deployments/preparations/cancelled-a0' }),
    );
    assert.match(legacy, /正在读取所属部署记录/);
    assert.doesNotMatch(legacy, /checkpoint-transfer-card|停止服务|启动服务/);
    const missing = render(
      React.createElement(PolicyDeployments, { path: '/deployments/missing-record' }),
    );
    assert.match(missing, /正在读取部署记录/);
    assert.doesNotMatch(missing, /持久部署记录|SERVICE_LOG_MARKER|robot-mock/);
  });

  await test('standalone runtime receipt compatibility keeps service actions on its runtime while tabs use record_id', () => {
    const html = render(
      React.createElement(DeploymentDetail, { id: runtime.id, initialItem: runtime, changed() {} }),
    );
    assert.match(html, /href="#\/deployments\/parent-095f\?tab=overview"/);
    assert.doesNotMatch(html, /href="#\/deployments\/runtime-095f\?tab=/);
  });
} finally {
  await server.close();
}
