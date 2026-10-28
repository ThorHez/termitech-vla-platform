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

const buttons = (html) =>
  [...html.matchAll(/<button\b[^>]*>([\s\S]*?)<\/button>/g)].map((match) =>
    match[1].replace(/<[^>]*>/g, '').trim(),
  );
const noRealServiceActions = (html) => {
  for (const label of ['启动服务', '重试启动', '停止服务', '检查状态', '运行单次推理测试'])
    assert.ok(!buttons(html).includes(label), label);
};
const singleWorkspace = (html) => {
  assert.equal((html.match(/data-testid="data-return-mock"/g) || []).length, 1);
  assert.doesNotMatch(html, /class="robot-mock(?:\s|")/);
  assert.equal((html.match(/功能预览/g) || []).length, 1);
  assert.match(html, /尚未连接采集设备/);
  assert.match(html, /不写入资产或提交训练/);
};
const interfaceCopy = (html) =>
  [
    html.replace(/<[^>]*>/g, ' '),
    ...[...html.matchAll(/\b(?:aria-label|title|placeholder)="([^"]*)"/g)].map((match) => match[1]),
  ].join(' ');

try {
  const { DeploymentRecordView, deploymentDetailTabs } = await server.ssrLoadModule(
    '/src/PolicyDeployments.tsx',
  );
  const { default: DataReturnMock, workflowLogText } =
    await server.ssrLoadModule('/src/DataReturnMock.tsx');
  const { createRobotMock, transitionRobotMock } = await server.ssrLoadModule(
    '/src/robotDeploymentSimulation.ts',
  );
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const runtime = {
    id: 'runtime-real-stopped',
    record_id: 'record-unified',
    name: '实际停止的推理服务',
    node_id: 'node-a',
    profile_id: 'profile-a',
    state: 'STOPPED',
    checkpoint_path: '/machine/weights',
    config_name: 'fixture',
    port: 8000,
    checks: [],
    logs: 'REAL_RUNTIME_LOG_MUST_NOT_APPEAR',
    test_result: { marker: 'REAL_TEST_RESULT_MUST_NOT_APPEAR' },
    pending_command: null,
    created_at: '2026-10-27T12:00:00.000000+00:00',
    updated_at: '2026-10-27T12:00:00.000000+00:00',
  };
  const record = {
    id: runtime.record_id,
    name: '单记录真机闭环',
    node_id: runtime.node_id,
    profile_id: runtime.profile_id,
    source_run_id: null,
    state: runtime.state,
    deployment_id: runtime.id,
    selected_transfer_id: null,
    transfer_ids: [],
    transfers: [],
    deployment: runtime,
    created_at: runtime.created_at,
    updated_at: runtime.updated_at,
  };
  const renderRecord = (item, tab) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: (permission) => permission === 'training.write' } },
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(DeploymentRecordView, {
            record: item,
            tab,
            changed() {
              assert.fail('Preview rendering must not queue parent operations');
            },
            continueConfig() {
              assert.fail('Preview rendering must not configure a real service');
            },
          }),
        ),
      ),
    );

  await test('deployment navigation has one unified workflow entry and both legacy names resolve to it', () => {
    assert.equal(deploymentDetailTabs.filter((tab) => tab.id === 'workflow').length, 1);
    assert.ok(!deploymentDetailTabs.some((tab) => ['mock', 'data-return'].includes(tab.id)));
    for (const alias of ['workflow', 'mock', 'data-return']) {
      const html = renderRecord(record, alias);
      singleWorkspace(html);
      assert.match(html, /href="#\/deployments\/record-unified\?tab=workflow"/);
      assert.match(html, /role="tab" aria-selected="true"[^>]*>真机运行与回流</);
      assert.doesNotMatch(html, /href="#\/deployments\/record-unified\?tab=(?:mock|data-return)"/);
    }
  });

  await test('a stopped real runtime stays stopped while all unified preview aliases hide real service operations', () => {
    const before = JSON.stringify(record);
    for (const alias of ['workflow', 'mock', 'data-return']) {
      const html = renderRecord(record, alias);
      singleWorkspace(html);
      noRealServiceActions(html);
      assert.match(html, /已停止/);
      assert.doesNotMatch(
        html,
        /REAL_RUNTIME_LOG_MUST_NOT_APPEAR|REAL_TEST_RESULT_MUST_NOT_APPEAR/,
      );
    }
    assert.equal(JSON.stringify(record), before);
    assert.equal(record.deployment.state, 'STOPPED');
  });

  await test('a preparation-only record can open the same workflow without creating a runtime', () => {
    const preparation = {
      ...record,
      id: 'record-without-runtime',
      deployment_id: null,
      deployment: null,
      state: 'CANCELLED',
    };
    const before = JSON.stringify(preparation);
    for (const alias of ['workflow', 'mock', 'data-return']) {
      const html = renderRecord(preparation, alias);
      singleWorkspace(html);
      noRealServiceActions(html);
      assert.match(html, /href="#\/deployments\/record-without-runtime\?tab=workflow"/);
      assert.doesNotMatch(html, /继续配置部署|实际停止的推理服务|REAL_RUNTIME_LOG/);
    }
    assert.equal(JSON.stringify(preparation), before);
  });

  await test('one workspace presents SDK and Eval preparation, shared capture/replay, curation and versions', () => {
    const html = renderToStaticMarkup(
      React.createElement(DataReturnMock, {
        deploymentRecordId: record.id,
        deploymentName: record.name,
        machineName: '采集机器',
        runtimeId: runtime.id,
      }),
    );
    singleWorkspace(html);
    assert.match(html, /真机运行与数据回流/);
    assert.ok(buttons(html).includes('启动 SDK'));
    assert.ok(buttons(html).includes('初始化 Eval'));
    assert.match(html, /运行与采集/);
    assert.match(html, /打标与审核/);
    assert.match(html, /数据版本/);
    assert.match(html, /aria-label="运行遥测"/);
    assert.match(html, /关节角度/);
    assert.match(html, /TCP 位姿/);
    assert.match(html, /aria-label="运行日志"/);
    assert.match(html, /aria-label="筛选运行日志"/);
    assert.equal((html.match(/aria-label="双相机预览"/g) || []).length, 1);
    assert.equal((html.match(/aria-label="俯视相机画面"/g) || []).length, 1);
    assert.equal((html.match(/aria-label="腕部相机画面"/g) || []).length, 1);
    assert.match(html, /aria-label="回放时间"/);
    assert.doesNotMatch(interfaceCopy(html), /\bmock\b|模拟|示例|演示/i);
    assert.doesNotMatch(html, /<(?:iframe|video|img)\b[^>]*\bsrc=/);
    noRealServiceActions(html);
  });

  await test('default deployment navigation and feature labels use formal copy while preserving one preview notice', () => {
    const html = renderRecord(record, 'workflow');
    singleWorkspace(html);
    assert.match(html, /aria-label="真机运行与回流工作区"/);
    assert.match(html, /role="tab" aria-selected="true"[^>]*>真机运行与回流</);
    assert.ok(buttons(html).includes('新建采集会话'));
    assert.doesNotMatch(interfaceCopy(html), /\bmock\b|模拟|示例|演示/i);
    noRealServiceActions(html);
    assert.equal(record.deployment.state, 'STOPPED');
  });

  await test('shared SDK, Eval, execution and stream event logs use formal display copy without losing the connection boundary', () => {
    let robot = createRobotMock(1_000_000);
    for (const [index, action] of [
      'start_sdk',
      'sdk_ready',
      'start_eval',
      'eval_ready',
      'start',
      'pause',
      'stream_disconnect',
      'stream_reconnect',
    ].entries())
      robot = transitionRobotMock(robot, action, 1_000_000 + (index + 1) * 100);
    assert.equal(robot.phase, 'paused');
    assert.equal(robot.streamConnected, true);
    assert.equal(robot.logs.length, 8);
    assert.ok(robot.logs.some((entry) => /\[MOCK\]/.test(entry.message)));
    const displayed = robot.logs.map((entry) => workflowLogText(entry.message));
    for (const message of displayed) assert.doesNotMatch(message, /\bmock\b|模拟/i);
    assert.match(displayed[0], /未连接 CAN/);
    assert.match(displayed.join(' '), /SDK|arx5-sdk/);
    assert.match(displayed.join(' '), /Eval/);
    assert.match(displayed.join(' '), /暂停/);
    assert.match(displayed.join(' '), /中断|断流/);
    assert.match(displayed.join(' '), /恢复/);
  });
} finally {
  await server.close();
}
