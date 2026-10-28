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

const freeze = (value) => {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
};
const buttons = (html) =>
  [...html.matchAll(/<button\b[^>]*>([\s\S]*?)<\/button>/g)].map((match) =>
    match[1].replace(/<[^>]*>/g, '').trim(),
  );
const noServiceActions = (html) => {
  for (const label of ['启动服务', '重试启动', '停止服务', '检查状态', '运行单次推理测试'])
    assert.ok(!buttons(html).includes(label), label);
};
const previewNotice = (html) => {
  assert.equal((html.match(/功能预览/g) || []).length, 1);
  assert.match(html, /尚未连接采集设备/);
  assert.match(html, /不写入资产或提交训练/);
};

try {
  const { default: DataReturnMock } = await server.ssrLoadModule('/src/DataReturnMock.tsx');
  const { DeploymentRecordView } = await server.ssrLoadModule('/src/PolicyDeployments.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const render = (element) => renderToStaticMarkup(element);
  const renderRecord = (record) =>
    render(
      React.createElement(
        AuthContext.Provider,
        { value: { can: (permission) => permission === 'training.write' } },
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(DeploymentRecordView, {
            record,
            tab: 'data-return',
            changed() {
              assert.fail('Rendering a Mock must not mutate or refresh the parent');
            },
            continueConfig() {
              assert.fail('Rendering a Mock must not create a real deployment');
            },
          }),
        ),
      ),
    );
  const props = freeze({
    deploymentRecordId: 'parent-collection-a',
    deploymentName: '折叠衣服部署 <演示甲>',
    machineName: '示例机器甲',
    runtimeId: 'runtime-stopped-a',
  });
  const runtime = freeze({
    id: props.runtimeId,
    record_id: props.deploymentRecordId,
    name: '实际已停止的服务',
    node_id: 'node-a',
    profile_id: 'profile-a',
    checkpoint_path: '/machine/model',
    config_name: 'fixture',
    port: 8000,
    state: 'STOPPED',
    checks: [],
    logs: 'REAL_SERVICE_LOG_NOT_A_MOCK',
    test_result: { marker: 'REAL_TEST_RESULT_NOT_A_MOCK' },
    pending_command: null,
    created_at: '2026-10-27T12:00:00.000000+00:00',
    updated_at: '2026-10-27T12:00:00.000000+00:00',
  });
  const record = freeze({
    id: props.deploymentRecordId,
    name: props.deploymentName,
    node_id: runtime.node_id,
    profile_id: runtime.profile_id,
    source_run_id: null,
    state: 'STOPPED',
    deployment_id: runtime.id,
    selected_transfer_id: null,
    transfer_ids: [],
    deployment: runtime,
    transfers: [],
    created_at: runtime.created_at,
    updated_at: runtime.updated_at,
  });

  await test('the standalone workspace opens usable corrections with one preview notice and accessible replay controls', () => {
    const html = render(React.createElement(DataReturnMock, props));
    assert.match(html, /aria-label="真机运行与回流工作区"/);
    previewNotice(html);
    assert.ok((html.match(/class="dr-session-item(?:\s|\")/g) || []).length >= 2);
    assert.match(html, /aria-label="双相机预览"/);
    assert.match(html, /role="img" aria-label="俯视相机画面"/);
    assert.match(html, /role="img" aria-label="腕部相机画面"/);
    assert.match(html, /aria-label="搜索采集会话"/);
    assert.match(html, /aria-label="选择 Episode"/);
    assert.match(html, /aria-label="播放回放"/);
    assert.match(html, /type="range"[^>]*aria-label="回放时间"/);
    assert.match(html, /expert_label_valid = true/);
    assert.match(html, /按当前控制来源选择动作/);
    assert.ok(buttons(html).includes('新建采集会话'));
    assert.ok(buttons(html).includes('审核片段'));
    assert.doesNotMatch(html, /<(?:iframe|video|img)\b[^>]*\bsrc=/);
    noServiceActions(html);
  });

  await test('two parent workspaces preserve their own labels and unique SVG/control IDs without mutating context', () => {
    const other = freeze({
      deploymentRecordId: 'parent-collection-b',
      deploymentName: '装配部署乙',
      machineName: '示例机器乙',
      runtimeId: 'runtime-b',
    });
    const before = JSON.stringify([props, other]);
    const first = render(React.createElement(DataReturnMock, props));
    const second = render(React.createElement(DataReturnMock, other));
    assert.match(first, /折叠衣服部署 &lt;演示甲&gt;/);
    assert.match(first, /示例机器甲/);
    assert.doesNotMatch(first, /装配部署乙|示例机器乙/);
    assert.match(second, /装配部署乙/);
    assert.match(second, /示例机器乙/);
    assert.doesNotMatch(second, /折叠衣服部署|示例机器甲/);
    const together = render(
      React.createElement(
        React.Fragment,
        null,
        React.createElement(DataReturnMock, { ...props, key: props.deploymentRecordId }),
        React.createElement(DataReturnMock, { ...other, key: other.deploymentRecordId }),
      ),
    );
    const ids = [...together.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
    assert.ok(ids.length > 0, 'the synthetic previews have per-instance SVG definitions');
    assert.equal(new Set(ids).size, ids.length);
    assert.equal(JSON.stringify([props, other]), before);
  });

  await test('the parent tab keeps a real stopped runtime stopped and exposes no real service action or log', () => {
    const before = JSON.stringify(record);
    const html = renderRecord(record);
    assert.match(html, /href="#\/deployments\/parent-collection-a\?tab=workflow"/);
    assert.match(html, /role="tab" aria-selected="true"[^>]*>真机运行与回流</);
    assert.match(html, /aria-label="真机运行与回流工作区"/);
    assert.match(html, /已停止/);
    assert.doesNotMatch(
      html,
      /REAL_SERVICE_LOG_NOT_A_MOCK|REAL_TEST_RESULT_NOT_A_MOCK|class="robot-mock/,
    );
    noServiceActions(html);
    assert.equal(JSON.stringify(record), before);
    assert.equal(record.deployment.state, 'STOPPED');
  });

  await test('a preparation-only parent can preview data return without inventing a running service', () => {
    const preparation = freeze({
      ...record,
      id: 'parent-not-started',
      name: '尚未部署的示例',
      state: 'CANCELLED',
      deployment_id: null,
      deployment: null,
    });
    const before = JSON.stringify(preparation);
    const html = renderRecord(preparation);
    assert.match(html, /href="#\/deployments\/parent-not-started\?tab=workflow"/);
    assert.match(html, /aria-label="真机运行与回流工作区"/);
    previewNotice(html);
    assert.doesNotMatch(html, /服务就绪|runtime-stopped-a|继续配置部署/);
    noServiceActions(html);
    assert.equal(JSON.stringify(preparation), before);
    assert.equal(preparation.deployment, null);
  });
} finally {
  await server.close();
}
