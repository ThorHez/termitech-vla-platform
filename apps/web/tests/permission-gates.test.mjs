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
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const { NodesPage, NodePage, AssetsPage, OverviewPage, RunsPage } =
    await server.ssrLoadModule('/src/pages.tsx');
  const { NodeManagement } = await server.ssrLoadModule('/src/NodeManagement.tsx');
  const { NodeOnboarding } = await server.ssrLoadModule('/src/NodeOnboarding.tsx');
  const { default: RunWizard } = await server.ssrLoadModule('/src/RunWizard.tsx');
  const { RecipeLibrary } = await server.ssrLoadModule('/src/RecipeLibrary.tsx');
  const grants = {
    admin: [
      'machines.manage',
      'training.write',
      'assets.write',
      'recipes.write',
      'recipes.writeback',
    ],
    operator: ['training.write', 'assets.write', 'recipes.write'],
    viewer: [],
  };
  const auth = (role) => ({
    permissions: grants[role],
    loading: false,
    expired: false,
    can: (permission) => grants[role].includes(permission),
    refresh: async () => {},
    login: async () => {},
    logout: async () => {},
    changePassword: async () => {},
  });
  const render = (role, component, props = {}, withData = true) => {
    const element = React.createElement(component, props);
    return renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: auth(role) },
        withData ? React.createElement(DataProvider, { enabled: true }, element) : element,
      ),
    );
  };

  await test('machine connection actions are visible only to administrators', () => {
    assert.match(render('admin', NodesPage), />接入机器</);
    for (const role of ['operator', 'viewer']) {
      const list = render(role, NodesPage);
      assert.match(list, /机器资源/);
      assert.doesNotMatch(list, /<button[^>]*>[\s\S]*?接入机器/);
      const detail = render(role, NodePage, { id: 'fixture', createRun() {} });
      assert.doesNotMatch(detail, /管理连接/);
    }
  });

  await test('operators retain asset and training actions while viewers retain the pages', () => {
    for (const role of ['admin', 'operator']) {
      assert.match(render(role, AssetsPage), />登记资产</);
      assert.match(render(role, RunsPage, { createRun() {} }), />新建训练</);
      assert.match(render(role, OverviewPage, { createRun() {} }), />新建训练</);
    }
    assert.doesNotMatch(render('viewer', AssetsPage), />登记资产</);
    assert.doesNotMatch(render('viewer', RunsPage, { createRun() {} }), />新建训练</);
    assert.doesNotMatch(render('viewer', OverviewPage, { createRun() {} }), />新建训练</);
    assert.match(render('viewer', AssetsPage), /数据与模型/);
    assert.match(render('viewer', RunsPage, { createRun() {} }), /训练任务/);
  });

  await test('privileged dialogs reject direct mounts before business resources are accessed', () => {
    for (const role of ['operator', 'viewer']) {
      assert.match(render(role, NodeOnboarding, { close() {} }, false), /权限不足/);
      assert.match(
        render(
          role,
          NodeManagement,
          { node: { id: 'fixture' }, close() {}, reconnect() {} },
          false,
        ),
        /权限不足/,
      );
    }
    assert.match(render('viewer', RunWizard, { close() {} }, false), /权限不足/);
    assert.match(renderToStaticMarkup(React.createElement(RunWizard, { close() {} })), /权限不足/);
  });

  await test('readonly recipe library never mounts the live code search input', () => {
    const html = render(
      'viewer',
      RecipeLibrary,
      {
        node: { id: 'fixture', status: 'ONLINE' },
        profile: { id: 'profile', node_id: 'fixture' },
        recipe: '',
        onSelect() {},
        onDefaults() {},
        onDatasets() {},
        onDirtyChange() {},
        onLoadingChange() {},
      },
      false,
    );
    assert.match(html, /代码配方读取需要操作员或管理员权限/);
    assert.doesNotMatch(html, /role="combobox"|type="search"/);
  });
} finally {
  await server.close();
}
