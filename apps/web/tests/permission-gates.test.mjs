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
  const { NodesPage, NodePage, AssetsPage, OverviewPage, RunsPage, RegisterAsset } =
    await server.ssrLoadModule('/src/pages.tsx');
  const { DatasetProfileEditor } = await server.ssrLoadModule('/src/DatasetProfileForm.tsx');
  const { DatasetNormBindingDialog } = await server.ssrLoadModule('/src/DatasetNormBinding.tsx');
  const { NodeManagement } = await server.ssrLoadModule('/src/NodeManagement.tsx');
  const { NodeOnboarding } = await server.ssrLoadModule('/src/NodeOnboarding.tsx');
  const { default: RunWizard } = await server.ssrLoadModule('/src/RunWizard.tsx');
  const { RecipeLibrary } = await server.ssrLoadModule('/src/RecipeLibrary.tsx');
  const { AssetDeletionsPage } = await server.ssrLoadModule('/src/AssetDeletions.tsx');
  const grants = {
    admin: [
      'machines.manage',
      'training.write',
      'assets.write',
      'assets.register',
      'assets.archive',
      'assets.purge',
      'datasets.register',
      'datasets.archive',
      'assets.transfer',
      'dataset-templates.write',
      'recipes.query',
      'recipes.write',
      'recipes.writeback',
    ],
    operator: [
      'training.write',
      'assets.register',
      'assets.archive',
      'datasets.register',
      'datasets.archive',
      'datasets.write',
      'assets.query',
      'assets.transfer',
      'dataset-templates.write',
      'recipes.query',
    ],
    viewer: [],
  };
  const auth = (role) => ({
    user: { id: role, role, username: role, display_name: role },
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

  await test('operators can register all asset types while viewers keep a shared readonly catalog', () => {
    for (const role of ['admin', 'operator'])
      assert.match(render(role, AssetsPage), />数据集格式模板</);
    assert.doesNotMatch(render('viewer', AssetsPage), />数据集格式模板</);
    assert.match(render('admin', AssetsPage), />登记资产</);
    assert.match(render('admin', AssetsPage), />批量登记数据集</);
    assert.match(render('operator', AssetsPage), />登记资产</);
    assert.match(render('operator', AssetsPage), />批量登记数据集</);
    for (const role of ['admin', 'operator']) {
      assert.match(render(role, RunsPage, { createRun() {} }), />新建训练</);
      assert.match(render(role, OverviewPage, { createRun() {} }), />新建训练</);
    }
    assert.doesNotMatch(render('viewer', AssetsPage), />登记资产</);
    assert.doesNotMatch(render('viewer', AssetsPage), />登记数据集<|>批量登记数据集</);
    assert.doesNotMatch(render('viewer', RunsPage, { createRun() {} }), />新建训练</);
    assert.doesNotMatch(render('viewer', OverviewPage, { createRun() {} }), />新建训练</);
    assert.match(render('viewer', AssetsPage), /数据与模型/);
    assert.match(render('viewer', RunsPage, { createRun() {} }), /训练任务/);
  });

  await test('only administrators with the file deletion permission can open the deletion ledger', () => {
    assert.match(render('admin', AssetsPage), /href="#\/assets\/deletions"/);
    assert.match(render('admin', AssetDeletionsPage, {}, false), /<h1>已删除资产<\/h1>/);
    for (const role of ['operator', 'viewer']) {
      assert.doesNotMatch(render(role, AssetsPage), /href="#\/assets\/deletions"/);
      assert.match(render(role, AssetDeletionsPage, {}, false), /<h1>权限不足<\/h1>/);
    }
    for (const [role, permission] of [
      ['admin', false],
      ['operator', true],
    ]) {
      const denied = renderToStaticMarkup(
        React.createElement(
          AuthContext.Provider,
          { value: { ...auth(role), can: () => permission } },
          React.createElement(AssetDeletionsPage),
        ),
      );
      assert.match(denied, /<h1>权限不足<\/h1>/);
      assert.doesNotMatch(denied, /刷新列表|永久删除文件/);
    }
  });

  await test('operator registration offers all three asset types and complete dataset configuration without existing edits', () => {
    const operator = render('operator', RegisterAsset, { close() {} });
    assert.match(operator, /登记资产/);
    for (const kind of ['DATASET', 'MODEL', 'NORM'])
      assert.match(operator, new RegExp(`<option value="${kind}"`));
    assert.match(operator, /机器人类型|机器人/);
    assert.match(operator, /任务描述/);
    assert.match(operator, /选择格式模板/);
    assert.match(operator, /norm_stats/);
    assert.match(operator, /稍后补录数据集特征/);
    assert.doesNotMatch(operator, /确认删除登记|保存数据特征/);
    const admin = render('admin', RegisterAsset, { close() {} });
    for (const kind of ['DATASET', 'MODEL', 'NORM'])
      assert.match(admin, new RegExp(`<option value="${kind}"`));
    const denied = render('viewer', RegisterAsset, { close() {} }, false);
    assert.match(denied, /权限不足/);
    assert.doesNotMatch(denied, /<form|所在机器|资产类型/);
    const dataset = { id: 'dataset', node_id: 'node', kind: 'DATASET', state: 'REGISTERED' };
    assert.match(
      render('operator', DatasetProfileEditor, { asset: dataset, close() {} }, false),
      /权限不足/,
    );
    assert.match(
      render('operator', DatasetNormBindingDialog, { asset: dataset, close() {} }),
      /无法修改绑定/,
    );
  });

  await test('dataset editing dialogs admit their operator registrant and reject other accounts', () => {
    const owned = {
      id: 'dataset', name: 'Dataset', node_id: 'node', kind: 'DATASET', state: 'REGISTERED',
      path: '/datasets/demo', registered_by_user_id: 'operator',
    };
    const other = { ...owned, registered_by_user_id: 'someone-else' };
    for (const [component, action, denial] of [
      [DatasetProfileEditor, /保存数据特征/, /权限不足/],
      [DatasetNormBindingDialog, /保存为不绑定/, /无法修改绑定/],
    ]) {
      assert.match(render('operator', component, { asset: owned, close() {} }), action);
      assert.match(render('operator', component, { asset: other, close() {} }, false), denial);
      assert.match(render('viewer', component, { asset: { ...owned, registered_by_user_id: 'viewer' }, close() {} }, false), denial);
    }
  });

  await test('single registration requires the common asset permission while batch keeps its dataset scope', () => {
    const single = {
      user: { id: 'single', role: 'operator' },
      can: (permission) => permission === 'assets.register',
    };
    const element = (Component, auth, withData = true) =>
      renderToStaticMarkup(
        React.createElement(
          AuthContext.Provider,
          { value: auth },
          withData
            ? React.createElement(
                DataProvider,
                { enabled: false },
                React.createElement(Component, { close() {} }),
              )
            : React.createElement(Component, { close() {} }),
        ),
      );
    const form = element(RegisterAsset, single);
    for (const kind of ['DATASET', 'NORM', 'MODEL'])
      assert.match(form, new RegExp(`<option value="${kind}"`));
    const page = element(AssetsPage, single);
    assert.match(page, />登记资产</);
    assert.doesNotMatch(page, />批量登记数据集</);
    const legacy = { ...single, can: (permission) => permission === 'datasets.register' };
    assert.match(element(RegisterAsset, legacy, false), /权限不足/);
    assert.doesNotMatch(element(RegisterAsset, legacy, false), /<form/);
    assert.match(element(AssetsPage, legacy), />批量登记数据集</);
    assert.doesNotMatch(element(AssetsPage, legacy), />登记资产</);
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
