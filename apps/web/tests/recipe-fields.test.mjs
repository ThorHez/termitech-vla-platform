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
  const { RecipeDetails } = await server.ssrLoadModule('/src/RecipeDetails.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const granted = ['training.write', 'recipes.write', 'recipes.writeback'];
  const auth = (permissions) => ({
    permissions,
    loading: false,
    expired: false,
    can: (permission) => permissions.includes(permission),
    refresh: async () => {},
    login: async () => {},
    logout: async () => {},
    changePassword: async () => {},
  });
  const { assetManagedOverridePaths, isAssetManagedField } = await server.ssrLoadModule(
    '/src/recipeFieldPolicy.ts',
  );
  const parent = 'data.datasets[0].base_config';
  const prompt = `${parent}.prompt_from_task`;
  const quantile = `${parent}.use_quantile_norm`;
  const keys = `${parent}.action_sequence_keys`;
  const source = {
    path: 'src/openpi/training/config.py',
    code_dir: '/fixture/openpi',
    line: 1,
    sha256: 'a'.repeat(64),
    commit_sha: 'b'.repeat(40),
    changes_digest: 'c'.repeat(64),
    dirty: false,
  };
  const detail = {
    source,
    fields: [
      { path: parent, kind: 'expression', expression: 'DataConfig(...)', origin: 'recipe' },
      { path: prompt, kind: 'literal', value: true, expression: 'True', origin: 'recipe' },
      { path: quantile, kind: 'literal', value: true, expression: 'True', origin: 'recipe' },
      { path: keys, kind: 'literal', value: [], expression: '()', origin: 'recipe' },
      {
        path: 'data.datasets[0].repo_id',
        kind: 'literal',
        value: '/source/dataset',
        origin: 'recipe',
      },
      {
        path: 'data.datasets[0].assets.assets_dir',
        kind: 'literal',
        value: '/source/norm',
        origin: 'recipe',
      },
      { path: 'data.datasets[0].assets.asset_id', kind: 'literal', value: '.', origin: 'recipe' },
      { path: 'data.weights', kind: 'literal', value: [1, 2], origin: 'recipe' },
      { path: 'weight_loader.path', kind: 'literal', value: '/source/model', origin: 'recipe' },
      { path: 'data.state_pad_dim', kind: 'literal', value: 128, origin: 'recipe' },
    ],
    source_code: 'TrainConfig(name="fixture")',
    warnings: [],
  };
  const render = (overrides = {}, permissions = granted) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: auth(permissions) },
        React.createElement(RecipeDetails, {
          node: { id: 'node', status: 'ONLINE' },
          profile: { id: 'profile', node_id: 'node', adapter: 'openpi-sft' },
          recipe: 'fixture',
          saved: {
            id: 'saved',
            name: 'Fixture',
            latest_revision: {
              id: 'revision',
              revision: 1,
              detail,
              definition: { overrides, source },
            },
          },
          onSaved() {},
          onDefaults() {},
          onDatasets() {},
          onDirtyChange() {},
          onLoadingChange() {},
        }),
      ),
    );
  const allFields = (html) => {
    const panel = html.split('aria-label="全部字段参数"', 2)[1];
    let depth = 1;
    for (const match of panel.matchAll(/<\/?details\b[^>]*>/g)) {
      depth += match[0].startsWith('</') ? -1 : 1;
      if (!depth) return panel.slice(0, match.index);
    }
    throw new Error('Missing all-fields container');
  };
  const escape = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const control = (panel, path, tag) => {
    const match = panel.match(new RegExp(`<${tag}[^>]*aria-label="修改${escape(path)}"[^>]*>`));
    assert.ok(match, `${path} must have a ${tag} in all fields`);
    assert.doesNotMatch(match[0], /\bdisabled(?:=|\s|>)/);
    return match;
  };

  await test('all fields contains enabled boolean and list editors for nested data config', () => {
    const panel = allFields(render());
    control(panel, prompt, 'select');
    control(panel, quantile, 'select');
    control(panel, keys, 'textarea');
  });

  await test('grouped and all-field editors use distinct IDs with valid labels', () => {
    const html = render();
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
    assert.equal(new Set(ids).size, ids.length);
    const labels = [...html.matchAll(/\bfor="([^"]+)"/g)].map((match) => match[1]);
    assert.ok(labels.length >= 6);
    for (const label of labels) assert.ok(ids.includes(label), `Missing input for label ${label}`);
  });

  await test('all fields uses saved boolean and list overrides', () => {
    const panel = allFields(render({ [quantile]: false, [keys]: ['actions'] }));
    const selection = panel.match(
      new RegExp(`<select[^>]*aria-label="修改${escape(quantile)}"[^>]*>(.*?)</select>`),
    );
    assert.match(selection[1], /<option value="false" selected="">False<\/option>/);
    const list = panel.match(
      new RegExp(`<textarea[^>]*aria-label="修改${escape(keys)}"[^>]*>(.*?)</textarea>`),
    );
    assert.match(list[1], /actions/);
  });

  await test('constructor expressions stay read-only while their literal children are editable', () => {
    const panel = allFields(render());
    assert.ok(panel.includes('Python 表达式（只读）'));
    assert.ok(!panel.includes(`aria-label="修改${parent}"`));
    control(panel, prompt, 'select');
  });

  await test('asset resource fields have no editors in grouped or all-field views', () => {
    const html = render();
    const panel = allFields(html);
    for (const path of [
      'data.datasets[0].repo_id',
      'data.datasets[0].assets.assets_dir',
      'data.datasets[0].assets.asset_id',
      'data.weights',
      'weight_loader.path',
    ]) {
      assert.ok(!html.includes(`aria-label="修改${path}"`), `${path} must have no input anywhere`);
      assert.ok(panel.includes(path));
    }
    assert.match(panel, /由训练资产绑定/);
    assert.match(panel, /查看代码默认值（仅供参考）/);
    assert.match(html, /本次训练以“资产与参数”中选择的数据集、归一化参数、初始模型及混采权重为准/);
    control(panel, 'data.state_pad_dim', 'input');
    control(panel, prompt, 'select');
  });

  await test('legacy asset overrides are disclosed and require explicit review', () => {
    const html = render({
      'data.datasets[0].repo_id': '/legacy/dataset',
      'data.weights': [2, 3],
      [quantile]: false,
    });
    assert.match(html, /此版本含有旧的资产覆盖，需要先核对/);
    assert.match(html, /核对移除项并准备新版本/);
    assert.match(html, /旧版本的资产覆盖（需移除）/);
    assert.match(html, /\/legacy\/dataset/);
    assert.ok(!html.includes('aria-label="修改data.datasets[0].repo_id"'));
    control(allFields(html), quantile, 'select');
  });

  await test('asset policy protects containers and aliases without locking processing options', () => {
    for (const path of [
      'data.datasets',
      'data.datasets[1]',
      'data.datasets[1].repo_id',
      'data.datasets[1].assets',
      'data.datasets[1].assets.asset_id',
      'data.datasets[1].norm_stats_path',
      'data.rlds_data_dir',
      'data.filter_dict_path',
      'data.weights[0]',
      'model.weights',
      'model.pretrained_path',
      'weight_loader',
      'weight_loader.path',
    ])
      assert.equal(isAssetManagedField(path), true, path);
    for (const path of [prompt, quantile, keys, 'data.state_pad_dim', 'model.action_dim'])
      assert.equal(isAssetManagedField(path), false, path);
    assert.deepEqual(
      assetManagedOverridePaths({ [prompt]: false, 'data.weights': [1], 'data.repo_id': '/data' }),
      ['data.weights', 'data.repo_id'],
    );
  });

  await test('legacy parent dictionary resource edits are disclosed for explicit removal', () => {
    const html = render({
      'data.base_config': { prompt_from_task: true, repo_id: '/legacy/embedded-dataset' },
      [quantile]: false,
    });
    assert.match(html, /此版本含有旧的资产覆盖，需要先核对/);
    assert.match(html, /核对移除项并准备新版本/);
    assert.match(html, /<code>data\.base_config<\/code>/);
    control(allFields(html), quantile, 'select');
  });

  await test('embedded resource dictionaries and lists use the same review boundary as direct leaves', () => {
    assert.deepEqual(
      assetManagedOverridePaths({
        'data.base_config': { repo_id: '/legacy/data', prompt_from_task: true },
        'model.context': [{ checkpoint_path: '/legacy/model' }],
        'data.processing': [{ prompt_from_task: true, action_sequence_keys: ['actions'] }],
        'data.prompt': 'repo_id is just prompt text',
      }),
      ['data.base_config', 'model.context'],
    );
  });

  await test('viewer sees saved configuration without any edit or save controls', () => {
    const html = render({}, []);
    assert.match(html, /当前为只读配方快照/);
    assert.match(html, /DataConfig/);
    assert.match(html, /fixture\/openpi/);
    assert.doesNotMatch(html, /aria-label="修改/);
    assert.doesNotMatch(html, /另存为新配方|保存新版本|核对移除项并准备新版本/);
    assert.doesNotMatch(html, /其他可覆盖参数/);
  });

  await test('operator can edit platform recipes and is told code writes require an admin', () => {
    const html = render({}, ['training.write', 'recipes.write']);
    control(allFields(html), prompt, 'select');
    assert.match(html, /保存将创建平台配方版本；写入训练代码需要管理员权限/);
    assert.match(html, /另存为新配方/);
    assert.doesNotMatch(html, /保存会在 config\.py 中新增独立配方/);
  });
} finally {
  await server.close();
}
