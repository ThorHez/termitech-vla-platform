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
    DatasetProfileForm,
    DatasetProfileEditor,
    emptyDatasetProfile,
    datasetProfileErrors,
    normalizedDatasetProfile,
    readDatasetProfile,
    datasetProfileSummary,
  } = await server.ssrLoadModule('/src/DatasetProfileForm.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const profile = {
    ...emptyDatasetProfile(),
    robot_type: 'ARX5',
    action_representation: 'eef',
    arm_mode: 'bimanual',
    cameras: [
      { name: '左腕', key: 'observation.images.left_wrist', kind: 'rgb' },
      { name: '右腕', key: 'observation.images.right_wrist', kind: 'rgb' },
      { name: '头部深度', key: 'observation.depth', kind: 'depth' },
    ],
    action_dim: 20,
    state_dim: 40,
    fps: 30,
    task_description: '双臂折叠衣物\n放入收纳盒。',
    feature_keys: ['actions', 'task'],
  };
  const render = (component, props) => renderToStaticMarkup(React.createElement(component, props));
  const renderEditor = (permissions, metadata = { dataset_profile: profile }) => renderToStaticMarkup(
    React.createElement(AuthContext.Provider, { value: { can: (name) => permissions.includes(name) } },
      React.createElement(DataProvider, { enabled: true }, React.createElement(DatasetProfileEditor, {
        asset: { id: 'dataset', name: '衣物数据 <v1>', kind: 'DATASET', node_id: 'node', path: '/data/datasets/clothes', version: 'v1', metadata }, close() {},
      }))),
  );

  await test('dataset registration form covers robot, arms, actions, cameras and task', () => {
    const html = render(DatasetProfileForm, { value: profile, onChange() {} });
    for (const label of ['机器人类型', '机械臂配置', '视觉输入', '数据字段名', '任务描述', '采样频率', '状态与动作字段标签']) assert.ok(html.includes(label), label);
    assert.doesNotMatch(html, /动作表示|动作编码与数据字段|动作维度|状态维度|数据字段清单/);
    assert.equal((html.match(/采样频率（Hz）/g) || []).length, 1);
    assert.match(html, /value="bimanual" selected=""/);
    assert.match(html, /value="observation.images.left_wrist"/);
    assert.match(html, /value="depth" selected=""/);
    assert.match(html, /相机数量相同，字段映射也可能不同/);
    assert.match(html, /双臂折叠衣物\n放入收纳盒。/);
    assert.match(html, />添加视角<\/button>/);
  });

  await test('empty profile keeps actual camera keys and robot unset instead of guessing them', () => {
    const empty = emptyDatasetProfile();
    assert.equal(empty.robot_type, '');
    assert.equal(empty.cameras[0].key, '');
    assert.equal(empty.action_mode, 'unknown');
    assert.equal(empty.action_representation, 'unknown');
    assert.deepEqual(empty.field_annotations, []);
    assert.equal(empty.rotation_representation, 'unknown');
    assert.ok(datasetProfileErrors(empty).length > 0);
    assert.match(render(DatasetProfileForm, { value: empty, onChange() {} }), /aria-label="移除视觉输入 1"[^>]*disabled=""/);
  });

  await test('robot and task metadata are normalized without mutating the saved source', () => {
    const draft = { ...profile, robot_type: '  ARX5  ', task_description: '  折叠衣物\n 放入盒子。  ', cameras: [{ name: ' 腕部 ', key: ' observation.image ', kind: 'rgb' }] };
    assert.deepEqual(datasetProfileErrors(draft), []);
    const clean = normalizedDatasetProfile(draft);
    assert.equal(clean.robot_type, 'ARX5');
    assert.equal(clean.task_description, '折叠衣物\n 放入盒子。');
    assert.equal(clean.cameras[0].key, 'observation.image');
    assert.equal(draft.robot_type, '  ARX5  ');
    assert.equal(draft.cameras[0].name, ' 腕部 ');
    assert.equal(datasetProfileSummary(profile), 'ARX5 · 双臂 · EEF 末端 · 3 个视觉输入');
  });

  await test('source field labels retain independent state/action semantics without a processing plan', () => {
    const labels = [
      {key:'observation.state',role:'state',representation:'joint',value_mode:'absolute',component:'joint',arm:'left',shape:[7],dtype:'float32'},
      {key:'actions',role:'action',representation:'joint',value_mode:'delta',component:'other',arm:'unknown',shape:[16,7],dtype:'float32'},
    ];
    const source = {...profile, action_representation:'unknown', field_annotations:labels};
    assert.deepEqual(datasetProfileErrors(source), []);
    const clean = normalizedDatasetProfile(source);
    assert.equal(clean.field_annotations[0].value_mode, 'absolute');
    assert.equal(clean.field_annotations[1].value_mode, 'delta');
    assert.notEqual(clean.field_annotations[0].shape, labels[0].shape);
    assert.equal(clean.processing, undefined);
    const html = render(DatasetProfileForm,{value:source,onChange(){}});
    assert.match(html,/状态与动作字段标签/);
    assert.match(html,/字段数值方式 2/);
    assert.doesNotMatch(html,/使用组件组合数据处理|<legend>组件组合处理/);
    for (const fields of [[...labels,labels[0]], [{...labels[0],value_mode:'guess'}], [{...labels[0],shape:[0]}], [{...labels[0],unexpected:true}]])
      assert.ok(datasetProfileErrors({...source,field_annotations:fields}).length);
  });
  await test('only source action fields expose a horizon and it must match the stored sequence shape', () => {
    const fields = [
      {key:'observation.state',role:'state',representation:'joint',value_mode:'absolute',component:'joint',arm:'left',shape:[7]},
      {key:'actions',role:'action',representation:'joint',value_mode:'delta',component:'other',arm:'unknown',shape:[16,7],horizon:16},
    ];
    const source={...profile,field_annotations:fields};
    assert.deepEqual(datasetProfileErrors(source),[]);
    const html=render(DatasetProfileForm,{value:source,onChange(){}});
    assert.match(html,/aria-label="动作 Horizon 2"/);
    assert.doesNotMatch(html,/aria-label="动作 Horizon 1"/);
    assert.match(html,/状态字段 1 无动作 Horizon/);
    assert.equal(normalizedDatasetProfile({...source,field_annotations:[{...fields[1],shape:[7],horizon:undefined}]}).field_annotations[0].horizon,1);
    for(const horizon of [0,-1,1.5,true,'16',15])
      assert.ok(datasetProfileErrors({...source,field_annotations:[fields[0],{...fields[1],horizon}]}).length);
    assert.ok(datasetProfileErrors({...source,field_annotations:[{...fields[0],horizon:1},fields[1]]}).length);
    assert.deepEqual(datasetProfileErrors({...source,field_annotations:[{...fields[1],shape:undefined,horizon:undefined}]}),[]);
  });
  await test('standalone gripper opening labels round-trip independently from arm representation', () => {
    const field = {key:'observation.gripper_width',role:'state',representation:'gripper',value_mode:'unknown',component:'gripper',arm:'left',shape:[1]};
    const source = {...profile,field_annotations:[field]};
    assert.deepEqual(datasetProfileErrors(source),[]);
    assert.equal(normalizedDatasetProfile(source).field_annotations[0].representation,'gripper');
    assert.equal(readDatasetProfile(source).action_representation,'eef');
    assert.match(render(DatasetProfileForm,{value:source,onChange(){}}),/value="gripper" selected="">夹爪开度/);
    assert.ok(datasetProfileErrors({...source,field_annotations:[{...field,component:'position'}]}).some(error=>error.includes('夹爪分量')));
    for(const representation of ['eef','unknown'])
      assert.deepEqual(datasetProfileErrors({...source,field_annotations:[{...field,representation}]}),[]);
  });

  await test('duplicate or malformed visual inputs cannot be accepted as dataset facts', () => {
    for (const cameras of [[], Array.from({ length: 17 }, (_, index) => ({ name: String(index), key: String(index), kind: 'rgb' })), [{ name: '左', key: 'image', kind: 'rgb' }, { name: '右', key: 'image', kind: 'rgb' }], [{ name: '同名', key: 'left', kind: 'rgb' }, { name: '同名', key: 'right', kind: 'rgb' }], [{ name: '视角', key: 'bad\nkey', kind: 'rgb' }]]) {
      assert.ok(datasetProfileErrors({ ...profile, cameras }).length > 0);
    }
  });

  await test('invalid dimensions, sampling rates and control text are rejected', () => {
    for (const change of [{ action_dim: 0 }, { action_dim: 3.5 }, { state_dim: 513 }, { fps: 0 }, { fps: Infinity }, { fps: 1001 }, { robot_type: 'robot\u0000' }, { task_description: 'bad\u007f' }, { feature_keys: ['actions', 'actions'] }, { feature_keys: ['bad\nkey'] }, { schema_version: 2 }]) assert.ok(datasetProfileErrors({ ...profile, ...change }).length > 0, JSON.stringify(change));
    assert.deepEqual(datasetProfileErrors(profile), []);
  });

  await test('legacy or malformed metadata falls back to supplementing without rendering errors', () => {
    for (const value of [undefined, null, {}, [], 'profile', { ...profile, cameras: null }, { ...profile, cameras: [null] }, { ...profile, cameras: [{ name: 1, key: 3, kind: 'rgb' }] }, { ...profile, unexpected: true }]) assert.equal(readDatasetProfile(value), undefined);
    const normalized = readDatasetProfile({ ...profile, action_mode: undefined });
    assert.equal(normalized, undefined);
    const { action_mode: _mode, rotation_representation: _rotation, feature_keys: _keys, ...legacy } = profile;
    assert.equal(readDatasetProfile(legacy).action_mode, 'unknown');
    assert.ok(readDatasetProfile(profile));
    assert.match(renderEditor(['assets.write'], {}), /请填写机器人类型/);
  });

  await test('profile editing retains original path as read-only text and preserves multiline metadata', () => {
    const html = renderEditor(['assets.write']);
    assert.match(html, /衣物数据 &lt;v1&gt;/);
    assert.match(html, /<code class="dataset-profile-path">\/data\/datasets\/clothes<\/code>/);
    assert.doesNotMatch(html, /<input[^>]*value="\/data\/datasets\/clothes"/);
    assert.match(html, /已创建训练保留原配置/);
    assert.match(html, /<button[^>]*type="submit"[^>]*>保存数据特征<\/button>/);
  });

  await test('viewers cannot access the metadata editor and pending forms disable all fields', () => {
    const html = renderEditor(['platform.read']);
    assert.match(html, /编辑数据特征需要操作员或管理员权限/);
    assert.doesNotMatch(html, /保存数据特征|dataset-profile-form/);
    assert.match(render(DatasetProfileForm, { value: profile, onChange() {}, disabled: true }), /<fieldset[^>]*disabled=""/);
  });
} finally {
  await server.close();
}
