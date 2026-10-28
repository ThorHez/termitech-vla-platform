import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const fixtures = JSON.parse(readFileSync(new URL('../../../tests/fixtures/composed-processing-v1.json', import.meta.url)));
const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false }, optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
try {
  const helpers = await server.ssrLoadModule('/src/composedProcessing.ts');
  const { composedProcessingErrors: errors, normalizeComposedProcessing: normalize, compositionSchemaKey } = helpers;
  const { readDatasetProfile, datasetProfileErrors, applyComposedProcessing, DatasetProfileForm } = await server.ssrLoadModule('/src/DatasetProfileForm.tsx');
  const { compositionBindingState, composedChoice, compositionToken } = await server.ssrLoadModule('/src/ComposedDatasetProcessing.tsx');
  const { processorParameterErrors, filterDataProcessors } = await server.ssrLoadModule('/src/DataProcessorPicker.tsx');
  const { DatasetProcessing } = await server.ssrLoadModule('/src/DatasetProcessing.tsx');
  const { assetManagedOverridePaths, isAssetManagedField } = await server.ssrLoadModule('/src/recipeFieldPolicy.ts');
  const clone = (value) => structuredClone(value);
  const spec = clone(fixtures.valid[0]);
  const profileFor = (processing) => ({ schema_version: 1, format: 'lerobot', robot_type: processing.robot_model,
    arm_mode: processing.arm_mode, action_representation: processing.action_representation,
    action_mode: processing.action_mode, rotation_representation: processing.rotation_representation,
    cameras: processing.cameras.map((camera, index) => ({ name: `相机 ${index + 1}`, key: camera.key, kind: camera.kind })),
    action_dim: processing.action_layout.reduce((sum, block) => sum + block.width, 0), state_dim: processing.state_fields.reduce((sum, field) => sum + field.width, 0),
    task_description: '组件组合测试', feature_keys: [], processing });
  const asset = (processing = spec) => ({ id: 'dataset', name: '组合数据', kind: 'DATASET', node_id: 'node', path: '/data/dataset', version: '1', metadata: { dataset_profile: profileFor(clone(processing)) } });
  const capability = { name: 'ComposableDataConfig', description: 'registered composition', available: true, composition_schema_version: 1,
    parameters: [], unsupported_parameters: [], source: { path: 'src/openpi/training/composable_data_config.py', sha256: 'a'.repeat(64) } };
  const binding = { dataset_asset_id: 'dataset', norm_asset_id: 'norm', weight: 20, recipe_index: 0 };
  const configuredBinding = (processing = spec) => ({ ...binding, recipe_index: undefined, processor: composedChoice(processing) });
  const stateFor = (changes = {}) => compositionBindingState({ binding: configuredBinding(), dataset: asset(), entries: [capability], slots: [], index: 0, stage: 'sft', ...changes });

  for (const [index, value] of fixtures.valid.entries()) await test(`shared composition v1 valid fixture ${index + 1}`, () => {
    assert.deepEqual(errors(value), []); assert.deepEqual(errors(normalize(value)), []);
  });
  for (const [index, value] of fixtures.invalid.entries()) await test(`shared composition v1 invalid fixture ${index + 1}`, () => {
    assert.ok(errors(value).length > 0, `invalid fixture ${index}`);
  });
  await test('optional metadata stays blank and omitted keys normalize to a complete binding', () => {
    const omitted = clone(spec);
    for (const key of ['position_unit', 'joint_unit', 'coordinate_frame', 'gripper_range']) delete omitted[key];
    assert.deepEqual(errors(omitted), []);
    const canonical = normalize(omitted);
    assert.equal(canonical.position_unit, ''); assert.equal(canonical.joint_unit, '');
    assert.equal(canonical.coordinate_frame, ''); assert.equal(canonical.gripper_range, null);
    assert.deepEqual(normalize(canonical), canonical);
    const rawBinding = { ...binding, processor: { name: 'ComposableDataConfig', parameters: { composition: omitted } } };
    assert.equal(stateFor({ binding: rawBinding }).valid, true);
    assert.deepEqual(stateFor({ binding: rawBinding }).spec, canonical);
    assert.deepEqual(composedChoice(omitted).parameters.composition, canonical);
    const blanks = { ...omitted, position_unit: '\uFEFF \t', joint_unit: null, coordinate_frame: ' \u00a0', gripper_range: null };
    assert.deepEqual(errors(blanks), []); assert.deepEqual(normalize(blanks), canonical);
    assert.deepEqual(normalize(spec), spec);
  });
  await test('explicit invalid metadata and unfinished enabled bounds still block validation', () => {
    for (const [key, value] of [['position_unit', 'cm'], ['position_unit', ' m '], ['joint_unit', 'turn'], ['coordinate_frame', 'base\u001c'], ['gripper_range', [NaN, NaN]], ['gripper_range', [0, Infinity]], ['gripper_range', [1, 1]]]) {
      assert.ok(errors({ ...spec, [key]: value }).length, key);
    }
    for (const key of ['state_history', 'normalization']) {
      const missing = clone(spec); delete missing[key]; assert.ok(errors(missing).length, key);
    }
    const restored = { ...spec, gripper_range: [NaN, NaN] };
    assert.ok(stateFor({ binding: configuredBinding(restored) }).spec);
    assert.equal(stateFor({ binding: configuredBinding(restored) }).valid, false);
    assert.equal(stateFor({ binding: configuredBinding({ ...restored, gripper_range: null }) }).valid, true);
  });
  await test('encoding declarations and prompt source may be omitted without inventing a rotation encoding', () => {
    const omitted = clone(spec);
    for (const key of ['action_mode', 'rotation_representation', 'prompt_key']) delete omitted[key];
    assert.deepEqual(errors(omitted), []);
    const normalized = normalize(omitted);
    for (const key of ['action_mode', 'rotation_representation', 'prompt_key']) assert.equal(normalized[key], '');
    assert.equal(stateFor({ binding: { ...binding, processor: { name: 'ComposableDataConfig', parameters: { composition: omitted } } } }).valid, true);
    const blank = { ...omitted, action_mode: '\ufeff \t', rotation_representation: null, prompt_key: '\u00a0 ' };
    assert.deepEqual(errors(blank), []); assert.deepEqual(normalize(blank), normalized);
    for (const [key, value] of [['action_mode', 'unknown'], ['rotation_representation', 'euler'], ['prompt_key', '/etc/passwd'], ['prompt_key', 123]])
      assert.ok(errors({ ...normalized, [key]: value }).length, key);
    const invalidLayout = clone(normalized); invalidLayout.action_layout.find(block => block.kind === 'rotation').width = 5;
    assert.ok(errors(invalidLayout).some(error => /3、4 或 6/.test(error)));
    const explicitMismatch = { ...normalized, rotation_representation: 'quaternion' };
    assert.ok(errors(explicitMismatch).some(error => /宽度.*旋转编码/.test(error)));
    assert.deepEqual(errors({ ...omitted, action_key: 'undefined' }), []);
  });
  await test('editor marks declarations optional and hides bounds until explicitly enabled', async () => {
    const { ComposedProcessingEditor } = await server.ssrLoadModule('/src/ComposedProcessingEditor.tsx');
    const optional = { ...spec, position_unit: '', joint_unit: '', coordinate_frame: '', gripper_range: null, action_mode: '', rotation_representation: '', prompt_key: '' };
    const render = (value) => renderToStaticMarkup(React.createElement(ComposedProcessingEditor, { value, profile: profileFor(spec), onChange() {} }));
    const html = render(optional);
    for (const label of ['位置单位（可选）', '关节单位（可选）', '动作坐标系（可选）', '组合动作模式（可选）', '组合动作旋转编码（可选）', '任务文本真实字段（可选）']) {
      const start = html.indexOf(label);
      const field = html.slice(start, html.indexOf('</label>', start));
      assert.doesNotMatch(field, /required|field-required/);
    }
    assert.match(html, /启用夹爪范围校验（可选）/);
    assert.match(html, /无文本时使用空提示/);
    assert.doesNotMatch(html, /夹爪原始最小值|夹爪原始最大值/);
    const enabled = render(spec);
    assert.match(enabled, /夹爪原始最小值/); assert.match(enabled, /夹爪原始最大值/);
  });
  await test('composition schema keys allow real nested fields and reject paths or code', () => {
    for (const key of ['observation.state', 'observation/robot0/joints', 'camera_0_rgb', ' 字段 ']) assert.equal(compositionSchemaKey(key), true, key);
    for (const key of ['/etc/passwd', '~/data', 'C:\\data', 'a//b', 'a/../b', '.', 'robot[0]', 'f()', 'a;rm', 'a$HOME', 'a\nkey']) assert.equal(compositionSchemaKey(key), false, key);
  });
  await test('trimmed action and prompt keys cannot occupy an auxiliary input', () => {
    for (const key of ['action_key', 'prompt_key']) {
      const conflict = { ...spec, [key]: ' value_target ', aux_keys: ['value_target'] };
      assert.ok(errors(conflict).some((error) => error.includes('辅助字段')));
    }
  });
  await test('canonical text uses ECMAScript trim including BOM without swallowing edge controls', () => {
    const bom = clone(spec); bom.state_fields[0].key = '\uFEFF' + bom.state_fields[0].key + '\uFEFF';
    bom.robot_model = '\uFEFF' + bom.robot_model + '\uFEFF';
    assert.deepEqual(errors(bom), []);
    assert.equal(normalize(bom).state_fields[0].key, spec.state_fields[0].key);
    assert.equal(normalize(bom).robot_model, spec.robot_model);
    for (const char of ['\u001c', '\u0085']) {
      const controlled = clone(spec); controlled.state_fields[0].key = char + controlled.state_fields[0].key + char;
      assert.ok(errors(controlled).length); assert.equal(compositionSchemaKey(controlled.state_fields[0].key), false);
    }
  });
  await test('ARX5 creates an incomplete draft without guessing DOF or physical units', () => {
    const seed = profileFor(spec); seed.action_mode = 'unknown'; seed.rotation_representation = 'unknown';
    const draft = helpers.draftComposedProcessing(seed);
    assert.equal(draft.robot_model, 'ARX5'); assert.equal(draft.state_fields[0].width, 0);
    assert.equal(draft.action_layout[0].width, 0); assert.equal(draft.coordinate_frame, '');
    assert.equal(draft.position_unit, ''); assert.equal(draft.action_mode, '');
    assert.deepEqual(draft.cameras.map((camera) => camera.key), seed.cameras.map((camera) => camera.key));
    assert.ok(errors(draft).length);
  });
  await test('UMI sample buttons provide explicit structure while critical facts remain user supplied', () => {
    const seed = profileFor(spec); seed.action_mode = 'unknown';
    const single = helpers.umiV4ComposedPreset(seed, 'single');
    const bimanual = helpers.umiV4ComposedPreset(seed, 'bimanual');
    assert.equal(single.state_fields.reduce((sum, field) => sum + field.width, 0), 16);
    assert.equal(bimanual.state_fields.reduce((sum, field) => sum + field.width, 0), 56);
    assert.equal(bimanual.action_layout.reduce((sum, block) => sum + block.width, 0), 20);
    assert.equal(single.state_history, 2); assert.equal(bimanual.action_mode, '');
    assert.equal(single.coordinate_frame, ''); assert.equal(single.cameras[0].layout, '');
    assert.ok(errors(single).some(error => /布局/.test(error)));
    seed.image_features = [{ key: seed.cameras[0].key, dtype: 'image', shape: [3, 240, 320] }];
    assert.deepEqual(errors(helpers.umiV4ComposedPreset(seed, 'single')), []);
    assert.deepEqual(single.cameras.map((camera) => camera.key), seed.cameras.map((camera) => camera.key));
  });
  await test('profile composition persists state/action independently and frozen row order', () => {
    const profile = profileFor(spec); assert.equal(profile.processing.state_representation, 'joint');
    assert.equal(profile.processing.action_representation, 'eef'); assert.deepEqual(datasetProfileErrors(profile), []);
    assert.deepEqual(readDatasetProfile(profile).processing, normalize(spec));
    const source = clone(profile); const changed = clone(spec); changed.state_history = 2;
    const synced = applyComposedProcessing(profile, changed);
    assert.equal(synced.state_dim, 7); assert.equal(synced.processing.state_history, 2);
    assert.equal(synced.action_dim, 10); assert.deepEqual(profile, source);
    const reversed = { ...changed, state_fields: [...changed.state_fields].reverse() };
    assert.notEqual(compositionToken(changed), compositionToken(reversed));
  });
  await test('clearing optional training declarations does not clear asset source semantics', () => {
    const profile = profileFor(clone(spec));
    const changed = { ...spec, action_mode: '', rotation_representation: '', prompt_key: '' };
    const updated = applyComposedProcessing(profile, changed);
    assert.equal(updated.action_mode, profile.action_mode);
    assert.equal(updated.rotation_representation, profile.rotation_representation);
    assert.equal(updated.processing.action_mode, '');
    assert.equal(updated.processing.rotation_representation, '');
  });
  await test('unrelated composition edits preserve unsynchronized basic metadata drafts', () => {
    const profile = profileFor(clone(spec));
    profile.robot_type = 'ARX5-v2'; profile.arm_mode = 'bimanual'; profile.action_representation = 'joint';
    profile.action_mode = 'delta'; profile.rotation_representation = 'none';
    profile.state_dim = 40; profile.action_dim = 32;
    profile.cameras.push({ name: '额外相机', key: 'observation.images.extra', kind: 'depth' });
    const changed = clone(spec); changed.state_history = 2; changed.state_fields[0].normalize = false;
    changed.cameras[0].slot = 'base_0_rgb'; changed.cameras[0].layout = 'hwc';
    const source = clone(profile); const synced = applyComposedProcessing(profile, changed);
    assert.deepEqual({ ...synced, processing: source.processing }, source);
    assert.deepEqual(profile, source); assert.deepEqual(datasetProfileErrors(synced), []);
  });
  await test('only an actual composition field or width change syncs its own shared metadata', () => {
    const profile = profileFor(clone(spec)); profile.robot_type = 'basic draft';
    profile.state_dim = 40; profile.action_dim = 32;
    const changed = clone(spec); changed.state_fields[0].width = 8;
    const stateChanged = applyComposedProcessing(profile, changed);
    assert.equal(stateChanged.state_dim, 9); assert.equal(stateChanged.action_dim, 32);
    assert.equal(stateChanged.robot_type, 'basic draft');
    const actionChanged = applyComposedProcessing(profile, { ...spec, action_layout: spec.action_layout.map((block) => ({ ...block, width: block.width + 1 })) });
    assert.equal(actionChanged.action_dim, 13); assert.equal(actionChanged.state_dim, 40);
    const robotChanged = applyComposedProcessing(profile, { ...spec, robot_model: 'composition draft' });
    assert.equal(robotChanged.robot_type, 'composition draft'); assert.equal(robotChanged.state_dim, 40);
  });
  await test('explicit basic-description sync retains vector rows and existing camera roles and display names', () => {
    const profile = profileFor(clone(spec)); profile.robot_type = 'ARX5-v2';
    profile.cameras = [{ name: '新增深度', key: 'observation.images.extra', kind: 'depth' }, ...profile.cameras];
    const changed = helpers.updateComposedProcessingFromProfile(profile, spec);
    assert.equal(changed.robot_model, 'ARX5-v2');
    assert.deepEqual(changed.state_fields, spec.state_fields); assert.deepEqual(changed.action_layout, spec.action_layout);
    assert.equal(changed.cameras[1].slot, spec.cameras[0].slot);
    assert.notEqual(changed.cameras[0].slot, changed.cameras[1].slot);
    const synced = applyComposedProcessing(profile, changed);
    assert.deepEqual(synced.cameras.map((camera) => camera.name), ['新增深度', '相机 1']);
    assert.deepEqual(datasetProfileErrors(synced), []);
    const sameDimension = { ...profile, arm_mode: 'bimanual', action_representation: 'joint', rotation_representation: 'none' };
    const incompatible = helpers.updateComposedProcessingFromProfile(sameDimension, changed);
    assert.deepEqual(incompatible.action_layout, changed.action_layout);
    assert.ok(errors(incompatible).length); // no guessed joint width or right-arm block
  });
  await test('historical composition cannot lock source descriptions to one training plan', () => {
    const profile = profileFor(spec); profile.robot_type = 'Another robot';
    assert.deepEqual(datasetProfileErrors(profile), []); assert.equal(readDatasetProfile(profile).robot_type, 'Another robot');
    const invalid = profileFor({ ...spec, extra: 'no expressions' }); assert.equal(readDatasetProfile(invalid), undefined);
  });
  await test('registration exposes source labels and keeps combination editor in training', () => {
    const html = renderToStaticMarkup(React.createElement(DatasetProfileForm, { value: profileFor(spec), onChange() {} }));
    for (const text of ['使用组件组合数据处理', '状态字段拼接顺序', '动作向量分块顺序', '模型视觉槽位']) assert.ok(!html.includes(text), text);
    assert.match(html, /字段|标签/);
    assert.doesNotMatch(html, /name="(?:python|expression|module)"/);
  });
  await test('combination parameter validation requires capability 1 and exactly one data-only field', () => {
    const choice = composedChoice(spec);
    assert.deepEqual(processorParameterErrors(choice, capability), []);
    assert.ok(processorParameterErrors(choice, { ...capability, composition_schema_version: undefined }).length);
    assert.ok(processorParameterErrors({ ...choice, parameters: { ...choice.parameters, stage: 'value' } }, capability).length);
    assert.ok(processorParameterErrors({ ...choice, parameters: { composition: { ...spec, expression: 'eval()' } } }, capability).length);
    assert.deepEqual(filterDataProcessors([capability], ''), []);
  });
  await test('recipe overrides cannot replace asset-owned composition directly or through a parent object', () => {
    assert.equal(isAssetManagedField('data.composition'), true);
    assert.equal(isAssetManagedField('data.composition.action_layout[0].width'), true);
    assert.deepEqual(assetManagedOverridePaths({ 'data.base_config': { composition: spec } }), ['data.base_config']);
    assert.deepEqual(processorParameterErrors(composedChoice(spec), capability), []);
  });
  await test('source image shapes fill camera layout defaults instead of guessing CHW', () => {
    const profile = profileFor(spec); delete profile.processing;
    profile.image_features = [{ key: spec.cameras[0].key, dtype: 'video', shape: [240, 320, 3] }];
    const before = clone(profile);
    assert.equal(helpers.trainingCompositionDraft(profile).cameras[0].layout, 'hwc');
    profile.image_features[0].shape = [3, 240, 320];
    assert.equal(helpers.trainingCompositionDraft(profile).cameras[0].layout, 'chw');
    profile.image_features[0].shape = [3, 240, 3];
    assert.equal(helpers.trainingCompositionDraft(profile).cameras[0].layout, '');
    delete profile.image_features;
    assert.equal(helpers.trainingCompositionDraft(profile).cameras[0].layout, '');
    assert.deepEqual(before.image_features[0].shape, [240, 320, 3]);
  });
  await test('known image metadata blocks wrong channels and missing history frame before preflight', () => {
    const imageFeatures = [{ key: spec.cameras[0].key, dtype: 'image', shape: [240, 320, 3] }];
    const wrong = stateFor({ imageFeatures });
    assert.equal(wrong.valid, false); assert.ok(wrong.errors.some(error => /RGB 三通道/.test(error)));
    const corrected = clone(spec); corrected.cameras[0].layout = 'hwc';
    assert.equal(stateFor({ binding: configuredBinding(corrected), imageFeatures }).valid, true);
    imageFeatures[0].shape = [4, 240, 320, 3];
    assert.ok(stateFor({ binding: configuredBinding(corrected), imageFeatures }).errors.some(error => /图像时间索引/.test(error)));
    corrected.cameras[0].time_index = 1;
    assert.equal(stateFor({ binding: configuredBinding(corrected), imageFeatures }).valid, true);
  });
  await test('training uses the binding plan even when source asset has no composition', () => {
    const initial = stateFor({ binding }); assert.equal(initial.valid, false); assert.ok(initial.errors.length);
    const configured = { ...binding, recipe_index: undefined, processor: composedChoice(spec) };
    assert.equal(stateFor({ binding: configured }).valid, true);
    const plain = asset(); delete plain.metadata.dataset_profile.processing;
    assert.equal(stateFor({ binding: configured, dataset: plain }).valid, true);
    assert.equal(stateFor({ binding: { ...configured, processor: { name: 'ManualFactory', parameters: {} } } }).valid, false);
    const html = renderToStaticMarkup(React.createElement(DatasetProcessing, { binding: configured, dataset: asset(), entries: [capability], slots: [], index: 0, stage: 'sft', managedParameters: {}, onChange() {} }));
    assert.match(html, /本次训练组合方案已配置/); assert.match(html, /状态 JOINT/); assert.match(html, /动作 EEF/);
    assert.match(html, /自定义字段组合/); assert.match(html, /状态字段拼接顺序/);
    assert.match(html, /导入资产中的历史组合方案/);
  });
  await test('missing, old or unavailable catalogs block final composition without locking editing', () => {
    for (const entries of [[], [{ ...capability, available: false }], [{ ...capability, composition_schema_version: undefined }]]) assert.ok(stateFor({ entries }).errors.length);
    const malformed = asset(); malformed.metadata.dataset_profile.processing.action_key = '/data/secret';
    const html = renderToStaticMarkup(React.createElement(DatasetProcessing, { binding: configuredBinding(), dataset: malformed, entries: [], slots: [], index: 0, stage: 'sft', managedParameters: {}, onChange() {} }));
    assert.match(html, /尚未确认当前训练环境支持/); assert.match(html, /状态字段拼接顺序/);
    assert.doesNotMatch(html, /导入资产中的历史组合方案/);
    const unbound = renderToStaticMarkup(React.createElement(DatasetProcessing, { binding, dataset: asset(), entries: [], slots: [], index: 0, stage: 'sft', managedParameters: {}, onChange() {} }));
    assert.match(unbound, /自动推荐处理器/); assert.match(unbound, /选择现有处理器/);
    assert.doesNotMatch(unbound, /状态字段拼接顺序/);
  });
  await test('MEM/Hybrid and missing phase supervision cannot slip through a standard composition', () => {
    for (const family of ['memory', 'hybrid']) assert.ok(stateFor({ slots: [{ recipe_index: 0, processing: 'Special' }], entries: [capability, { name: 'Special', dataset_contract: { model_family: family } }] }).errors.some((error) => error.includes('MEM/Hybrid')));
    assert.ok(stateFor({ stage: 'value' }).errors.some((error) => error.includes('value_target')));
    assert.ok(stateFor({ stage: 'offline_rl' }).errors.some((error) => error.includes('is_positive')));
    const supervised = { ...spec, aux_keys: ['value_target', 'is_positive'] };
    assert.deepEqual(stateFor({ stage: 'value', binding: configuredBinding(supervised) }).errors, []);
  });
  await test('canonical comparison preserves row order but tolerates JSON property order', () => {
    const reordered = Object.fromEntries(Object.entries(spec).reverse());
    assert.equal(compositionToken(spec), compositionToken(reordered));
    const selected = { ...binding, processor: { name: 'ComposableDataConfig', parameters: { composition: reordered } } };
    assert.equal(stateFor({ binding: selected }).valid, true);
    const changed = clone(spec); changed.state_fields[0].normalize = !changed.state_fields[0].normalize;
    assert.equal(stateFor({ binding: selected, dataset: asset(changed) }).valid, true);
  });
  const sourceLabels = () => [
    { key: 'observation.joints', role: 'state', representation: 'joint', value_mode: 'absolute', component: 'joint', arm: 'left', shape: [6], dtype: 'float32' },
    { key: 'observation.gripper', role: 'state', representation: 'unknown', value_mode: 'absolute', component: 'gripper', arm: 'left', shape: [1], dtype: 'float32' },
    { key: 'actions', role: 'action', representation: 'eef', value_mode: 'absolute', component: 'other', arm: 'unknown', shape: [16, 10], dtype: 'float32', names: [
      ...['x', 'y', 'z'].map((name) => 'left_position_' + name),
      ...Array.from({length:6}, (_, index) => 'left_rotation_6d_' + index), 'left_gripper_width',
    ] },
  ];
  const taggedProfile = () => ({ ...profileFor(spec), processing: undefined,
    feature_keys: [...spec.state_fields.map((field) => field.key), 'actions', 'task', ...spec.cameras.map((camera) => camera.key)], field_annotations: sourceLabels() });
  await test('source action layout reads width from the last stored axis and complete coordinate names', () => {
    const profile=taggedProfile();profile.action_dim=25;
    const before=clone(profile),inferred=helpers.inferSourceActionLayout(profile,'actions');
    assert.equal(inferred.width,10);assert.deepEqual(inferred.shape,[16,10]);
    assert.deepEqual(inferred.blocks.map(block=>[block.arm,block.kind,block.width]),[['left','position',3],['left','rotation',6],['left','gripper',1]]);
    assert.deepEqual(inferred.blocks.map(block=>block.normalize),[true,false,false]);
    assert.equal(inferred.blocks.reduce((sum,block)=>sum+block.width,0),10);
    assert.deepEqual(profile,before);assert.notEqual(inferred.shape,profile.field_annotations[2].shape);
    profile.field_annotations[2].shape=[10];
    assert.equal(helpers.inferSourceActionLayout(profile,'actions').width,10);
  });
  await test('a field-level actions name proves total width but never invents EEF semantic partitions', () => {
    const profile=taggedProfile();profile.action_dim=25;profile.robot_type='UMI';
    profile.field_annotations[2].names=['actions'];
    const inferred=helpers.inferSourceActionLayout(profile,'actions');
    assert.equal(inferred.width,10);assert.equal(inferred.blocks,undefined);assert.match(inferred.reason,/逐维坐标名称/);
    assert.equal(helpers.trainingCompositionDraft(profile).action_layout[0].width,0);
    profile.field_annotations[2].names=[];
    assert.equal(helpers.inferSourceActionLayout(profile,'actions').blocks,undefined);
    profile.field_annotations[2].shape=undefined;
    assert.equal(helpers.inferSourceActionLayout(profile,'actions').width,undefined);
  });
  await test('explicit single-component tags seed their real width and right-arm attribution', () => {
    const profile=taggedProfile();profile.field_annotations[2]={key:'actions',role:'action',representation:'joint',value_mode:'absolute',component:'joint',arm:'right',shape:[7]};
    const inferred=helpers.inferSourceActionLayout(profile,'actions');
    assert.equal(inferred.width,7);assert.deepEqual(inferred.blocks,[{arm:'right',kind:'joint',width:7,normalize:true}]);
    assert.deepEqual(helpers.trainingCompositionDraft(profile).action_layout,inferred.blocks);
  });
  await test('selected action keys and coordinate order stay independent across multiple action fields', () => {
    const profile=taggedProfile();profile.arm_mode='bimanual';
    profile.field_annotations.push({key:'future.actions',role:'action',representation:'joint',value_mode:'delta',component:'other',arm:'unknown',shape:[20,4],names:['right_joint_0','right_joint_1','left_joint_0','left_joint_1']});
    const inferred=helpers.inferSourceActionLayout(profile,'future.actions');
    assert.equal(inferred.width,4);assert.deepEqual(inferred.blocks.map(block=>[block.arm,block.kind,block.width]),[['right','joint',2],['left','joint',2]]);
    assert.equal(helpers.inferSourceActionLayout(profile,'actions').width,10);
    assert.equal(helpers.inferSourceActionLayout(profile,'observation.joints').width,undefined);
    assert.equal(helpers.inferSourceActionLayout(profile,'missing').width,undefined);
    const ambiguous=clone(profile);ambiguous.field_annotations.push(clone(ambiguous.field_annotations[2]));
    assert.equal(helpers.inferSourceActionLayout(ambiguous,'actions').width,undefined);
  });
  await test('incomplete or conflicting coordinate evidence never produces partial action blocks', () => {
    for(const names of [sourceLabels()[2].names.slice(0,-1),[...sourceLabels()[2].names.slice(0,-1),'unknown'],['left_right_position_x',...sourceLabels()[2].names.slice(1)]]){
      const profile=taggedProfile();profile.field_annotations[2].names=names;
      const inferred=helpers.inferSourceActionLayout(profile,'actions');assert.equal(inferred.width,10);assert.equal(inferred.blocks,undefined);assert.ok(inferred.reason);
    }
    const profile=taggedProfile();profile.arm_mode='bimanual';profile.field_annotations[2].names=Array.from({length:10},(_,i)=>'joint_'+i);
    assert.equal(helpers.inferSourceActionLayout(profile,'actions').blocks,undefined);
    profile.field_annotations[2].names=sourceLabels()[2].names;profile.field_annotations[2].arm='right';
    assert.equal(helpers.inferSourceActionLayout(profile,'actions').blocks,undefined);
    profile.field_annotations[2].arm='unknown';profile.field_annotations[2].component='joint';
    assert.equal(helpers.inferSourceActionLayout(profile,'actions').blocks,undefined);
    profile.field_annotations[2].component='other';profile.field_annotations[2].representation='joint';
    assert.equal(helpers.inferSourceActionLayout(profile,'actions').blocks,undefined);
  });
  await test('invalid or unsupported action shapes cannot masquerade as source dimensions', () => {
    for(const shape of [[],[0,10],[16,0],[true,10],[16,1.5],[16,Infinity],[16,2**31]]){
      const profile=taggedProfile();profile.field_annotations[2].shape=shape;
      const inferred=helpers.inferSourceActionLayout(profile,'actions');assert.equal(inferred.width,undefined);assert.equal(inferred.blocks,undefined);
    }
    const profile=taggedProfile();profile.field_annotations[2].shape=[2,16,10];
    const inferred=helpers.inferSourceActionLayout(profile,'actions');assert.deepEqual(inferred.shape,[2,16,10]);assert.equal(inferred.width,undefined);assert.equal(inferred.blocks,undefined);
  });
  await test('ambiguous component names and noncontiguous duplicate blocks cannot become inferred layouts', () => {
    const profile=taggedProfile();
    profile.field_annotations[2]={key:'actions',role:'action',representation:'unknown',value_mode:'unknown',component:'other',arm:'left',shape:[1],names:['left_joint_position_0']};
    const ambiguous=helpers.inferSourceActionLayout(profile,'actions');
    assert.equal(ambiguous.width,1);assert.equal(ambiguous.blocks,undefined);assert.ok(ambiguous.reason);
    profile.field_annotations[2]=sourceLabels()[2];
    const names=profile.field_annotations[2].names;
    profile.field_annotations[2].names=[names[0],names[3],names[1],names[2],...names.slice(4)];
    const duplicate=helpers.inferSourceActionLayout(profile,'actions');
    assert.equal(duplicate.width,10);assert.equal(duplicate.blocks,undefined);assert.match(duplicate.reason,/重复分块/);
  });
  await test('training drafts seed exact source widths and preserve independent state/action semantics', () => {
    const profile = taggedProfile();
    const before = clone(profile);
    const draft = helpers.trainingCompositionDraft(profile);
    assert.equal(draft.state_representation, 'joint');assert.equal(draft.action_representation, 'eef');
    assert.deepEqual(draft.state_fields.map((field) => field.width), [6, 1]);
    assert.deepEqual(draft.action_layout.map((block) => [block.kind, block.width]), [['position',3], ['rotation',6], ['gripper',1]]);
    assert.equal(draft.position_unit, '');assert.equal(draft.coordinate_frame, '');assert.equal(draft.gripper_range, null);
    assert.deepEqual(profile, before);
    profile.field_annotations[0].shape = undefined;
    assert.equal(helpers.trainingCompositionDraft(profile).state_fields[0].width, 0);
  });
  await test('optional encoding permits unknown action modes while source representation stays checked', () => {
    const profile = taggedProfile();profile.field_annotations[2].value_mode = 'unknown';
    const draft = helpers.trainingCompositionDraft(profile);
    assert.equal(draft.action_mode, '');
    assert.deepEqual(helpers.trainingCompositionProfileErrors(profile, spec), []);
    assert.deepEqual(helpers.trainingCompositionProfileErrors(profile, { ...spec, action_mode: '', rotation_representation: '', prompt_key: '' }), []);
    profile.field_annotations[2].value_mode = 'delta';
    assert.ok(helpers.trainingCompositionProfileErrors(profile, spec).some(error => /绝对值\/增量/.test(error)));
    assert.deepEqual(helpers.trainingCompositionProfileErrors(profile, { ...spec, action_mode: '' }), []);
    profile.field_annotations[2].value_mode = 'absolute';profile.field_annotations[2].representation = 'unknown';
    assert.ok(helpers.trainingCompositionProfileErrors(profile, spec).some((error) => /EEF\/Joint.*尚未确认/.test(error)));
    profile.field_annotations[2].representation = 'eef';profile.field_annotations[1].value_mode = 'unknown';
    assert.ok(helpers.trainingCompositionProfileErrors(profile, spec).some((error) => error.includes('observation.gripper')));
  });
  await test('gripper state tags do not change Joint arm representation or satisfy complete arm actions', () => {
    const profile=taggedProfile();profile.field_annotations[1].representation='gripper';
    const draft=helpers.trainingCompositionDraft(profile);
    assert.equal(draft.state_representation,'joint');assert.equal(draft.action_representation,'eef');
    assert.deepEqual(helpers.trainingCompositionProfileErrors(profile,spec),[]);
    const action=profile.field_annotations[2];action.representation='gripper';action.component='gripper';
    assert.ok(['eef','joint'].includes(helpers.trainingCompositionDraft(profile).action_representation));
    assert.ok(helpers.trainingCompositionProfileErrors(profile,spec).some(error=>error.includes('仅包含夹爪开度')));
  });
  await test('binding validation checks individual source fields instead of asset-owned order or total state width', () => {
    const profile = taggedProfile();profile.state_dim = 500;
    assert.deepEqual(helpers.trainingCompositionProfileErrors(profile, spec), []);
    const reordered = { ...spec, state_fields: [...spec.state_fields].reverse() };
    assert.deepEqual(helpers.trainingCompositionProfileErrors(profile, reordered), []);
    const changed = clone(spec);changed.state_fields[0].width = 7;
    assert.ok(helpers.trainingCompositionProfileErrors(profile, changed).some((error) => /宽度.*原始形状/.test(error)));
    const unknown = { ...spec, state_fields: [{ ...spec.state_fields[0], key: 'observation.absent' }] };
    assert.ok(helpers.trainingCompositionProfileErrors(profile, unknown).some((error) => /缺少登记标签/.test(error)));
  });
  await test('step-two read-only composition keeps current stage gates and never mounts another editor', () => {
    const html = renderToStaticMarkup(React.createElement(DatasetProcessing, { binding: configuredBinding(), dataset: asset(), entries: [capability], slots: [], index: 0, stage: 'value', managedParameters: {}, readOnly: true, onChange() {} }));
    assert.match(html, /value_target/);assert.match(html, /本次训练/);
    assert.doesNotMatch(html, /状态真实字段|上移状态字段|填入 UMI/);
    assert.equal(stateFor({ stage: 'value' }).valid, false);
    assert.equal(stateFor({ stage: 'offline_rl' }).valid, false);
    assert.equal(stateFor({ stage: 'sft' }).valid, true);
  });
} finally { await server.close(); }
