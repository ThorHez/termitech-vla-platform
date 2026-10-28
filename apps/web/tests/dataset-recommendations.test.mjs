import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom',
});
try {
  const { recommendDatasetProcessor: rawRecommend } = await server.ssrLoadModule('/src/datasetRecommendations.ts');
  const recommend = (asset, entries, stage = 'sft', family = 'standard', parameters = {}) => rawRecommend(asset, entries, stage, family, parameters);
  const { DatasetProcessing } = await server.ssrLoadModule('/src/DatasetProcessing.tsx');
  const { DatasetBindingSummary } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
  const profile = {
    schema_version: 1, format: 'lerobot', robot_type: 'ARX5', arm_mode: 'bimanual',
    action_representation: 'eef', cameras: [
      { name: '左腕', key: 'observation.left_rgb', kind: 'rgb' },
      { name: '右腕', key: 'observation.right_rgb', kind: 'rgb' },
    ], task_description: '拿起杯子', action_mode: 'unknown', rotation_representation: 'unknown',
    feature_keys: ['robot0_eef_pos', 'robot1_eef_pos', 'actions', 'task'],
  };
  const asset = { id: 'data', kind: 'DATASET', name: '双臂数据', node_id: 'node', path: '/data/train', version: '1', metadata: { dataset_profile: profile } };
  const entry = {
    name: 'ArbitraryRegisteredName', description: 'actual code', available: true,
    source: { path: 'src/openpi/training/config.py', sha256: 'a'.repeat(64) },
    parameters: [{ path: 'base_config.prompt_from_task', type: 'boolean', required: false, default: true }], unsupported_parameters: [],
    dataset_contract: { evidence: 'static_repack', arm_mode: 'bimanual', action_representation: 'eef',
      camera_keys: ['observation.left_rgb', 'observation.right_rgb'],
      required_features: ['robot0_eef_pos', 'robot1_eef_pos', 'actions', 'task', 'observation.left_rgb', 'observation.right_rgb'],
      training_stage: 'sft', model_family: 'standard' },
  };
  const withProfile = (changes) => ({ ...asset, metadata: { dataset_profile: { ...profile, ...changes } } });
  const withContract = (changes) => ({ ...entry, dataset_contract: { ...entry.dataset_contract, ...changes } });

  await test('complete registered data matches actual contract regardless of class name', () => {
    const result = recommend(asset, [entry]);
    assert.equal(result.status, 'unique'); assert.equal(result.selected.name, entry.name);
    assert.equal(recommend(withProfile({ robot_type: 'custom-robot' }), [entry]).status, 'unique');
  });
  await test('old, malformed and unsupported metadata never silently auto-select', () => {
    assert.equal(recommend({ ...asset, metadata: {} }, [entry]).status, 'missing_profile');
    assert.equal(recommend(withProfile({ schema_version: 2 }), [entry]).status, 'missing_profile');
    assert.equal(recommend(withProfile({ format: 'rlds' }), [entry]).status, 'no_match');
    assert.equal(recommend(asset, [{ ...entry, dataset_contract: undefined }]).status, 'no_match');
  });
  await test('two cameras are insufficient when keys, arm mode or action representation differ', () => {
    assert.equal(recommend(withProfile({ arm_mode: 'single' }), [entry]).status, 'no_match');
    assert.equal(recommend(withProfile({ action_representation: 'joint' }), [entry]).status, 'no_match');
    assert.equal(recommend(withProfile({ cameras: profile.cameras.map((camera) => ({ ...camera, key: camera.key + '_wrong' })) }), [entry]).status, 'no_match');
    assert.equal(recommend(withProfile({ cameras: profile.cameras.slice(0, 1) }), [entry]).status, 'no_match');
  });
  await test('description never substitutes for real task or action feature declarations', () => {
    const result = recommend(withProfile({ feature_keys: [] }), [entry]);
    assert.equal(result.status, 'incomplete'); assert.equal(result.selected, undefined);
    assert.ok(result.candidates[0].missing.includes('数据字段 task'));
    assert.ok(result.candidates[0].missing.includes('数据字段 actions'));
  });
  await test('confirmed per-field action labels drive recommendations without changing the asset summary', () => {
    const annotated = withProfile({ action_representation: 'unknown', action_mode: 'unknown', field_annotations: [
      {key:'actions',role:'action',representation:'eef',value_mode:'delta',component:'other',arm:'unknown',shape:[16,20]},
    ] });
    const compatible = withContract({action_mode:'delta'});
    assert.equal(recommend(annotated,[compatible]).status,'unique');
    assert.equal(annotated.metadata.dataset_profile.action_representation,'unknown');
    const unresolved = structuredClone(annotated);
    unresolved.metadata.dataset_profile.field_annotations[0].value_mode='unknown';
    assert.equal(recommend(unresolved,[compatible]).status,'incomplete');
    assert.equal(recommend(unresolved,[compatible]).selected,undefined);
  });
  await test('ambiguous or incomplete alternatives never select the first entry', () => {
    const second = { ...entry, name: 'AnotherVersion' };
    assert.equal(recommend(asset, [entry, second]).status, 'ambiguous');
    assert.equal(recommend(asset, [second, entry]).selected, undefined);
    assert.equal(recommend(asset, [entry, { ...second, dataset_contract: { ...entry.dataset_contract, required_features: ['missing_state'] } }]).status, 'ambiguous');
    assert.equal(recommend(asset, [entry, { ...second, dataset_contract: { ...entry.dataset_contract, model_family: undefined } }]).status, 'ambiguous');
  });
  await test('standalone gripper labels neither override complete arm actions nor auto-select for gripper-only data', () => {
    const gripper={key:'gripper_action',role:'action',representation:'gripper',value_mode:'absolute',component:'gripper',arm:'left',shape:[1]};
    const annotated=withProfile({field_annotations:[gripper,{key:'actions',role:'action',representation:'eef',value_mode:'delta',component:'other',arm:'unknown'}]});
    assert.equal(recommend(annotated,[withContract({action_mode:'delta'})]).status,'unique');
    assert.equal(recommend(withProfile({field_annotations:[gripper]}),[entry]).status,'incomplete');
  });
  await test('phase, model family and explicit encoding conflicts are respected', () => {
    assert.equal(recommend(asset, [entry], 'value').status, 'no_match');
    assert.equal(recommend(asset, [entry], 'sft', 'memory').status, 'no_match');
    assert.equal(recommend(withProfile({ action_mode: 'absolute' }), [withContract({ action_mode: 'delta' })]).status, 'no_match');
    assert.equal(recommend(withProfile({ rotation_representation: 'axis_angle' }), [withContract({ rotation_representation: 'rotation_6d' })]).status, 'no_match');
    assert.equal(recommend(asset, [withContract({ model_family: undefined })]).status, 'incomplete');
    assert.equal(rawRecommend(asset, [entry], 'sft').status, 'incomplete');
    assert.equal(recommend(withProfile({ action_mode: 'delta' }), [entry]).status, 'incomplete');
    assert.equal(recommend(withProfile({ rotation_representation: 'quaternion' }), [entry]).status, 'incomplete');
  });
  await test('unavailable source and constructors requiring manual parameters cannot auto-apply', () => {
    assert.equal(recommend(asset, [{ ...entry, source: undefined }]).status, 'no_match');
    assert.equal(recommend(asset, [{ ...entry, available: false }]).status, 'no_match');
    assert.equal(recommend(asset, [{ ...entry, parameters: [{ path: 'horizon', type: 'integer', required: true }] }]).status, 'no_match');
    const acp = { ...withContract({ training_stage: 'offline_rl' }), parameters: [{ path: 'acp_dropout', type: 'number', required: false, default: 0.5 }] };
    assert.equal(recommend(asset, [acp], 'offline_rl', undefined, { acp_dropout: 0.2 }).status, 'unique');
  });
  await test('semantic summary and advanced fallback remain available without changing assets or weights', () => {
    const binding = { dataset_asset_id: 'data', norm_asset_id: 'norm', weight: 20, recipe_index: 0 };
    const before = JSON.stringify({ asset, binding, entry });
    const html = renderToStaticMarkup(React.createElement(DatasetProcessing, {
      binding, dataset: asset, entries: [entry], slots: [{ recipe_index: 0, processing: entry.name, weight: 1 }],
      index: 0, stage: 'sft', managedParameters: {}, onChange() {},
    }));
    assert.match(html, /双臂/); assert.match(html, /ARX5/); assert.match(html, /处理器推荐/);
    assert.match(html, /高级：手动选择处理器与参数/);
    assert.equal(JSON.stringify({ asset, binding, entry }), before);
  });
  await test('training summary displays frozen semantics instead of edited current asset metadata', () => {
    const html = renderToStaticMarkup(React.createElement(DatasetBindingSummary, {
      bindings: [{ dataset_asset_id: asset.id, norm_asset_id: 'norm', recipe_index: 0, weight: 1 }],
      assets: [withProfile({ robot_type: 'later-robot', task_description: 'later task' })],
      profiles: { dataset_0: { ...profile, robot_type: 'frozen-robot', task_description: 'original task' } },
    }));
    assert.match(html, /frozen-robot/); assert.match(html, /original task/);
    assert.doesNotMatch(html, /later-robot|later task/);
  });
} finally { await server.close(); }
