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
  const { trainingTemplateEntry } = await server.ssrLoadModule('/src/trainingTemplateEntry.ts');
  const { encodeTrainingDraft, decodeTrainingDraft } = await server.ssrLoadModule('/src/trainingDraft.ts');
  const { NormDatasetEditor } = await server.ssrLoadModule('/src/NormDatasetEditor.tsx');
  const { DEFAULT_AUTO_CONFIG } = await server.ssrLoadModule('/src/trainingConfiguration.ts');
  const resources = {
    nodes: [{ id: '169', metrics: { cpu_count: 112, available_cpu_count: 112 } }],
    profiles: [
      { id: 'openpi-native', node_id: '169', adapter: 'openpi-sft', execution_mode: 'native' },
      { id: 'smoke-native', node_id: '169', adapter: 'smoke', execution_mode: 'native' },
    ],
    assets: [
      { id: 'cloth', node_id: '169', kind: 'DATASET', name: 'Cloth', norm_asset_id: 'cloth-norm' },
      { id: 'cloth-two', node_id: '169', kind: 'DATASET', name: 'Cloth two', norm_asset_id: 'cloth-norm-two' },
    ],
  };
  const templates = {
    norm: { id: 'openpi-norm', name: '计算 norm_stats', status: 'AVAILABLE', runtime_adapter: 'openpi-sft', min_gpus: 0,
      parameters: { batch_size: { default: 32 }, num_workers: { default: 0 } } },
    sft: { id: 'openpi-sft', name: '监督微调', status: 'AVAILABLE', min_gpus: 1,
      parameters: { steps: { default: 10000 }, batch_size: { default: 8 }, num_workers: { default: 0 } } },
    recap: { id: 'recap', name: 'ReCap', status: 'AVAILABLE', runtime_adapter: 'openpi-sft', min_gpus: 1,
      parameters: { steps: { default: 10000 }, batch_size: { default: 8 }, num_workers: { default: 0 } } },
    smoke: { id: 'smoke', name: '平台链路测试', status: 'AVAILABLE', min_gpus: 0,
      parameters: { steps: { default: 30 }, batch_size: { default: 8 } } },
  };
  const source = () => ({
    request: {
      name: '叠衣服数据norm值计算', node_id: '169', profile_id: 'openpi-native', template_id: 'openpi-sft',
      inputs: { model: 'weights', dataset: 'cloth', norm: 'old-norm' },
      dataset_bindings: [
        { dataset_asset_id: 'cloth', norm_asset_id: 'old-norm', weight: 7,
          processor: { name: 'ComposableDataConfig', parameters: { composition: {
            action_key: 'actions', action_layout: [{ arm: 'shared', kind: 'joint', width: 14 }],
            cameras: [{ key: 'observation.images.top_head', slot: 'base_0_rgb' }],
            state_fields: [{ key: 'observation.state', width: 14 }],
          } } } },
        { dataset_asset_id: 'cloth-two', norm_asset_id: 'old-norm-two', weight: 3,
          processor: { name: 'CustomNumericDataConfig', parameters: { action_key: 'robot.actions', scale: 2 } } },
      ],
      auto_config: { ...DEFAULT_AUTO_CONFIG, finetune_mode: 'lora', action_horizon: 20, action_dim: 24 },
      parameters: { steps: 100, batch_size: 64, learning_rate: 0.01, num_workers: 24 },
      gpu_count: 2, requested_gpu_uuids: ['GPU-a', 'GPU-b'], normalization_method: 'zscore',
      recipe: 'old_training_recipe', recipe_revision_id: 'stale-revision',
      recipe_definition: { name: 'old-recipe' }, entrypoint: 'scripts/train.py',
      training_stage: 'offline_rl', stage_config: { steps: 999 },
    },
    step: 1, advancedRecipe: true, editedParameters: ['batch_size', 'steps', 'num_workers', 'learning_rate'],
    editedStageParameters: ['steps'], entrypointEdited: true,
    recipeEditorDraft: { scope: 'old-scope', fields: { 'model.checkpoint': 'old weights' } },
    recipeDatasets: [{ recipe_index: 0, weight: 7, processing: 'OldFactory' }],
    recipeDefaults: { batch_size: 1024 }, stageRecipeDefaults: { steps: 999 },
  });

  await test('choosing the norm card overrides a restored SFT mode and retains numeric processing for each dataset', () => {
    const saved = source(), before = structuredClone(saved);
    const selected = trainingTemplateEntry(saved, templates.norm, resources);
    assert.equal(selected.request.template_id, 'openpi-norm');
    assert.equal(selected.request.name, saved.request.name);
    assert.equal(selected.request.node_id, '169');
    assert.equal(selected.request.profile_id, 'openpi-native');
    assert.equal(selected.step, 0);
    assert.equal(selected.advancedRecipe, false);
    assert.equal(selected.request.gpu_count, 0);
    assert.deepEqual(selected.request.inputs, {});
    assert.deepEqual(selected.request.parameters, { batch_size: 64, num_workers: 8 });
    assert.equal(selected.request.auto_config.action_horizon, 20);
    assert.equal(selected.request.auto_config.action_dim, 24);
    assert.equal(selected.request.auto_config.finetune_mode, 'full');
    assert.deepEqual(selected.request.norm_datasets.map(row => row.dataset_asset_id), ['cloth', 'cloth-two']);
    const composition = selected.request.norm_datasets[0].processor.parameters.composition;
    assert.deepEqual(composition.cameras, []);
    assert.deepEqual(composition.action_layout, saved.request.dataset_bindings[0].processor.parameters.composition.action_layout);
    assert.deepEqual(composition.state_fields, saved.request.dataset_bindings[0].processor.parameters.composition.state_fields);
    assert.deepEqual(selected.request.norm_datasets[1].processor, saved.request.dataset_bindings[1].processor);
    for (const key of ['dataset_bindings', 'normalization_method', 'requested_gpu_uuids', 'recipe', 'recipe_revision_id',
      'recipe_definition', 'entrypoint', 'training_stage', 'stage_config']) assert.equal(Object.hasOwn(selected.request, key), false, key);
    assert.equal(selected.recipeEditorDraft, undefined);
    assert.equal(selected.recipeDefaults, undefined);
    assert.equal(selected.recipeDatasets, undefined);
    assert.deepEqual(selected.editedStageParameters, []);
    assert.ok(selected.editedParameters.includes('resource_recommendation.auto.num_workers'));
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(selected)), selected);
    assert.deepEqual(saved, before, 'converting the entry must not mutate the saved source');
    const html = renderToStaticMarkup(React.createElement(NormDatasetEditor, {
      rows: selected.request.norm_datasets, assets: resources.assets, automatic: true, onChange() {},
    }));
    assert.match(html, /批量计算模式/);
    assert.doesNotMatch(html, /选择归一化参数|混采权重|基础模型|视觉输入与模型槽位/);
  });

  await test('ordinary new-training restoration and repeated matching template entries keep the existing draft intact', () => {
    assert.equal(trainingTemplateEntry(undefined, templates.norm, resources), undefined);
    const saved = source();
    assert.equal(trainingTemplateEntry(saved, templates.sft, resources), saved);
    const selected = trainingTemplateEntry(saved, templates.norm, resources);
    selected.request.parameters.num_workers = 3;
    selected.editedParameters = ['num_workers'];
    assert.equal(trainingTemplateEntry(selected, templates.norm, resources), selected);
    assert.equal(selected.request.parameters.num_workers, 3, 'reopening the same norm template retains explicit edits');
  });

  await test('all explicit template cards take priority over a different draft, without carrying old recipe editor state', () => {
    for (const target of [templates.recap, templates.smoke]) {
      const selected = trainingTemplateEntry(source(), target, resources);
      assert.equal(selected.request.template_id, target.id);
      assert.equal(selected.step, 0);
      assert.equal(selected.recipeEditorDraft, undefined);
      assert.equal(selected.request.recipe_revision_id, undefined);
      assert.equal(selected.request.training_stage, undefined);
      assert.equal(selected.request.profile_id, target.id === 'smoke' ? 'smoke-native' : 'openpi-native');
    }
    const smoke = trainingTemplateEntry(source(), templates.smoke, resources);
    assert.equal(smoke.request.recipe, 'smoke');
    assert.deepEqual(smoke.request.inputs, {});
    assert.equal(smoke.request.auto_config, undefined);
    assert.equal(smoke.request.gpu_count, 0);
    const sft = trainingTemplateEntry(smoke, templates.sft, resources);
    assert.equal(sft.request.template_id, 'openpi-sft');
    assert.equal(sft.request.profile_id, 'openpi-native');
    assert.equal(sft.request.recipe, undefined);
    assert.equal(sft.request.gpu_count, 1);
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(sft)), sft);
  });

  await test('switching from statistics to SFT restores dataset norm bindings and requires training assets again', () => {
    const norm = trainingTemplateEntry(source(), templates.norm, resources);
    const sft = trainingTemplateEntry(norm, templates.sft, resources);
    assert.equal(sft.request.template_id, 'openpi-sft');
    assert.equal(sft.request.norm_datasets, undefined);
    assert.deepEqual(sft.request.dataset_bindings.map(row => row.norm_asset_id), ['cloth-norm', 'cloth-norm-two']);
    assert.deepEqual(sft.request.dataset_bindings.map(row => row.weight), [1, 1]);
    assert.deepEqual(sft.request.inputs, {});
    assert.equal(sft.request.gpu_count, 1);
    assert.equal(sft.request.requested_gpu_uuids, undefined);
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(sft)), sft);
  });

  await test('legacy named-recipe and unresolvable dataset IDs stay selected while new processing is seeded from metadata', () => {
    const saved = source();
    saved.request.dataset_bindings = [{ dataset_asset_id: 'cloth', norm_asset_id: 'old', weight: 1, recipe_index: 0 },
      { dataset_asset_id: 'missing-dataset', norm_asset_id: 'old', weight: 1, recipe_index: 1 }];
    const selected = trainingTemplateEntry(saved, templates.norm, resources);
    assert.deepEqual(selected.request.norm_datasets.map(row => row.dataset_asset_id), ['cloth', 'missing-dataset']);
    assert.ok(selected.request.norm_datasets.every(row => row.processor.name === 'ComposableDataConfig'));
    assert.ok(selected.request.norm_datasets.every(row => row.processor.parameters.composition.cameras.length === 0));
    const otherProfile = trainingTemplateEntry({ ...saved, request: { ...saved.request, profile_id: 'stale-profile' } }, templates.norm, resources);
    assert.equal(otherProfile.request.profile_id, 'openpi-native');
  });
} finally {
  await server.close();
}
