import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const helper = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const modes = await server.ssrLoadModule('/src/actionHorizonDefaults.ts');
  const configuration = await server.ssrLoadModule('/src/trainingConfiguration.ts');
  const resources = () => ({
    nodes: [{ id: 'node-a', name: 'Machine A', hostname: 'machine-a', status: 'ONLINE' }],
    profiles: [{ id: 'native', node_id: 'node-a', name: 'Native', adapter: 'openpi-sft', execution_mode: 'native', code_dir: '/code/a', python: '/code/a/.venv/bin/python' }],
    assets: [{ id: 'model-a', node_id: 'node-a', name: 'Initial policy', kind: 'MODEL', state: 'READY', path: '/models/a', version: '1' }],
  });
  const snapshot = () => ({
    request: {
      name: 'old-run', node_id: 'node-a', profile_id: 'native', template_id: 'recap',
      inputs: { model: 'model-a', dataset: 'old-data', norm: 'old-norm', dataset_0: 'data-a', norm_0: 'norm-a', dataset_1: 'data-b', norm_1: 'norm-b' },
      parameters: { batch_size: 72, steps: 60000, learning_rate: 3e-5 }, gpu_count: 4,
      auto_config: { model_type: 'pi05', finetune_mode: 'full', action_horizon: 16, action_dim: 32 },
      training_stage: 'offline_rl', stage_config: { acp_dropout: .5 },
      dataset_bindings: [
        { dataset_asset_id: 'data-a', norm_asset_id: 'norm-a', weight: 20,
          processor: { name: 'ComposableDataConfig', parameters: { composition: {
            action_key: 'actions', state_fields: [{ key: 'observation.joints', width: 6, arm: 'left', kind: 'joint', normalize: true }],
            action_layout: [{ width: 3, kind: 'position', arm: 'left' }, { width: 6, kind: 'rotation', arm: 'left' }, { width: 1, kind: 'gripper', arm: 'left' }],
            cameras: [{ key: 'observation.images.wrist', slot: 'left_wrist_0_rgb', kind: 'rgb', layout: 'chw', time_index: null }],
            gripper_range: null, nested: { list: [1, { text: 'preserve arbitrary nested processor settings' }] },
          } } } },
        { dataset_asset_id: 'data-b', norm_asset_id: 'norm-b', weight: 50, processor: { name: 'OtherProcessor', parameters: { image_horizon: 2, options: { normalized: true } } } },
        { dataset_asset_id: 'data-c', norm_asset_id: 'norm-c', weight: 1, recipe_index: 2 },
      ],
      entrypoint: 'scripts/train_multi_dataset.py',
    },
    step: 1, advancedRecipe: false, editedParameters: ['batch_size', 'auto_config.action_horizon'],
    editedStageParameters: ['acp_dropout'], entrypointEdited: true,
    recipeDatasets: [{ recipe_index: 2, weight: 1, processing: 'RecipeProcessor', multiple: true }],
    recipeDefaults: { batch_size: 32 }, stageRecipeDefaults: { acp_dropout: .1 },
    stageRecipeInfo: { values: { acp_dropout: .1 }, mixedDropout: false, valueHead: [] },
  });
  const record = (content) => ({ id: 'template-a', name: 'My mixed training', description: 'Three datasets', revision: 1,
    created_at: '2026-10-27T00:00:00Z', updated_at: '2026-10-27T00:00:00Z', content });

  await test('saving keeps model, stages, parameters and independent processor groups while clearing every data reference', () => {
    const source = snapshot(), before = structuredClone(source);
    const content = helper.prepareTrainingTemplate(source);
    const restored = codec.decodeTrainingDraft(content);
    assert.deepEqual(source, before);
    assert.equal(restored.request.name, '');
    assert.equal(restored.step, 0);
    assert.deepEqual(restored.request.inputs, { model: 'model-a' });
    assert.equal(restored.request.node_id, 'node-a');
    assert.equal(restored.request.profile_id, 'native');
    for (const key of ['parameters', 'auto_config', 'training_stage', 'stage_config', 'gpu_count', 'entrypoint']) assert.deepEqual(restored.request[key], source.request[key]);
    restored.request.dataset_bindings.forEach((binding, index) => {
      assert.equal(binding.dataset_asset_id, '');
      assert.equal(binding.norm_asset_id, '');
      assert.equal(binding.weight, source.request.dataset_bindings[index].weight);
      assert.deepEqual(binding.processor, source.request.dataset_bindings[index].processor);
      assert.equal(binding.recipe_index, source.request.dataset_bindings[index].recipe_index);
    });
    assert.equal(restored.request.dataset_profiles, undefined);
    assert.ok(restored.editedParameters.includes(helper.TRAINING_TEMPLATE_PROCESSING_MARKER));
  });

  await test('each application creates an independent editable snapshot and never mutates stored content', () => {
    const stored = record(helper.prepareTrainingTemplate(snapshot()));
    const before = structuredClone(stored);
    const first = helper.applyTrainingTemplate(stored, resources());
    const second = helper.applyTrainingTemplate(stored, resources());
    assert.deepEqual(first.issues, []);
    first.snapshot.request.dataset_bindings[0].processor.parameters.composition.state_fields[0].width = 99;
    first.snapshot.request.dataset_bindings[1].processor.parameters.options.normalized = false;
    first.snapshot.request.parameters.batch_size = 1;
    assert.equal(second.snapshot.request.parameters.batch_size, 72);
    assert.equal(second.snapshot.request.dataset_bindings[0].processor.parameters.composition.state_fields[0].width, 6);
    assert.equal(second.snapshot.request.dataset_bindings[1].processor.parameters.options.normalized, true);
    assert.deepEqual(stored, before);
  });

  await test('application also clears legacy dataset IDs and unsaved editor fields from otherwise valid draft content', () => {
    const source = snapshot();
    source.recipeEditorDraft = { scope: 'old-code', fields: { 'model.name': 'unsaved' }, saveMode: 'new' };
    const applied = helper.applyTrainingTemplate(record(codec.encodeTrainingDraft(source)), resources()).snapshot;
    assert.equal(applied.recipeEditorDraft, undefined);
    assert.equal(applied.step, 0);
    assert.equal(applied.request.name, '');
    assert.deepEqual(applied.request.inputs, { model: 'model-a' });
    assert.ok(applied.request.dataset_bindings.every((binding) => !binding.dataset_asset_id && !binding.norm_asset_id));
  });

  await test('saved recipe references survive only inside their original machine and environment scope', () => {
    const source = snapshot();source.advancedRecipe = true;
    source.request.recipe = 'saved_code_recipe';source.request.recipe_revision_id = 'recipe-revision-1';
    source.request.recipe_definition = { schema_version: 1, node_id: 'node-a', profile_id: 'native', overrides: { batch_size: 72 }, source: { sha256: 'a'.repeat(64) } };
    source.recipeEditorDraft = { scope: 'saved_code_recipe', fields: { batch_size: 'unfinished editor value' } };
    const stored = record(helper.prepareTrainingTemplate(source));
    const applied = helper.applyTrainingTemplate(stored, resources());
    assert.equal(applied.snapshot.advancedRecipe, true);
    assert.equal(applied.snapshot.request.recipe_revision_id, 'recipe-revision-1');
    assert.deepEqual(applied.snapshot.request.recipe_definition, source.request.recipe_definition);
    assert.equal(applied.snapshot.recipeEditorDraft, undefined);
    stored.content.request.recipe_definition.node_id = 'other-node';
    const mismatched = helper.applyTrainingTemplate(stored, resources());
    assert.equal(mismatched.snapshot.request.recipe_definition, undefined);
    assert.equal(mismatched.snapshot.request.recipe_revision_id, undefined);
    assert.equal(mismatched.snapshot.request.recipe, undefined);
    assert.ok(mismatched.issues.some((message) => message.includes('配方')));
  });

  await test('launch state and approvals never become reusable template content', () => {
    const source = { ...snapshot(), step: 2, previewId: 'old-preview', confirmedConfigSha256: 'a'.repeat(64),
      idempotencyKey: 'old-key', runId: 'old-run', codeSnapshot: { commit_sha: 'old-code' } };
    const content = helper.prepareTrainingTemplate(source);
    const serialized = JSON.stringify(content);
    assert.doesNotMatch(serialized, /old-preview|old-key|old-run|old-code|confirmedConfig|previewId|runId/);
    const applied = helper.applyTrainingTemplate(record(content), resources()).snapshot;
    assert.equal(applied.step, 0);
    assert.equal(applied.previewId, undefined);
    assert.equal(applied.confirmedConfigSha256, undefined);
  });

  await test('unknown schema, wire approval fields and credentials are rejected', () => {
    for (const change of [
      (content) => content.schema_version = 0,
      (content) => content.schema_version = 2,
      (content) => content.preview_id = 'old-preview',
      (content) => content.request.confirmed_config_sha256 = 'a'.repeat(64),
      (content) => content.request.password = 'do-not-save',
      (content) => content.request.dataset_bindings[0].processor.parameters.authorization = 'do-not-save',
    ]) {
      const content = helper.prepareTrainingTemplate(snapshot());change(content);
      assert.throws(() => helper.applyTrainingTemplate(record(content), resources()));
    }
    for (const change of [
      (source) => source.password = 'do-not-save',
      (source) => source.request.access_token = 'do-not-save',
      (source) => source.request.dataset_bindings[0].processor.parameters.nested = { 'Ａuthorization': 'do-not-save' },
    ]) {
      const source = snapshot();change(source);
      assert.throws(() => helper.prepareTrainingTemplate(source));
    }
  });

  await test('missing or removed machine clears scoped environment and model instead of selecting alternatives', () => {
    for (const nodes of [[], [{ ...resources().nodes[0], status: 'REMOVED' }], [{ ...resources().nodes[0], status: 'DISCONNECTED' }]]) {
      const available = resources();available.nodes = nodes;
      available.nodes.push({ id: 'other-node', name: 'Other', status: 'ONLINE' });
      const applied = helper.applyTrainingTemplate(record(helper.prepareTrainingTemplate(snapshot())), available);
      assert.equal(applied.snapshot.request.node_id, '');
      assert.equal(applied.snapshot.request.profile_id, '');
      assert.deepEqual(applied.snapshot.request.inputs, {});
      assert.ok(applied.issues.length);
    }
  });

  await test('temporarily offline machine is reported while retaining the exact model and profile', () => {
    const available = resources();available.nodes[0].status = 'OFFLINE';
    const applied = helper.applyTrainingTemplate(record(helper.prepareTrainingTemplate(snapshot())), available);
    assert.equal(applied.snapshot.request.node_id, 'node-a');
    assert.equal(applied.snapshot.request.profile_id, 'native');
    assert.equal(applied.snapshot.request.inputs.model, 'model-a');
    assert.ok(applied.issues.some((message) => message.includes('未在线')));
  });

  await test('missing, cross-machine or incompatible environment is cleared without changing the valid model', () => {
    for (const update of [[], [{ ...resources().profiles[0], node_id: 'other' }],
      [{ ...resources().profiles[0], adapter: 'smoke' }], [{ ...resources().profiles[0], execution_mode: 'docker' }]]) {
      const available = resources();available.profiles = update;
      const applied = helper.applyTrainingTemplate(record(helper.prepareTrainingTemplate(snapshot())), available);
      assert.equal(applied.snapshot.request.profile_id, '');
      assert.equal(applied.snapshot.request.inputs.model, 'model-a');
      assert.ok(applied.issues.some((message) => message.includes('运行环境')));
    }
  });

  await test('unavailable, wrong-kind and cross-machine model IDs are cleared with no replacement', () => {
    for (const update of [[], [{ ...resources().assets[0], node_id: 'other' }],
      [{ ...resources().assets[0], kind: 'DATASET' }], [{ ...resources().assets[0], state: 'MISSING' }]]) {
      const available = resources();available.assets = [...update, { id: 'replacement', kind: 'MODEL', node_id: 'node-a', state: 'READY' }];
      const applied = helper.applyTrainingTemplate(record(helper.prepareTrainingTemplate(snapshot())), available);
      assert.deepEqual(applied.snapshot.request.inputs, {});
      assert.ok(applied.issues.some((message) => message.includes('基础模型')));
    }
  });

  await test('manual, automatic and unlabelled legacy horizon provenance survive template use', () => {
    for (const marker of ['auto_config.action_horizon', 'auto_config.action_horizon:automatic', undefined]) {
      const source = snapshot();source.editedParameters = ['batch_size', ...(marker ? [marker] : [])];
      const stored = record(helper.prepareTrainingTemplate(source));
      const applied = helper.applyTrainingTemplate(stored, resources()).snapshot;
      assert.deepEqual(applied.editedParameters, [...source.editedParameters, helper.TRAINING_TEMPLATE_PROCESSING_MARKER]);
      assert.equal(applied.request.auto_config.action_horizon, 16);
      assert.equal(modes.restoreActionHorizonMode(applied), marker?.endsWith(':automatic') ? 'automatic' : 'manual');
      const nextAssets = [{ id: 'new-data', kind: 'DATASET', metadata: { dataset_profile: { field_annotations: [{ key: 'actions', role: 'action', shape: [32, 10] }] } } }];
      applied.request.dataset_bindings[0].dataset_asset_id = 'new-data';
      const inferred = configuration.inferDatasetActionHorizon([applied.request.dataset_bindings[0]], nextAssets);
      assert.equal(inferred.recommendedHorizon, 32);
      assert.equal(modes.defaultActionHorizon(16, modes.restoreActionHorizonMode(applied), inferred.recommendedHorizon, inferred.errors), marker?.endsWith(':automatic') ? 32 : 16);
    }
  });

  await test('draft special-value codec is reused without losing unfinished nested processor values', () => {
    const source = snapshot();source.request.dataset_bindings[0].processor.parameters.composition.gripper_range = [NaN, NaN];
    source.request.parameters.learning_rate = NaN;
    const stored = record(helper.prepareTrainingTemplate(source));
    assert.doesNotMatch(JSON.stringify(stored), /NaN|Infinity/);
    const applied = helper.applyTrainingTemplate(stored, resources()).snapshot;
    assert.ok(Number.isNaN(applied.request.parameters.learning_rate));
    assert.ok(applied.request.dataset_bindings[0].processor.parameters.composition.gripper_range.every(Number.isNaN));
  });

  await test('reuse-processing marker survives drafts and repeated saving without duplication or mode loss', () => {
    const source = snapshot();
    const stored = record(helper.prepareTrainingTemplate(source));
    const first = helper.applyTrainingTemplate(stored, resources()).snapshot;
    const restoredDraft = codec.decodeTrainingDraft(codec.encodeTrainingDraft(first));
    const repeated = helper.applyTrainingTemplate(record(helper.prepareTrainingTemplate(restoredDraft)), resources()).snapshot;
    assert.equal(helper.TRAINING_TEMPLATE_PROCESSING_MARKER, 'training_template.reuse_processing');
    assert.equal(repeated.editedParameters.filter((key) => key === helper.TRAINING_TEMPLATE_PROCESSING_MARKER).length, 1);
    assert.ok(repeated.editedParameters.includes('auto_config.action_horizon'));
    assert.ok(repeated.editedParameters.includes('batch_size'));
    assert.deepEqual(repeated.request.dataset_bindings[0].processor, source.request.dataset_bindings[0].processor);
  });
} finally {
  await server.close();
}
