import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom',
});
try {
  const helper = await server.ssrLoadModule('/src/runRetry.ts');
  const fixture = () => ({
    id: 'failed-run', state: 'FAILED', name: 'Dual arm SFT', node_id: 'node-a', profile_id: 'native-a',
    template_id: 'openpi-sft', created_at: '2026-10-30T00:00:00Z', gpu_count: 4,
    inputs: { model: 'initial-model', dataset_0: 'data-a', norm_0: 'norm-a', dataset_1: 'data-b', norm_1: 'norm-b' },
    parameters: { steps: 80000, batch_size: 128, learning_rate: .00005, num_workers: 0,
      fsdp_devices: 2, save_interval: 2000, keep_period: 20000, log_interval: 100,
      warmup_steps: 1000, decay_steps: 80000, decay_lr: .0000025, ema_decay: .99, clip_gradient_norm: 1 },
    auto_config: { model_type: 'pi05', finetune_mode: 'full', action_horizon: 50, action_dim: 32 },
    dataset_bindings: [
      { dataset_asset_id: 'data-a', norm_asset_id: 'norm-a', weight: 1, processor: { name: 'ComposableDataConfig', parameters: { composition: {
        action_key: 'action', state_fields: [{ key: 'observation.state', width: 14, kind: 'joint', arm: 'shared', normalize: true }],
        action_layout: [{ arm: 'left', width: 7, kind: 'joint' }, { arm: 'right', width: 7, kind: 'joint' }],
        cameras: [{ key: 'observation.images.head', slot: 'base_0_rgb', layout: 'hwc', kind: 'rgb', time_index: null }],
      } } } },
      { dataset_asset_id: 'data-b', norm_asset_id: 'norm-b', weight: 3, processor: { name: 'ComposableDataConfig', parameters: { composition: { action_key: 'actions', action_layout: [{ width: 14, kind: 'joint' }] } } } },
    ],
    outputs: [{ id: 'last-checkpoint', kind: 'MODEL', path: '/outputs/last-checkpoint/params' }],
  });

  await test('failed training produces a fresh first-step configuration with exact datasets, model, processing and explicit parameters', () => {
    const source = fixture(), before = structuredClone(source);
    const snapshot = helper.retryTrainingSnapshot(source);
    assert.equal(snapshot.step, 0);
    assert.equal(snapshot.advancedRecipe, false);
    assert.deepEqual(snapshot.request.inputs, { model: 'initial-model' });
    assert.deepEqual(snapshot.request.dataset_bindings, source.dataset_bindings);
    assert.deepEqual(snapshot.request.parameters, source.parameters);
    assert.deepEqual(snapshot.request.auto_config, source.auto_config);
    assert.equal(snapshot.request.gpu_count, 4);
    assert.equal(snapshot.request.name, source.name);
    assert.ok(Object.keys(source.parameters).every((key) => snapshot.editedParameters.includes(key)));
    assert.ok(snapshot.editedParameters.includes('training_template.reuse_processing'));
    assert.ok(snapshot.editedParameters.every((key) => !key.startsWith('resource_recommendation.auto.')));
    assert.deepEqual(source, before);
  });

  await test('repeated retraining keeps the base title and cleans only trailing retry markers', () => {
    for (const [name, expected] of [
      ['叠衣服双臂训练 训练（重新训练）（重新训练）（重新训练）', '叠衣服双臂训练 训练'],
      ['叠衣服训练 （重新训练） (重新训练) ', '叠衣服训练'],
      ['任务（重新训练）对比（实验组）', '任务（重新训练）对比（实验组）'],
      ['普通训练', '普通训练'],
    ]) {
      const source = { ...fixture(), name };
      const original = structuredClone(source);
      for (let retry = 0; retry < 3; retry++) {
        const snapshot = helper.retryTrainingSnapshot(source);
        assert.equal(snapshot.request.name, expected);
        source.name = snapshot.request.name;
      }
      assert.deepEqual({ ...source, name: original.name }, original);
    }
    assert.equal(helper.retryTrainingName('（重新训练）'), '训练');
    assert.equal(helper.retryTrainingName('😀'.repeat(81)), '😀'.repeat(80));
  });

  await test('saved advanced recipe uses its server revision identifier and explicit entrypoint without sending a browser definition', () => {
    const source = fixture();
    source.auto_config = null;
    source.recipe = 'original_config';
    source.entrypoint = 'scripts/train_multi_dataset.py';
    source.recipe_definition = { id: 'revision-a', base_recipe: 'original_config', overrides: { num_workers: 8 }, source: { code_dir: '/code/old' } };
    const snapshot = helper.retryTrainingSnapshot(source);
    assert.equal(snapshot.advancedRecipe, true);
    assert.equal(snapshot.request.recipe_revision_id, 'revision-a');
    assert.equal(snapshot.request.recipe, 'original_config');
    assert.equal(snapshot.request.entrypoint, 'scripts/train_multi_dataset.py');
    assert.equal(snapshot.entrypointEdited, true);
    assert.equal(Object.hasOwn(snapshot.request, 'recipe_definition'), false);
  });

  await test('legacy single-dataset input is normalized to one binding without losing the original registered asset IDs', () => {
    const source = fixture();
    source.inputs = { model: 'initial-model', dataset: 'old-data', norm: 'old-norm' };
    source.dataset_bindings = null;
    source.auto_config = null;
    source.recipe = 'single_dataset_config';
    const request = helper.retryTrainingSnapshot(source).request;
    assert.deepEqual(request.inputs, { model: 'initial-model' });
    assert.deepEqual(request.dataset_bindings, [{ dataset_asset_id: 'old-data', norm_asset_id: 'old-norm', recipe_index: 0, weight: 1 }]);
    assert.equal(request.recipe, 'single_dataset_config');
  });

  await test('ReCap stage settings remain explicit on a new request', () => {
    const source = { ...fixture(), template_id: 'recap', training_stage: 'offline_rl', stage_config: { acp_dropout: .4 } };
    const snapshot = helper.retryTrainingSnapshot(source);
    assert.equal(snapshot.request.training_stage, 'offline_rl');
    assert.deepEqual(snapshot.request.stage_config, { acp_dropout: .4 });
    assert.deepEqual(snapshot.editedStageParameters, ['acp_dropout']);
  });

  await test('ambiguous or conflicting old dataset mappings require manual configuration rather than inventing a recipe index', () => {
    const expanded = fixture(); delete expanded.dataset_bindings;
    assert.throws(() => helper.retryTrainingSnapshot(expanded), /无法确定处理器/);
    const empty = fixture(); empty.dataset_bindings = [];
    assert.throws(() => helper.retryTrainingSnapshot(empty), /绑定无效/);
    const conflicted = fixture(); conflicted.dataset_bindings[0].recipe_index = 0;
    assert.throws(() => helper.retryTrainingSnapshot(conflicted), /冲突/);
    const invalidIndex = fixture(); invalidIndex.dataset_bindings[0] = { dataset_asset_id: 'data', norm_asset_id: 'norm', weight: 1, recipe_index: 64 };
    assert.throws(() => helper.retryTrainingSnapshot(invalidIndex), /处理配置不完整/);
  });

  await test('terminal state labels support new training while active states stay unavailable', () => {
    for (const state of ['FAILED', 'CANCELLED', 'SUCCEEDED']) assert.equal(helper.retryableRun({ state }), true);
    for (const state of ['RUNNING', 'UNKNOWN', 'QUEUED', 'CANCELLING']) assert.equal(helper.retryableRun({ state }), false);
  });

  await test('draft comparison detects changed training settings but ignores wizard UI position and field tracking', () => {
    const first = helper.retryTrainingSnapshot(fixture()), second = structuredClone(first);
    second.step = 1; second.editedParameters = ['resource_recommendation.auto.num_workers'];
    assert.equal(helper.sameTrainingConfiguration(first, second), true);
    second.request.parameters.num_workers = 8;
    assert.equal(helper.sameTrainingConfiguration(first, second), false);
    second.request.parameters.num_workers = NaN;
    assert.equal(helper.sameTrainingConfiguration(first, second), false);
  });

  await test('retry entry uses the normal wizard and preserves revision-checked draft replacement before fresh preflight', async () => {
    const [detail, app, wizard] = await Promise.all(['RunDetail.tsx', 'App.tsx', 'RunWizard.tsx'].map((path) => readFile(new URL(`../src/${path}`, import.meta.url), 'utf8')));
    assert.match(detail, /can\('training\.write'\)\s*&&\s*onRetry\s*&&\s*retryableRun\(run\)/);
    assert.match(app, /sourceRun=\{wizard\.sourceRun\}/);
    assert.match(wizard, /await draft\.replace\(snapshot\)/);
    assert.match(wizard, /consumedSourceRun\.current === source\.id/);
    assert.match(wizard, /sameTrainingConfiguration\(previous, snapshot\)/);
    assert.match(wizard, /载入并替换草稿/);
    assert.match(wizard, /confirmed_config_sha256: hasLocalTraining \? confirmedConfigSha256 : undefined/);
    assert.doesNotMatch(detail, /post[\s\S]{0,40}\/retry/);
  });
} finally { await server.close(); }
