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
  const { retryTrainingSnapshot: retry, retryableRun, sameTrainingConfiguration } =
    await server.ssrLoadModule('/src/runRetry.ts');
  const auto = { model_type: 'pi05', finetune_mode: 'full', action_horizon: 50, action_dim: 32 };
  const run = () => ({
    id: 'old-run', name: '失败训练', node_id: 'node-a', profile_id: 'profile-a',
    template_id: 'openpi-sft', state: 'FAILED', created_at: '2026-10-30T06:00:00Z',
    inputs: { model: 'model-a', dataset_0: 'data-a', norm_0: 'norm-a' },
    parameters: { steps: 80000, batch_size: 128, learning_rate: 3e-5, num_workers: 0, ema_decay: 0 },
    gpu_count: 4, auto_config: { ...auto }, entrypoint: null,
    dataset_bindings: [{ dataset_asset_id: 'data-a', norm_asset_id: 'norm-a', weight: 1,
      processor: { name: 'ComposableDataConfig', parameters: { composition: {
        schema_version: 1, action_key: 'action', action_layout: [{ arm: 'left', kind: 'joint', width: 6 }],
        cameras: [{ key: 'observation.images.head', slot: 'base_0_rgb' }],
      } } } }],
  });

  await test('retry copies only input settings and strips every old launch artifact', () => {
    const source = run();
    Object.assign(source, {
      confirmed_config_sha256: 'a'.repeat(64), preflight_id: 'old-preview', idempotency_key: 'old-key',
      output_dir: '/old-attempt', gpu_uuids: ['GPU-old'], execution_token: 'never-copy-token',
      outputs: [{ id: 'later-checkpoint', kind: 'MODEL', path: '/old-checkpoint' }],
      attempts: [{ id: 'old-attempt', output_dir: '/old-attempt', gpu_uuids: ['GPU-old'],
        code_snapshot: { generated_config: { source: 'never execute this' } } }],
      code_expectation: { commit_sha: 'b'.repeat(40), expected_plan_digest: 'c'.repeat(64),
        automatic_preflight: { plan: { output_dir: '/old-attempt', generated_config: { source: 'old' } } } },
    });
    Object.assign(source.parameters, { resume: 1, overwrite: 1, output_dir: 5, confirmed_config_sha256: 9 });
    const copied = retry(source);
    const serialized = JSON.stringify(copied);
    for (const key of ['confirmed_config_sha256', 'preflight_id', 'idempotency_key', 'execution_token',
      'output_dir', 'gpu_uuids', 'code_expectation', 'automatic_preflight', 'generated_config', 'resume', 'overwrite']) {
      assert.equal(Object.hasOwn(copied.request, key), false, key);
      assert.equal(Object.hasOwn(copied.request.parameters, key), false, key);
    }
    assert.equal(serialized.includes('never-copy-token'), false);
    assert.equal(serialized.includes('/old-attempt'), false);
    assert.equal(copied.request.inputs.model, 'model-a');
    assert.equal(copied.request.parameters.num_workers, 0);
    assert.equal(copied.request.parameters.ema_decay, 0);
  });

  await test('processing layout and weights are deep cloned without modifying source history', () => {
    const source = run(), before = structuredClone(source);
    const copy = retry(source);
    copy.request.dataset_bindings[0].weight = 2;
    copy.request.dataset_bindings[0].processor.parameters.composition.action_layout[0].width = 7;
    copy.request.dataset_bindings[0].processor.parameters.composition.cameras[0].key = 'new-camera';
    copy.request.auto_config.action_horizon = 16;
    copy.request.parameters.steps = 2;
    assert.deepEqual(source, before);
    assert.notEqual(copy.request.dataset_bindings, source.dataset_bindings);
    assert.ok(copy.editedParameters.includes('num_workers'));
  });

  await test('input asset objects use only IDs and never infer IDs from paths', () => {
    const source = run();
    source.inputs.model = { id: 'model-a', path: '/untrusted/path', name: 'display-only' };
    assert.equal(retry(source).request.inputs.model, 'model-a');
    for (const invalid of [{ path: '/machine/model' }, { id: '' }, { id: '/machine/model' },
      { id: 'https://host/model' }, { id: 'file:checkpoint' }, null]) {
      assert.throws(() => retry({ ...run(), inputs: { ...run().inputs, model: invalid } }), /资产 ID|初始模型/);
    }
  });

  await test('flat explicit null settings are authoritative over legacy snapshots', () => {
    const source = run();
    source.auto_config = null;
    source.dataset_bindings = null;
    source.recipe = 'current_recipe';
    source.recipe_definition = null;
    source.inputs = { model: 'model-a', dataset: 'data-a', norm: 'norm-a' };
    source.code_expectation = { auto_config: { ...auto }, dataset_bindings: run().dataset_bindings,
      recipe_definition: { id: 'stale-revision', base_recipe: 'stale_recipe' },
      requested_entrypoint: 'scripts/custom.py' };
    const copy = retry(source);
    assert.equal(copy.advancedRecipe, true);
    assert.equal(Object.hasOwn(copy.request, 'auto_config'), false);
    assert.deepEqual(copy.request.dataset_bindings, [{ dataset_asset_id: 'data-a', norm_asset_id: 'norm-a', weight: 1, recipe_index: 0 }]);
    assert.equal(Object.hasOwn(copy.request, 'recipe_revision_id'), false);
    assert.equal(Object.hasOwn(copy.request, 'entrypoint'), false);
    assert.equal(copy.request.recipe, 'current_recipe');
    assert.deepEqual(copy.request.inputs, { model: 'model-a' });
  });

  await test('saved recipes reuse only server revision identity, never a raw browser definition', () => {
    const source = run();
    source.auto_config = null;
    source.recipe_definition = { id: 'revision-a', base_recipe: 'recipe_a',
      source: { path: '/arbitrary-code' }, overrides: { batch_size: 1 } };
    const copy = retry(source);
    assert.equal(copy.request.recipe_revision_id, 'revision-a');
    assert.equal(copy.request.recipe, 'recipe_a');
    assert.equal(Object.hasOwn(copy.request, 'recipe_definition'), false);
    assert.equal(JSON.stringify(copy).includes('/arbitrary-code'), false);
  });

  await test('legacy snapshots may restore settings but cannot change the original asset IDs', () => {
    const source = run();
    delete source.auto_config;
    delete source.dataset_bindings;
    source.code_expectation = { auto_config: { ...auto }, dataset_bindings: run().dataset_bindings,
      automatic_preflight: { plan: { inputs: { model: '/old-path', dataset_0: '/old-data' } } } };
    const copy = retry(source);
    assert.deepEqual(copy.request.auto_config, auto);
    assert.deepEqual(copy.request.dataset_bindings, run().dataset_bindings);
    assert.deepEqual(copy.request.inputs, { model: 'model-a' });
  });

  await test('legacy indexed mixtures without processor bindings are rejected rather than guessed', () => {
    const source = run();
    source.auto_config = null;
    source.dataset_bindings = null;
    source.recipe = 'legacy_mixed_recipe';
    source.inputs = { ...source.inputs, dataset_1: 'data-b', norm_1: 'norm-b' };
    assert.throws(() => retry(source), /多数据集|数据绑定|手动/);
  });

  await test('automatic retry requires the original independent processors instead of inventing them', () => {
    const source = run();
    source.dataset_bindings = null;
    source.inputs = { model: 'model-a', dataset: 'data-a', norm: 'norm-a' };
    assert.throws(() => retry(source), /自动配置|处理器|独立/);
    source.dataset_bindings = [{ dataset_asset_id: 'data-a', norm_asset_id: 'norm-a', recipe_index: 0, weight: 1 }];
    assert.throws(() => retry(source), /自动配置|处理器|独立/);
  });

  await test('an automatic run with a custom requested entrypoint must never silently switch scripts', () => {
    const source = run();
    source.entrypoint = 'scripts/custom_train.py';
    assert.throws(() => retry(source), /自动配置.*自定义训练入口|高级设置/);
    delete source.entrypoint;
    source.code_expectation = { requested_entrypoint: 'scripts/custom_train.py' };
    assert.throws(() => retry(source), /自动配置.*自定义训练入口|高级设置/);
  });

  await test('active and unknown processes cannot be cloned through the terminal-run helper', () => {
    for (const state of ['QUEUED', 'STARTING', 'RUNNING', 'CANCELLING', 'UNKNOWN']) {
      assert.equal(retryableRun({ state }), false);
      assert.throws(() => retry({ ...run(), state }), /已结束/);
    }
    for (const state of ['FAILED', 'CANCELLED', 'SUCCEEDED']) assert.equal(retryableRun({ state }), true);
  });

  await test('retry equality compares request content rather than UI state and ignores undefined keys', () => {
    const a = retry(run()), b = structuredClone(a);
    b.step = 1;
    b.editedParameters = [];
    b.recipeEditorDraft = { scope: 'ui-only', fields: { width: 'unfinished' } };
    b.request.entrypoint = undefined;
    assert.equal(sameTrainingConfiguration(a, b), true);
    b.request.parameters.num_workers = 1;
    assert.equal(sameTrainingConfiguration(a, b), false);
  });
} finally {
  await server.close();
}
