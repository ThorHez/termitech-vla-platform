import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const parameters = await server.ssrLoadModule('/src/trainingParameters.ts');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const templates = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const optional = { type: 'integer', optional: true, default: 2, min: 0 };
  const allSpecs = {
    steps: { type: 'integer', min: 1, max: 1e6, default: 10000 },
    batch_size: { type: 'integer', min: 1, max: 4096, default: 8 },
    learning_rate: { type: 'number', min: 1e-8, max: .1, default: .00005 },
    warmup_steps: { ...optional, min: 0, default: 1000, max: 1e7 },
    decay_steps: { ...optional, min: 1, default: 30000, max: 1e8 },
    decay_lr: { ...optional, type: 'number', min: 0, max: .1, default: .0000025 },
    clip_gradient_norm: { ...optional, type: 'number', default: 1, min: 0, max: 1000, exclusive_min: true },
    ema_decay: { ...optional, type: 'number', default: null, min: 0, max: 1, exclusive_max: true },
    fsdp_devices: { ...optional, default: null, min: 1, max: 4096 },
    log_interval: { ...optional, min: 1, default: 100, max: 1e6 },
    save_interval: { ...optional, min: 1, default: 1000, max: 1e6 },
    keep_period: { ...optional, min: 1, default: 5000, max: 1e7 },
    num_workers: { ...optional, min: 0, default: 2, max: 128 },
  };
  const snapshot = (extra = {}) => ({
    request: {
      name: 'new-data-training', node_id: 'node-a', profile_id: 'profile-a',
      template_id: 'openpi-sft', inputs: { model: 'model-a' }, gpu_count: 4,
      parameters: { steps: 80000, batch_size: 128, learning_rate: .00005, ...extra },
      auto_config: { model_type: 'pi05', finetune_mode: 'full', action_horizon: 50, action_dim: 32 },
      dataset_bindings: [{ dataset_asset_id: 'data-new', norm_asset_id: 'norm-new', weight: 1,
        processor: { name: 'ComposableDataConfig', parameters: { composition: { action_key: 'action' } } } }],
    },
    step: 1, advancedRecipe: false,
    editedParameters: ['save_interval', 'keep_period', 'num_workers'],
    editedStageParameters: [], entrypointEdited: false,
  });
  const resources = {
    nodes: [{ id: 'node-a', name: 'Machine A', hostname: 'machine-a', status: 'ONLINE' }],
    profiles: [{ id: 'profile-a', node_id: 'node-a', name: 'Native', adapter: 'openpi-sft', execution_mode: 'native' }],
    assets: [{ id: 'model-a', node_id: 'node-a', name: 'Initial model', kind: 'MODEL', state: 'READY', path: '/models/a', version: '1' }],
  };
  const record = (content) => ({ id: 'template-a', name: 'Training options', description: '', revision: 1,
    created_at: '2026-10-29T00:00:00Z', updated_at: '2026-10-29T00:00:00Z', content });

  await test('clearing an optional override restores inheritance without changing other settings', () => {
    const source = { steps: 80000, num_workers: 8, save_interval: 2000 };
    const changed = parameters.editTrainingParameter(source, 'num_workers', '', optional);
    assert.deepEqual(changed, { steps: 80000, save_interval: 2000 });
    assert.equal(Object.hasOwn(changed, 'num_workers'), false);
    assert.deepEqual(source, { steps: 80000, num_workers: 8, save_interval: 2000 });
    assert.equal(parameters.editTrainingParameter(source, 'num_workers', '0', optional).num_workers, 0);
    assert.ok(Number.isNaN(parameters.editTrainingParameter(source, 'steps', '', { type: 'integer' }).steps));
  });

  await test('controls explain their concrete initial values and real recipe references', () => {
    const spec = { ...optional, hint: '0 表示在主进程读取数据。' };
    assert.match(parameters.trainingParameterHint('num_workers', spec, false, {}, false), /已填入初始值/);
    const inherited = parameters.trainingParameterHint('num_workers', spec, true, { num_workers: 8 }, false);
    assert.match(inherited, /配方参考值：8/);
    assert.match(inherited, /本次训练使用输入框中的数值/);
    assert.match(inherited, /0 表示/);
    assert.doesNotMatch(inherited, /当前为本次覆盖/);
    assert.match(parameters.trainingParameterHint('num_workers', spec, true, {}, false), /已填入初始值/);
    assert.equal(parameters.optionalTrainingParameter({ required: false }), true);
  });

  await test('draft round trip retains runtime overrides including zero workers and the selected datasets', () => {
    const source = snapshot({ save_interval: 2000, keep_period: 20000, num_workers: 0 });
    const restored = codec.decodeTrainingDraft(codec.encodeTrainingDraft(source));
    assert.deepEqual(restored.request.parameters, source.request.parameters);
    assert.deepEqual(restored.request.dataset_bindings, source.request.dataset_bindings);
    assert.deepEqual(restored.editedParameters, source.editedParameters);
    assert.equal(restored.advancedRecipe, false);
  });

  await test('saved training templates preserve runtime options while substituting dataset assets', () => {
    const source = snapshot({ save_interval: 2000, keep_period: 20000, num_workers: 8 });
    const applied = templates.applyTrainingTemplate(record(templates.prepareTrainingTemplate(source)), resources);
    assert.deepEqual(applied.issues, []);
    assert.deepEqual(applied.snapshot.request.parameters, source.request.parameters);
    assert.equal(applied.snapshot.request.dataset_bindings[0].dataset_asset_id, '');
    assert.equal(applied.snapshot.request.dataset_bindings[0].norm_asset_id, '');
    assert.deepEqual(applied.snapshot.request.dataset_bindings[0].processor, source.request.dataset_bindings[0].processor);
    assert.equal(applied.snapshot.advancedRecipe, false);
  });

  await test('legacy draft and template restoration keep missing overrides absent for native inheritance', () => {
    for (const advancedRecipe of [false, true]) {
      const source = { ...snapshot(), advancedRecipe };
      const restored = codec.decodeTrainingDraft(codec.encodeTrainingDraft(source));
      const applied = templates.applyTrainingTemplate(record(templates.prepareTrainingTemplate(source)), resources);
      for (const key of ['save_interval', 'keep_period', 'num_workers']) {
        assert.equal(Object.hasOwn(restored.request.parameters, key), false);
        assert.equal(Object.hasOwn(applied.snapshot.request.parameters, key), false);
      }
      assert.deepEqual(restored.request.parameters, source.request.parameters);
      assert.deepEqual(applied.snapshot.request.parameters, source.request.parameters);
    }
  });

  await test('new automatic defaults leave FSDP and EMA inherited without manufacturing one or enabling LoRA EMA', () => {
    const defaults = parameters.trainingParameterDefaults(allSpecs);
    assert.equal(defaults.warmup_steps, 1000);
    assert.equal(defaults.decay_steps, 30000);
    assert.equal(defaults.clip_gradient_norm, 1);
    assert.equal(defaults.log_interval, 100);
    assert.equal(Object.hasOwn(defaults, 'fsdp_devices'), false);
    assert.equal(Object.hasOwn(defaults, 'ema_decay'), false);
    assert.deepEqual(parameters.trainingParameterDefaults({ zero: { ...optional, default: 0 } }), { zero: 0 });
  });

  await test('all runtime controls belong to exactly one of four ordered sections', () => {
    const groups = parameters.trainingParameterGroups(allSpecs);
    assert.deepEqual(groups.map((group) => group.title), ['基本训练与资源', '学习率计划', '优化器与稳定性', '检查点与日志']);
    const keys = groups.flatMap((group) => group.entries.map(([key]) => key));
    assert.equal(new Set(keys).size, keys.length);
    assert.deepEqual([...keys].sort(), Object.keys(allSpecs).sort());
    assert.ok(groups.find((group) => group.id === 'resources').entries.some(([key]) => key === 'fsdp_devices'));
    assert.ok(groups.find((group) => group.id === 'schedule').entries.some(([key]) => key === 'decay_lr'));
  });

  await test('FSDP checks explicit divisors of allocated GPU count and clearing returns to GPU-count inheritance', () => {
    const defaults = parameters.trainingParameterDefaults(allSpecs);
    for (const fsdp_devices of [1, 2, 4])
      assert.deepEqual(parameters.trainingParameterErrors({ ...defaults, fsdp_devices }, allSpecs, 4), []);
    for (const fsdp_devices of [0, -1, 1.5, 3, 8])
      assert.ok(parameters.trainingParameterErrors({ ...defaults, fsdp_devices }, allSpecs, 4).some((issue) => issue.key === 'fsdp_devices'));
    const cleared = parameters.editTrainingParameter({ ...defaults, fsdp_devices: 8 }, 'fsdp_devices', '', allSpecs.fsdp_devices);
    assert.deepEqual(parameters.trainingParameterErrors(cleared, allSpecs, 4, { fsdp_devices: 8 }), []);
    const hint = parameters.trainingParameterHint('fsdp_devices', allSpecs.fsdp_devices, true, { fsdp_devices: 8 }, false);
    assert.match(hint, /跟随申请的 GPU 数量/);
    assert.doesNotMatch(hint, /沿用配方值/);
  });

  await test('native recipe defaults retain schedule values without silently importing an old GPU topology', () => {
    const request = { steps: 80000, learning_rate: .00005, num_workers: 0 };
    const defaults = { fsdp_devices: 8, num_workers: 8, warmup_steps: 500, decay_steps: 80000 };
    const inherited = parameters.inheritTrainingRecipeParameters(request, defaults, new Set(['num_workers']));
    assert.deepEqual(inherited, { ...request, warmup_steps: 500, decay_steps: 80000 });
    assert.deepEqual(request, { steps: 80000, learning_rate: .00005, num_workers: 0 });
    assert.equal(parameters.inheritTrainingRecipeParameters({ fsdp_devices: 2 }, defaults, new Set()).fsdp_devices, 2);
  });

  await test('schedule relations and finite numeric bounds block invalid values before preflight', () => {
    const defaults = parameters.trainingParameterDefaults(allSpecs);
    assert.ok(parameters.trainingParameterErrors({ ...defaults, warmup_steps: 30000 }, allSpecs, 4).some((issue) => issue.key === 'decay_steps'));
    assert.ok(parameters.trainingParameterErrors({ ...defaults, decay_lr: .0001 }, allSpecs, 4).some((issue) => issue.key === 'decay_lr'));
    for (const bad of [1, 1.1, -.1, NaN, Infinity])
      assert.ok(parameters.trainingParameterErrors({ ...defaults, ema_decay: bad }, allSpecs, 4).some((issue) => issue.key === 'ema_decay'));
    for (const bad of [0, -1, NaN, Infinity])
      assert.ok(parameters.trainingParameterErrors({ ...defaults, clip_gradient_norm: bad }, allSpecs, 4).some((issue) => issue.key === 'clip_gradient_norm'));
    assert.deepEqual(parameters.trainingParameterErrors({ ...defaults, decay_lr: 0, ema_decay: 0, clip_gradient_norm: 1e-10, num_workers: 0 }, allSpecs, 4), []);
    assert.deepEqual(parameters.trainingParameterErrors({ ...defaults, ema_decay: .99999999999 }, allSpecs, 4), []);
  });

  await test('old missing optional fields are not compared with guessed environment defaults', () => {
    const old = snapshot().request.parameters;
    assert.deepEqual(parameters.trainingParameterErrors(old, allSpecs, 4), []);
    assert.deepEqual(parameters.trainingParameterErrors({ ...old, decay_steps: 100 }, allSpecs, 4), []);
    assert.ok(parameters.trainingParameterErrors({ ...old, decay_steps: 100 }, allSpecs, 4, { warmup_steps: 1000 }).some((issue) => issue.key === 'decay_steps'));
  });

  await test('drafts and personal templates preserve every explicit optimizer schedule and resource control', () => {
    const source = snapshot({ warmup_steps: 2000, decay_steps: 80000, decay_lr: 0,
      clip_gradient_norm: .5, ema_decay: .999, fsdp_devices: 2, log_interval: 50,
      save_interval: 2000, keep_period: 20000, num_workers: 8 });
    const restored = codec.decodeTrainingDraft(codec.encodeTrainingDraft(source));
    const applied = templates.applyTrainingTemplate(record(templates.prepareTrainingTemplate(source)), resources);
    assert.deepEqual(restored.request.parameters, source.request.parameters);
    assert.deepEqual(applied.snapshot.request.parameters, source.request.parameters);
    assert.equal(applied.snapshot.advancedRecipe, false);
  });

  await test('wizard renders grouped fields with inline errors and blocks invalid preflight submission', async () => {
    const source = await readFile(new URL('../src/RunWizard.tsx', import.meta.url), 'utf8');
    assert.match(source, /parameterGroups\.map/);
    assert.match(source, /aria-invalid=\{\s*parameterIssues\.some/);
    assert.match(source, /\|\|\s*!!parameterBlocker/);
    assert.match(source, /if \(parameterIssues\.length\)/);
    assert.match(source, /trainingParameterDefaults\(specs\(template\)\)/);
  });
} finally {
  await server.close();
}
