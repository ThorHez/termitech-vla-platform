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
  const {
    initializeVisibleTrainingParameters: initialize,
    recipeNumericParameterDefaults,
    recipeParameterCompatibility,
    unsupportedTrainingParameters,
  } = await server.ssrLoadModule('/src/trainingParameterInitialization.ts');
  const { automaticResourceRecommendation, resourceRecommendationMarker } =
    await server.ssrLoadModule('/src/trainingResourceRecommendations.ts');
  const specs = {
    steps: { type: 'integer', default: 10000 },
    batch_size: { type: 'integer', default: 8 },
    learning_rate: { type: 'number', default: 0.00005 },
    warmup_steps: { type: 'integer', optional: true, default: 1000 },
    decay_steps: { type: 'integer', optional: true, default: 30000 },
    decay_lr: { type: 'number', optional: true, default: 0.0000025 },
    clip_gradient_norm: { type: 'number', optional: true, default: 1 },
    ema_decay: { type: 'number', optional: true, default: null },
    fsdp_devices: { type: 'integer', optional: true, default: null },
    log_interval: { type: 'integer', optional: true, default: 100 },
    save_interval: { type: 'integer', optional: true, default: 1000 },
    keep_period: { type: 'integer', optional: true, default: 5000 },
    num_workers: { type: 'integer', optional: true, default: 2 },
  };
  const resources = { gpuCount: 4, fsdp_devices: 4, num_workers: 8 };
  const options = (extra = {}) => ({
    resources,
    defaultedKeys: new Set(),
    edited: new Set(),
    phase: 'initial',
    ...extra,
  });
  const literal = (path, value) => ({
    path,
    kind: 'literal',
    value,
    expression: String(value),
    origin: 'recipe',
  });
  const expression = (path, value) => ({
    path,
    kind: 'expression',
    expression: value,
    origin: 'recipe',
  });
  const resolve = (previous, extra = {}) =>
    initialize(previous.parameters, specs, {
      ...options(),
      ...previous,
      phase: 'resolved',
      ...extra,
    });

  await test('cold start fills all thirteen visible controls without mutating inputs', () => {
    const parameters = Object.freeze({});
    const initial = options();
    const result = initialize(parameters, specs, initial);
    assert.deepEqual(result.parameters, {
      steps: 10000,
      batch_size: 8,
      learning_rate: 0.00005,
      warmup_steps: 1000,
      decay_steps: 30000,
      decay_lr: 0.0000025,
      clip_gradient_norm: 1,
      ema_decay: 0.99,
      fsdp_devices: 4,
      log_interval: 100,
      save_interval: 1000,
      keep_period: 5000,
      num_workers: 8,
    });
    assert.deepEqual([...result.defaultedKeys].sort(), Object.keys(specs).sort());
    assert.deepEqual(parameters, {});
    assert.equal(initial.defaultedKeys.size, 0);
    assert.equal(initial.edited.size, 0);
  });

  await test('legacy advanced drafts gain missing peak learning rate and optional fields', () => {
    const parameters = { steps: 80000, batch_size: 128, learning_rate: NaN };
    const result = initialize(
      parameters,
      { ...specs, learning_rate: { default: null } },
      options(),
    );
    assert.equal(result.parameters.steps, 80000);
    assert.equal(result.parameters.batch_size, 128);
    assert.equal(result.parameters.learning_rate, 5e-5);
    assert.equal(result.parameters.ema_decay, 0.99);
    assert.equal(Object.keys(result.parameters).length, 13);
    assert.ok(Number.isNaN(parameters.learning_rate));
    assert.equal(result.defaultedKeys.has('steps'), false);
  });

  await test('every pre-existing finite number including zero remains explicit', () => {
    const parameters = Object.fromEntries(Object.keys(specs).map((key) => [key, 0]));
    const result = initialize(
      parameters,
      specs,
      options({
        recipeDefaults: Object.fromEntries(Object.keys(specs).map((key) => [key, 9])),
        compatibility: { emaDisabled: true, cosineDecay: false, gradientClip: false },
        lora: true,
      }),
    );
    assert.equal(result.parameters, parameters);
    assert.equal(result.defaultedKeys.size, 0);
    assert.deepEqual(result.parameters, parameters);
  });

  await test('late actual recipe values replace only values defaulted in this session', () => {
    const initial = initialize({ steps: 80000, save_interval: 2222 }, specs, options());
    const result = resolve(initial, {
      recipeDefaults: {
        steps: 1,
        save_interval: 1,
        learning_rate: 0.0001,
        warmup_steps: 500,
        decay_steps: 60000,
        decay_lr: 0.000003,
        ema_decay: 0,
        num_workers: 32,
        fsdp_devices: 8,
      },
    });
    assert.equal(result.parameters.steps, 80000);
    assert.equal(result.parameters.save_interval, 2222);
    assert.equal(result.parameters.learning_rate, 0.0001);
    assert.equal(result.parameters.warmup_steps, 500);
    assert.equal(result.parameters.decay_steps, 60000);
    assert.equal(result.parameters.decay_lr, 0.000003);
    assert.equal(result.parameters.ema_decay, 0);
    assert.equal(result.parameters.fsdp_devices, 4);
    assert.equal(result.parameters.num_workers, 8);
    assert.deepEqual(result.defaultedKeys, initial.defaultedKeys);
  });

  await test('resolved phase fills untouched missing controls but respects manual clearing', () => {
    const initial = initialize({}, specs, options());
    const parameters = { ...initial.parameters, learning_rate: NaN };
    delete parameters.save_interval;
    delete parameters.keep_period;
    delete parameters.num_workers;
    const defaultedKeys = new Set(initial.defaultedKeys);
    for (const key of ['learning_rate', 'save_interval', 'keep_period', 'num_workers'])
      defaultedKeys.delete(key);
    const edited = new Set([...initial.edited, 'learning_rate', 'save_interval', 'num_workers']);
    const result = resolve(
      { parameters, defaultedKeys, edited },
      {
        recipeDefaults: {
          learning_rate: 0.0001,
          save_interval: 500,
          keep_period: 20000,
          num_workers: 16,
        },
      },
    );
    assert.ok(Number.isNaN(result.parameters.learning_rate));
    assert.equal(Object.hasOwn(result.parameters, 'save_interval'), false);
    assert.equal(Object.hasOwn(result.parameters, 'num_workers'), false);
    assert.equal(result.parameters.keep_period, 20000);
    assert.equal(result.defaultedKeys.has('keep_period'), true);
  });

  await test('recipe numeric defaults map only finite literal fields including zero', () => {
    assert.deepEqual(
      recipeNumericParameterDefaults({
        fields: [
          literal('num_train_steps', 80000),
          literal('batch_size', 128),
          literal('lr_schedule.peak_lr', 0.00005),
          literal('lr_schedule.warmup_steps', 0),
          literal('lr_schedule.decay_steps', 80000),
          literal('lr_schedule.decay_lr', 0),
          literal('optimizer.clip_gradient_norm', 0.5),
          literal('ema_decay', 0),
          literal('num_workers', 0),
          literal('fsdp_devices', 8),
          literal('save_interval', 2000),
          literal('keep_period', 20000),
          literal('log_interval', 50),
          literal('unknown', 2),
          literal('steps', 5),
          literal('ema_decay', null),
          literal('batch_size', Infinity),
          expression('num_train_steps', '100 * 800'),
        ],
      }),
      {
        steps: 80000,
        batch_size: 128,
        learning_rate: 0.00005,
        warmup_steps: 0,
        decay_steps: 80000,
        decay_lr: 0,
        clip_gradient_norm: 0.5,
        ema_decay: 0,
        num_workers: 0,
        fsdp_devices: 8,
        save_interval: 2000,
        keep_period: 20000,
        log_interval: 50,
      },
    );
  });

  await test('EMA None and LoRA leave EMA blank while explicit zero remains enabled', () => {
    for (const fields of [[literal('ema_decay', null)], [expression('ema_decay', ' None ')]]) {
      const compatibility = recipeParameterCompatibility({ fields });
      assert.equal(compatibility.emaDisabled, true);
      const cold = initialize({}, specs, options({ compatibility }));
      assert.equal(Object.hasOwn(cold.parameters, 'ema_decay'), false);
      const initial = initialize({}, specs, options());
      const late = resolve(initial, { compatibility });
      assert.equal(Object.hasOwn(late.parameters, 'ema_decay'), false);
      assert.equal(late.defaultedKeys.has('ema_decay'), true);
      assert.equal(
        resolve(
          { parameters: { ema_decay: 0 }, defaultedKeys: new Set(), edited: new Set() },
          { compatibility },
        ).parameters.ema_decay,
        0,
      );
    }
    assert.equal(
      recipeParameterCompatibility({ fields: [literal('ema_decay', 0)] }).emaDisabled,
      false,
    );
    assert.equal(
      recipeParameterCompatibility({ fields: [literal('ema_decay', 0.99)] }).emaDisabled,
      false,
    );
    assert.equal(
      recipeParameterCompatibility({ fields: [expression('ema_decay', 'get_ema()')] }).emaDisabled,
      undefined,
    );
    assert.equal(
      Object.hasOwn(initialize({}, specs, options({ lora: true })).parameters, 'ema_decay'),
      false,
    );
    const initial = initialize({}, specs, options());
    assert.equal(Object.hasOwn(resolve(initial, { lora: true }).parameters, 'ema_decay'), false);
    assert.equal(
      initialize({ ema_decay: 0 }, specs, options({ lora: true })).parameters.ema_decay,
      0,
    );
  });

  await test('schedule and optimizer compatibility reads constructors and child-field evidence', () => {
    assert.deepEqual(
      recipeParameterCompatibility({
        fields: [
          expression('lr_schedule', '_optimizer.CosineDecaySchedule(warmup_steps=1000)'),
          expression('optimizer', 'optimizer.AdamW()'),
        ],
      }),
      { cosineDecay: true, gradientClip: true },
    );
    const compatibility = recipeParameterCompatibility({
      fields: [
        expression('lr_schedule', 'RsqrtDecaySchedule()'),
        expression('optimizer', '_optimizer.SGD(momentum=0.9)'),
        literal('lr_schedule.decay_steps', 10000),
        literal('optimizer.clip_gradient_norm', 1),
      ],
    });
    assert.deepEqual(compatibility, { cosineDecay: false, gradientClip: false });
    assert.deepEqual(unsupportedTrainingParameters(compatibility), [
      'decay_steps',
      'decay_lr',
      'clip_gradient_norm',
    ]);
    assert.deepEqual(
      recipeParameterCompatibility({
        fields: [literal('lr_schedule.decay_lr', 0), literal('optimizer.clip_gradient_norm', 1)],
      }),
      { cosineDecay: true, gradientClip: true },
    );
    assert.deepEqual(
      recipeParameterCompatibility({
        fields: [
          expression('lr_schedule', 'CustomSchedule()'),
          expression('optimizer', 'CustomOptimizer()'),
        ],
      }),
      {},
    );
    assert.deepEqual(unsupportedTrainingParameters({}), []);
  });

  await test('initial LoRA and disabled EMA discard stale nonfinite payload values', () => {
    for (const mode of [{ lora: true }, { compatibility: { emaDisabled: true } }]) {
      for (const value of [NaN, Infinity, -Infinity]) {
        const parameters = { ema_decay: value };
        const result = initialize(
          parameters,
          specs,
          options({ ...mode, edited: new Set(['ema_decay']) }),
        );
        assert.equal(Object.hasOwn(result.parameters, 'ema_decay'), false);
        assert.equal(Object.hasOwn(parameters, 'ema_decay'), true);
        assert.equal(result.defaultedKeys.has('ema_decay'), false);
      }
      assert.equal(initialize({ ema_decay: 0 }, specs, options(mode)).parameters.ema_decay, 0);
    }
  });

  await test('initial unsupported schedule and optimizer discard nonfinite values while retaining finite overrides', () => {
    const compatibility = { cosineDecay: false, gradientClip: false };
    const parameters = { decay_steps: NaN, decay_lr: Infinity, clip_gradient_norm: -Infinity };
    const result = initialize(
      parameters,
      specs,
      options({ compatibility, edited: new Set(Object.keys(parameters)) }),
    );
    for (const key of Object.keys(parameters)) {
      assert.equal(Object.hasOwn(result.parameters, key), false);
      assert.equal(result.defaultedKeys.has(key), false);
    }
    const finiteValues = { decay_steps: 10000, decay_lr: 0, clip_gradient_norm: 0.5 };
    const retained = initialize(finiteValues, specs, options({ compatibility }));
    for (const key of Object.keys(finiteValues))
      assert.equal(retained.parameters[key], finiteValues[key]);
  });

  await test('Rsqrt and SGD suppress automatic controls but retain explicit incompatible overrides', () => {
    const compatibility = { cosineDecay: false, gradientClip: false };
    const keys = unsupportedTrainingParameters(compatibility);
    const cold = initialize(
      {},
      specs,
      options({
        compatibility,
        recipeDefaults: { decay_steps: 9000, decay_lr: 0.000001, clip_gradient_norm: 1 },
      }),
    );
    for (const key of keys) assert.equal(Object.hasOwn(cold.parameters, key), false);
    const initial = initialize({}, specs, options());
    const late = resolve(initial, { compatibility });
    for (const key of keys) {
      assert.equal(Object.hasOwn(late.parameters, key), false);
      assert.equal(late.defaultedKeys.has(key), true);
    }
    const parameters = { decay_steps: 10000, decay_lr: 0, clip_gradient_norm: 0.5 };
    const explicit = initialize(parameters, specs, options({ compatibility }));
    for (const key of keys) assert.equal(explicit.parameters[key], parameters[key]);
  });

  await test('resource defaults follow current recommendations and carry automatic edit markers', () => {
    const result = initialize(
      {},
      specs,
      options({
        recipeDefaults: { fsdp_devices: 8, num_workers: 64 },
      }),
    );
    for (const key of ['fsdp_devices', 'num_workers']) {
      assert.equal(result.parameters[key], resources[key]);
      assert.equal(result.edited.has(key), true);
      assert.equal(automaticResourceRecommendation(result.edited, key), true);
    }
    const existing = initialize({ fsdp_devices: 2, num_workers: 0 }, specs, options());
    assert.equal(existing.parameters.fsdp_devices, 2);
    assert.equal(existing.parameters.num_workers, 0);
    assert.equal(existing.edited.has(resourceRecommendationMarker('num_workers')), false);
    const unknown = initialize({}, specs, options({ resources: {} }));
    assert.equal(Object.hasOwn(unknown.parameters, 'fsdp_devices'), false);
    assert.equal(Object.hasOwn(unknown.parameters, 'num_workers'), false);
  });

  await test('only schema candidates are made consistent with visible peak and warmup', () => {
    const parameters = { learning_rate: 0.000001, warmup_steps: 50000 };
    const schema = initialize(parameters, specs, options());
    assert.equal(schema.parameters.decay_lr, 0.000001);
    assert.equal(schema.parameters.decay_steps, 50001);
    const actual = initialize(
      parameters,
      specs,
      options({
        recipeDefaults: { decay_lr: 0.00001, decay_steps: 10000 },
      }),
    );
    assert.equal(actual.parameters.decay_lr, 0.00001);
    assert.equal(actual.parameters.decay_steps, 10000);
    const manual = initialize(
      { ...parameters, decay_lr: 0.00001, decay_steps: 10000 },
      specs,
      options(),
    );
    assert.equal(manual.parameters.decay_lr, 0.00001);
    assert.equal(manual.parameters.decay_steps, 10000);
    const invalid = initialize({ learning_rate: 0, warmup_steps: -1 }, specs, options());
    assert.equal(invalid.parameters.decay_lr, specs.decay_lr.default);
    assert.equal(invalid.parameters.decay_steps, specs.decay_steps.default);
  });

  await test('React updater replay retains automatic provenance for replacements and removals', () => {
    const initial = initialize({}, specs, options());
    const original = initial.parameters;
    const lateOptions = {
      ...options(),
      ...initial,
      phase: 'resolved',
      recipeDefaults: { learning_rate: 0.0001, warmup_steps: 2000 },
      compatibility: { emaDisabled: true, cosineDecay: false, gradientClip: false },
    };
    const first = initialize(original, specs, lateOptions);
    const replay = initialize(original, specs, {
      ...lateOptions,
      defaultedKeys: first.defaultedKeys,
      edited: first.edited,
    });
    assert.deepEqual(replay.parameters, first.parameters);
    assert.deepEqual(replay.defaultedKeys, first.defaultedKeys);
    assert.equal(replay.defaultedKeys.has('ema_decay'), true);
    assert.equal(replay.defaultedKeys.has('learning_rate'), true);
    assert.equal(original.ema_decay, 0.99);
    assert.equal(original.learning_rate, 0.00005);
    const stable = initialize(first.parameters, specs, {
      ...lateOptions,
      defaultedKeys: first.defaultedKeys,
      edited: first.edited,
    });
    assert.equal(stable.parameters, first.parameters);
    assert.deepEqual(stable.defaultedKeys, first.defaultedKeys);
    assert.deepEqual(stable.edited, first.edited);
  });
} finally {
  await server.close();
}
