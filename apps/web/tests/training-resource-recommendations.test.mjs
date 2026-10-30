import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const helper = await server.ssrLoadModule('/src/trainingResourceRecommendations.ts');
  const controls = await server.ssrLoadModule('/src/trainingParameters.ts');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const templates = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const horizon = await server.ssrLoadModule('/src/actionHorizonDefaults.ts');
  const metrics = (available_cpu_count, cpu_count = 96) => ({ available_cpu_count, cpu_count });
  const bothAutomatic = () => {
    const markers = new Set();
    helper.TRAINING_RESOURCE_FIELDS.forEach((key) => helper.setResourceRecommendationMode(markers, key, true));
    return markers;
  };

  await test('GPU count sets FSDP and starts two loading workers per GPU with a sixteen-worker ceiling', () => {
    for (const [gpus, workers] of [[1, 2], [2, 4], [4, 8], [8, 16], [16, 16]]) {
      const result = helper.recommendTrainingResources(gpus, metrics(96));
      assert.equal(result.fsdp_devices, gpus);
      assert.equal(result.num_workers, workers);
    }
  });

  await test('visible CPU budget limits workers using CPU count minus one including a valid zero-worker result', () => {
    for (const [cpus, workers] of [[1, 0], [2, 1], [4, 3], [6, 5], [8, 7], [16, 8]])
      assert.equal(helper.recommendTrainingResources(4, metrics(cpus)).num_workers, workers);
  });

  await test('unknown visible CPU count uses the GPU heuristic without treating total host CPUs as a quota', () => {
    for (const value of [undefined, null, '4', 0, -1, 1.5, NaN, Infinity]) {
      const result = helper.recommendTrainingResources(4, metrics(value, 1));
      assert.equal(result.availableCpuCount, undefined);
      assert.equal(result.cpuCount, 1);
      assert.equal(result.num_workers, 8);
      assert.match(helper.trainingResourceRecommendationHint('num_workers', result), /可见 CPU 数未知/);
      assert.match(helper.trainingResourceRecommendationHint('num_workers', result), /主机总计 1 个/);
    }
    assert.equal(helper.recommendTrainingResources(8).num_workers, 16);
  });

  await test('unfinished or invalid GPU counts never manufacture FSDP or worker recommendations', () => {
    for (const value of [0, -1, 1.5, NaN, Infinity]) {
      const result = helper.recommendTrainingResources(value, metrics(96));
      assert.equal(result.fsdp_devices, undefined);
      assert.equal(result.num_workers, undefined);
    }
  });

  await test('automatic resource values follow both GPU and newly reported node CPU counts without touching training settings', () => {
    const markers = bothAutomatic();
    const source = { steps: 80000, batch_size: 128, fsdp_devices: 1, num_workers: 2 };
    const four = helper.applyResourceRecommendations(source, markers, helper.recommendTrainingResources(4, metrics(96)));
    assert.deepEqual(four, { ...source, fsdp_devices: 4, num_workers: 8 });
    const constrained = helper.applyResourceRecommendations(four, markers, helper.recommendTrainingResources(4, metrics(3)));
    assert.deepEqual(constrained, { ...source, fsdp_devices: 4, num_workers: 2 });
    const eight = helper.applyResourceRecommendations(constrained, markers, helper.recommendTrainingResources(8, metrics(96)));
    assert.deepEqual(eight, { ...source, fsdp_devices: 8, num_workers: 16 });
    assert.deepEqual(source, { steps: 80000, batch_size: 128, fsdp_devices: 1, num_workers: 2 });
  });

  await test('manual zero and legacy explicit values remain manual even when they equal an old recommendation', () => {
    for (const source of [{ fsdp_devices: 4, num_workers: 0 }, { fsdp_devices: 4, num_workers: 8 }, {}]) {
      const result = helper.applyResourceRecommendations(source, new Set(['num_workers']), helper.recommendTrainingResources(8, metrics(96)));
      assert.equal(result, source);
    }
  });

  await test('manual entry or clearing exits automatic mode until the user explicitly chooses the current recommendation', () => {
    const markers = bothAutomatic();
    let values = helper.applyResourceRecommendations({}, markers, helper.recommendTrainingResources(4, metrics(96)));
    helper.setResourceRecommendationMode(markers, 'num_workers', false);
    values = controls.editTrainingParameter(values, 'num_workers', '0', { optional: true });
    values = helper.applyResourceRecommendations(values, markers, helper.recommendTrainingResources(8, metrics(96)));
    assert.deepEqual(values, { fsdp_devices: 8, num_workers: 0 });
    helper.setResourceRecommendationMode(markers, 'fsdp_devices', false);
    values = controls.editTrainingParameter(values, 'fsdp_devices', '', { optional: true });
    assert.equal(helper.applyResourceRecommendations(values, markers, helper.recommendTrainingResources(4, metrics(96))), values);
    assert.equal(Object.hasOwn(values, 'fsdp_devices'), false);
    helper.setResourceRecommendationMode(markers, 'fsdp_devices', true);
    values = helper.applyResourceRecommendations(values, markers, helper.recommendTrainingResources(4, metrics(96)));
    assert.deepEqual(values, { fsdp_devices: 4, num_workers: 0 });
    assert.equal(helper.automaticResourceRecommendation(markers, 'fsdp_devices'), true);
    assert.equal(helper.automaticResourceRecommendation(markers, 'num_workers'), false);
  });

  await test('resource recommendation flags are independent and automatic no-ops retain object identity', () => {
    const markers = new Set();
    helper.setResourceRecommendationMode(markers, 'num_workers', true);
    const source = { fsdp_devices: 2, num_workers: 8 };
    assert.equal(helper.applyResourceRecommendations(source, markers, helper.recommendTrainingResources(4, metrics(96))), source);
    assert.deepEqual(helper.applyResourceRecommendations(source, markers, helper.recommendTrainingResources(4, metrics(1))), { fsdp_devices: 2, num_workers: 0 });
  });

  await test('late native recipe defaults cannot overwrite automatic workers and field edits protect a saved manual zero', () => {
    const markers = bothAutomatic();
    const source = helper.applyResourceRecommendations({}, markers, helper.recommendTrainingResources(4, metrics(96)));
    assert.deepEqual(controls.inheritTrainingRecipeParameters(source, { num_workers: 2, fsdp_devices: 8 }, markers), source);
    helper.setResourceRecommendationMode(markers, 'num_workers', false);
    assert.equal(controls.inheritTrainingRecipeParameters({ num_workers: 0 }, { num_workers: 8 }, markers).num_workers, 0);
    assert.equal(controls.inheritTrainingRecipeParameters({ num_workers: 0 }, { num_workers: 8 }, new Set()).num_workers, 0);
    assert.equal(Object.hasOwn(controls.inheritTrainingRecipeParameters({}, { num_workers: 8 }, new Set()), 'num_workers'), false);
  });

  const snapshot = (parameters, editedParameters, advancedRecipe = false) => ({
    request: { name: 'resource-test', template_id: 'openpi-sft', node_id: 'node-a', profile_id: 'profile-a',
      inputs: { model: 'model-a' }, parameters: { steps: 80000, batch_size: 128, learning_rate: .00005, ...parameters },
      gpu_count: 4, auto_config: advancedRecipe ? undefined : { model_type: 'pi05', finetune_mode: 'full', action_horizon: 50, action_dim: 32 } },
    step: 1, advancedRecipe, editedParameters: horizon.actionHorizonDraftParameters(editedParameters, 'automatic'),
    editedStageParameters: [], entrypointEdited: false,
  });
  const resources = { nodes: [{ id: 'node-a', status: 'ONLINE' }],
    profiles: [{ id: 'profile-a', node_id: 'node-a', adapter: 'openpi-sft', execution_mode: 'native' }],
    assets: [{ id: 'model-a', node_id: 'node-a', kind: 'MODEL', state: 'READY' }] };
  const record = (content) => ({ id: 'template', name: 'Resource options', description: '', revision: 1,
    created_at: '2026-10-30T00:00:00Z', updated_at: '2026-10-30T00:00:00Z', content });

  await test('automatic mode survives account draft and personal template round trips and resumes only the automatic fields', () => {
    const markers = bothAutomatic();
    helper.setResourceRecommendationMode(markers, 'num_workers', false);
    const source = snapshot({ fsdp_devices: 4, num_workers: 0 }, markers);
    const restored = codec.decodeTrainingDraft(codec.encodeTrainingDraft(source));
    const applied = templates.applyTrainingTemplate(record(templates.prepareTrainingTemplate(source)), resources).snapshot;
    for (const state of [restored, applied]) {
      const result = helper.applyResourceRecommendations(state.request.parameters, new Set(state.editedParameters), helper.recommendTrainingResources(8, metrics(96)));
      assert.equal(result.fsdp_devices, 8);
      assert.equal(result.num_workers, 0);
      assert.equal(result.steps, 80000);
      assert.ok(Object.keys(result).every((key) => !key.startsWith(helper.RESOURCE_RECOMMENDATION_AUTO_PREFIX)));
    }
  });

  await test('legacy and advanced snapshots without mode flags keep explicit and absent values unchanged', () => {
    for (const advanced of [false, true]) for (const values of [{ fsdp_devices: 2, num_workers: 0 }, {}]) {
      const source = snapshot(values, [], advanced);
      const restored = codec.decodeTrainingDraft(codec.encodeTrainingDraft(source));
      const result = helper.applyResourceRecommendations(restored.request.parameters, new Set(restored.editedParameters), helper.recommendTrainingResources(8, metrics(96)));
      assert.equal(result, restored.request.parameters);
      assert.deepEqual(result, source.request.parameters);
    }
  });

  await test('resource tracking can survive a node or stage scope change without inferring automatic mode from field values', () => {
    const source = bothAutomatic();
    source.add('auto_config.action_horizon:automatic');
    source.add('save_interval');
    const tracked = helper.resourceRecommendationTracking(source);
    assert.equal(tracked.has('save_interval'), false);
    assert.equal(tracked.has('auto_config.action_horizon:automatic'), false);
    assert.equal(helper.automaticResourceRecommendation(tracked, 'fsdp_devices'), true);
    assert.deepEqual(helper.retainedTrainingResources({ num_workers: 0, fsdp_devices: 2, steps: 80000 }), { fsdp_devices: 2, num_workers: 0 });
    assert.equal(helper.automaticResourceRecommendation(helper.resourceRecommendationTracking(['num_workers']), 'num_workers'), false);
  });

  await test('CPU copy describes visible counts and a starting heuristic rather than free or reserved CPU resources', () => {
    const hint = helper.trainingResourceRecommendationHint('num_workers', helper.recommendTrainingResources(4, metrics(12)));
    assert.match(hint, /Agent 可见 CPU 12 个/);
    assert.match(hint, /CPU 数减 1/);
    assert.match(hint, /起始建议/);
    assert.doesNotMatch(hint, /空闲|预留/);
  });
} finally {
  await server.close();
}
