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
  const defaults = await server.ssrLoadModule('/src/actionHorizonDefaults.ts');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const snapshot = () => ({
    request: {
      name: 'source horizon', node_id: 'node', profile_id: 'native', template_id: 'openpi-sft',
      inputs: {}, parameters: {}, gpu_count: 1,
      auto_config: { model_type: 'pi05', finetune_mode: 'full', action_horizon: 16, action_dim: 32 },
    },
    step: 1, advancedRecipe: false, editedParameters: ['batch_size'],
    editedStageParameters: [], entrypointEdited: false,
  });

  await test('new wizards infer defaults while unmarked legacy drafts preserve saved choices', () => {
    assert.equal(defaults.restoreActionHorizonMode(), 'automatic');
    for (const horizon of [16, 50, 64]) {
      const draft = snapshot();draft.request.auto_config.action_horizon = horizon;
      assert.equal(defaults.restoreActionHorizonMode(draft), 'manual');
      assert.equal(defaults.defaultActionHorizon(horizon, 'manual', 32, []), horizon);
    }
  });
  await test('automatic horizons follow new packed datasets and return to the single-step training default', () => {
    assert.equal(defaults.defaultActionHorizon(50, 'automatic', 16, []), 16);
    assert.equal(defaults.defaultActionHorizon(16, 'automatic', 32, []), 32);
    assert.equal(defaults.defaultActionHorizon(16, 'automatic', 1, []), 1);
    assert.equal(defaults.defaultActionHorizon(16, 'automatic', undefined, []), 50);
    assert.equal(defaults.defaultActionHorizon(16, 'automatic', undefined, ['conflicting sources']), 16);
  });
  await test('manual values, including a cleared field, are never silently replaced by metadata refresh', () => {
    assert.equal(defaults.defaultActionHorizon(64, 'manual', 16, []), 64);
    assert.ok(Number.isNaN(defaults.defaultActionHorizon(NaN, 'manual', 16, [])));
  });
  for (const mode of ['automatic', 'manual']) {
    await test(`${mode} provenance survives the existing account draft JSON codec`, () => {
      const draft = snapshot();
      draft.editedParameters = defaults.actionHorizonDraftParameters(draft.editedParameters, mode);
      const restored = codec.parseTrainingDraft(codec.stringifyTrainingDraft(draft));
      assert.equal(defaults.restoreActionHorizonMode(restored), mode);
      assert.equal(restored.request.auto_config.action_horizon, 16);
      assert.ok(restored.editedParameters.includes('batch_size'));
      assert.equal(restored.request.confirmed_config_sha256, undefined);
      const switched = defaults.actionHorizonDraftParameters(restored.editedParameters, mode === 'manual' ? 'automatic' : 'manual');
      assert.equal(switched.filter(key => key.startsWith('auto_config.action_horizon')).length, 1);
    });
  }
} finally {
  await server.close();
}
