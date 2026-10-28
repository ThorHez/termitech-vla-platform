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
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const snapshot = () => ({
    request: {
      name: '', node_id: 'node-a', profile_id: 'profile-a', template_id: 'openpi-sft',
      inputs: { model: 'model-a' }, parameters: { batch_size: 72 }, gpu_count: 4,
      auto_config: { model_type: 'pi05', finetune_mode: 'full', action_horizon: 16, action_dim: 32 },
      dataset_bindings: [{ dataset_asset_id: 'dataset-a', norm_asset_id: 'norm-a', weight: 1,
        processor: { name: 'ComposableDataConfig', parameters: { composition: {
          action_key: 'actions', action_layout: [{ arm: 'left', kind: 'position', width: 0 }],
          coordinate_frame: '', gripper_range: [NaN, NaN], cameras: [],
        } } } }],
    },
    step: 0, advancedRecipe: false, editedParameters: ['batch_size'],
    editedStageParameters: [], entrypointEdited: false,
  });
  const clone = structuredClone;

  await test('unfinished composition and blank form inputs round trip without normalization or mutation', () => {
    const draft = snapshot(), before = clone(draft);
    const encoded = codec.encodeTrainingDraft(draft);
    assert.equal(encoded.schema_version, 1);
    assert.doesNotMatch(JSON.stringify(encoded), /NaN|Infinity/);
    assert.deepEqual(codec.decodeTrainingDraft(JSON.parse(JSON.stringify(encoded))), before);
    assert.deepEqual(draft, before);
    assert.equal(draft.request.dataset_bindings[0].processor.parameters.composition.action_layout[0].width, 0);
  });

  await test('disabled optional gripper bounds survive server draft JSON round trip', () => {
    const draft = snapshot();
    draft.request.dataset_bindings[0].processor.parameters.composition.gripper_range = null;
    assert.deepEqual(codec.parseTrainingDraft(codec.stringifyTrainingDraft(draft)), draft);
    assert.equal(codec.parseTrainingDraft(codec.stringifyTrainingDraft(draft)).request.dataset_bindings[0].processor.parameters.composition.gripper_range, null);
  });

  await test('nonfinite numbers, undefined object and array values, and negative zero survive JSON storage', () => {
    const draft = snapshot();
    draft.request.parameters = { empty: NaN, high: Infinity, low: -Infinity, negative_zero: -0 };
    draft.request.recipe = undefined;
    const parameters = draft.request.dataset_bindings[0].processor.parameters;
    parameters.nested = { missing: undefined, list: [undefined, null, NaN, Infinity, -Infinity, -0] };
    const restored = codec.parseTrainingDraft(codec.stringifyTrainingDraft(draft));
    assert.deepEqual(restored, draft);
    assert.ok(Object.hasOwn(restored.request, 'recipe'));
    assert.equal(restored.request.recipe, undefined);
    assert.equal(restored.request.parameters.high, Infinity);
    assert.ok(Object.is(restored.request.parameters.negative_zero, -0));
    assert.equal(restored.request.dataset_bindings[0].processor.parameters.nested.list[1], null);
  });

  await test('ordinary user parameter objects cannot collide with the special-value codec', () => {
    const draft = snapshot();
    draft.request.dataset_bindings[0].processor.parameters.literal = {
      value: 'nan', path: ['request', 'parameters', 'batch_size'], $number: 'Infinity', marker: 'undefined',
    };
    assert.deepEqual(codec.parseTrainingDraft(codec.stringifyTrainingDraft(draft)), draft);
  });

  await test('preflight step always restores to parameter review and never carries launch approvals', () => {
    const draft = { ...snapshot(), step: 2, confirmedConfigSha256: 'a'.repeat(64), previewId: 'old-preview' };
    const encoded = codec.encodeTrainingDraft(draft);
    assert.equal(encoded.step, 1);
    assert.equal(codec.decodeTrainingDraft(encoded).step, 1);
    assert.equal(encoded.confirmedConfigSha256, undefined);
    assert.equal(encoded.previewId, undefined);
    const old = codec.encodeTrainingDraft(snapshot()); old.step = 2;
    assert.equal(codec.decodeTrainingDraft(old).step, 1);
    old.step = 3;
    assert.equal(codec.decodeTrainingDraft(old), undefined);
  });

  await test('recipe UI context and unfinished editor strings round trip without reusable writeback confirmation', () => {
    const draft = { ...snapshot(), step: 1, advancedRecipe: true, entrypointEdited: true,
      recipeDatasets: [{ recipe_index: 0, weight: 1, processing: 'Factory', multiple: true }],
      recipeDefaults: { batch_size: 72, missing: NaN }, stageRecipeDefaults: { acp_dropout: 0.5 },
      stageRecipeInfo: { values: { acp_dropout: 0.5 }, mixedDropout: false,
        valueHead: [{ path: 'model.freeze_backbone', value: 'true' }] },
      recipeEditorDraft: { scope: 'node/profile/recipe', sourceSha256: 'a'.repeat(64),
        fields: { 'optimizer.lr': '[unfinished', 'model.name': '' }, saveMode: 'new',
        saveName: 'My recipe', configName: 'new_config', description: 'work in progress' },
    };
    draft.request.recipe_revision_id = 'saved-revision';
    assert.deepEqual(codec.parseTrainingDraft(codec.stringifyTrainingDraft(draft)), draft);
    const bad = clone(draft); bad.recipeEditorDraft.pendingWriteback = { id: 'job' };
    assert.throws(() => codec.encodeTrainingDraft(bad));
  });

  await test('request authentication, execution, preview and unknown fields are rejected rather than cached', () => {
    for (const key of ['csrf_token', 'execution_token', 'session_token', 'preview_id', 'code_snapshot', 'idempotency_key', 'arbitrary']) {
      const draft = snapshot(); draft.request[key] = 'must-not-persist';
      assert.throws(() => codec.encodeTrainingDraft(draft), key);
      const encoded = codec.encodeTrainingDraft(snapshot()); encoded.request[key] = 'must-not-restore';
      assert.equal(codec.decodeTrainingDraft(encoded), undefined, key);
    }
    const nested = snapshot(); nested.request.dataset_bindings[0].processor.parameters.execution_token = 'secret';
    assert.throws(() => codec.encodeTrainingDraft(nested));
  });

  await test('source recipe provenance remains editable data rather than a preflight snapshot', () => {
    const draft = snapshot();
    draft.request.recipe_definition = { schema_version: 1, source: { path: 'config.py', sha256: 'a'.repeat(64) }, overrides: {} };
    assert.deepEqual(codec.parseTrainingDraft(codec.stringifyTrainingDraft(draft)), draft);
  });

  await test('nested normalized credential and approval aliases are rejected before local outbox persistence', () => {
    const forbidden = [
      'preview_id', 'Preview-ID', 'confirmedConfigSha256', 'confirmed_config_sha256',
      'codeSnapshot', 'CODE_SNAPSHOT', 'idempotencyKey', 'idempotency_key',
      'executionToken', 'EXECUTION-TOKEN', 'csrfToken', 'csrf_token', 'PASSWORD',
      'adminToken', 'enrollment_token', 'access_token', 'accessToken', 'ACCESS.TOKEN',
      'refresh_token', 'refreshToken', 'Refresh Token', 'sessionToken', 'session_token',
      'Authorization', 'Cookie', 'passwordHash', 'paßword', 'paſſword',
    ];
    for (const key of forbidden) {
      const draft = snapshot();
      draft.request.dataset_bindings[0].processor.parameters.nested = { [key]: 'must-not-persist' };
      assert.throws(() => codec.encodeTrainingDraft(draft), key);
      const encoded = codec.encodeTrainingDraft(snapshot());
      encoded.request.dataset_bindings[0].processor.parameters.nested = { [key]: 'must-not-restore' };
      assert.equal(codec.decodeTrainingDraft(encoded), undefined, key);
    }
    const permitted = snapshot();
    permitted.request.dataset_bindings[0].processor.parameters.tokenizer_name = 'model tokenizer';
    assert.deepEqual(codec.parseTrainingDraft(codec.stringifyTrainingDraft(permitted)), permitted);
  });

  await test('wrong schemas, corrupt JSON and malformed required request fields fail closed', () => {
    for (const value of ['', '{', 'null', '[]', '3']) assert.equal(codec.parseTrainingDraft(value), undefined);
    for (const update of [{ schema_version: 2 }, { runtime_state: {} }, { step: '1' }, { advanced_recipe: 'false' }]) {
      const encoded = { ...codec.encodeTrainingDraft(snapshot()), ...update };
      assert.equal(codec.decodeTrainingDraft(encoded), undefined);
    }
    for (const update of [{ name: null }, { inputs: [] }, { parameters: { batch_size: '72' } }, { gpu_count: '4' },
      { dataset_bindings: [{ dataset_asset_id: '', norm_asset_id: '', weight: '1' }] }]) {
      const draft = snapshot(); Object.assign(draft.request, update);
      assert.throws(() => codec.encodeTrainingDraft(draft));
    }
  });

  await test('prototype keys and path metadata cannot mutate prototypes or overwrite real values', () => {
    const base = codec.encodeTrainingDraft(snapshot());
    const pollution = JSON.parse(JSON.stringify(base));
    pollution.request.dataset_bindings[0].processor.parameters = JSON.parse('{"__proto__":{"polluted":true}}');
    assert.equal(codec.decodeTrainingDraft(pollution), undefined);
    assert.equal({}.polluted, undefined);
    for (const path of [['__proto__', 'polluted'], ['request', 'constructor', 'prototype'], ['request', 'name'],
      ['special_values', 0], ['request', 'missing'], ['request', 'dataset_bindings', -1]]) {
      const changed = clone(base); changed.special_values = [{ path, value: 'nan' }];
      assert.equal(codec.decodeTrainingDraft(changed), undefined, JSON.stringify(path));
    }
    const repeated = clone(base); repeated.special_values.push(clone(repeated.special_values[0]));
    assert.equal(codec.decodeTrainingDraft(repeated), undefined);
  });

  await test('size limit uses UTF-8 bytes and oversized nesting or node counts are rejected', () => {
    const oversized = snapshot(); oversized.request.dataset_bindings[0].processor.parameters.large = '汉'.repeat(180000);
    assert.throws(() => codec.encodeTrainingDraft(oversized));
    assert.equal(codec.parseTrainingDraft(' '.repeat(codec.MAX_TRAINING_DRAFT_BYTES + 1)), undefined);
    const deep = snapshot(); let at = deep.request.dataset_bindings[0].processor.parameters;
    for (let index = 0; index <= codec.MAX_TRAINING_DRAFT_DEPTH; index++) at = at.child = {};
    assert.throws(() => codec.encodeTrainingDraft(deep));
    const many = snapshot(); many.request.dataset_bindings[0].processor.parameters.large = Array(codec.MAX_TRAINING_DRAFT_NODES).fill(0);
    assert.throws(() => codec.encodeTrainingDraft(many));
  });

  await test('cyclic, executable and non-JSON objects cannot be persisted as draft data', () => {
    for (const value of [() => 1, 1n, new Date(), new Map()]) {
      const draft = snapshot(); draft.request.dataset_bindings[0].processor.parameters.bad = value;
      assert.throws(() => codec.encodeTrainingDraft(draft));
    }
    const cyclic = snapshot(); cyclic.request.dataset_bindings[0].processor.parameters.self = cyclic;
    assert.throws(() => codec.encodeTrainingDraft(cyclic));
  });
} finally {
  await server.close();
}
