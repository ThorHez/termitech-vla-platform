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
  const { copyDatasetBinding, selectDatasetBinding } =
    await server.ssrLoadModule('/src/datasetBindingHelpers.ts');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const composed = () => ({
    dataset_asset_id: 'source-data', norm_asset_id: 'source-norm', weight: 20,
    processor: { name: 'ComposableDataConfig', parameters: { composition: {
      schema_version: 1, robot_model: 'ARX5', arm_mode: 'single',
      action_key: 'actions', action_mode: 'absolute', rotation_representation: 'none',
      state_history: 2, normalization: 'quantile',
      state_fields: [{ key: 'observation.state', width: 7, arm: 'left', kind: 'joint', normalize: false }],
      action_layout: [
        { arm: 'left', kind: 'joint', width: 6, normalize: true },
        { arm: 'left', kind: 'gripper', width: 1, normalize: false },
      ],
      cameras: [{ key: 'observation.images.wrist', slot: 'left_wrist_0_rgb', kind: 'rgb', layout: 'chw', time_index: null }],
      gripper_range: [0, 1], aux_keys: ['task_phase'], prompt_key: 'task',
    } } },
  });
  const selection = { dataset_asset_id: 'new-data', norm_asset_id: 'new-norm' };

  await test('new dataset copies processing and relative weight while clearing both asset identities', () => {
    const first = composed(), before = structuredClone(first);
    const copied = copyDatasetBinding(first, 7);
    assert.deepEqual(copied, { ...before, dataset_asset_id: '', norm_asset_id: '' });
    assert.deepEqual(first, before);
    assert.equal(copied.processor.parameters.composition.state_fields[0].normalize, false);
    assert.equal(copied.processor.parameters.composition.action_layout[1].normalize, false);
    assert.equal(Object.hasOwn(copied, 'recipe_index'), false);
  });

  await test('nested edits to one copied dataset cannot mutate the first dataset or another copy', () => {
    const first = composed(), before = structuredClone(first);
    const left = copyDatasetBinding(first), right = copyDatasetBinding(first);
    const spec = left.processor.parameters.composition;
    spec.state_fields[0].normalize = true;
    spec.action_layout[1].normalize = true;
    spec.action_layout.push({ arm: 'right', kind: 'joint', width: 6, normalize: false });
    spec.cameras[0].layout = 'hwc';
    spec.gripper_range[0] = -1;
    spec.aux_keys.push('value_target');
    assert.deepEqual(first, before);
    assert.deepEqual(right.processor, before.processor);
    assert.notStrictEqual(left.processor, right.processor);
    assert.notStrictEqual(left.processor.parameters.composition.cameras, right.processor.parameters.composition.cameras);
  });

  await test('recipe-based first dataset preserves its selected slot and weight instead of fallback defaults', () => {
    const first = { dataset_asset_id: 'source', norm_asset_id: 'norm', recipe_index: 4, weight: 0.25 };
    assert.deepEqual(copyDatasetBinding(first, 1), {
      dataset_asset_id: '', norm_asset_id: '', recipe_index: 4, weight: 0.25,
    });
    assert.equal(first.dataset_asset_id, 'source');
  });

  await test('an absent first dataset gets explicit blank asset identities and the default recipe slot', () => {
    assert.deepEqual(copyDatasetBinding(undefined), {
      dataset_asset_id: '', norm_asset_id: '', recipe_index: 0, weight: 1,
    });
    assert.deepEqual(copyDatasetBinding(undefined, 3), {
      dataset_asset_id: '', norm_asset_id: '', recipe_index: 3, weight: 1,
    });
  });

  await test('copying does not silently repair invalid unfinished weights', () => {
    const first = { ...composed(), weight: NaN };
    assert.ok(Number.isNaN(copyDatasetBinding(first).weight));
    assert.ok(Number.isNaN(first.weight));
  });

  await test('first selection on the blank copied row retains all processing values and new norm identity', () => {
    const first = composed(), before = structuredClone(first);
    const blank = copyDatasetBinding(first);
    const selected = selectDatasetBinding(blank, selection, 0);
    assert.deepEqual(selected, { ...blank, ...selection });
    assert.equal(selected.processor.parameters.composition.action_layout[1].normalize, false);
    assert.equal(selected.processor.parameters.composition.state_fields[0].normalize, false);
    assert.equal(blank.dataset_asset_id, '');
    assert.equal(blank.norm_asset_id, '');
    assert.deepEqual(first, before);
  });

  await test('first selection on a recipe-based copy keeps its inherited recipe slot', () => {
    const blank = copyDatasetBinding({ dataset_asset_id: 'old', norm_asset_id: 'old-norm', recipe_index: 5, weight: 2 });
    assert.deepEqual(selectDatasetBinding(blank, selection, 0), {
      ...selection, recipe_index: 5, weight: 2,
    });
  });

  await test('choosing another dataset never falls back to the first dataset normalization identity', () => {
    for (const preserve of [false, true]) {
      const first = composed();
      const selected = selectDatasetBinding(first,
        { dataset_asset_id: 'other-data', norm_asset_id: '' }, 2, preserve);
      assert.equal(selected.norm_asset_id, '');
      assert.equal(selected.dataset_asset_id, 'other-data');
      assert.equal(first.norm_asset_id, 'source-norm');
    }
  });

  await test('changing a populated dataset still clears its processor and resets recipe selection', () => {
    const previous = composed(), before = structuredClone(previous);
    const changed = selectDatasetBinding(previous, selection, 3);
    assert.deepEqual(changed, { ...selection, recipe_index: 3, weight: previous.weight });
    assert.equal(Object.hasOwn(changed, 'processor'), false);
    assert.deepEqual(previous, before);
  });

  await test('inherited processing is preserved only for the first selection, not a later dataset replacement', () => {
    const firstSelection = selectDatasetBinding(copyDatasetBinding(composed()), selection, 0);
    const nextSelection = selectDatasetBinding(firstSelection,
      { dataset_asset_id: 'third-data', norm_asset_id: 'third-norm' }, 2);
    assert.deepEqual(nextSelection, {
      dataset_asset_id: 'third-data', norm_asset_id: 'third-norm', recipe_index: 2, weight: 20,
    });
  });

  await test('explicit template preservation still keeps processor or recipe slot on populated rows', () => {
    const processor = composed(), before = structuredClone(processor);
    assert.deepEqual(selectDatasetBinding(processor, selection, 0, true), { ...before, ...selection });
    const recipe = { dataset_asset_id: 'old', norm_asset_id: 'old-norm', recipe_index: 6, weight: 3 };
    assert.deepEqual(selectDatasetBinding(recipe, selection, 0, true), { ...recipe, ...selection });
    assert.deepEqual(processor, before);
  });

  await test('clearing an existing dataset also clears stale processing before another choice', () => {
    const cleared = selectDatasetBinding(composed(), { dataset_asset_id: '', norm_asset_id: '' }, 2);
    assert.deepEqual(cleared, { dataset_asset_id: '', norm_asset_id: '', recipe_index: 2, weight: 20 });
    assert.deepEqual(selectDatasetBinding(cleared, selection, 0), { ...cleared, ...selection });
  });

  await test('copied unfinished rows and false normalization flags survive account draft serialization', () => {
    const first = composed();
    const copied = copyDatasetBinding(first);
    copied.weight = NaN;
    const snapshot = {
      request: {
        name: 'draft', node_id: 'node', profile_id: 'profile', template_id: 'openpi-sft',
        inputs: { model: 'model' }, parameters: {}, gpu_count: 1,
        dataset_bindings: [first, copied],
      },
      step: 0, advancedRecipe: false, editedParameters: [],
      editedStageParameters: [], entrypointEdited: false,
    };
    const restored = codec.parseTrainingDraft(codec.stringifyTrainingDraft(snapshot));
    assert.deepEqual(restored, snapshot);
    assert.equal(restored.request.dataset_bindings[1].dataset_asset_id, '');
    assert.equal(restored.request.dataset_bindings[1].norm_asset_id, '');
    assert.ok(Number.isNaN(restored.request.dataset_bindings[1].weight));
    assert.equal(restored.request.dataset_bindings[1].processor.parameters.composition.action_layout[1].normalize, false);
    restored.request.dataset_bindings[1].processor.parameters.composition.cameras[0].layout = 'hwc';
    assert.equal(restored.request.dataset_bindings[0].processor.parameters.composition.cameras[0].layout, 'chw');
  });
} finally {
  await server.close();
}
