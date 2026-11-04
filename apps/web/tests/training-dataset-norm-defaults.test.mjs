import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
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
  const {
    trainingDatasetNormId,
    selectTrainingDatasetBinding,
    datasetNormalizationMode,
    copyDatasetBinding,
  } = await server.ssrLoadModule('/src/datasetBindingHelpers.ts');
  const {
    applyTrainingTemplate,
    prepareTrainingTemplate,
    TRAINING_TEMPLATE_PROCESSING_MARKER,
  } = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const { trainingTemplateEntry } = await server.ssrLoadModule('/src/trainingTemplateEntry.ts');
  const { decodeTrainingDraft, encodeTrainingDraft } = await server.ssrLoadModule('/src/trainingDraft.ts');
  const { DatasetBindings } = await server.ssrLoadModule('/src/DatasetBindings.tsx');

  const dataset = (id, normId = `norm-${id}`) => ({
    id, kind: 'DATASET', node_id: 'node-a', name: `Dataset ${id}`,
    path: `/data/${id}`, state: 'REGISTERED', version: 'v1', norm_asset_id: normId,
  });
  const norm = (id, extra = {}) => ({
    id, kind: 'NORM', node_id: 'node-a', name: `Statistics ${id}`,
    path: `/data/statistics/${id}/norm_stats.json`, state: 'REGISTERED', version: 'v1',
    ...extra,
  });
  const assets = () => [
    dataset('cloth-a'), dataset('cloth-b'), norm('norm-cloth-a'), norm('norm-cloth-b'),
    norm('manual-norm'), norm('shared-norm'),
    { id: 'model-a', kind: 'MODEL', node_id: 'node-a', name: 'Initial weights',
      path: '/models/a', state: 'READY', version: 'v1' },
  ];
  const processor = () => ({
    name: 'ComposableDataConfig', parameters: { composition: {
      schema_version: 1, action_key: 'actions', state_history: 2,
      state_fields: [{ key: 'observation.state', width: 7, kind: 'joint', arm: 'left', normalize: true }],
      action_layout: [{ width: 6, kind: 'joint', arm: 'left', normalize: true },
        { width: 1, kind: 'gripper', arm: 'left', normalize: false }],
      cameras: [{ key: 'observation.images.wrist', slot: 'left_wrist_0_rgb', kind: 'rgb', layout: 'chw' }],
    } },
  });
  const binding = (datasetId = '', normId = '') => ({
    dataset_asset_id: datasetId, norm_asset_id: normId, weight: 3, processor: processor(),
  });
  const resources = () => ({
    assets: assets(),
    nodes: [{ id: 'node-a', name: 'Machine A', status: 'ONLINE' }],
    profiles: [{ id: 'native-a', node_id: 'node-a', name: 'OpenPI', adapter: 'openpi-sft', execution_mode: 'native' }],
  });
  const sft = {
    id: 'openpi-sft', name: 'SFT', status: 'AVAILABLE', runtime_adapter: 'openpi-sft', min_gpus: 1,
    parameters: { steps: { default: 10000 }, batch_size: { default: 8 } },
  };
  const snapshot = () => ({
    request: {
      name: 'Original training', node_id: 'node-a', profile_id: 'native-a', template_id: sft.id,
      inputs: { model: 'model-a', dataset: 'old-data', norm: 'old-norm' },
      dataset_bindings: [binding('old-data', 'old-norm')],
      parameters: { steps: 10000, batch_size: 8 }, gpu_count: 1,
      auto_config: { model_type: 'pi05', finetune_mode: 'full', action_horizon: 50, action_dim: 32 },
    },
    step: 1, advancedRecipe: false, editedParameters: ['batch_size'],
    editedStageParameters: [], entrypointEdited: false,
  });

  await test('a registered same-machine norm is authoritative even outside the dataset directory', () => {
    const available = assets();
    available.push(norm('directory-alternative', { path: '/data/cloth-a/norm_stats.json' }));
    const before = structuredClone(available);
    assert.equal(trainingDatasetNormId(available[0], available), 'norm-cloth-a');
    assert.deepEqual(available, before);
    assert.equal(trainingDatasetNormId(undefined, available), '');
    assert.equal(trainingDatasetNormId(available.find((asset) => asset.id === 'model-a'), available), '');
  });

  await test('invalid explicit bindings never fall back to another matching directory or asset name', () => {
    for (const invalid of [
      { id: 'missing' },
      norm('cross-machine', { node_id: 'node-b' }),
      norm('wrong-kind', { kind: 'MODEL' }),
      norm('archived-state', { state: 'ARCHIVED' }),
      norm('unavailable-state', { state: 'UNAVAILABLE' }),
      norm('archived-status', { status: 'ARCHIVED' }),
      norm('unavailable-status', { status: 'UNAVAILABLE' }),
    ]) {
      const data = dataset('cloth-a', invalid.id);
      const available = [data, norm('directory-alternative', { path: '/data/cloth-a/norm_stats.json' }),
        ...(invalid.kind ? [invalid] : [])];
      assert.equal(trainingDatasetNormId(data, available), '', invalid.id);
      assert.equal(selectTrainingDatasetBinding([binding()], 0, data.id, available)[0].norm_asset_id, '', invalid.id);
    }
  });

  await test('an unbound dataset uses only one available same-machine descendant NORM as a legacy fallback', () => {
    const data = { ...dataset('cloth-a', null), path: '/data/cloth-a///' };
    const available = [
      data, norm('descendant', { path: '/data/cloth-a/./stats/norm_stats.json/' }),
      norm('cross-machine', { node_id: 'node-b', path: '/data/cloth-a/other/norm_stats.json' }),
      norm('wrong-kind', { kind: 'MODEL', path: '/data/cloth-a/model' }),
      norm('archived', { state: 'ARCHIVED', path: '/data/cloth-a/archived' }),
      norm('unavailable', { state: 'UNAVAILABLE', path: '/data/cloth-a/unavailable' }),
      norm('status-archived', { status: 'ARCHIVED', path: '/data/cloth-a/status-archived' }),
      norm('status-unavailable', { status: 'UNAVAILABLE', path: '/data/cloth-a/status-unavailable' }),
      norm('prefix-neighbor', { path: '/data/cloth-a-copy/norm_stats.json' }),
      norm('equal-path', { path: '/data/cloth-a' }),
    ];
    assert.equal(trainingDatasetNormId(data, available), 'descendant');
    assert.equal(trainingDatasetNormId({ ...data, norm_asset_id: '' }, available), 'descendant');
    assert.equal(trainingDatasetNormId({ ...data, path: undefined }, available), '');
    assert.equal(trainingDatasetNormId(data, [data, ...available.slice(-2)]), '');
  });

  await test('ambiguous directory descendants leave normalization unselected', () => {
    const data = dataset('cloth-a', null);
    assert.equal(trainingDatasetNormId(data, [data,
      norm('one', { path: '/data/cloth-a/one/norm_stats.json' }),
      norm('two', { path: '/data/cloth-a/two/norm_stats.json' }),
    ]), '');
  });

  await test('applying a real saved template clears old assets, then choosing data restores its bound norm and processing', () => {
    const original = snapshot();
    const stored = { id: 'saved', name: 'Saved format', description: '', revision: 1,
      created_at: '2026-11-04T00:00:00Z', updated_at: '2026-11-04T00:00:00Z',
      content: prepareTrainingTemplate(original) };
    const before = structuredClone(stored);
    const applied = applyTrainingTemplate(stored, resources());
    assert.deepEqual(applied.issues, []);
    assert.equal(applied.snapshot.request.dataset_bindings[0].dataset_asset_id, '');
    assert.equal(applied.snapshot.request.dataset_bindings[0].norm_asset_id, '');
    assert.ok(applied.snapshot.editedParameters.includes(TRAINING_TEMPLATE_PROCESSING_MARKER));
    const selected = selectTrainingDatasetBinding(applied.snapshot.request.dataset_bindings, 0, 'cloth-a', assets());
    assert.equal(selected[0].dataset_asset_id, 'cloth-a');
    assert.equal(selected[0].norm_asset_id, 'norm-cloth-a');
    assert.equal(selected[0].weight, 3);
    assert.deepEqual(selected[0].processor, original.request.dataset_bindings[0].processor);
    assert.equal(selected[0].processor.parameters.composition.action_layout[1].normalize, false);
    assert.equal(applied.snapshot.request.dataset_bindings[0].norm_asset_id, '');
    assert.deepEqual(stored, before);
  });

  await test('a later explicit norm edit survives draft serialization and reopening the same training mode', () => {
    const saved = snapshot();
    saved.request.dataset_bindings = selectTrainingDatasetBinding([binding()], 0, 'cloth-a', assets());
    saved.request.dataset_bindings[0] = { ...saved.request.dataset_bindings[0], norm_asset_id: 'manual-norm' };
    const restored = decodeTrainingDraft(encodeTrainingDraft(saved));
    const reopened = trainingTemplateEntry(restored, sft, resources());
    assert.strictEqual(reopened, restored);
    assert.equal(reopened.request.dataset_bindings[0].norm_asset_id, 'manual-norm');
    // Loading updated asset metadata is a read, not a new dataset selection.
    assert.equal(trainingDatasetNormId(assets()[0], assets()), 'norm-cloth-a');
    assert.equal(reopened.request.dataset_bindings[0].norm_asset_id, 'manual-norm');
  });

  await test('replacing one populated dataset refreshes its norm and applies the existing processing reset rules', () => {
    const previous = [binding('cloth-a', 'manual-norm')], before = structuredClone(previous);
    const changed = selectTrainingDatasetBinding(previous, 0, 'cloth-b', assets(), 4);
    assert.deepEqual(changed, [{ dataset_asset_id: 'cloth-b', norm_asset_id: 'norm-cloth-b', weight: 3, recipe_index: 4 }]);
    const preserved = selectTrainingDatasetBinding(previous, 0, 'cloth-b', assets(), 4, true);
    assert.deepEqual(preserved, [{ ...before[0], dataset_asset_id: 'cloth-b', norm_asset_id: 'norm-cloth-b' }]);
    assert.deepEqual(previous, before);
    const cleared = selectTrainingDatasetBinding(previous, 0, '', assets(), 2);
    assert.deepEqual(cleared, [{ dataset_asset_id: '', norm_asset_id: '', weight: 3, recipe_index: 2 }]);
  });

  await test('a copied blank recipe row keeps its inherited slot while adopting the newly selected norm', () => {
    const copied = copyDatasetBinding({ dataset_asset_id: 'cloth-a', norm_asset_id: 'norm-cloth-a', weight: 7, recipe_index: 5 });
    const selected = selectTrainingDatasetBinding([copied], 0, 'cloth-b', assets(), 0);
    assert.deepEqual(selected, [{ dataset_asset_id: 'cloth-b', norm_asset_id: 'norm-cloth-b', weight: 7, recipe_index: 5 }]);
  });

  await test('empty shared normalization defaults only the selected row and cannot spread a single-dataset norm', () => {
    const rows = [binding(), binding(), binding()], before = structuredClone(rows);
    assert.equal(datasetNormalizationMode(rows), 'shared');
    const first = selectTrainingDatasetBinding(rows, 1, 'cloth-a', assets());
    assert.deepEqual(first.map((row) => row.norm_asset_id), ['', 'norm-cloth-a', '']);
    assert.equal(datasetNormalizationMode(first), 'legacy');
    assert.strictEqual(first[0], rows[0]);
    assert.strictEqual(first[2], rows[2]);
    const second = selectTrainingDatasetBinding(first, 2, 'cloth-b', assets());
    assert.deepEqual(second.map((row) => row.norm_asset_id), ['', 'norm-cloth-a', 'norm-cloth-b']);
    assert.deepEqual(rows, before);
  });

  await test('an explicit nonempty shared norm remains shared when another dataset is chosen', () => {
    const rows = [binding('cloth-a', 'shared-norm'), binding('', 'shared-norm')];
    const selected = selectTrainingDatasetBinding(rows, 1, 'cloth-b', assets());
    assert.deepEqual(selected.map((row) => row.norm_asset_id), ['shared-norm', 'shared-norm']);
    assert.equal(datasetNormalizationMode(selected), 'shared');
    assert.strictEqual(selected[0], rows[0]);
    assert.deepEqual(selected[1].processor, rows[1].processor);
  });

  await test('changing a different legacy row leaves the manually selected normalization untouched', () => {
    const rows = [binding('cloth-a', 'manual-norm'), binding('', '')];
    const selected = selectTrainingDatasetBinding(rows, 1, 'cloth-b', assets());
    assert.deepEqual(selected.map((row) => row.norm_asset_id), ['manual-norm', 'norm-cloth-b']);
    assert.strictEqual(selected[0], rows[0]);
  });

  await test('entering training fills empty norms through validated bindings but retains nonempty explicit overrides', () => {
    for (const target of [sft, { ...sft, id: 'recap', name: 'ReCap' }]) {
      const saved = snapshot();
      saved.request.template_id = 'openpi-norm';
      saved.request.dataset_bindings = [binding('cloth-a', ''), binding('cloth-b', 'manual-norm')];
      const selected = trainingTemplateEntry(saved, target, resources());
      assert.deepEqual(selected.request.dataset_bindings.map((row) => row.norm_asset_id), ['norm-cloth-a', 'manual-norm']);
      assert.deepEqual(selected.request.dataset_bindings[0].processor.parameters.composition.action_layout,
        saved.request.dataset_bindings[0].processor.parameters.composition.action_layout);
    }
    const invalidResources = resources();
    invalidResources.assets = invalidResources.assets.map((asset) => asset.id === 'norm-cloth-a'
      ? { ...asset, node_id: 'node-b' } : asset);
    invalidResources.assets.push(norm('path-alternative', { path: '/data/cloth-a/norm_stats.json' }));
    const saved = snapshot();
    saved.request.template_id = 'openpi-norm';
    saved.request.dataset_bindings = [binding('cloth-a', '')];
    assert.equal(trainingTemplateEntry(saved, sft, invalidResources).request.dataset_bindings[0].norm_asset_id, '');
    assert.equal(saved.request.dataset_bindings[0].norm_asset_id, '');
  });

  await test('SSR describes automatic binding and per-dataset norm choices without labelling them old configuration', () => {
    const single = renderToStaticMarkup(React.createElement(DatasetBindings, {
      bindings: [binding('cloth-a', 'norm-cloth-a')], slots: [], assets: assets(),
      node: resources().nodes[0], showProcessing: false, onChange() {},
    }));
    assert.match(single, /选择数据集后自动带出已绑定的归一化参数，也可手动更换/);
    const grouped = renderToStaticMarkup(React.createElement(DatasetBindings, {
      bindings: [binding('cloth-a', 'norm-cloth-a'), binding('cloth-b', 'norm-cloth-b')],
      slots: [], assets: assets(), node: resources().nodes[0], showProcessing: false, onChange() {},
    }));
    assert.match(grouped, /逐组归一化/);
    assert.match(grouped, /每组可使用数据集已绑定的参数，也可手动更换/);
    assert.doesNotMatch(grouped, /逐组归一化（旧配置）|已保留草稿或模板中原有的逐组选择/);
  });
} finally {
  await server.close();
}
