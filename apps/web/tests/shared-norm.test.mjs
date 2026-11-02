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
const row = (dataset, norm = '', weight = 1) => ({
  dataset_asset_id: dataset, norm_asset_id: norm, recipe_index: 0, weight,
});
const asset = (id, kind, changes = {}) => ({
  id, kind, node_id: 'node', name: id, path: `/data/${id}`, version: 'v1',
  state: 'REGISTERED', metadata: {}, ...changes,
});
const assets = [
  asset('data-a', 'DATASET'), asset('data-b', 'DATASET'),
  asset('shared-norm', 'NORM'), asset('norm-a', 'NORM'), asset('norm-b', 'NORM'),
];

try {
  const helpers = await server.ssrLoadModule('/src/datasetBindingHelpers.ts');
  const { datasetNormalizationMode: mode, sharedNormalizationAssetId: commonNorm,
    bindSharedNormalization: bind, appendDatasetBinding: append,
    copyDatasetBinding: copy, selectDatasetBinding: select } = helpers;
  const { DatasetBindings } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const render = (bindings, props = {}) => renderToStaticMarkup(React.createElement(DatasetBindings, {
    bindings, assets, slots: [], showProcessing: false, node: { id: 'node' }, onChange() {}, ...props,
  }));
  const snapshot = (bindings) => ({
    request: { name: 'Shared statistics', node_id: 'node', profile_id: 'profile',
      template_id: 'openpi-sft', inputs: { model: 'model' }, parameters: {}, gpu_count: 1,
      dataset_bindings: bindings },
    step: 0, advancedRecipe: false, editedParameters: [], editedStageParameters: [], entrypointEdited: false,
  });

  await test('blank and identical multi-source norm selections share one mode without coercing legacy drafts', () => {
    assert.equal(mode([row('data-a')]), 'single');
    assert.equal(mode([row('data-a'), row('data-b')]), 'shared');
    assert.equal(commonNorm([row('data-a'), row('data-b')]), '');
    const shared = [row('data-a', 'shared-norm'), row('data-b', 'shared-norm')];
    assert.equal(mode(shared), 'shared'); assert.equal(commonNorm(shared), 'shared-norm');
    for (const bindings of [[row('data-a', 'norm-a'), row('data-b', 'norm-b')],
      [row('data-a', 'norm-a'), row('data-b')]]) {
      const before = structuredClone(bindings);
      assert.equal(mode(bindings), 'legacy'); assert.equal(commonNorm(bindings), '');
      assert.deepEqual(bindings, before);
    }
  });

  await test('choosing one shared norm updates all row identities without rewriting processing or weights', () => {
    const bindings = [row('data-a', 'norm-a', 0.5),
      { dataset_asset_id: 'data-b', norm_asset_id: 'norm-b', weight: 2,
        processor: { name: 'ComposableDataConfig', parameters: { composition: { normalize: false } } } }];
    const before = structuredClone(bindings);
    const changed = bind(bindings, 'shared-norm');
    assert.deepEqual(changed, before.map((binding) => ({ ...binding, norm_asset_id: 'shared-norm' })));
    assert.equal(mode(changed), 'shared'); assert.deepEqual(bindings, before);
    assert.deepEqual(Object.keys(changed[0]).sort(), Object.keys(before[0]).sort());
    const switched = bind(bindings, '');
    assert.equal(mode(switched), 'shared'); assert.equal(commonNorm(switched), '');
    assert.ok(switched.every((binding) => binding.norm_asset_id === ''));
  });

  await test('adding a second dataset inherits the chosen norm but ordinary copying still clears asset identities', () => {
    const first = { dataset_asset_id: 'data-a', norm_asset_id: 'shared-norm', weight: 2,
      processor: { name: 'ComposableDataConfig', parameters: { composition: { normalize: false } } } };
    const before = structuredClone(first);
    const changed = append([first]);
    assert.equal(mode(changed), 'shared'); assert.equal(changed[1].dataset_asset_id, '');
    assert.equal(changed[1].norm_asset_id, 'shared-norm'); assert.equal(changed[1].weight, 2);
    assert.deepEqual(changed[1].processor, before.processor);
    changed[1].processor.parameters.composition.normalize = true;
    assert.deepEqual(first, before, 'copied processing must not alias its source');
    assert.deepEqual(copy(first), { ...before, dataset_asset_id: '', norm_asset_id: '' });
  });

  await test('appending preserves shared mode and leaves old per-source identities untouched', () => {
    const shared = [row('data-a', 'shared-norm'), row('data-b', 'shared-norm')];
    assert.equal(commonNorm(append(shared)), 'shared-norm');
    const legacy = [row('data-a', 'norm-a'), row('data-b', 'norm-b')];
    const before = structuredClone(legacy), changed = append(legacy, 3);
    assert.deepEqual(changed.slice(0, 2), before); assert.equal(changed[2].norm_asset_id, '');
    assert.equal(mode(changed), 'legacy'); assert.deepEqual(legacy, before);
    assert.deepEqual(append([], 3), [row('', '', 1)].map((binding) => ({ ...binding, recipe_index: 3 })));
  });

  await test('changing a dataset retains the explicit shared norm rather than a dataset-local guess', () => {
    const previous = row('data-a', 'shared-norm');
    const detected = { dataset_asset_id: 'data-b', norm_asset_id: 'norm-b' };
    assert.equal(select(previous, detected, 0, false, 'shared-norm').norm_asset_id, 'shared-norm');
    assert.equal(select(previous, detected, 0, false, '').norm_asset_id, '', 'unselected shared norm stays unselected');
    assert.equal(select(previous, detected, 0).norm_asset_id, 'norm-b', 'single and legacy selection behavior is retained');
    assert.equal(previous.norm_asset_id, 'shared-norm');
  });

  await test('shared norm UI offers one selector and removes per-row selectors while retaining row evidence', () => {
    const html = render([row('data-a', 'shared-norm'), row('data-b', 'shared-norm')]);
    assert.equal([...html.matchAll(/<option value="shared-norm"/g)].length, 1);
    assert.match(html, /value="shared-norm" selected=""/);
    assert.match(html, /共享归一化参数/);
    assert.doesNotMatch(html, /归一化参数 [12]/);
    assert.match(html, /单源参数不能冒充多源统计/);
    assert.match(html, /不会启动全量统计任务/);
    assert.match(html, /每条样本的重复权重；实际来源占比还取决于样本数/);
    assert.match(html, /权重占比（未考虑样本数）约/);
    assert.doesNotMatch(html, /采样比例约/);
    assert.match(html, /数据集 1/); assert.match(html, /数据集 2/);
  });

  await test('shared norm choices exclude other machines and archived or unavailable assets', () => {
    const html = render([row('data-a'), row('data-b')], { assets: [
      ...assets, asset('remote-norm', 'NORM', { node_id: 'other' }),
      asset('archived-norm', 'NORM', { state: 'ARCHIVED' }),
      asset('unavailable-norm', 'NORM', { state: 'UNAVAILABLE' }),
    ] });
    for (const id of ['remote-norm', 'archived-norm', 'unavailable-norm'])
      assert.doesNotMatch(html, new RegExp(`<option value="${id}"`));
    assert.match(html, /<option value="" selected="">选择共享归一化参数/);
  });

  await test('account draft restoration keeps legacy row norms until an explicit switch', () => {
    const original = snapshot([row('data-a', 'norm-a'), row('data-b', 'norm-b')]);
    const restored = codec.decodeTrainingDraft(JSON.parse(JSON.stringify(codec.encodeTrainingDraft(original))));
    assert.deepEqual(restored, original);
    const html = render(restored.request.dataset_bindings);
    assert.match(html, /逐组归一化（旧配置）/);
    assert.match(html, /切换为共享归一化/);
    assert.match(html, /value="norm-a" selected=""/); assert.match(html, /value="norm-b" selected=""/);
    assert.match(html, /归一化参数 1/); assert.match(html, /归一化参数 2/);
    assert.deepEqual(restored.request.dataset_bindings, original.request.dataset_bindings);
    const switched = bind(restored.request.dataset_bindings, 'shared-norm');
    const sharedSnapshot = snapshot(switched);
    const sharedRestored = codec.decodeTrainingDraft(codec.encodeTrainingDraft(sharedSnapshot));
    assert.equal(mode(sharedRestored.request.dataset_bindings), 'shared');
    assert.equal(commonNorm(sharedRestored.request.dataset_bindings), 'shared-norm');
  });

  await test('single dataset and read-only reviews retain their existing editing boundaries', () => {
    const single = render([row('data-a', 'norm-a')]);
    assert.match(single, /归一化参数 1/); assert.match(single, /value="norm-a" selected=""/);
    assert.match(single, /每条样本的重复权重；实际来源占比还取决于样本数/);
    assert.doesNotMatch(single, /共享归一化参数|逐组归一化（旧配置）/);
    const shared = render([row('data-a', 'shared-norm'), row('data-b', 'shared-norm')], { readOnly: true });
    assert.match(shared, /共享归一化参数/); assert.match(shared, /shared-norm/);
    assert.doesNotMatch(shared, /<select\b|添加数据集|切换为共享归一化/);
    const legacy = render([row('data-a', 'norm-a'), row('data-b', 'norm-b')], { readOnly: true });
    assert.match(legacy, /逐组归一化（旧配置）/); assert.doesNotMatch(legacy, /切换为共享归一化/);
  });
} finally {
  await server.close();
}
