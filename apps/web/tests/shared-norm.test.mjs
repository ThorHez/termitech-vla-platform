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
  const { sharedNormalizationAssetId: commonNorm, bindSharedNormalization: bind,
    appendDatasetBinding: append } = helpers;
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

  await test('choosing one norm in bulk updates only the row references', () => {
    const bindings = [row('data-a', 'norm-a', 0.5),
      { dataset_asset_id: 'data-b', norm_asset_id: 'norm-b', weight: 2,
        processor: { name: 'ComposableDataConfig', parameters: { composition: { normalize: false } } } }];
    const before = structuredClone(bindings);
    const changed = bind(bindings, 'shared-norm');
    assert.deepEqual(changed, before.map((binding) => ({ ...binding, norm_asset_id: 'shared-norm' })));
    assert.equal(commonNorm(changed), 'shared-norm'); assert.deepEqual(bindings, before);
    assert.deepEqual(Object.keys(changed[0]).sort(), Object.keys(before[0]).sort());
    const switched = bind(bindings, '');
    assert.equal(commonNorm(switched), '');
    assert.ok(switched.every((binding) => binding.norm_asset_id === ''));
  });

  await test('adding a second dataset keeps both normalization selectors and starts the new reference empty', () => {
    const first = { dataset_asset_id: 'data-a', norm_asset_id: 'shared-norm', weight: 2,
      processor: { name: 'ComposableDataConfig', parameters: { composition: { normalize: false } } } };
    const before = structuredClone(first);
    const changed = append([first]);
    assert.equal(changed[1].dataset_asset_id, '');
    assert.equal(changed[1].norm_asset_id, ''); assert.equal(changed[1].weight, 2);
    assert.deepEqual(changed[1].processor, before.processor);
    const html = render(changed);
    assert.match(html, /归一化参数参考 1/);
    assert.match(html, /归一化参数参考 2/);
    assert.equal([...html.matchAll(/value="shared-norm" selected=""/g)].length, 1);
    changed[1].processor.parameters.composition.normalize = true;
    assert.deepEqual(first, before, 'copied processing must not alias its source');
  });

  await test('appending keeps old row references whether they are identical or different', () => {
    for (const bindings of [
      [row('data-a', 'shared-norm'), row('data-b', 'shared-norm')],
      [row('data-a', 'norm-a'), row('data-b', 'norm-b')],
    ]) {
      const before = structuredClone(bindings);
      const changed = append(bindings, 3);
      assert.deepEqual(changed.slice(0, 2), before);
      assert.deepEqual(changed[2], row(''));
      assert.deepEqual(bindings, before);
    }
    assert.deepEqual(append([], 3), [{ ...row(''), recipe_index: 3 }]);
  });

  await test('bulk and per-row choices coexist for empty, matching, and different references', () => {
    for (const [first, second] of [['', ''], ['shared-norm', 'shared-norm'], ['norm-a', 'norm-b']]) {
      const html = render([row('data-a', first), row('data-b', second)]);
      assert.equal([...html.matchAll(/<option value="shared-norm"/g)].length, 3);
      assert.match(html, /批量选择归一化参考/);
      assert.match(html, /归一化参数参考 1/);
      assert.match(html, /归一化参数参考 2/);
    }
    const html = render([row('data-a', 'norm-a'), row('data-b', 'norm-b')]);
    assert.match(html, /训练启动时按全部源数据与本次混采权重计算一份共享 norm_stats/);
    assert.match(html, /每条样本的重复权重；实际来源占比还取决于样本数/);
    assert.match(html, /权重占比约/);
    assert.match(html, /样本数占比/);
    assert.match(html, /预计采样占比/);
    assert.doesNotMatch(html, /采样比例约/);
    assert.match(html, /数据集 1/); assert.match(html, /数据集 2/);
  });

  await test('normalization choices exclude other machines and archived or unavailable assets', () => {
    const html = render([row('data-a'), row('data-b')], { assets: [
      ...assets, asset('remote-norm', 'NORM', { node_id: 'other' }),
      asset('archived-norm', 'NORM', { state: 'ARCHIVED' }),
      asset('unavailable-norm', 'NORM', { state: 'UNAVAILABLE' }),
    ] });
    for (const id of ['remote-norm', 'archived-norm', 'unavailable-norm'])
      assert.doesNotMatch(html, new RegExp(`<option value="${id}"`));
    assert.match(html, /<option value="" disabled="" selected="">批量选择归一化参考/);
  });

  await test('account draft restoration keeps identical and distinct references independently editable', () => {
    for (const [first, second] of [['shared-norm', 'shared-norm'], ['norm-a', 'norm-b']]) {
      const original = snapshot([row('data-a', first), row('data-b', second)]);
      const restored = codec.decodeTrainingDraft(JSON.parse(JSON.stringify(codec.encodeTrainingDraft(original))));
      assert.deepEqual(restored, original);
      const html = render(restored.request.dataset_bindings);
      assert.match(html, new RegExp(`value="${first}" selected=""`));
      assert.match(html, new RegExp(`value="${second}" selected=""`));
      assert.match(html, /归一化参数参考 1/);
      assert.match(html, /归一化参数参考 2/);
      assert.deepEqual(restored.request.dataset_bindings, original.request.dataset_bindings);
    }
  });

  await test('single dataset and read-only reviews retain their existing editing boundaries', () => {
    const single = render([row('data-a', 'norm-a')]);
    assert.match(single, /归一化参数 1/); assert.match(single, /value="norm-a" selected=""/);
    assert.match(single, /每条样本的重复权重；实际来源占比还取决于样本数/);
    assert.doesNotMatch(single, /共享归一化参数|逐组归一化（旧配置）/);
    const review = render([row('data-a', 'norm-a'), row('data-b', 'norm-b')], { readOnly: true });
    assert.match(review, /norm-a/); assert.match(review, /norm-b/);
    assert.doesNotMatch(review, /<select\b|添加数据集|批量选择归一化参考/);
  });
} finally {
  await server.close();
}
