import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom',
});
const composition = JSON.parse(readFileSync(new URL('../../../tests/fixtures/composed-processing-v1.json', import.meta.url))).valid[0];
const dataset = (index = 0) => ({
  id: `data-${index}`, kind: 'DATASET', node_id: 'node', name: `Data ${index}`, path: `/data/${index}`, version: '1',
  metadata: { dataset_profile: { schema_version: 1, format: 'lerobot', robot_type: composition.robot_model,
    arm_mode: composition.arm_mode, action_representation: composition.action_representation,
    action_mode: composition.action_mode, rotation_representation: composition.rotation_representation,
    action_dim: 10, state_dim: 7, task_description: 'test', feature_keys: [],
    cameras: composition.cameras.map((camera) => ({ name: camera.key, key: camera.key, kind: camera.kind })) } },
});
const binding = (index = 0) => ({ dataset_asset_id: `data-${index}`, norm_asset_id: 'norm', weight: 1,
  processor: { name: 'ComposableDataConfig', parameters: { composition: structuredClone(composition) } } });
const request = () => ({
  name: 'Training method', node_id: 'node', profile_id: 'profile', template_id: 'openpi-sft',
  inputs: { model: 'model' }, parameters: { steps: 1000, batch_size: 2, learning_rate: 0.0001 }, gpu_count: 1,
  auto_config: { model_type: 'pi05', finetune_mode: 'full', action_horizon: 16, action_dim: 32 },
  dataset_bindings: [binding()],
});
const snapshot = (value) => ({ request: value, step: 1, advancedRecipe: false,
  editedParameters: [], editedStageParameters: [], entrypointEdited: false });
const render = (Component, props) => renderToStaticMarkup(React.createElement(Component, props));

try {
  const methods = await server.ssrLoadModule('/src/trainingNormalization.ts');
  const { TrainingNormalizationField } = await server.ssrLoadModule('/src/TrainingNormalizationField.tsx');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const templates = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const retries = await server.ssrLoadModule('/src/runRetry.ts');
  const { configurationConfirmed } = await server.ssrLoadModule('/src/trainingConfiguration.ts');
  const { GeneratedConfigReview } = await server.ssrLoadModule('/src/GeneratedConfigReview.tsx');
  const { ComposedProcessingEditor } = await server.ssrLoadModule('/src/ComposedProcessingEditor.tsx');
  const { DatasetBindings, DatasetBindingSummary } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
  const { DatasetProcessing } = await server.ssrLoadModule('/src/DatasetProcessing.tsx');
  const { composedChoice } = await server.ssrLoadModule('/src/ComposedDatasetProcessing.tsx');
  const normAsset = { id: 'norm', kind: 'NORM', node_id: 'node', name: 'Shared statistics', path: '/norm', version: '1' };

  await test('global normalization choices expose true Min-Max and q01/q99 while retaining Z-score and inheritance', () => {
    assert.equal(methods.DEFAULT_TRAINING_NORMALIZATION_METHOD, 'quantile');
    for (const method of ['minmax', 'quantile', 'zscore']) {
      const html = render(TrainingNormalizationField, { value: method, onChange() {} });
      assert.match(html, new RegExp(`<option value="${method}" selected="">`));
      assert.match(html, /Min-Max（最小\/最大值）/); assert.match(html, /q01\/q99（1%–99%分位数）/);
      assert.match(html, /Z-score（均值\/标准差）/); assert.match(html, /沿用配方 \/ 旧配置/);
    }
    const inherited = render(TrainingNormalizationField, { onChange() {} });
    assert.match(inherited, /<option value="" selected="">沿用配方 \/ 旧配置/);
    assert.doesNotMatch(inherited, /value="quantile" selected/);
  });

  await test('selecting and clearing the actual field callback updates only the top-level method', () => {
    const original = request(), before = structuredClone(original);
    let selected = original;
    const field = TrainingNormalizationField({ onChange(method) {
      selected = methods.withTrainingNormalizationMethod(selected, method);
    } });
    const select = field.props.children;
    select.props.onChange({ target: { value: 'minmax' } });
    assert.equal(selected.normalization_method, 'minmax');
    assert.deepEqual(selected.parameters, before.parameters);
    assert.deepEqual(selected.auto_config, before.auto_config);
    assert.deepEqual(selected.dataset_bindings, before.dataset_bindings);
    select.props.onChange({ target: { value: '' } });
    assert.equal(Object.hasOwn(selected, 'normalization_method'), false);
    assert.deepEqual(selected, original); assert.deepEqual(original, before);
  });

  await test('drafts and reusable templates preserve explicit methods and keep old absence unchanged', () => {
    for (const method of [undefined, 'minmax', 'quantile', 'zscore']) {
      const source = snapshot(methods.withTrainingNormalizationMethod(request(), method));
      const restored = codec.decodeTrainingDraft(JSON.parse(JSON.stringify(codec.encodeTrainingDraft(source))));
      assert.deepEqual(restored, source);
      assert.equal(Object.hasOwn(restored.request, 'normalization_method'), method !== undefined);
      const content = templates.prepareTrainingTemplate(source);
      const applied = templates.applyTrainingTemplate({ content }, {
        nodes: [{ id: 'node', name: 'Node', status: 'ONLINE' }],
        profiles: [{ id: 'profile', node_id: 'node', name: 'Native', adapter: 'openpi-sft', code_dir: '/code', python: '/python' }],
        assets: [{ id: 'model', node_id: 'node', kind: 'MODEL', name: 'Model', path: '/model', version: '1' }],
      }).snapshot;
      assert.equal(applied.request.normalization_method, method);
      assert.equal(Object.hasOwn(applied.request, 'normalization_method'), method !== undefined);
      assert.equal(applied.request.dataset_bindings[0].dataset_asset_id, '');
    }
    for (const invalid of ['', null, 'min_max', 'q01_q99', 0, {}, '__proto__']) {
      const source = snapshot({ ...request(), normalization_method: invalid });
      assert.throws(() => codec.encodeTrainingDraft(source));
    }
    assert.throws(() => codec.encodeTrainingDraft(snapshot({ ...request(), parameters: { normalization_method: 'minmax' } })));
  });

  await test('retraining restores explicit run or frozen methods without inventing one for historical tasks', () => {
    const source = { id: 'run', state: 'FAILED', ...request() };
    for (const method of ['minmax', 'quantile', 'zscore']) {
      for (const recorded of [{ ...source, normalization_method: method },
        { ...source, code_expectation: { normalization_method: method } },
        { ...source, attempts: [{ code_snapshot: { normalization_method: method } }] }]) {
        assert.equal(retries.retryTrainingSnapshot(recorded).request.normalization_method, method);
      }
    }
    const old = retries.retryTrainingSnapshot(source);
    assert.equal(Object.hasOwn(old.request, 'normalization_method'), false);
    assert.throws(() => retries.retryTrainingSnapshot({ ...source, normalization_method: 'min_max' }), /归一化方式/);
    assert.equal(retries.sameTrainingConfiguration(old, snapshot({ ...old.request, normalization_method: 'minmax' })), false);
  });

  await test('configuration confirmation is bound to the displayed normalization source and digest', () => {
    const config = (method) => {
      const source = `FROZEN_REQUEST = {"normalization_method": "${method}"}\n`;
      return { schema_version: 1, language: 'python', filename: 'training_config.py', source,
        sha256: createHash('sha256').update(source).digest('hex') };
    };
    const minmax = config('minmax'), quantile = config('quantile');
    assert.equal(configurationConfirmed(minmax, minmax.sha256), true);
    assert.equal(configurationConfirmed(quantile, minmax.sha256), false);
    const html = render(GeneratedConfigReview, { config: quantile, confirmedSha256: minmax.sha256, onConfirm() {} });
    assert.match(html, /normalization_method/); assert.match(html, /quantile/);
    assert.doesNotMatch(html, /checked=""/);
  });

  await test('explicit global methods replace the local composition selector without changing persisted composition', () => {
    for (const local of ['quantile', 'zscore']) {
      const spec = { ...structuredClone(composition), normalization: local }, before = structuredClone(spec);
      for (const method of ['minmax', 'quantile', 'zscore']) {
        const html = render(ComposedProcessingEditor, { value: spec, profile: dataset().metadata.dataset_profile,
          normalizationOverride: method, onChange() {} });
        const start = html.indexOf('组合归一化方式'), field = html.slice(start, html.indexOf('</label>', start));
        assert.match(field, /本次训练统一使用/); assert.ok(field.includes(methods.trainingNormalizationLabel(method)));
        assert.doesNotMatch(field, /<select\b/);
        assert.deepEqual(spec, before); assert.equal(composedChoice(spec).parameters.composition.normalization, local);
      }
      const legacy = render(ComposedProcessingEditor, { value: spec, profile: dataset().metadata.dataset_profile, onChange() {} });
      const start = legacy.indexOf('组合归一化方式'), field = legacy.slice(start, legacy.indexOf('</label>', start));
      assert.match(field, /<select\b/); assert.match(field, new RegExp(`value="${local}" selected=""`));
    }
  });

  await test('global method presentation reaches single and multi-dataset editors and read-only summaries', () => {
    for (const count of [1, 2]) {
      const bindings = Array.from({ length: count }, (_, index) => binding(index)), before = structuredClone(bindings);
      const assets = [...Array.from({ length: count }, (_, index) => dataset(index)), normAsset];
      for (const readOnly of [false, true]) {
        const html = render(DatasetBindings, { bindings, assets, slots: [], normalizationOverride: 'minmax', readOnly, onChange() {} });
        assert.match(html, /本次训练统一使用：Min-Max/);
        assert.doesNotMatch(html, /分位数归一化/);
      }
      const review = render(DatasetBindingSummary, { bindings, assets, normalizationOverride: 'minmax' });
      assert.match(review, /本次训练统一使用：Min-Max/);
      assert.match(review, /实际公式以本次训练选择为准/);
      assert.deepEqual(bindings, before);
    }
    const legacy = render(DatasetProcessing, { binding: { dataset_asset_id: 'data-0', norm_asset_id: 'norm', recipe_index: 0, weight: 1 },
      entries: [], slots: [], index: 0, stage: 'sft', managedParameters: {}, normalizationOverride: 'minmax', onChange() {} });
    assert.match(legacy, /本次训练统一使用：Min-Max/);
  });
} finally {
  await server.close();
}
