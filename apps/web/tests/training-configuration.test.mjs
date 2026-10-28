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
  const { configurationConfirmed, generatedConfigAvailable, automaticConfigErrors, DEFAULT_AUTO_CONFIG, inferDatasetActionHorizon } = await server.ssrLoadModule('/src/trainingConfiguration.ts');
  const { GeneratedConfigReview } = await server.ssrLoadModule('/src/GeneratedConfigReview.tsx');
  const { DataProcessorPicker } = await server.ssrLoadModule('/src/DataProcessorPicker.tsx');
  const { DatasetBindings } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
  const config = { schema_version: 1, language: 'python', filename: 'training_config.py', source: 'config = TrainConfig(name="<review>", batch_size=72)\n', sha256: 'a'.repeat(64) };

  await test('launch confirmation belongs to the exact generated configuration digest', () => {
    assert.equal(configurationConfirmed(config), false);
    assert.equal(configurationConfirmed(config, config.sha256), true);
    assert.equal(configurationConfirmed({ ...config, sha256: 'b'.repeat(64) }, config.sha256), false);
    assert.equal(configurationConfirmed(undefined, config.sha256), false);
  });
  await test('older or incomplete preview evidence cannot enable confirmation', () => {
    for (const bad of [undefined, { ...config, source: '' }, { ...config, source: ' ' }, { ...config, sha256: '' }, { ...config, sha256: 'not-a-digest' }, { ...config, schema_version: 2 }, { ...config, language: 'json' }]) {
      assert.equal(generatedConfigAvailable(bad), false);
      assert.equal(configurationConfirmed(bad, config.sha256), false);
    }
    const html = renderToStaticMarkup(React.createElement(GeneratedConfigReview, {}));
    assert.match(html, /暂时无法确认启动/);
    assert.doesNotMatch(html, /type="checkbox"/);
  });
  await test('review shows complete escaped Python, digest and an unchecked explicit confirmation', () => {
    const html = renderToStaticMarkup(React.createElement(GeneratedConfigReview, { config, onConfirm() {} }));
    assert.match(html, /TrainConfig/);
    assert.match(html, /&lt;review&gt;/);
    assert.match(html, new RegExp(config.sha256));
    assert.match(html, /已核对本次 config，确认启动训练/);
    assert.match(html, /下载 config/);
    assert.match(html, /type="checkbox"/);
    assert.doesNotMatch(html, /checked=""/);
    const confirmed = renderToStaticMarkup(React.createElement(GeneratedConfigReview, { config, confirmedSha256: config.sha256, onConfirm() {} }));
    assert.match(confirmed, /checked=""/);
  });
  await test('run history keeps downloadable configuration without a new launch confirmation', () => {
    const html = renderToStaticMarkup(React.createElement(GeneratedConfigReview, { config, confirmedSha256: config.sha256, readOnly: true }));
    assert.match(html, /本次训练 config/);
    assert.match(html, /TrainConfig/);
    assert.match(html, /下载 config/);
    assert.doesNotMatch(html, /type="checkbox"/);
  });
  await test('automatic processing cannot offer a nonexistent recipe fallback', () => {
    const binding = { dataset_asset_id: 'data', norm_asset_id: 'norm', recipe_index: 0, weight: 1 };
    const html = renderToStaticMarkup(React.createElement(DataProcessorPicker, { binding, entries: [], slots: [], index: 0, allowRecipeDefault: false, onChange() {} }));
    assert.match(html, /选择数据处理器/);
    assert.doesNotMatch(html, /legacy:/);
    assert.doesNotMatch(html, /沿用当前配方数据项/);
  });
  await test('asset selection precedes processor confirmation and retains independent norms', () => {
    const html = renderToStaticMarkup(React.createElement(DatasetBindings, {
      bindings: [{ dataset_asset_id: 'data', norm_asset_id: 'norm', recipe_index: 0, weight: 20 }],
      assets: [
        { id: 'data', kind: 'DATASET', node_id: 'node', name: 'Training data', path: '/data/demo', version: '1' },
        { id: 'norm', kind: 'NORM', node_id: 'node', name: 'Statistics', path: '/data/demo/norm_stats.json', version: '1' },
      ],
      slots: [], automatic: true, showProcessing: false, onChange() {},
    }));
    assert.match(html, /value="data" selected/);
    assert.match(html, /value="norm" selected/);
    assert.match(html, /添加数据集/);
    assert.doesNotMatch(html, /搜索数据处理器/);
    assert.doesNotMatch(html, /未能静态读取/);
  });
  await test('model fields reject invalid integers and preserve published adapter bounds', () => {
    assert.deepEqual(automaticConfigErrors(DEFAULT_AUTO_CONFIG), []);
    assert.deepEqual(automaticConfigErrors({ ...DEFAULT_AUTO_CONFIG, action_horizon: 1000, action_dim: 4096 }), []);
    for (const update of [{ action_horizon: 0 }, { action_horizon: 1001 }, { action_horizon: 1.5 }, { action_dim: 0 }, { action_dim: 4097 }, { action_dim: NaN }, { model_type: 'memory' }, { finetune_mode: 'unknown' }])
      assert.ok(automaticConfigErrors({ ...DEFAULT_AUTO_CONFIG, ...update }).length > 0);
  });

  const action = (shape, extra = {}) => ({ key: 'actions', role: 'action', ...(shape === undefined ? {} : { shape }), ...extra });
  const dataset = (id, fields, extra = {}) => ({ id, kind: 'DATASET', name: `Dataset ${id}`, node_id: 'node', path: `/data/${id}`, version: '1', metadata: { dataset_profile: { field_annotations: fields, ...extra } } });
  const binding = (id, actionKey) => ({ dataset_asset_id: id, norm_asset_id: 'norm', weight: 1,
    ...(actionKey ? { processor: { name: 'ComposableDataConfig', parameters: { composition: { action_key: actionKey } } } } : { recipe_index: 0 }) });

  await test('raw per-step actions do not change the training default to source horizon one', () => {
    const inferred = inferDatasetActionHorizon([binding('raw')], [dataset('raw', [action([10], { horizon: 1 })], { action_horizon: 77 })]);
    assert.equal(inferred.recommendedHorizon, undefined);
    assert.equal(inferred.sources[0].kind, 'single-step');
    assert.equal(inferred.sources[0].sourceHorizon, 1);
    assert.deepEqual(inferred.errors, []);
    assert.equal(DEFAULT_AUTO_CONFIG.action_horizon, 50);
  });
  await test('stored two-dimensional action sequences recommend their real horizon including one', () => {
    for (const horizon of [1, 16, 50, 1000]) {
      const inferred = inferDatasetActionHorizon([binding('packed')], [dataset('packed', [action([horizon, 10])])]);
      assert.equal(inferred.recommendedHorizon, horizon);
      assert.equal(inferred.sources[0].kind, 'packed');
      assert.deepEqual(inferred.sources[0].shape, [horizon, 10]);
      assert.equal(inferred.sources[0].actionKey, 'actions');
      assert.deepEqual(inferred.errors, []);
    }
  });
  await test('mixing raw and consistently packed datasets uses the packed horizon in any order', () => {
    const assets = [dataset('raw', [action([10], { horizon: 1 })]), dataset('one', [action([16, 10])]), dataset('two', [action([16, 20], { horizon: 16 })])];
    for (const selected of [[binding('raw'), binding('one'), binding('two')], [binding('two'), binding('raw'), binding('one')]]) {
      const inferred = inferDatasetActionHorizon(selected, assets);
      assert.equal(inferred.recommendedHorizon, 16);
      assert.equal(inferred.sources.length, 3);
      assert.deepEqual(inferred.errors, []);
    }
  });
  await test('mixed packed horizons are a blocking conflict rather than a guessed default', () => {
    const inferred = inferDatasetActionHorizon([binding('one'), binding('two')], [dataset('one', [action([16, 10])]), dataset('two', [action([50, 10])])]);
    assert.equal(inferred.recommendedHorizon, undefined);
    assert.ok(inferred.errors.length);
    assert.ok(inferred.diagnostics.some((item) => item.code === 'ACTION_HORIZON_CONFLICT' && item.severity === 'error'));
    assert.match(inferred.errors.join('\n'), /Dataset one：16.*Dataset two：50/);
  });
  await test('the current training composition selects its action field before inference', () => {
    const asset = dataset('multi', [action([16, 10]), action([32, 7], { key: 'action.joint' })]);
    const inferred = inferDatasetActionHorizon([binding('multi', 'action.joint')], [asset]);
    assert.equal(inferred.recommendedHorizon, 32);
    assert.equal(inferred.sources[0].actionKey, 'action.joint');
    assert.deepEqual(inferred.errors, []);
  });
  await test('multiple action candidates need an explicit field selection and never a name heuristic', () => {
    const inferred = inferDatasetActionHorizon([binding('multi')], [dataset('multi', [action([16, 10]), action([50, 7], { key: 'alternative' })])]);
    assert.equal(inferred.recommendedHorizon, undefined);
    assert.equal(inferred.sources[0].kind, 'unknown');
    assert.deepEqual(inferred.errors, []);
    assert.ok(inferred.diagnostics.some((item) => item.code === 'ACTION_FIELD_AMBIGUOUS'));
  });
  await test('an unlabelled selected composition field does not fall back to another action', () => {
    const inferred = inferDatasetActionHorizon([binding('one', 'not-labelled')], [dataset('one', [action([16, 10])])]);
    assert.equal(inferred.recommendedHorizon, undefined);
    assert.equal(inferred.sources[0].actionKey, 'not-labelled');
    assert.ok(inferred.diagnostics.some((item) => item.code === 'ACTION_FIELD_UNAVAILABLE'));
  });
  await test('duplicate annotations for the selected key cannot provide unique source evidence', () => {
    const inferred = inferDatasetActionHorizon([binding('one', 'actions')], [dataset('one', [action([16, 10]), action([16, 10])])]);
    assert.equal(inferred.recommendedHorizon, undefined);
    assert.ok(inferred.warnings.length);
  });
  await test('unselected datasets, state fields, dimensions and legacy profile processing do not constrain horizon', () => {
    const inferred = inferDatasetActionHorizon([binding('selected')], [
      dataset('selected', [action([16, 10]), { key: 'observation.state', role: 'state', shape: [8, 14] }],
        { action_dim: 10, state_dim: 8, fps: 60, processing: { action_key: 'legacy', action_horizon: 88 } }),
      dataset('not-selected', [action([50, 10])]),
    ]);
    assert.equal(inferred.recommendedHorizon, 16);
    assert.equal(inferred.sources.length, 1);
    assert.deepEqual(inferred.errors, []);
  });
  await test('horizon-only metadata without source shape remains a warning without forced rejection', () => {
    const inferred = inferDatasetActionHorizon([binding('one')], [dataset('one', [action(undefined, { horizon: 16 })])]);
    assert.equal(inferred.recommendedHorizon, undefined);
    assert.equal(inferred.sources[0].sourceHorizon, undefined);
    assert.deepEqual(inferred.errors, []);
    assert.ok(inferred.diagnostics.some((item) => item.code === 'ACTION_SHAPE_MISSING' && item.severity === 'warning'));
  });
  await test('shape evidence wins and contradictory source horizon metadata blocks use until corrected', () => {
    for (const [shape, horizon, expected] of [[[16, 10], 50, 16], [[10], 50, 1], [[16, 10], '16', 16], [[16, 10], null, 16]]) {
      const inferred = inferDatasetActionHorizon([binding('one')], [dataset('one', [action(shape, { horizon })])]);
      assert.equal(inferred.sources[0].sourceHorizon, expected);
      assert.equal(inferred.recommendedHorizon, undefined);
      assert.ok(inferred.diagnostics.some((item) => item.code === 'SOURCE_HORIZON_INCONSISTENT' && item.severity === 'error'));
    }
  });
  await test('packed horizon outside the training adapter range is a blocking error', () => {
    for (const horizon of [1001, 2 ** 31 - 1]) {
      const inferred = inferDatasetActionHorizon([binding('one')], [dataset('one', [action([horizon, 10])])]);
      assert.equal(inferred.sources[0].sourceHorizon, horizon);
      assert.equal(inferred.recommendedHorizon, undefined);
      assert.ok(inferred.diagnostics.some((item) => item.code === 'ACTION_HORIZON_OUT_OF_RANGE' && item.severity === 'error'));
    }
  });
  await test('malformed source shapes do not become dimensions or horizons', () => {
    for (const shape of [null, [], [0, 10], [-1, 10], [16, 0], [1.5, 10], ['16', 10], [NaN, 10], [16, 2 ** 31]]) {
      const inferred = inferDatasetActionHorizon([binding('one')], [dataset('one', [action(shape)])]);
      assert.equal(inferred.recommendedHorizon, undefined);
      assert.ok(inferred.diagnostics.some((item) => item.code === 'ACTION_SHAPE_INVALID' && item.severity === 'error'));
    }
  });
  await test('higher-rank source arrays do not establish the action time axis even with a horizon tag', () => {
    const inferred = inferDatasetActionHorizon([binding('one')], [dataset('one', [action([2, 16, 10], { horizon: 2 })])]);
    assert.equal(inferred.recommendedHorizon, undefined);
    assert.deepEqual(inferred.sources[0].shape, [2, 16, 10]);
    assert.equal(inferred.sources[0].sourceHorizon, undefined);
    assert.deepEqual(inferred.errors, []);
    assert.ok(inferred.diagnostics.some((item) => item.code === 'ACTION_SHAPE_UNSUPPORTED'));
  });
  await test('incomplete registration is diagnostic rather than a crash or manufactured horizon', () => {
    for (const asset of [undefined, { ...dataset('one', []), metadata: null }, dataset('one', [null, {}, { role: 'state', key: 'actions', shape: [16, 10] }])]) {
      const inferred = inferDatasetActionHorizon([binding('one')], asset ? [asset] : []);
      assert.equal(inferred.recommendedHorizon, undefined);
      assert.deepEqual(inferred.errors, []);
      assert.ok(inferred.warnings.length);
    }
    const empty = inferDatasetActionHorizon([binding('')], []);
    assert.deepEqual(empty, { sources: [], diagnostics: [], errors: [], warnings: [] });
  });
  await test('known packed facts remain a recommendation while missing facts on another dataset require review', () => {
    const inferred = inferDatasetActionHorizon([binding('known'), binding('unknown')], [dataset('known', [action([16, 10])]), dataset('unknown', [action(undefined)])]);
    assert.equal(inferred.recommendedHorizon, 16);
    assert.deepEqual(inferred.errors, []);
    assert.ok(inferred.warnings.length);
  });
  await test('inference is pure and returns independent copied shape evidence', () => {
    const assets = [dataset('one', [action([16, 10], { horizon: 16 })])];
    const selected = [binding('one')];
    const snapshot = JSON.stringify({ assets, selected });
    const manualConfig = { ...DEFAULT_AUTO_CONFIG, action_horizon: 32 };
    const inferred = inferDatasetActionHorizon(selected, assets);
    inferred.sources[0].shape[0] = 99;
    assert.equal(JSON.stringify({ assets, selected }), snapshot);
    assert.equal(manualConfig.action_horizon, 32);
    assert.equal(DEFAULT_AUTO_CONFIG.action_horizon, 50);
  });
} finally {
  await server.close();
}
