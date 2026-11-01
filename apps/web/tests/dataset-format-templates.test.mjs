import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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
const fixture = JSON.parse(readFileSync(new URL('./fixtures/dataset-inspections.json', import.meta.url))).complete;

try {
  const { datasetFormatProfile, datasetTemplateNameError } = await server.ssrLoadModule('/src/datasetFormatTemplateData.ts');
  const { emptyDatasetProfile } = await server.ssrLoadModule('/src/DatasetProfileForm.tsx');
  const { replaceInspectionDraftFromTemplate, mergeDatasetInspection, clearAutomaticInspectionDefaults } =
    await server.ssrLoadModule('/src/datasetInspectionHelpers.ts');
  const { DatasetFormatTemplates, DatasetFormatTemplateDialog } = await server.ssrLoadModule('/src/DatasetFormatTemplates.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const profile = {
    ...emptyDatasetProfile(), robot_type: 'ARX5', task_description: 'Fold the cloth.',
    cameras: [{ name: '头部', key: 'observation.images.top_head', kind: 'rgb' }],
    field_annotations: [{ key: 'actions', role: 'action', representation: 'joint', value_mode: 'absolute',
      component: 'joint', arm: 'left', shape: [50, 7], horizon: 50 }],
  };
  const state = () => ({ initial: emptyDatasetProfile(), registered: false, touched: new Set(), automatic: new Map(), originals: new Map() });
  const render = (component, props, permission = true, user = 'user-1') => renderToStaticMarkup(
    React.createElement(AuthContext.Provider, { value: { user: user ? { id: user } : undefined, can: () => permission } },
      React.createElement(component, props)),
  );

  await test('format copy strips even invalid legacy processing before validating source metadata', () => {
    const source = { ...profile, processing: { invalid_old_composition: true } };
    const copy = datasetFormatProfile(source);
    assert.ok(copy); assert.equal(copy.processing, undefined); assert.equal(source.processing.invalid_old_composition, true);
    assert.equal(copy.robot_type, 'ARX5'); assert.deepEqual(copy.cameras, profile.cameras);
    assert.equal(copy.field_annotations[0].horizon, 50);
  });
  await test('template format never accepts an asset object or location, normalization, model and account references', () => {
    for (const [key, value] of Object.entries({ node_id: 'node', path: '/private/data', norm_asset_id: 'norm', user_id: 'other', action_horizon: 80 }))
      assert.equal(datasetFormatProfile({ ...profile, [key]: value }), undefined, key);
    assert.equal(datasetFormatProfile({ metadata: { dataset_profile: profile } }), undefined);
  });
  await test('copy retains raw fields and stored horizon while isolating all nested objects', () => {
    const copy = datasetFormatProfile(profile);
    copy.cameras[0].key = 'changed'; copy.field_annotations[0].shape[0] = 1;
    assert.equal(profile.cameras[0].key, 'observation.images.top_head');
    assert.equal(profile.field_annotations[0].shape[0], 50);
    const single = datasetFormatProfile({ ...profile, field_annotations: [{ ...profile.field_annotations[0], shape: [7], horizon: 1 }] });
    assert.equal(single.field_annotations[0].horizon, 1);
  });
  await test('invalid profiles and invalid names cannot be saved as usable format templates', () => {
    assert.equal(datasetFormatProfile({ ...profile, cameras: [] }), undefined);
    assert.equal(datasetFormatProfile({ ...profile, field_annotations: [{ ...profile.field_annotations[0], horizon: 10 }] }), undefined);
    for (const name of ['', '   ', 'bad\nname', 'x'.repeat(101)]) assert.ok(datasetTemplateNameError(name));
    assert.equal(datasetTemplateNameError(' ARX5 双臂 '), undefined);
  });
  await test('applying explicit default-like fields and source labels resists a later inspection response', () => {
    const s = state();
    const applied = replaceInspectionDraftFromTemplate(datasetFormatProfile(profile), s);
    const result = mergeDatasetInspection(applied, fixture, s).profile;
    for (const key of ['robot_type', 'arm_mode', 'action_representation', 'action_mode', 'rotation_representation', 'cameras', 'field_annotations'])
      assert.deepEqual(result[key], applied[key], key);
    assert.equal(result.field_annotations[0].horizon, 50);
    assert.equal(result.fps, fixture.suggestions.fps);
    assert.equal(result.action_dim, fixture.suggestions.action_dim);
  });
  await test('empty provided label lists remain explicit while absent optional facts can still be inspected', () => {
    const s = state(), raw = { ...profile, field_annotations: [], fps: undefined };
    const current = replaceInspectionDraftFromTemplate(raw, s);
    const result = mergeDatasetInspection(current, fixture, s).profile;
    assert.deepEqual(result.field_annotations, []);
    assert.equal(result.fps, fixture.suggestions.fps);
    assert.equal(s.touched.has('fps'), false);
  });
  await test('changing source path cannot roll template choices back to old automatic values', () => {
    const s = state();
    const prior = mergeDatasetInspection(emptyDatasetProfile(), fixture, s).profile;
    assert.ok(s.automatic.size > 0);
    const next = replaceInspectionDraftFromTemplate(datasetFormatProfile(profile), s);
    assert.equal(s.automatic.size, 0); assert.equal(s.originals.size, 0);
    assert.deepEqual(clearAutomaticInspectionDefaults(next, s), next);
    assert.notEqual(prior.cameras[0].key, next.cameras[0].key);
    next.cameras[0].name = 'mutated'; assert.equal(profile.cameras[0].name, '头部');
  });
  await test('real-account editors expose explicit save and apply controls without requiring a nested form', () => {
    const html = render(DatasetFormatTemplates, { value: profile, onApply() {}, initialName: 'ARX5 格式' });
    assert.match(html, /选择格式模板/); assert.match(html, /套用模板/); assert.match(html, /保存当前格式为模板/);
    assert.match(html, /模板仅当前账户可见/); assert.doesNotMatch(html, /<form/);
    assert.ok((html.match(/type="button"/g) || []).length >= 4);
  });
  await test('viewer and legacy-style contexts keep registration usable without template mutations', () => {
    for (const html of [render(DatasetFormatTemplates, { value: profile }, false), render(DatasetFormatTemplates, { value: profile }, true, null)]) {
      assert.match(html, /个人账户/); assert.doesNotMatch(html, /<button|<select/);
    }
  });
  await test('existing dataset save dialog excludes apply controls and displays untrusted names as text', () => {
    const html = render(DatasetFormatTemplateDialog, { asset: { id: 'asset', name: '<script>name</script>', kind: 'DATASET', metadata: { dataset_profile: profile } }, close() {} });
    assert.match(html, /保存数据集格式模板/); assert.match(html, /&lt;script&gt;name&lt;\/script&gt;/);
    assert.doesNotMatch(html, /<script>|选择格式模板|套用模板/);
  });
} finally { await server.close(); }
