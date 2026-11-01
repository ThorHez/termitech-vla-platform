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
const fixture = JSON.parse(readFileSync(new URL('../../../tests/fixtures/composed-processing-v1.json', import.meta.url))).valid[0];
const label = (role = 'action', changes = {}) => ({
  key: role === 'state' ? 'observation.packed_state' : 'actions', role,
  representation: 'composite', component: 'other', value_mode: 'absolute', arm: 'left',
  shape: role === 'state' ? [2, 10] : [16, 10],
  ...(role === 'action' ? { horizon: 16 } : {}), ...changes,
});
const profile = () => ({
  schema_version: 1, format: 'lerobot', robot_type: 'ARX5', arm_mode: 'single',
  action_representation: 'unknown', action_mode: 'unknown', rotation_representation: 'unknown',
  cameras: [{ name: 'wrist', key: 'observation.images.wrist', kind: 'rgb' }],
  task_description: 'Fold cloth', feature_keys: ['observation.packed_state', 'actions', 'observation.images.wrist', 'task'],
  field_annotations: [label('state'), label()],
});
const spec = () => ({ ...structuredClone(fixture), state_representation: 'eef', state_history: 2,
  state_fields: [{ key: 'observation.packed_state', width: 10, arm: 'left', kind: 'other', normalize: true }] });
const html = (Component, props) => renderToStaticMarkup(React.createElement(Component, props));

try {
  const fields = await server.ssrLoadModule('/src/datasetFieldAnnotations.ts');
  const { DatasetFieldLabelsEditor } = await server.ssrLoadModule('/src/DatasetFieldLabelsEditor.tsx');
  const profiles = await server.ssrLoadModule('/src/DatasetProfileForm.tsx');
  const templates = await server.ssrLoadModule('/src/datasetFormatTemplateData.ts');
  const processing = await server.ssrLoadModule('/src/composedProcessing.ts');
  const { DatasetProcessing } = await server.ssrLoadModule('/src/DatasetProcessing.tsx');

  await test('composite state and action tags persist raw shape and source horizon through registration and format templates', () => {
    const source = profile();
    assert.deepEqual(fields.fieldAnnotationErrors(source.field_annotations), []);
    assert.deepEqual(profiles.datasetProfileErrors(source), []);
    const registered = profiles.readDatasetProfile(source);
    assert.deepEqual(registered.field_annotations, source.field_annotations);
    const saved = templates.datasetFormatProfile(source);
    assert.deepEqual(saved.field_annotations, source.field_annotations);
    assert.equal(saved.field_annotations[1].horizon, 16);
    saved.field_annotations[1].shape[0] = 50;
    assert.equal(source.field_annotations[1].shape[0], 16);
  });
  await test('composite cannot claim a single scalar component or bypass source horizon validation', () => {
    for (const component of ['position', 'rotation', 'joint', 'gripper'])
      assert.match(fields.fieldAnnotationErrors([label('action', { component })]).join(), /组合向量/);
    assert.match(fields.fieldAnnotationErrors([label('action', { horizon: 50 })]).join(), /原始形状/);
    assert.match(fields.fieldAnnotationErrors([label('state', { horizon: 16 })]).join(), /仅用于动作/);
  });
  await test('registration offers the composite choice and locks its component while keeping field facts editable', () => {
    const rendered = html(DatasetFieldLabelsEditor, { value: [label()], onChange() {} });
    assert.match(rendered, /value="composite" selected="">组合向量（多分量）/);
    assert.match(rendered, /aria-label="字段分量 1"[^>]*disabled=""/);
    assert.match(rendered, /value="other" selected="">组合向量/);
    assert.match(rendered, /aria-label="动作 Horizon 1"[^>]*value="16"/);
    assert.doesNotMatch(rendered, /aria-label="字段数值方式 1"[^>]*disabled/);
    assert.match(rendered, /字段内部顺序与分块在新建训练中配置，不会自动猜测/);
    const plain = html(DatasetFieldLabelsEditor, { value: [label('action', { representation: 'eef' })], onChange() {} });
    assert.doesNotMatch(plain, /aria-label="字段分量 1"[^>]*disabled=""/);
  });
  await test('total width alone does not infer mixed action partitions or an EEF versus Joint representation', () => {
    const source = profile(), before = structuredClone(source);
    const inferred = processing.inferSourceActionLayout(source, 'actions');
    assert.equal(inferred.width, 10);
    assert.equal(inferred.blocks, undefined);
    const draft = processing.trainingCompositionDraft(source);
    assert.equal(draft.action_representation, '');
    assert.equal(draft.state_representation, '');
    assert.equal(draft.action_layout[0].width, 0);
    assert.deepEqual(draft.state_fields, [{ key: 'observation.packed_state', width: 10, arm: 'left', kind: 'other', normalize: true }]);
    assert.deepEqual(source, before);
  });
  await test('complete coordinate names may seed ordered action blocks without treating the source as a scalar component', () => {
    const source = profile();
    source.field_annotations[1].names = [
      'left_position_x', 'left_position_y', 'left_position_z',
      ...Array.from({ length: 6 }, (_, index) => 'left_rotation_6d_' + index), 'left_gripper_width',
    ];
    assert.deepEqual(processing.inferSourceActionLayout(source, 'actions').blocks.map(block => [block.kind, block.width]),
      [['position', 3], ['rotation', 6], ['gripper', 1]]);
  });
  await test('explicit EEF processing accepts composite facts while retaining total-width, mode, arm and state-kind checks', () => {
    assert.deepEqual(processing.composedProcessingErrors(spec()), []);
    assert.deepEqual(processing.trainingCompositionProfileErrors(profile(), spec()), []);
    const wrongWidth = spec(); wrongWidth.action_layout[0].width = 4;
    assert.match(processing.trainingCompositionProfileErrors(profile(), wrongWidth).join(), /动作分块总宽度/);
    const wrongMode = spec(); wrongMode.action_mode = 'delta';
    assert.match(processing.trainingCompositionProfileErrors(profile(), wrongMode).join(), /绝对值\/增量/);
    const wrongKind = spec(); wrongKind.state_fields[0].kind = 'position';
    assert.match(processing.trainingCompositionProfileErrors(profile(), wrongKind).join(), /必须整体保留/);
    const wrongArm = spec(); wrongArm.state_fields[0].arm = 'shared';
    assert.match(processing.trainingCompositionProfileErrors(profile(), wrongArm).join(), /归属/);
    const wrongStateWidth = spec(); wrongStateWidth.state_fields[0].width = 9;
    assert.match(processing.trainingCompositionProfileErrors(profile(), wrongStateWidth).join(), /宽度与原始形状/);
  });
  await test('explicit Joint plus gripper layouts support a composite source without weakening ordinary EEF versus Joint labels', () => {
    const source = profile(), target = spec();
    source.field_annotations[0].shape = [7];
    source.field_annotations[1].shape = [50, 7];source.field_annotations[1].horizon = 50;
    target.state_history = 1;target.state_fields[0].width = 7;target.state_representation = 'joint';
    target.action_representation = 'joint';target.rotation_representation = 'none';
    target.action_layout = [{ arm: 'left', kind: 'joint', width: 6, normalize: true }, { arm: 'left', kind: 'gripper', width: 1, normalize: false }];
    assert.deepEqual(processing.composedProcessingErrors(target), []);
    assert.deepEqual(processing.trainingCompositionProfileErrors(source, target), []);
    source.field_annotations[1].representation = 'eef';
    assert.match(processing.trainingCompositionProfileErrors(source, target).join(), /动作表示与源字段标签不一致/);
  });
  await test('training review displays a Chinese composite label instead of an implied EEF or Joint label', () => {
    const rendered = html(DatasetProcessing, { binding: { dataset_asset_id: 'dataset', norm_asset_id: 'norm', weight: 1 },
      dataset: { id: 'dataset', name: 'packed', kind: 'DATASET', node_id: 'node', path: '/data/packed', version: 'v1', metadata: { dataset_profile: profile() } },
      entries: [], slots: [], index: 0, stage: 'sft', managedParameters: {}, readOnly: true, onChange() {} });
    assert.match(rendered, /组合向量（多分量）/);
    assert.doesNotMatch(rendered, />COMPOSITE</);
  });
} finally {
  await server.close();
}
