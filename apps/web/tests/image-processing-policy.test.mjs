import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false }, optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
try {
  const { imageProcessingErrors, imageSourceResolution, imageRegistrationErrors, retainMatchingImageProcessing } = await server.ssrLoadModule('/src/imageProcessingPolicy.ts');
  const { DatasetProfileForm, emptyDatasetProfile, readDatasetProfile, normalizedDatasetProfile } = await server.ssrLoadModule('/src/DatasetProfileForm.tsx');
  const { datasetFormatProfile } = await server.ssrLoadModule('/src/datasetFormatTemplateData.ts');
  const { DatasetBindingSummary } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
  const { ComposedProcessingEditor } = await server.ssrLoadModule('/src/ComposedProcessingEditor.tsx');
  const { composedProcessingErrors, normalizeComposedProcessing, trainingCompositionDraft, updateComposedProcessingFromProfile } = await server.ssrLoadModule('/src/composedProcessing.ts');
  const { batchRegistrationError, batchRegistrationEntries, scanRows } = await server.ssrLoadModule('/src/batchDatasetRegistrationData.ts');
  const policy = { resize_mode: 'pad', augmentation: 'crop_rotate_color' };
  const camera = { name: '头部', key: 'observation.images.head', kind: 'rgb' };
  const profile = { ...emptyDatasetProfile(), robot_type: 'AR5', task_description: '抓取放置', cameras: [camera] };
  const feature = (shape, key = camera.key) => ({ key, dtype: 'video', shape, names: [] });
  const render = (component, props) => renderToStaticMarkup(React.createElement(component, props));

  await test('real CHW, HWC, history and depth metadata determine dimensions without guessing ambiguous layouts', () => {
    assert.deepEqual(imageSourceResolution([feature([3, 360, 640])], camera.key, 'rgb'), [360, 640]);
    assert.deepEqual(imageSourceResolution([feature([360, 640, 3])], camera.key, 'rgb'), [360, 640]);
    assert.deepEqual(imageSourceResolution([feature([2, 3, 360, 640])], camera.key, 'rgb'), [360, 640]);
    assert.deepEqual(imageSourceResolution([feature([360, 640])], camera.key, 'depth'), [360, 640]);
    assert.equal(imageSourceResolution([feature([3, 224, 3])], camera.key, 'rgb'), undefined);
    assert.equal(imageSourceResolution(undefined, camera.key, 'rgb'), undefined);
  });

  await test('non-224 images require explicit strategy including raw cameras omitted from a profile', () => {
    assert.match(imageRegistrationErrors([camera], [feature([3, 360, 640])])[0], /640×360.*明确选择/);
    assert.deepEqual(imageRegistrationErrors([{ ...camera, image_processing: policy }], [feature([3, 360, 640])]), []);
    assert.deepEqual(imageRegistrationErrors([camera], [feature([3, 224, 224])]), []);
    assert.match(imageRegistrationErrors(undefined, [feature([3, 360, 640])])[0], /登记该视觉输入/);
    assert.match(imageRegistrationErrors([camera], undefined)[0], /先读取/);
    assert.match(imageRegistrationErrors([camera], [feature([3, 224, 3])])[0], /无法确认/);
    assert.deepEqual(imageRegistrationErrors([{ ...camera, image_processing: policy }], [feature([3, 224, 3])]), []);
    assert.match(imageRegistrationErrors([{ ...camera, image_processing: policy }], [feature([3, 360, 640]), feature([3, 480, 640], 'other.camera')])[0], /other.camera/);
  });

  await test('asset profiles accept geometry and legacy policies while format templates retain only geometry', () => {
    for (const resize_mode of ['pad', 'center_crop', 'stretch']) {
      assert.deepEqual(imageProcessingErrors({ resize_mode }), []);
      const resizeOnly = { ...profile, cameras: [{ ...camera, image_processing: { resize_mode } }] };
      assert.deepEqual(readDatasetProfile(resizeOnly).cameras[0].image_processing, { resize_mode });
      for (const augmentation of ['default', 'none', 'color', 'crop_rotate_color']) {
        const source = { ...profile, cameras: [{ ...camera, image_processing: { resize_mode, augmentation } }] };
        assert.deepEqual(imageProcessingErrors(source.cameras[0].image_processing), []);
        assert.deepEqual(readDatasetProfile(source).cameras[0].image_processing, { resize_mode, augmentation });
        const template = datasetFormatProfile(source);
        assert.deepEqual(template.cameras[0].image_processing, { resize_mode });
        template.cameras[0].image_processing.resize_mode = 'stretch';
        assert.equal(source.cameras[0].image_processing.resize_mode, resize_mode);
        assert.deepEqual(normalizedDatasetProfile(source).cameras[0].image_processing, source.cameras[0].image_processing);
      }
    }
    for (const value of [null, {}, [], { ...policy, resize_mode: 'resize' }, { ...policy, augmentation: 'random' }, { ...policy, target: [224, 224] }]) {
      assert.ok(imageProcessingErrors(value).length);
      assert.equal(readDatasetProfile({ ...profile, cameras: [{ ...camera, image_processing: value }] }), undefined);
    }
    assert.equal(readDatasetProfile({ ...profile, cameras: [{ ...camera, kind: 'depth', image_processing: policy }] }), undefined);
    const depth = render(DatasetProfileForm, { value: { ...profile, cameras: [{ ...camera, kind: 'depth', image_processing: { resize_mode: 'pad', augmentation: 'none' } }] }, onChange() {} });
    assert.doesNotMatch(depth, /value="color"|value="crop_rotate_color"/);
  });

  await test('registration exposes processing choices and never silently preselects a resize', () => {
    const html = render(DatasetProfileForm, { value: profile, features: [feature([3, 360, 640])], onChange() {} });
    for (const label of ['图像处理方式 1', '等比缩放并补黑边', '中心裁剪后缩放', '直接拉伸', '224×224']) assert.ok(html.includes(label), label);
    assert.match(html, /<option value="" selected="">请选择处理方式/);
    assert.doesNotMatch(html, /训练图像增强 1/);
    const selected = render(DatasetProfileForm, { value: { ...profile, cameras: [{ ...camera, image_processing: policy }] }, features: [feature([3, 360, 640])], onChange() {} });
    assert.match(selected, /value="pad" selected=""/);
    assert.doesNotMatch(selected, /训练图像增强|95% 随机裁剪/);
    assert.equal(profile.cameras[0].image_processing, undefined);
  });

  await test('training composition drafts seed portable choices and retain legacy absence distinctly from explicit default', () => {
    const source = {
      ...profile,
      cameras: [{ ...camera, image_processing: policy }, { ...camera, name: '左腕', key: 'observation.images.wrist' }],
      image_features: [feature([3, 360, 640]), feature([3, 224, 224], 'observation.images.wrist')],
      field_annotations: [
        { key: 'observation.state', role: 'state', representation: 'composite', value_mode: 'absolute', component: 'other', arm: 'unknown', shape: [7] },
        { key: 'action', role: 'action', representation: 'composite', value_mode: 'delta', component: 'other', arm: 'unknown', shape: [7] },
      ],
    };
    const draft = trainingCompositionDraft(source);
    assert.deepEqual(composedProcessingErrors(draft), []);
    assert.deepEqual(draft.cameras.map((row) => row.augmentation), ['crop_rotate_color', 'default']);
    const old = structuredClone(draft);
    delete old.cameras[0].augmentation;
    assert.deepEqual(composedProcessingErrors(old), []);
    assert.equal(Object.hasOwn(normalizeComposedProcessing(old).cameras[0], 'augmentation'), false);
    const oldHtml = render(ComposedProcessingEditor, { value: old, profile: source, onChange() {} });
    assert.match(oldHtml, /沿用旧配置 · 95% 随机裁剪/);
    old.cameras[0].augmentation = 'default';
    assert.equal(updateComposedProcessingFromProfile(source, normalizeComposedProcessing(old)).cameras[0].augmentation, 'default');
    assert.deepEqual(normalizeComposedProcessing(old, 'norm').cameras, []);
    assert.doesNotMatch(render(ComposedProcessingEditor, { value: old, profile: source, purpose: 'norm', onChange() {} }), /训练图像增强/);
    old.cameras[0].augmentation = 'invalid';
    assert.match(composedProcessingErrors(old).join(), /训练增强方式无效/);
    old.cameras[0].augmentation = 'color';
    old.cameras[0].kind = 'depth';
    assert.match(composedProcessingErrors(old).join(), /深度图不支持 RGB 训练增强/);
  });

  await test('older templates retain an existing matching image choice without copying policies to other cameras', () => {
    const current = { ...profile, cameras: [{ ...camera, image_processing: policy }] };
    const retained = retainMatchingImageProcessing(profile, current);
    assert.deepEqual(retained.cameras[0].image_processing, policy);
    retained.cameras[0].image_processing.resize_mode = 'stretch';
    assert.equal(current.cameras[0].image_processing.resize_mode, 'pad');
    assert.equal(profile.cameras[0].image_processing, undefined);
    assert.equal(retainMatchingImageProcessing({ ...profile, cameras: [{ ...camera, key: 'different.camera' }] }, current).cameras[0].image_processing, undefined);
    const explicit = { ...profile, cameras: [{ ...camera, image_processing: { resize_mode: 'center_crop', augmentation: 'none' } }] };
    assert.deepEqual(retainMatchingImageProcessing(explicit, current).cameras[0].image_processing, { resize_mode: 'center_crop', augmentation: policy.augmentation });
    assert.deepEqual(retainMatchingImageProcessing(explicit, profile).cameras[0].image_processing, { resize_mode: 'center_crop' });
  });

  await test('batch validates every selected dataset and preserves strategy in registration payloads', () => {
    const entries = [
      { path: '/data/one', relative_path: 'one', name: 'one', features: [feature([3, 224, 224])], summary: {}, error: null },
      { path: '/data/two', relative_path: 'two', name: 'two', features: [feature([3, 360, 640])], summary: {}, error: null },
    ];
    const rows = scanRows({ node_id: 'n', entries }, []);
    const args = { nodeId: 'n', rows, version: 'v1', metadataOnly: true, profile, assets: [] };
    assert.match(batchRegistrationError(args), /two.*640×360/);
    assert.match(batchRegistrationError({ ...args, metadataOnly: false }), /two.*明确选择/);
    const selectedProfile = { ...profile, cameras: [{ ...camera, image_processing: policy }] };
    assert.equal(batchRegistrationError({ ...args, metadataOnly: false, profile: selectedProfile }), undefined);
    assert.deepEqual(batchRegistrationEntries(rows, 'v1', false, selectedProfile)[1].profile.cameras[0].image_processing, { resize_mode: 'pad' });
    rows[1].selected = false;
    assert.equal(batchRegistrationError(args), undefined);
    assert.match(batchRegistrationError({ ...args, rows: [{ ...rows[0], entry: { ...rows[0].entry, features: undefined } }] }), /先读取/);
  });

  await test('training review displays frozen per-camera policies rather than changed live asset metadata', () => {
    const frozen = { ...profile, cameras: [{ ...camera, image_processing: policy }] };
    const live = { ...profile, cameras: [{ ...camera, image_processing: { resize_mode: 'stretch', augmentation: 'none' } }] };
    const html = render(DatasetBindingSummary, {
      bindings: [{ dataset_asset_id: 'd', norm_asset_id: 's', weight: 1, recipe_index: 0 }],
      assets: [{ id: 'd', name: 'data', path: '/data', metadata: { dataset_profile: live } }],
      profiles: { dataset: frozen },
    });
    assert.match(html, /图像处理 · 目标 224×224/);
    assert.match(html, /等比缩放并补黑边/);
    assert.match(html, /逐路训练增强在本次训练的组合视觉输入中设置/);
    assert.doesNotMatch(html, /直接拉伸|关闭训练增强|95% 随机裁剪/);
  });
} finally {
  await server.close();
}
