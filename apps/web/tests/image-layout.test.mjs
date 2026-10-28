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
  const { inferImageLayout, imageLayoutErrors } = await server.ssrLoadModule('/src/imageLayout.ts');
  const features = (shape, dtype = 'image') => [{ key: 'observation.camera', dtype, shape, names: [] }];
  const camera = (layout = 'chw', kind = 'rgb', time_index = null) => ({ key: 'observation.camera', kind, layout, time_index });

  await test('RGB layout comes only from a unique three-channel axis for images and videos', () => {
    for (const dtype of ['image', 'video']) for (const [shape, layout] of [[[224, 320, 3], 'hwc'], [[3, 224, 320], 'chw']]) {
      const result = inferImageLayout(features(shape, dtype), 'observation.camera', 'rgb');
      assert.equal(result.layout, layout);
      assert.deepEqual(result.shape, shape);
      assert.equal(result.requiresTimeIndex, false);
      assert.equal(result.timeLength, undefined);
      assert.deepEqual(imageLayoutErrors(camera(layout), features(shape, dtype)), []);
    }
  });

  await test('depth layout requires a unique singleton channel axis', () => {
    for (const [shape, layout] of [[[224, 320, 1], 'hwc'], [[1, 224, 320], 'chw']]) {
      assert.equal(inferImageLayout(features(shape), 'observation.camera', 'depth').layout, layout);
      assert.deepEqual(imageLayoutErrors(camera(layout, 'depth'), features(shape)), []);
    }
  });

  await test('both possible channel axes and absent expected channels never create a guessed layout', () => {
    for (const [shape, kind] of [[[3, 224, 3], 'rgb'], [[1, 224, 1], 'depth'], [[224, 320, 4], 'rgb'], [[8, 224, 320], 'depth']]) {
      const result = inferImageLayout(features(shape), 'observation.camera', kind);
      assert.equal(result.layout, undefined);
      assert.ok(result.reason);
    }
    // A manually chosen interpretation is still valid when both axes fit.
    assert.deepEqual(imageLayoutErrors(camera('chw'), features([3, 224, 3])), []);
    assert.deepEqual(imageLayoutErrors(camera('hwc'), features([3, 224, 3])), []);
  });

  await test('an extra leading axis permits layout inference while always requiring explicit frame selection', () => {
    for (const [shape, kind, layout] of [
      [[2, 224, 320, 3], 'rgb', 'hwc'], [[2, 3, 224, 320], 'rgb', 'chw'],
      [[1, 224, 320, 3], 'rgb', 'hwc'], [[4, 224, 320, 1], 'depth', 'hwc'], [[4, 1, 224, 320], 'depth', 'chw'],
    ]) {
      const result = inferImageLayout(features(shape), 'observation.camera', kind);
      assert.equal(result.layout, layout);
      assert.equal(result.requiresTimeIndex, true);
      assert.equal(result.timeLength, shape[0]);
      assert.deepEqual(result.frameShape, shape.slice(1));
      assert.equal(result.time_index, undefined);
      assert.match(imageLayoutErrors(camera(layout, kind), features(shape)).join(' '), /显式填写图像时间索引/);
      assert.deepEqual(imageLayoutErrors(camera(layout, kind, 0), features(shape)), []);
      assert.deepEqual(imageLayoutErrors(camera(layout, kind, -1), features(shape)), []);
    }
  });

  await test('history shape with ambiguous channel positions still needs manual layout and time index', () => {
    const result = inferImageLayout(features([2, 3, 224, 3]), 'observation.camera', 'rgb');
    assert.equal(result.layout, undefined);
    assert.equal(result.requiresTimeIndex, true);
    assert.match(result.reason, /无法唯一判断/);
    assert.deepEqual(imageLayoutErrors(camera('hwc', 'rgb', 1), features([2, 3, 224, 3])), []);
  });

  await test('invalid, fractional and out of range frame indexes are reported before preflight', () => {
    for (const index of [2, -3]) assert.match(imageLayoutErrors(camera('hwc', 'rgb', index), features([2, 224, 320, 3])).join(' '), /越界/);
    for (const index of [NaN, Infinity, .5, -65, 64]) assert.match(imageLayoutErrors(camera('hwc', 'rgb', index), features([100, 224, 320, 3])).join(' '), /整数/);
    assert.deepEqual(imageLayoutErrors(camera('hwc', 'rgb', -2), features([2, 224, 320, 3])), []);
  });

  await test('static images cannot silently lose a spatial or channel axis through a time index', () => {
    assert.match(imageLayoutErrors(camera('hwc', 'rgb', 0), features([224, 320, 3])).join(' '), /清空图像时间索引/);
    assert.match(imageLayoutErrors(camera('hwc', 'depth', 0), features([224, 320])).join(' '), /清空图像时间索引/);
  });

  await test('two dimensional depth has no layout to infer and remains a supported source', () => {
    const result = inferImageLayout(features([224, 320]), 'observation.camera', 'depth');
    assert.equal(result.layout, undefined);
    assert.equal(result.depthWithoutChannel, true);
    assert.equal(result.requiresTimeIndex, false);
    for (const layout of ['chw', 'hwc']) assert.deepEqual(imageLayoutErrors(camera(layout, 'depth'), features([224, 320])), []);
  });

  await test('channel-free depth history is accepted only when the user explicitly selects its first axis', () => {
    assert.equal(inferImageLayout(features([4, 224, 320]), 'observation.camera', 'depth').layout, undefined);
    assert.ok(imageLayoutErrors(camera('hwc', 'depth'), features([4, 224, 320])).length);
    assert.deepEqual(imageLayoutErrors(camera('hwc', 'depth', -1), features([4, 224, 320])), []);
  });

  await test('missing inspection does not introduce a blocker or a fallback CHW recommendation', () => {
    for (const value of [undefined, []]) {
      const result = inferImageLayout(value, 'observation.camera', 'rgb');
      assert.equal(result.layout, undefined);
      assert.match(result.reason, /尚未读取/);
      assert.deepEqual(imageLayoutErrors(camera('chw'), value), []);
    }
  });

  await test('matching the exact selected key prevents reuse of another cameras layout', () => {
    const source = [...features([224, 320, 3]), { key: 'other.camera', dtype: 'video', shape: [3, 224, 320] }];
    assert.equal(inferImageLayout(source, 'observation.camera', 'rgb').layout, 'hwc');
    assert.equal(inferImageLayout(source, 'other.camera', 'rgb').layout, 'chw');
    assert.equal(inferImageLayout(source, 'missing.camera', 'rgb').layout, undefined);
    assert.ok(imageLayoutErrors({ ...camera(), key: 'missing.camera' }, source).length);
    assert.equal(inferImageLayout([...source, source[0]], 'observation.camera', 'rgb').layout, undefined);
  });

  await test('metadata dtype and shape must be valid, and higher ranks are never flattened implicitly', () => {
    for (const shape of [null, [], [0, 320, 3], [-1, 320, 3], [224.5, 320, 3], ['224', 320, 3], [224, Infinity, 3], [224, 2 ** 31, 3]]) {
      assert.equal(inferImageLayout(features(shape), 'observation.camera', 'rgb').layout, undefined);
      assert.ok(imageLayoutErrors(camera(), features(shape)).length);
    }
    assert.equal(inferImageLayout(features([224, 320, 3], 'float32'), 'observation.camera', 'rgb').layout, 'hwc');
    assert.deepEqual(imageLayoutErrors(camera('hwc'), features([224, 320, 3], 'float32')), []);
    assert.equal(inferImageLayout(features([224, 320, 3], 'string'), 'observation.camera', 'rgb').layout, undefined);
    assert.match(imageLayoutErrors(camera(), features([224, 320, 3], 'string')).join(' '), /不是图像或数值张量/);
    const unsupported = inferImageLayout(features([2, 2, 224, 320, 3]), 'observation.camera', 'rgb');
    assert.equal(unsupported.layout, undefined);
    assert.ok(imageLayoutErrors(camera('hwc', 'rgb', 0), features([2, 2, 224, 320, 3])).length);
  });

  await test('known source layout disagreement is visible without altering a manual or restored setting', () => {
    const manual = camera('chw');
    assert.equal(inferImageLayout(features([224, 320, 3]), manual.key, manual.kind).layout, 'hwc');
    assert.match(imageLayoutErrors(manual, features([224, 320, 3])).join(' '), /CHW.*RGB 三通道/);
    assert.equal(manual.layout, 'chw');
    assert.ok(imageLayoutErrors(camera(''), features([224, 320, 3])).length);
  });

  await test('inference produces independent copies and never edits camera selection or source metadata', () => {
    const source = features([2, 224, 320, 3]), before = structuredClone(source), selected = camera('chw');
    const result = inferImageLayout(source, selected.key, selected.kind);
    result.shape[0] = 99;result.frameShape[0] = 999;
    assert.deepEqual(source, before);
    assert.deepEqual(selected, camera('chw'));
  });
} finally {
  await server.close();
}
