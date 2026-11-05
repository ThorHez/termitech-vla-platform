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
  const { DatasetFieldLabelsEditor } = await server.ssrLoadModule(
    '/src/DatasetFieldLabelsEditor.tsx',
  );
  const label = (changes = {}) => ({
    key: 'observation.state',
    role: 'state',
    representation: 'unknown',
    value_mode: 'unknown',
    component: 'other',
    arm: 'unknown',
    shape: [7],
    dtype: 'float32',
    ...changes,
  });
  const render = (row, features) =>
    renderToStaticMarkup(
      React.createElement(DatasetFieldLabelsEditor, {
        value: [row],
        features,
        onChange() {},
      }),
    );

  await test('registered one-dimensional numeric fields show explicit shape, dtype and width', () => {
    const html = render(label());
    assert.match(html, /原始形状：<code>\[7\]<\/code>/);
    assert.match(html, /float32/);
    assert.match(html, /状态维度：7 维/);
    assert.match(html, /来源：字段标签记录/);
  });

  await test('fresh feature shape takes precedence without changing saved annotation or horizon', () => {
    const row = label({ key: 'action', role: 'action', shape: [16, 10], horizon: 16 });
    const before = structuredClone(row);
    const html = render(row, [{ key: 'action', dtype: 'float64', shape: [50, 14] }]);
    assert.match(html, /原始形状：<code>\[50, 14\]<\/code>/);
    assert.match(html, /动作维度：14 维\/步 · 原有 50 步/);
    assert.match(html, /float64/);
    assert.match(html, /来源：当前数据集元数据/);
    assert.deepEqual(row, before);
  });

  await test('per-frame actions show one stored step rather than a training horizon', () => {
    const html = render(label({ key: 'action', role: 'action', shape: [7] }));
    assert.match(html, /动作维度：7 维\/步 · 原有 1 步/);
  });

  await test('leading axes are retained without flattening state or inventing temporal semantics', () => {
    const state = render(label({ shape: [2, 7] }));
    assert.match(state, /原始形状：<code>\[2, 7\]<\/code>/);
    assert.match(state, /状态末轴宽度：7 维/);
    assert.doesNotMatch(state, /状态维度：14|原有 2 步|历史/);
    const action = render(label({ key: 'action', role: 'action', shape: [2, 16, 7] }));
    assert.match(action, /动作末轴宽度：7 维/);
    assert.doesNotMatch(action, /原有 \d+ 步/);
  });

  await test('missing fields in current metadata never fall back to facts from a previous dataset', () => {
    for (const features of [[], [{ key: 'different.state', dtype: 'float32', shape: [14] }]]) {
      const html = render(label(), features);
      assert.match(html, /当前元数据中未找到此字段/);
      assert.doesNotMatch(html, /<code>\[7\]<\/code>|状态维度：7/);
    }
  });

  await test('unknown or invalid source shapes remain visibly unknown', () => {
    for (const shape of [undefined, [], [0], [-1], [NaN]]) {
      const html = render(label({ shape, dtype: undefined }));
      assert.match(html, /原始形状：<code>未读取<\/code>/);
      assert.match(html, /状态维度：未读取/);
      assert.doesNotMatch(html, /维度：0|维度：NaN/);
    }
  });
} finally {
  await server.close();
}
