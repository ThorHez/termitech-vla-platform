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

const asset = (id, kind, name) => ({
  id, kind, name, node_id: 'node', path: `/machine/${id}`, version: 'v1',
  state: 'REGISTERED', created_at: '2026-11-01T00:00:00Z', metadata: {},
});
const assets = [
  asset('fold-data', 'DATASET', '叠衣数据'), asset('cup-data', 'DATASET', '取杯数据'),
  asset('fold-norm', 'NORM', '叠衣归一化'), asset('cup-norm', 'NORM', '取杯归一化'),
];
const bindings = () => [
  { dataset_asset_id: 'fold-data', norm_asset_id: 'fold-norm', weight: 3, recipe_index: 2 },
  { dataset_asset_id: 'cup-data', norm_asset_id: 'cup-norm', weight: 1,
    processor: { name: 'ActualRegisteredProcessor', parameters: { normalize: false } } },
];
const slots = [{ recipe_index: 2, weight: 1, processing: 'ActualRecipeFactory', multiple: true }];
const noop = () => {};
const attributes = (tag) => Object.fromEntries(
  [...tag.matchAll(/([\w-]+)="([^"]*)"/gu)].map((match) => [match[1], match[2]]),
);
const toggles = (html) => [...html.matchAll(/<button\b[^>]*aria-expanded="(?:true|false)"[^>]*>/gu)]
  .map(([tag]) => ({ tag, ...attributes(tag) }));
const openingTag = (html, id) => [...html.matchAll(/<[^>]+>/gu)]
  .map(([tag]) => tag).find((tag) => attributes(tag).id === id);

try {
  const { DatasetBindings } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
  const { createDatasetBindingDisclosure, datasetBindingDisclosureReducer: reduce } =
    await server.ssrLoadModule('/src/datasetBindingDisclosure.ts');
  const render = (props = {}) => renderToStaticMarkup(React.createElement(DatasetBindings, {
    bindings: bindings(), slots, assets, onChange: noop, ...props,
  }));

  await test('initialization opens the newest group and adding closes old groups', () => {
    assert.deepEqual(createDatasetBindingDisclosure(0), { count: 0, expanded: [] });
    const current = createDatasetBindingDisclosure(3);
    assert.deepEqual(current, { count: 3, expanded: [2] });
    const multipleOpen = reduce(current, { type: 'open', index: 0 });
    assert.deepEqual(reduce(multipleOpen, { type: 'append' }), { count: 4, expanded: [3] });
    assert.deepEqual(current, { count: 3, expanded: [2] });
  });

  await test('manual toggles can close all cards or open several independently', () => {
    let state = createDatasetBindingDisclosure(3);
    state = reduce(state, { type: 'toggle', index: 2 });
    assert.deepEqual(state.expanded, []);
    state = reduce(state, { type: 'toggle', index: 0 });
    state = reduce(state, { type: 'toggle', index: 1 });
    assert.deepEqual([...state.expanded].sort(), [0, 1]);
    state = reduce(state, { type: 'toggle', index: 0 });
    assert.deepEqual(state.expanded, [1]);
    assert.deepEqual(reduce(state, { type: 'open', index: 1 }).expanded, [1]);
  });

  await test('removing first or middle cards keeps the same surviving expanded groups', () => {
    const state = { count: 4, expanded: [0, 2, 3] };
    assert.deepEqual(reduce(state, { type: 'remove', index: 0 }), { count: 3, expanded: [1, 2] });
    assert.deepEqual(reduce(state, { type: 'remove', index: 1 }), { count: 3, expanded: [0, 1, 2] });
    assert.deepEqual(reduce(state, { type: 'remove', index: 2 }), { count: 3, expanded: [0, 2] });
    assert.deepEqual(state, { count: 4, expanded: [0, 2, 3] });
  });

  await test('removing the last expanded card opens its neighbor without reopening deliberately closed cards', () => {
    assert.deepEqual(reduce({ count: 4, expanded: [3] }, { type: 'remove', index: 3 }),
      { count: 3, expanded: [2] });
    assert.deepEqual(reduce({ count: 4, expanded: [] }, { type: 'remove', index: 1 }),
      { count: 3, expanded: [] });
    const removedMiddle = reduce({ count: 4, expanded: [1] }, { type: 'remove', index: 1 });
    assert.equal(removedMiddle.count, 3);
    assert.equal(removedMiddle.expanded.length, 1);
    assert.ok(removedMiddle.expanded[0] >= 0 && removedMiddle.expanded[0] < 3);
  });

  await test('external draft growth opens its latest row and shrink clamps stale disclosure targets', () => {
    assert.deepEqual(reduce({ count: 2, expanded: [0, 1] }, { type: 'sync', count: 4 }),
      { count: 4, expanded: [3] });
    const smaller = reduce({ count: 4, expanded: [0, 3] }, { type: 'sync', count: 2 });
    assert.equal(smaller.count, 2);
    assert.deepEqual(smaller.expanded, [0, 1]);
    assert.deepEqual(reduce({ count: 2, expanded: [1] }, { type: 'sync', count: 0 }),
      { count: 0, expanded: [] });
  });

  await test('each dataset has a unique disclosure control paired with its editor panel', () => {
    const html = render({ showProcessing: false });
    const controls = toggles(html);
    assert.equal(controls.length, 2);
    assert.equal(new Set(controls.map((control) => control['aria-controls'])).size, 2);
    assert.ok(controls.every((control) => control.type === 'button' && control['aria-controls']));
    for (const control of controls) {
      const panel = openingTag(html, control['aria-controls']);
      assert.ok(panel, 'every aria-controls must identify an existing editor panel');
      assert.equal(/\bhidden(?:="")?(?:\s|>)/u.test(panel), control['aria-expanded'] === 'false');
    }
    assert.equal(controls.filter((control) => control['aria-expanded'] === 'true').length, 1);
  });

  await test('two binding editors on the same page do not reuse disclosure panel IDs', () => {
    const props = { bindings: bindings(), slots, assets, showProcessing: false, onChange: noop };
    const html = renderToStaticMarkup(React.createElement(React.Fragment, null,
      React.createElement(DatasetBindings, props), React.createElement(DatasetBindings, props)));
    const controls = toggles(html);
    assert.equal(controls.length, 4);
    assert.equal(new Set(controls.map((control) => control['aria-controls'])).size, 4);
  });

  await test('collapsed editors retain dataset processing controls and validation errors in the DOM', () => {
    const html = render({ bindings: bindings().toReversed() });
    const controls = toggles(html);
    assert.equal(controls.length, 2);
    assert.ok(controls.some((control) => control['aria-expanded'] === 'false'));
    assert.equal([...html.matchAll(/aria-label="数据处理方式 [12]"/gu)].length, 2);
    assert.match(html, /处理器未出现在当前环境目录中/u);
    assert.match(html, /ActualRegisteredProcessor/u);
    const hiddenControl = controls.find((control) => control['aria-expanded'] === 'false');
    const hiddenStart = html.indexOf(openingTag(html, hiddenControl['aria-controls']));
    const nextControl = controls.find((control) => control['aria-expanded'] === 'true');
    const nextStart = html.indexOf(openingTag(html, nextControl['aria-controls']));
    const hiddenPanel = html.slice(hiddenStart, nextStart > hiddenStart ? nextStart : undefined);
    assert.match(hiddenPanel, /ActualRegisteredProcessor/u);
    assert.match(hiddenPanel, /处理器未出现在当前环境目录中/u);
    const requiredInputs = [...hiddenPanel.matchAll(/<(?:select|input)\b[^>]*>/gu)]
      .map(([tag]) => tag).filter((tag) => /\brequired(?:="")?(?:\s|>)/u.test(tag));
    assert.ok(requiredInputs.length >= 3, 'collapsed dataset, norm and weight inputs retain native validation');
    assert.ok(requiredInputs.every((tag) => !/\bdisabled(?:="")?(?:\s|>)/u.test(tag)));
  });

  await test('card summaries identify the actual dataset, norm, weight share and processing choice', () => {
    const html = render({ showProcessing: false });
    assert.match(html, /叠衣数据/u);
    assert.match(html, /取杯数据/u);
    assert.match(html, /叠衣归一化/u);
    assert.match(html, /取杯归一化/u);
    assert.match(html, /ActualRecipeFactory/u);
    assert.match(html, /ActualRegisteredProcessor/u);
    assert.match(html, /75(?:\.0+)?%/u);
    assert.match(html, /25(?:\.0+)?%/u);
    assert.match(html, /权重占比（未考虑样本数）约/u);
    assert.doesNotMatch(html, /采样比例约/u);
  });

  await test('collapsed summaries still call out missing asset selections and invalid weight', () => {
    const html = render({ showProcessing: false, bindings: [
      { dataset_asset_id: '', norm_asset_id: '', weight: NaN, recipe_index: 2 },
      bindings()[1],
    ] });
    assert.match(html, /未选择数据集|待选择数据集|请选择数据集/u);
    assert.match(html, /未选择归一化|待选择归一化|请选择归一化/u);
    assert.match(html, /权重.*(?:无效|大于|待)|(?:检查|填写).*权重/u);
    assert.equal(toggles(html)[0]['aria-expanded'], 'false');
  });

  await test('processor validation status remains visible even when its card starts collapsed', () => {
    const html = render({ bindings: bindings().toReversed() });
    assert.match(html, /处理配置待完善|处理配置校验中/u);
    assert.equal(toggles(html)[0]['aria-expanded'], 'false');
  });

  await test('read-only review keeps disclosures while hiding add and remove mutations', () => {
    const html = render({ readOnly: true, onEditProcessing: noop });
    assert.equal(toggles(html).length, 2);
    assert.doesNotMatch(html, /添加数据集|aria-label="移除数据集/u);
    assert.match(html, /返回第 1 步修改数据处理/u);
  });
} finally {
  await server.close();
}
