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
const assets = ['fold', 'cup', 'replacement'].map((id) => ({
  id, kind: 'DATASET', node_id: 'node', name: `${id} data`, path: `/data/${id}`, version: 'v1', metadata: {},
}));
const rows = () => [
  { dataset_asset_id: 'fold', recipe_index: 0 },
  { dataset_asset_id: 'cup', processor: { name: 'RegisteredNumericProcessor', parameters: { scale: 2 } } },
];
const attributes = (tag) => Object.fromEntries(
  [...tag.matchAll(/([\w-]+)="([^"]*)"/gu)].map((match) => [match[1], match[2]]),
);
const toggles = (html) => [...html.matchAll(/<button\b[^>]*aria-expanded="(?:true|false)"[^>]*>/gu)]
  .map(([tag]) => ({ tag, ...attributes(tag) }));
const openingTag = (html, id) => [...html.matchAll(/<[^>]+>/gu)]
  .map(([tag]) => tag).find((tag) => attributes(tag).id === id);
const findElement = (element, predicate) => {
  if (!React.isValidElement(element)) return undefined;
  if (predicate(element)) return element;
  return React.Children.toArray(element.props.children)
    .map((child) => findElement(child, predicate)).find(Boolean);
};

try {
  const { NormDatasetEditor, NormDatasetCard } = await server.ssrLoadModule('/src/NormDatasetEditor.tsx');
  const { createDatasetBindingDisclosure, datasetBindingDisclosureReducer: reduce } =
    await server.ssrLoadModule('/src/datasetBindingDisclosure.ts');
  const { appendNormDataset, selectNormDataset } = await server.ssrLoadModule('/src/normStats.ts');
  const render = (props = {}) => renderToStaticMarkup(React.createElement(NormDatasetEditor, {
    rows: rows(), assets, onChange() {}, ...props,
  }));
  const gestureHarness = () => {
    let currentRows = rows();
    let disclosure = createDatasetBindingDisclosure(currentRows.length);
    const card = (index) => NormDatasetCard({
      index, dataset: assets.find((asset) => asset.id === currentRows[index].dataset_asset_id),
      processingLabel: currentRows[index].processor?.name || '配方数据项 1',
      processingPending: false, issues: [], expanded: disclosure.expanded.includes(index),
      contentId: `norm-panel-${index}`, readOnly: false, removable: currentRows.length > 1,
      onToggle(position) { disclosure = reduce(disclosure, { type: 'toggle', index: position }); },
      onRemove(position) {
        disclosure = reduce(disclosure, { type: 'remove', index: position });
        currentRows = currentRows.filter((_, row) => row !== position);
      },
      onInvalid(position) { disclosure = reduce(disclosure, { type: 'open', index: position }); },
      children: React.createElement('input', { required: true, value: currentRows[index].dataset_asset_id, readOnly: true }),
    });
    return {
      card,
      state: () => disclosure,
      rows: () => currentRows,
      add() {
        disclosure = reduce(disclosure, { type: 'append' });
        currentRows = appendNormDataset(currentRows, false);
      },
      changeDataset(index, asset) {
        currentRows = currentRows.map((row, position) => position === index ? selectNormDataset(row, asset, false) : row);
      },
    };
  };
  const toggleButton = (card) => findElement(card, (element) => element.type === 'button' && 'aria-expanded' in element.props);
  const panel = (card) => findElement(card, (element) => element.props.className === 'dataset-binding-content');
  const removeButton = (card) => findElement(card, (element) => element.type === 'button' && element.props['aria-label']?.startsWith('移除统计数据集'));

  await test('norm cards expose unique accessible disclosure controls and keep collapsed editors mounted', () => {
    const html = render();
    const controls = toggles(html);
    assert.equal(controls.length, 2);
    assert.deepEqual(controls.map((control) => control['aria-expanded']), ['false', 'true']);
    assert.equal(new Set(controls.map((control) => control['aria-controls'])).size, 2);
    for (const control of controls) {
      const content = openingTag(html, control['aria-controls']);
      assert.ok(content);
      assert.equal(/\bhidden(?:="")?(?:\s|>)/u.test(content), control['aria-expanded'] === 'false');
      assert.equal(control.type, 'button');
    }
    assert.match(html, /数据集 1/u); assert.match(html, /展开配置/u); assert.match(html, /收起配置/u);
    assert.match(html, /待统计数据集 1/u); assert.match(html, /数据处理方式 1/u);
    const firstContent = html.slice(html.indexOf(openingTag(html, controls[0]['aria-controls'])), html.indexOf('</article>'));
    assert.match(firstContent, /<select[^>]*required/u);
    assert.doesNotMatch(firstContent, /<select[^>]*disabled/u);
  });

  await test('toggle gestures open and close each norm card independently without changing dataset processing', () => {
    const harness = gestureHarness(), before = structuredClone(harness.rows());
    toggleButton(harness.card(1)).props.onClick();
    assert.deepEqual(harness.state().expanded, []);
    assert.equal(panel(harness.card(1)).props.hidden, true);
    toggleButton(harness.card(0)).props.onClick();
    assert.deepEqual(harness.state().expanded, [0]);
    assert.equal(panel(harness.card(0)).props.hidden, false);
    assert.equal(panel(harness.card(1)).props.hidden, true);
    toggleButton(harness.card(1)).props.onClick();
    assert.deepEqual(harness.state().expanded, [0, 1]);
    assert.deepEqual(harness.rows(), before);
    assert.equal(toggleButton(harness.card(0)).props['aria-controls'], panel(harness.card(0)).props.id);
  });

  await test('adding, changing and deleting datasets keeps disclosure state associated with the surviving cards', () => {
    const harness = gestureHarness();
    toggleButton(harness.card(0)).props.onClick();
    harness.add();
    assert.deepEqual(harness.state().expanded, [2]);
    assert.equal(toggleButton(harness.card(2)).props['aria-expanded'], true);
    assert.equal(harness.rows()[2].dataset_asset_id, '');
    harness.changeDataset(2, assets[2]);
    assert.deepEqual(harness.state().expanded, [2]);
    assert.match(renderToStaticMarkup(harness.card(2)), /replacement data/u);
    removeButton(harness.card(0)).props.onClick();
    assert.deepEqual(harness.rows().map((row) => row.dataset_asset_id), ['cup', 'replacement']);
    assert.deepEqual(harness.state().expanded, [1]);
    assert.equal(harness.rows()[0].processor.parameters.scale, 2);
    removeButton(harness.card(1)).props.onClick();
    assert.deepEqual(harness.state().expanded, [0]);
    assert.equal(harness.rows()[0].dataset_asset_id, 'cup');
    assert.equal(removeButton(harness.card(0)).props.disabled, true);
  });

  await test('native invalid events reopen their collapsed card while leaving all source fields in place', () => {
    const harness = gestureHarness(), before = structuredClone(harness.rows());
    assert.equal(panel(harness.card(0)).props.hidden, true);
    panel(harness.card(0)).props.onInvalidCapture();
    assert.equal(panel(harness.card(0)).props.hidden, false);
    assert.equal(panel(harness.card(1)).props.hidden, false);
    assert.deepEqual(harness.rows(), before);
  });

  await test('collapsed norm summaries retain dataset identity, processing choice and visible validation messages', () => {
    const html = render({ rows: [
      { dataset_asset_id: 'missing', processor: { name: 'UnavailableProcessor', parameters: {} } },
      rows()[1],
    ] });
    const firstArticle = html.slice(html.indexOf('<article'), html.indexOf('</article>'));
    const contentStart = firstArticle.indexOf('class="dataset-binding-content"');
    const summary = firstArticle.slice(0, contentStart);
    assert.match(summary, /尚未选择数据集/u);
    assert.match(summary, /处理方案：UnavailableProcessor/u);
    assert.match(summary, /原数据集暂不可用，请重新选择/u);
    assert.match(summary, /处理配置待完善，请展开检查/u);
    assert.equal(toggles(html)[0]['aria-expanded'], 'false');
    assert.match(html, /cup data/u);
    assert.match(html, /处理器未出现在当前环境目录中/u);
    assert.doesNotMatch(html, /选择归一化参数|混采权重|初始模型|基础模型|视觉输入与模型槽位/u);
  });

  await test('norm editors have independent IDs and retain the original read-only summary behavior', () => {
    const props = { rows: rows(), assets, onChange() {} };
    const html = renderToStaticMarkup(React.createElement(React.Fragment, null,
      React.createElement(NormDatasetEditor, props), React.createElement(NormDatasetEditor, props)));
    assert.equal(new Set(toggles(html).map((control) => control['aria-controls'])).size, 4);
    const summary = render({ readOnly: true });
    assert.equal(toggles(summary).length, 0);
    assert.doesNotMatch(summary, /添加待统计数据集|移除统计数据集/u);
    assert.match(summary, /fold data/u); assert.match(summary, /cup data/u);
    assert.match(summary, /数据处理参数 JSON/u);
    const validatedReadOnly = render({ readOnly: true, onValidityChange() {} });
    assert.equal(toggles(validatedReadOnly).length, 0);
    assert.doesNotMatch(validatedReadOnly, /dataset-binding-content[^>]*hidden/u);
  });
} finally {
  await server.close();
}
