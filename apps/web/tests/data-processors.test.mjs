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
  const {
    DataProcessorPicker,
    DataProcessorCatalogStatus,
    processorDefaults,
    processorParameterErrors,
    selectBindingProcessor,
    filterDataProcessors,
  } = await server.ssrLoadModule('/src/DataProcessorPicker.tsx');
  const { DatasetBindingSummary } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
  const processor = {
    name: 'IndependentUmiFactory',
    description: '独立注册的双臂 UMI 图像、动作处理',
    available: true,
    source: { path: 'src/openpi/training/config.py', sha256: 'a'.repeat(64) },
    parameters: [
      { path: 'horizon', type: 'integer', required: true },
      { path: 'base_config.prompt_from_task', type: 'boolean', required: false, default: true },
      {
        path: 'base_config.action_sequence_keys',
        type: 'array',
        items: { type: 'string' },
        required: false,
        default: ['actions'],
      },
      { path: 'metadata', type: 'object', required: false, default: {} },
      { path: 'mode', type: 'string', required: false, default: 'umi', choices: ['umi', 'joint'] },
      { path: 'optional_dim', type: 'integer', required: false, default: null, nullable: true },
    ],
    unsupported_parameters: [
      { path: 'model_transforms', reason: 'Python 构造器，使用类默认值', required: false },
    ],
  };
  const other = {
    ...processor,
    name: 'AnotherRegisteredFactory',
    description: '机器人图像处理',
    parameters: [],
  };
  const legacy = { dataset_asset_id: 'data', norm_asset_id: 'norm', weight: 20, recipe_index: 0 };
  const slots = [{ recipe_index: 0, weight: 1, multiple: true, processing: 'CurrentRecipeOnly' }];

  await test('picker lists independent registered classes and explicit recipe fallback', () => {
    const html = renderToStaticMarkup(
      React.createElement(DataProcessorPicker, {
        binding: legacy,
        slots,
        entries: [processor, other],
        index: 0,
        onChange() {},
      }),
    );
    assert.match(html, /type="search"/);
    assert.match(html, /沿用当前配方数据项 1/);
    assert.match(html, /value="class:IndependentUmiFactory"/);
    assert.match(html, /value="class:AnotherRegisteredFactory"/);
    assert.deepEqual(
      filterDataProcessors([processor, other], '机器人').map((entry) => entry.name),
      [other.name],
    );
    assert.deepEqual(
      filterDataProcessors([processor, other], 'non-matching', processor.name).map(
        (entry) => entry.name,
      ),
      [processor.name],
    );
  });

  await test('switching processor modes drops stale indices and prior class parameters', () => {
    const selected = selectBindingProcessor(legacy, processor);
    assert.equal(Object.hasOwn(selected, 'recipe_index'), false);
    assert.equal(selected.processor.name, processor.name);
    assert.equal(selected.processor.parameters['base_config.prompt_from_task'], true);
    assert.equal(selected.weight, 20);
    assert.equal(selected.dataset_asset_id, 'data');
    const changed = selectBindingProcessor(
      { ...selected, processor: { ...selected.processor, parameters: { horizon: 50 } } },
      other,
    );
    assert.deepEqual(changed.processor.parameters, {});
    assert.deepEqual(selectBindingProcessor(changed, 0), legacy);
    assert.equal(Object.hasOwn(selectBindingProcessor(changed, 0), 'processor'), false);
  });

  await test('declared literal controls show settings and leave unsupported Python read-only', () => {
    const html = renderToStaticMarkup(
      React.createElement(DataProcessorPicker, {
        binding: {
          ...selectBindingProcessor(legacy, processor),
          processor: {
            ...processorDefaults(processor),
            parameters: { ...processorDefaults(processor).parameters, horizon: 16 },
          },
        },
        slots,
        entries: [processor],
        index: 0,
        onChange() {},
      }),
    );
    assert.match(html, /独立注册的双臂 UMI/);
    assert.match(html, /src\/openpi\/training\/config.py/);
    assert.match(html, /base_config.prompt_from_task/);
    assert.match(html, /<option value="true" selected="">True<\/option>/);
    assert.match(html, /<textarea[^>]+>\[&quot;actions&quot;\]<\/textarea>/);
    assert.match(html, /<textarea[^>]+>\{\}<\/textarea>/);
    assert.match(html, /复杂参数（只读）/);
    assert.match(html, /model_transforms/);
    assert.doesNotMatch(html, /(?:input|textarea)[^>]+(?:id|name)="[^"]*model_transforms/);
    assert.match(html, /当前为 None；改为填写值/);
  });

  await test('missing required values, bad types, removed fields and unavailable classes block preflight', () => {
    const defaults = processorDefaults(processor);
    assert.match(processorParameterErrors(defaults, processor).join(' '), /horizon/);
    assert.deepEqual(
      processorParameterErrors(
        { ...defaults, parameters: { ...defaults.parameters, horizon: 16 } },
        processor,
      ),
      [],
    );
    assert.match(
      processorParameterErrors(
        { name: processor.name, parameters: { horizon: 'sixteen', stale: true } },
        processor,
      ).join(' '),
      /horizon/,
    );
    assert.match(
      processorParameterErrors(
        { name: processor.name, parameters: { horizon: 16, mode: 'invalid' } },
        processor,
      ).join(' '),
      /mode/,
    );
    assert.ok(processorParameterErrors(defaults, undefined).length);
    assert.ok(
      processorParameterErrors(defaults, {
        ...processor,
        available: false,
        reason: 'required Python object',
      }).length,
    );
  });

  await test('nested parameter types and hidden asset paths are rejected before preflight', () => {
    assert.match(
      processorParameterErrors(
        {
          name: processor.name,
          parameters: { horizon: 16, 'base_config.action_sequence_keys': [16] },
        },
        processor,
      ).join(' '),
      /action_sequence_keys/,
    );
    assert.match(
      processorParameterErrors(
        {
          name: processor.name,
          parameters: { horizon: 16, metadata: { repo_id: '/hidden/data' } },
        },
        processor,
      ).join(' '),
      /资产绑定/,
    );
  });

  await test('run summary preserves selected processor name and parameter values', () => {
    const binding = {
      ...selectBindingProcessor(legacy, processor),
      processor: {
        name: processor.name,
        parameters: { horizon: 24, 'base_config.prompt_from_task': false,
          composition: { state_history: 2, state_fields: [{ key: 'observation.state', width: 7 }], gripper_range: null } },
      },
    };
    const html = renderToStaticMarkup(
      React.createElement(DatasetBindingSummary, { bindings: [binding], assets: [] }),
    );
    assert.match(html, /IndependentUmiFactory/);
    const serialized = html.match(/<pre[^>]*aria-label="数据处理参数 JSON"[^>]*><code>([\s\S]*?)<\/code><\/pre>/)?.[1]
      .replaceAll('&quot;', '"').replaceAll('&#x27;', "'").replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&');
    assert.ok(serialized);
    assert.deepEqual(JSON.parse(serialized), binding.processor.parameters);
    assert.match(serialized, /\n  "horizon": 24,/);
    assert.match(serialized, /\n  "composition": \{\n    "state_history": 2,/);
    assert.doesNotMatch(html, /NaN|undefined|配方数据项/);
  });

  await test('catalog failure exposes a retry button that cannot submit the training form', () => {
    const html = renderToStaticMarkup(
      React.createElement(DataProcessorCatalogStatus, {
        loading: false,
        error: new Error('旧 Agent 尚不支持数据处理器目录'),
        refresh() {},
      }),
    );
    assert.match(html, /role="alert"/);
    assert.match(html, /旧 Agent/);
    assert.match(html, /<button type="button"[^>]*>重试读取目录<\/button>/);
  });
  await test('ReCap stage owns ACP dropout instead of a hidden processor default', () => {
    const acp = {
      ...processor,
      parameters: [
        ...processor.parameters,
        { path: 'acp_dropout', type: 'number', required: false, default: 0.5 },
      ],
    };
    const managed = { acp_dropout: 0.2 };
    const binding = selectBindingProcessor(legacy, acp, managed);
    assert.equal(Object.hasOwn(binding.processor.parameters, 'acp_dropout'), false);
    assert.equal(processorDefaults(acp).parameters.acp_dropout, 0.5);
    assert.match(
      processorParameterErrors(
        { name: acp.name, parameters: { horizon: 16, acp_dropout: 0.5 } },
        acp,
        managed,
      ).join(' '),
      /阶段参数/,
    );
    const html = renderToStaticMarkup(
      React.createElement(DataProcessorPicker, {
        binding,
        slots,
        entries: [acp],
        index: 0,
        managedParameters: managed,
        onChange() {},
      }),
    );
    assert.match(html, /由本次 ReCap 阶段参数统一设置/);
    assert.match(html, /<code>0\.2<\/code>/);
    assert.doesNotMatch(html, /<input[^>]+id="[^"]*acp_dropout/);
  });
} finally {
  await server.close();
}
