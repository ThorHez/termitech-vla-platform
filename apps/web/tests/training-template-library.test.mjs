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
const originalFetch = globalThis.fetch;

try {
  const { setSessionCsrf, ApiError } = await server.ssrLoadModule('/src/api.ts');
  const { TrainingTemplateLibrary, createTrainingTemplateClient } = await server.ssrLoadModule(
    '/src/TrainingTemplateLibrary.tsx',
  );
  const snapshot = {
    request: {
      name: '叠衣训练',
      node_id: '',
      profile_id: '',
      template_id: 'openpi-sft',
      inputs: {},
      parameters: {},
      gpu_count: 1,
    },
    step: 0,
    advancedRecipe: false,
    editedParameters: [],
    editedStageParameters: [],
    entrypointEdited: false,
  };
  const record = {
    id: 'template-a',
    name: '双臂叠衣',
    description: '从已有配置开始',
    revision: 3,
    created_at: '2026-10-27T00:00:00Z',
    updated_at: '2026-10-27T01:00:00Z',
    content: { schema_version: 1 },
  };
  const response = (value, status = 200) =>
    new Response(JSON.stringify(value), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });

  await test('personal library uses explicit configuration actions and never submits its containing wizard', () => {
    const html = renderToStaticMarkup(
      React.createElement(TrainingTemplateLibrary, {
        userId: 'user-a',
        snapshot,
        onApply() {},
      }),
    );
    assert.match(html, /aria-label="我的训练模板"/);
    assert.match(html, /跨浏览器和设备恢复/);
    assert.match(html, /保存模板/);
    assert.match(html, /同名覆盖更新，新名称另存/);
    assert.match(html, /模板名称/);
    assert.match(html, /模板说明/);
    assert.match(html, /value="叠衣训练"/);
    assert.match(html, /重新预检及确认生成的 config/);
    assert.doesNotMatch(html, /<form|type="submit"|开始训练|access_token|csrf_token/);
    for (const button of html.match(/<button\b[^>]*>/g)) {
      assert.match(button, /type="button"/);
      assert.match(button, /disabled=""/);
    }
  });

  await test('template API carries account assertions, session cookies, CSRF and exact update/delete revisions', async () => {
    setSessionCsrf('csrf-library');
    const calls = [];
    globalThis.fetch = async (url, options) => {
      calls.push({ url, options });
      if (options.method === 'DELETE') return response({ id: record.id, deleted: true });
      if (!options.method && url.endsWith('/training-templates')) {
        const { content, ...metadata } = record;
        assert.ok(content);
        return response([metadata]);
      }
      return response(record, options.method === 'POST' ? 201 : 200);
    };
    const client = createTrainingTemplateClient('user-a');
    const body = { name: record.name, description: record.description, content: record.content };
    try {
      const rows = await client.list();
      assert.equal(rows.length, 1);
      assert.equal(Object.hasOwn(rows[0], 'content'), false);
      assert.deepEqual(await client.get(record.id), record);
      assert.deepEqual(await client.create(body), record);
      assert.deepEqual(await client.update(record.id, body, 3), record);
      assert.deepEqual(await client.remove(record.id, 3), { id: record.id, deleted: true });
      assert.deepEqual(
        calls.map(({ url, options }) => [url, options.method || 'GET']),
        [
          ['/api/v1/training-templates', 'GET'],
          ['/api/v1/training-templates/template-a', 'GET'],
          ['/api/v1/training-templates', 'POST'],
          ['/api/v1/training-templates/template-a', 'PUT'],
          ['/api/v1/training-templates/template-a', 'DELETE'],
        ],
      );
      for (const { options } of calls) {
        assert.equal(options.credentials, 'same-origin');
        assert.equal(options.headers.get('X-Training-Template-User'), 'user-a');
        assert.equal(options.headers.get('Authorization'), null);
        assert.ok(options.signal instanceof AbortSignal);
        assert.equal(options.headers.get('X-CSRF-Token'), options.method ? 'csrf-library' : null);
      }
      assert.deepEqual(JSON.parse(calls[2].options.body), body);
      assert.deepEqual(JSON.parse(calls[3].options.body), { ...body, expected_revision: 3 });
      assert.deepEqual(JSON.parse(calls[4].options.body), { expected_revision: 3 });
    } finally {
      client.dispose();
      setSessionCsrf();
    }
  });

  await test('account change or unmount cancels in-flight work and discards late responses even if fetch ignores abort', async () => {
    const calls = [];
    globalThis.fetch = (url, options) =>
      new Promise((resolve) => calls.push({ url, options, resolve }));
    const previous = createTrainingTemplateClient('previous-user');
    const pending = previous.get(record.id);
    previous.dispose();
    const rejection = assert.rejects(pending, (error) => error.name === 'AbortError');
    assert.equal(calls[0].options.signal.aborted, true);
    calls[0].resolve(response(record));
    await rejection;
    const current = createTrainingTemplateClient('current-user');
    const fresh = current.list();
    assert.equal(calls[1].options.headers.get('X-Training-Template-User'), 'current-user');
    calls[1].resolve(response([]));
    assert.deepEqual(await fresh, []);
    const count = calls.length;
    await assert.rejects(previous.list(), (error) => error.name === 'AbortError');
    assert.equal(calls.length, count);
    current.dispose();
  });

  await test('a revision conflict remains an explicit error without retrying a write or silently refreshing its version', async () => {
    let calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      return response(
        {
          error: { code: 'TRAINING_TEMPLATE_CONFLICT', message: '模板已被其他页面修改，请刷新。' },
        },
        409,
      );
    };
    const client = createTrainingTemplateClient('user-a');
    try {
      await assert.rejects(
        client.update(
          record.id,
          {
            name: record.name,
            description: record.description,
            content: record.content,
          },
          3,
        ),
        (error) => error instanceof ApiError && error.code === 'TRAINING_TEMPLATE_CONFLICT',
      );
      assert.equal(calls, 1);
    } finally {
      client.dispose();
    }
  });
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
