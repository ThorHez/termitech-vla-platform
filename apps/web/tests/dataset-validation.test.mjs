import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
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
  const helpers = await server.ssrLoadModule('/src/datasetValidationHelpers.ts');
  const { DatasetValidation, DatasetValidationBadge } = await server.ssrLoadModule(
    '/src/DatasetValidation.tsx',
  );
  const { AssetDetail } = await server.ssrLoadModule('/src/AssetDetail.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { ApiError, setSessionCsrf } = await server.ssrLoadModule('/src/api.ts');
  const node = { id: 'source', name: '训练机', hostname: 'source', status: 'ONLINE' };
  const profile = {
    id: 'native',
    node_id: node.id,
    name: 'OpenPI 环境',
    adapter: 'openpi-sft',
    execution_mode: 'native',
    active: true,
  };
  const asset = {
    id: 'dataset',
    node_id: node.id,
    kind: 'DATASET',
    name: '叠衣数据',
    path: '/data/cloth',
    version: 'v1',
    state: 'REGISTERED',
  };
  const report = (status, changes = {}) => ({
    id: 'validation',
    status,
    scope: 'dataset_source',
    profile_id: profile.id,
    requested_at: '2026-11-02T01:00:00Z',
    completed_at: status === 'PENDING' ? null : '2026-11-02T01:00:10Z',
    checks: [],
    ...changes,
  });
  const render = (element, writable = true, auth = {}) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: (permission) => writable && permission === 'assets.write', ...auth } },
        element,
      ),
    );
  const panel = (validation, options = {}) =>
    render(
      React.createElement(DatasetValidation, {
        asset: { ...asset, validation, ...options.asset },
        node: options.node || node,
        profiles: options.profiles || [profile],
        changed() {},
      }),
      options.writable !== false,
      options.auth,
    );
  const mutationButton = (html) =>
    [...html.matchAll(/<button\b([^>]*)>([\s\S]*?)<\/button>/g)]
      .map((match) => ({ attributes: match[1], text: match[2].replace(/<[^>]*>/g, '') }))
      .find((button) => /重新校验|开始基础校验/.test(button.text));

  await test('source validation states have explicit limited-scope labels, including old and unknown records', () => {
    const labels = {
      PENDING: '待校验 / 后台校验中',
      SUCCEEDED: '基础校验通过',
      FAILED: '基础校验失败',
      CANCELLED: '基础校验已取消',
      STALE: '需重验',
      BLOCKED: '暂无法校验',
    };
    for (const [status, label] of Object.entries(labels))
      assert.equal(helpers.datasetValidationPresentation(report(status)).label, label);
    assert.equal(helpers.datasetValidationPresentation(null).label, '尚未校验');
    assert.notEqual(
      helpers.datasetValidationPresentation(report('SUCCEEDED', { scope: 'other' })).state,
      'VALID',
    );
    assert.notEqual(helpers.datasetValidationPresentation(report('OTHER')).state, 'VALID');
    assert.match(renderToStaticMarkup(React.createElement(DatasetValidationBadge)), /尚未校验/);
  });

  await test('reports show checks and blocked reasons without claiming training configurations are verified', () => {
    const success = panel(report('SUCCEEDED'));
    assert.match(success, /基础校验通过/);
    assert.match(success, /相机映射、状态拼接等训练组合配置仍需在训练向导中校验/);
    assert.match(success, /最近完成/);
    const failed = panel(
      report('FAILED', {
        error: { code: 'DATASET_BAD', message: '时间戳不是有限数值' },
        checks: [{ name: 'timestamps', ok: false, message: '第 3 帧时间戳无效' }],
      }),
    );
    assert.match(failed, /DATASET_BAD.*时间戳不是有限数值/);
    assert.match(failed, /第 3 帧时间戳无效/);
    assert.match(failed, /<details[^>]* open=""/);
    assert.match(
      panel(report('BLOCKED', { error: { code: 'PROFILE_REQUIRED', message: '请选择校验环境' } })),
      /PROFILE_REQUIRED.*请选择校验环境/,
    );
    assert.match(panel(report('STALE')), /数据或配置已变化，请重新校验/);
  });

  await test('permissions, pending work and archived assets guard validation actions while preserving reports', () => {
    assert.ok(mutationButton(panel(report('SUCCEEDED'))));
    assert.doesNotMatch(mutationButton(panel(report('SUCCEEDED'))).attributes, /disabled/);
    assert.match(mutationButton(panel(report('PENDING'))).attributes, /disabled/);
    assert.match(panel(report('PENDING')), /已有后台校验任务，请等待结果/);
    assert.equal(mutationButton(panel(report('FAILED'), { writable: false })), undefined);
    assert.match(panel(report('FAILED'), { writable: false }), /基础校验失败/);
    assert.equal(
      mutationButton(panel(report('SUCCEEDED'), { asset: { state: 'ARCHIVED' } })),
      undefined,
    );
    assert.match(
      mutationButton(panel(null, { node: { ...node, status: 'OFFLINE' } })).attributes,
      /disabled/,
    );
    assert.ok(mutationButton(panel(null)));
    assert.match(panel(null), /尚未校验/);
  });

  await test('operators can retry their own registered datasets while others remain readable', () => {
    const auth = { user: { id: 'alice' }, can: (permission) => permission === 'datasets.validate' };
    const owned = { registered_by_user_id: 'alice' };
    assert.doesNotMatch(
      mutationButton(panel(report('SUCCEEDED'), { asset: owned, auth })).attributes,
      /disabled/,
    );
    assert.match(
      mutationButton(panel(report('PENDING'), { asset: owned, auth })).attributes,
      /disabled/,
    );
    for (const registrar of ['bob', null, undefined]) {
      const html = panel(report('SUCCEEDED'), { asset: { registered_by_user_id: registrar }, auth });
      assert.equal(mutationButton(html), undefined);
      assert.match(html, /基础校验通过/);
    }
    assert.equal(
      mutationButton(panel(report('SUCCEEDED'), { asset: owned, auth: { ...auth, can: () => false } })),
      undefined,
    );
  });

  await test('cancelled validation is neutral, retains its receipt and permits an authorized retry', () => {
    const cancelled = report('CANCELLED');
    assert.deepEqual(helpers.datasetValidationPresentation(cancelled), {
      state: 'CANCELLED',
      label: '基础校验已取消',
    });
    const html = panel(cancelled);
    assert.match(html, /基础校验已取消/);
    assert.match(html, /本次基础校验已取消，可重新提交/);
    assert.match(html, /最近取消/);
    assert.match(html, /校验环境：OpenPI 环境/);
    assert.doesNotMatch(html, /最近完成|等待 Agent 返回结果|已有后台校验任务/);
    assert.doesNotMatch(mutationButton(html).attributes, /disabled/);
    assert.equal(mutationButton(panel(cancelled, { writable: false })), undefined);
    assert.match(panel(cancelled, { writable: false }), /基础校验已取消/);
    assert.match(
      mutationButton(panel(cancelled, { node: { ...node, status: 'OFFLINE' } })).attributes,
      /disabled/,
    );
  });

  await test('validation environment selection excludes foreign, container and inactive profiles', () => {
    const profiles = [
      profile,
      { ...profile, id: 'foreign', node_id: 'other' },
      { ...profile, id: 'container', execution_mode: 'docker' },
      { ...profile, id: 'smoke', adapter: 'smoke' },
      { ...profile, id: 'inactive', active: false },
    ];
    assert.deepEqual(
      helpers.datasetValidationProfiles(profiles, node.id).map((item) => item.id),
      [profile.id],
    );
    assert.match(
      mutationButton(panel(null, { profiles: [profile, { ...profile, id: 'second' }] })).attributes,
      /disabled/,
    );
    assert.match(
      panel(null, { profiles: [profile, { ...profile, id: 'second' }] }),
      /请选择本次校验使用的训练环境/,
    );
    assert.match(panel(null, { profiles: [] }), /请先为此机器配置原生 OpenPI/);
  });

  await test('a fresh submission cannot be replaced by an older GET, and completed or stale receipts stay authoritative', () => {
    const old = report('SUCCEEDED');
    const pending = report('PENDING', { id: 'new', requested_at: '2026-11-02T02:00:00Z' });
    assert.equal(helpers.currentDatasetValidation(old, old, pending), pending);
    const complete = report('SUCCEEDED', {
      ...pending,
      status: 'SUCCEEDED',
      completed_at: '2026-11-02T02:00:10Z',
    });
    assert.equal(helpers.currentDatasetValidation(old, complete, pending), complete);
    const stale = { ...complete, status: 'STALE' };
    assert.equal(helpers.currentDatasetValidation(stale, complete, pending), stale);
    const cancelled = { ...pending, status: 'CANCELLED' };
    assert.equal(helpers.currentDatasetValidation(old, cancelled, pending), cancelled);
    assert.equal(helpers.currentDatasetValidation(cancelled, pending), cancelled);
    assert.equal(helpers.currentDatasetValidation(old, null), undefined);
  });

  await test('asset details integrate validation for datasets without adding mutation controls for viewers', () => {
    const props = {
      asset: { ...asset, validation: report('FAILED') },
      node,
      profiles: [profile],
      assets: [asset],
      close() {},
      editProfile() {},
      bindNorm() {},
      sync() {},
      remove() {},
    };
    assert.match(render(React.createElement(AssetDetail, props)), /数据集基础校验/);
    assert.match(render(React.createElement(AssetDetail, props)), /已登记/);
    assert.equal(mutationButton(render(React.createElement(AssetDetail, props), false)), undefined);
    assert.doesNotMatch(
      render(React.createElement(AssetDetail, { ...props, asset: { ...asset, kind: 'MODEL' } })),
      /aria-label="数据集基础校验"/,
    );
  });

  await test('report reads preserve empty legacy records and validation requests use encoded scoped endpoints', async () => {
    const paths = [];
    setSessionCsrf('test-csrf');
    globalThis.fetch = async (path, options) => {
      paths.push(path);
      if (path.endsWith('/validation')) return new Response('null', { status: 200 });
      assert.equal(options.method, 'POST');
      assert.equal(options.headers.get('X-CSRF-Token'), 'test-csrf');
      assert.deepEqual(JSON.parse(options.body), { profile_id: profile.id });
      return new Response(JSON.stringify(report('PENDING')), { status: 202 });
    };
    assert.equal(await helpers.readDatasetValidation('dataset/一'), null);
    assert.equal(
      (await helpers.requestDatasetValidation('dataset/一', profile.id)).status,
      'PENDING',
    );
    assert.deepEqual(paths, [
      '/api/v1/assets/dataset%2F%E4%B8%80/validation',
      '/api/v1/assets/dataset%2F%E4%B8%80/validations',
    ]);
    setSessionCsrf();
  });

  await test('a blocked 202 remains a report while rejected requests remain API errors', async () => {
    globalThis.fetch = async (_path, options) => {
      assert.deepEqual(JSON.parse(options.body), {});
      return new Response(
        JSON.stringify(
          report('BLOCKED', {
            id: null,
            error: { code: 'PROFILE_REQUIRED', message: '需选择环境' },
          }),
        ),
        { status: 202 },
      );
    };
    assert.equal((await helpers.requestDatasetValidation(asset.id)).status, 'BLOCKED');
    globalThis.fetch = async () =>
      new Response(
        JSON.stringify({ error: { code: 'PROFILE_INVALID', message: '环境不属于该机器' } }),
        { status: 422 },
      );
    await assert.rejects(
      helpers.requestDatasetValidation(asset.id, 'foreign'),
      (error) =>
        error instanceof ApiError && error.status === 422 && error.code === 'PROFILE_INVALID',
    );
  });
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
