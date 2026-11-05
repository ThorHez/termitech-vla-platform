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
const asset = {
  id: 'dataset',
  name: 'My dataset',
  kind: 'DATASET',
  state: 'REGISTERED',
  node_id: 'node',
  path: '/data/my-dataset',
  version: 'v1',
  registered_by_user_id: 'alice',
};
const user = (role = 'operator', id = 'alice') => ({ id, role, username: id, display_name: id });
const grants = {
  admin: ['assets.write', 'assets.archive', 'datasets.archive'],
  operator: ['assets.archive', 'datasets.archive'],
  viewer: [],
};
const context = (account = user(), permissions = grants[account?.role] || []) => ({
  user: account,
  can: (permission) => permissions.includes(permission),
});
const deferred = () => {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
};

try {
  const { canArchiveAsset, createAssetArchiveOperation } =
    await server.ssrLoadModule('/src/assetArchive.ts');
  const { AssetArchiveDialog } = await server.ssrLoadModule('/src/AssetArchiveDialog.tsx');
  const { AssetDetail } = await server.ssrLoadModule('/src/AssetDetail.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const transport = await server.ssrLoadModule('/src/api.ts');
  const render = (Component, props, auth = context()) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: auth },
        React.createElement(Component, props),
      ),
    );
  const allowed = (account, selected, permissions) =>
    canArchiveAsset(account, selected, context(account, permissions).can);

  await test('only the server-recorded manual registrant receives the operator archive grant for the three supported kinds', () => {
    for (const kind of ['DATASET', 'NORM', 'MODEL']) {
      assert.equal(allowed(user(), { ...asset, kind }), true);
      for (const registered_by_user_id of [undefined, null, '', 'bob'])
        assert.equal(allowed(user(), { ...asset, kind, registered_by_user_id }), false);
    }
    for (const kind of ['UNKNOWN', '', 'dataset', 'toString'])
      assert.equal(allowed(user(), { ...asset, kind }), false);
    assert.equal(
      allowed(user(), {
        ...asset,
        registered_by_user_id: null,
        metadata: { registered_by_user_id: 'alice' },
      }),
      false,
    );
    assert.equal(
      allowed(user(), { ...asset, registered_by_user_id: null, owner_user_id: 'alice' }),
      false,
    );
  });

  await test('permission alone, legacy dataset scope, missing identity and viewer role cannot claim an asset', () => {
    assert.equal(allowed(user(), asset, ['datasets.register']), false);
    assert.equal(allowed(user(), asset, ['datasets.archive']), false);
    assert.equal(allowed(user('viewer'), asset, ['assets.archive']), false);
    assert.equal(allowed(user('admin'), asset, ['assets.archive']), false);
    assert.equal(allowed(undefined, asset, ['assets.write', 'assets.archive']), false);
    assert.equal(allowed(user('operator', ''), asset, ['assets.archive']), false);
  });

  await test('administrators retain history and all asset kinds, while archived records have no delete action', () => {
    for (const kind of ['DATASET', 'MODEL', 'NORM']) {
      for (const registered_by_user_id of [undefined, null, 'bob'])
        assert.equal(allowed(user('admin'), { ...asset, kind, registered_by_user_id }), true);
    }
    for (const account of [user(), user('admin')]) {
      assert.equal(allowed(account, { ...asset, state: 'ARCHIVED' }), false);
      assert.equal(allowed(account, { ...asset, status: 'ARCHIVED' }), false);
      assert.equal(allowed(account, { ...asset, state: 'ARCHIVED', status: 'REGISTERED' }), false);
    }
  });

  await test('detail rendering offers own registration deletion without public metadata or validation editing', () => {
    const props = {
      asset,
      assets: [asset],
      close() {},
      editProfile() {},
      bindNorm() {},
      sync() {},
      remove() {},
    };
    for (const kind of ['DATASET', 'NORM', 'MODEL']) {
      const html = render(AssetDetail, { ...props, asset: { ...asset, kind } });
      assert.match(html, /删除登记/);
      assert.doesNotMatch(html, /编辑数据特征|补录数据特征|绑定 norm_stats|发起校验|重新校验/);
      for (const registered_by_user_id of [null, 'bob'])
        assert.doesNotMatch(
          render(AssetDetail, { ...props, asset: { ...asset, kind, registered_by_user_id } }),
          />删除登记</,
        );
      assert.doesNotMatch(
        render(AssetDetail, {
          ...props,
          asset: {
            ...asset,
            kind,
            registered_by_user_id: null,
            metadata: { registered_by_user_id: 'alice' },
          },
        }),
        />删除登记</,
      );
      assert.doesNotMatch(
        render(AssetDetail, { ...props, asset: { ...asset, kind } }, context(user('viewer'))),
        />删除登记</,
      );
    }
    assert.doesNotMatch(
      render(AssetDetail, { ...props, asset: { ...asset, kind: 'UNKNOWN' } }),
      />删除登记</,
    );
    assert.doesNotMatch(render(AssetDetail, props, context(user('viewer'))), />删除登记</);
  });

  await test('confirmation checks the same grant and the opening account even when the replacement is another admin', () => {
    const props = { asset, ownerUserId: 'alice', close() {}, onArchived() {} };
    const html = render(AssetArchiveDialog, props);
    assert.match(html, /确认删除登记/);
    assert.match(html, /历史训练关联和本机文件会继续保留/);
    for (const registered_by_user_id of [null, 'bob'])
      assert.equal(
        render(AssetArchiveDialog, { ...props, asset: { ...asset, registered_by_user_id } }),
        '',
      );
    assert.equal(render(AssetArchiveDialog, props, context(user('admin', 'bob'))), '');
    assert.equal(render(AssetArchiveDialog, props, context(user('viewer'))), '');
    assert.equal(render(AssetArchiveDialog, props, context(null)), '');
    assert.match(
      render(
        AssetArchiveDialog,
        { ...props, asset: { ...asset, kind: 'MODEL', registered_by_user_id: null } },
        context(user('admin')),
      ),
      /确认删除登记/,
    );
  });

  await test('a same-session confirmation archives only the requested registration and completes once', async () => {
    let request,
      completed = 0;
    globalThis.fetch = async (url, options) => {
      request = { url, options };
      return new Response('{}', { status: 200 });
    };
    for (const kind of ['DATASET', 'NORM', 'MODEL']) {
      transport.setSessionCsrf('alice-session', 'alice');
      const archive = createAssetArchiveOperation();
      await archive(`${kind}/specific`, {
        isActive: () => allowed(user(), { ...asset, kind }),
        signal: new AbortController().signal,
        onArchived: () => completed++,
      });
      assert.equal(request.url, `/api/v1/assets/${kind}%2Fspecific/archive`);
      assert.equal(request.options.method, 'POST');
      assert.equal(request.options.headers.get('X-Operation-User'), 'alice');
      assert.equal(request.options.headers.get('X-CSRF-Token'), 'alice-session');
      assert.equal(request.options.body, '{}');
    }
    assert.equal(completed, 3);
  });

  await test('an old confirmation cannot send a request using a replacement account session', async () => {
    transport.setSessionCsrf('alice-session', 'alice');
    const archive = createAssetArchiveOperation();
    transport.setSessionCsrf('bob-session', 'bob');
    let calls = 0,
      completed = 0;
    globalThis.fetch = async () => {
      calls++;
      return new Response('{}');
    };
    await assert.rejects(
      archive(asset.id, {
        isActive: () => true,
        signal: new AbortController().signal,
        onArchived: () => completed++,
      }),
      (error) => error.code === 'SESSION_CHANGED',
    );
    assert.equal(calls, 0);
    assert.equal(completed, 0);
  });

  await test('late successful deletion cannot close or refresh the replacement account', async () => {
    transport.setSessionCsrf('alice-session', 'alice');
    const archive = createAssetArchiveOperation(),
      held = deferred();
    let completed = 0;
    globalThis.fetch = async () => held.promise;
    const pending = archive(asset.id, {
      isActive: () => true,
      signal: new AbortController().signal,
      onArchived: () => completed++,
    });
    transport.setSessionCsrf('bob-session', 'bob');
    held.resolve(new Response('{}'));
    await assert.rejects(pending, (error) => error.code === 'SESSION_CHANGED');
    assert.equal(completed, 0);
  });

  await test('permission changes or a closed dialog suppress completion after an accepted response', async () => {
    transport.setSessionCsrf('alice-session', 'alice');
    const archive = createAssetArchiveOperation(),
      held = deferred();
    let active = true,
      completed = 0;
    globalThis.fetch = async () => held.promise;
    const pending = archive(asset.id, {
      isActive: () => active,
      signal: new AbortController().signal,
      onArchived: () => completed++,
    });
    active = false;
    held.resolve(new Response('{}'));
    await assert.rejects(pending, (error) => error.name === 'AbortError');
    assert.equal(completed, 0);
  });

  await test('aborted confirmations suppress both submission and late success even when fetch ignores abort', async () => {
    transport.setSessionCsrf('alice-session', 'alice');
    const archive = createAssetArchiveOperation(),
      held = deferred(),
      controller = new AbortController();
    let calls = 0,
      completed = 0;
    globalThis.fetch = async () => {
      calls++;
      return held.promise;
    };
    const pending = archive(asset.id, {
      isActive: () => true,
      signal: controller.signal,
      onArchived: () => completed++,
    });
    controller.abort();
    held.resolve(new Response('{}'));
    await assert.rejects(pending, (error) => error.name === 'AbortError');
    assert.equal(calls, 1);
    assert.equal(completed, 0);
    await assert.rejects(
      archive(asset.id, {
        isActive: () => true,
        signal: controller.signal,
        onArchived: () => completed++,
      }),
      (error) => error.name === 'AbortError',
    );
    assert.equal(calls, 1);
    assert.equal(completed, 0);
  });
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
