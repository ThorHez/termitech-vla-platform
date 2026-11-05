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
const alice = '11111111-1111-4111-8111-111111111111';
const bob = '22222222-2222-4222-8222-222222222222';
const event = (changes = {}) => ({
  id: 'event-1',
  created_at: '2026-11-05T10:00:00Z',
  actor_id: alice,
  actor_username: 'alice',
  action: 'asset.register',
  method: 'POST',
  path: '/api/v1/assets',
  status: 201,
  remote_addr: '192.0.2.1',
  details: {
    operation_label: '登记资产',
    target_type: 'asset',
    target_id: 'asset-1',
    target_name: 'My dataset',
    asset_kind: 'DATASET',
    request_id: 'request-1',
    duration_ms: 20,
    created_count: 1,
    skipped_count: 0,
  },
  ...changes,
});
const response = (items, next_cursor = null, has_more = false) =>
  new Response(JSON.stringify({ items, next_cursor, has_more }));
const deferred = () => {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
};
try {
  const presentation = await server.ssrLoadModule('/src/auditLogPresentation.ts');
  const { createAuditEventsClient, firstAuditPage, nextAuditPage, previousAuditPage } =
    await server.ssrLoadModule('/src/auditLogData.ts');
  const { AuditLog, AuditLogTable } = await server.ssrLoadModule('/src/AuditLog.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const transport = await server.ssrLoadModule('/src/api.ts');
  const {
    emptyAuditFilters,
    submitAuditFilters,
    auditEventsPath,
    auditOutcome,
    auditOperationLabel,
    safeAuditDetails,
    auditDetailRows,
    auditActionGroups,
  } = presentation;
  const render = (Component, props, role = 'admin', allowed = true) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: {
            user: { id: alice, role },
            can: (permission) => allowed && permission === 'audit.read',
          },
        },
        React.createElement(Component, props),
      ),
    );

  await test('only an authenticated audit administrator sees the independent panel and disabled-user filter', () => {
    const props = {
      users: [{ id: bob, username: 'bob', display_name: '已离职成员', enabled: false }],
    };
    const html = render(AuditLog, props);
    for (const label of ['操作记录', '操作账号', '操作类型', '关键词', '开始时间', '结束时间'])
      assert.match(html, new RegExp(label));
    assert.doesNotMatch(html, /placeholder="例如 asset.register"|aria-label="自定义操作标识"/);
    assert.match(html, /bob（已停用）/);
    assert.match(html, /每页 50 条/);
    assert.match(html, /首页每 15 秒刷新/);
    assert.doesNotMatch(html, /type="checkbox"[^>]*checked/);
    for (const role of ['operator', 'viewer']) assert.equal(render(AuditLog, props, role), '');
    assert.equal(render(AuditLog, props, 'admin', false), '');
    assert.equal(renderToStaticMarkup(React.createElement(AuditLog, props)), '');
  });

  await test('Chinese action choices use exact server identifiers and technical detail enums have readable labels', () => {
    const actions = auditActionGroups.flatMap((group) => Object.entries(group.actions));
    assert.equal(new Set(actions.map(([value]) => value)).size, actions.length);
    for (const [action, label] of [
      ['assets.register', '登记资产'],
      ['runs.cancel', '取消训练任务'],
      ['datasets.register_batch', '批量登记数据集'],
      ['transfers.create', '创建资产同步任务'],
      ['deployments.start', '启动推理服务'],
      ['auth.login_failed', '登录失败'],
      ['users.update', '修改用户信息或权限'],
    ])
      assert.equal(actions.find(([value]) => value === action)?.[1], label);
    for (const [category, target, kind, expectedCategory, expectedTarget, expectedKind] of [
      ['assets', 'asset', 'DATASET', '资产', '资产', '数据集'],
      ['training', 'run', 'NORM', '训练', '训练任务', '归一化统计（norm_stats）'],
      ['machines', 'node', 'MODEL', '机器', '机器', '模型'],
    ]) {
      const rows = auditDetailRows(
        event({ details: { category, target_type: target, asset_kind: kind } }),
      );
      assert.deepEqual(
        rows.map((row) => row.value),
        [expectedCategory, expectedTarget, expectedKind],
      );
      assert.deepEqual(
        rows.map((row) => row.identifier),
        [category, target, kind],
      );
    }
    assert.equal(
      auditDetailRows(event({ details: { target_type: 'future_resource' } }))[0].value,
      'future_resource',
    );
  });

  await test('result labels distinguish asynchronous acceptance and denials from execution success', () => {
    for (const [status, expected] of [
      [200, 'succeeded'],
      [201, 'succeeded'],
      [202, 'accepted'],
      [204, 'succeeded'],
      [401, 'denied'],
      [403, 'denied'],
      [404, 'failed'],
      [409, 'failed'],
      [422, 'failed'],
      [500, 'failed'],
    ])
      assert.equal(auditOutcome(status), expected);
    const html = render(AuditLogTable, {
      items: [
        event({ id: 'a', status: 202 }),
        event({ id: 'b', status: 403 }),
        event({ id: 'c', status: 409 }),
      ],
    });
    assert.match(html, /已受理 · 202/);
    assert.match(html, /拒绝 · 403/);
    assert.match(html, /失败 · 409/);
  });

  await test('semantic labels, target names and safe details render as escaped text without raw sensitive values', () => {
    const entry = event({
      details: {
        operation_label: '登记资产',
        target_name: '<script>never execute</script>',
        target_id: 'asset-1',
        request_id: 'safe-request',
        duration_ms: 1.5,
        created_count: 2,
        password: 'private-password',
        token: 'private-token',
        csrf_token: 'private-csrf',
        body: { secret: 'private-body' },
        query: 'private-query',
        terminal_input: 'private-terminal',
      },
    });
    const html = render(AuditLogTable, { items: [entry] });
    assert.match(html, /&lt;script&gt;never execute&lt;\/script&gt;/);
    assert.match(html, /safe-request|1.5 ms|新增数量/);
    assert.doesNotMatch(
      html,
      /<script>|private-password|private-token|private-csrf|private-body|private-query|private-terminal/,
    );
    assert.deepEqual(
      Object.keys(safeAuditDetails(entry.details)).sort(),
      [
        'operation_label',
        'target_name',
        'target_id',
        'request_id',
        'duration_ms',
        'created_count',
      ].sort(),
    );
  });

  await test('unattributed authentication attempts never impersonate an existing account', () => {
    const html = render(AuditLogTable, {
      items: [
        event({
          actor_id: null,
          actor_username: 'legacy-user',
          action: 'auth.login_failed',
          status: 401,
          details: { attempted_username: 'alice' },
        }),
      ],
      users: [{ id: alice, username: 'alice', display_name: '已认证 Alice', enabled: true }],
    });
    assert.match(html, /未归属账号/);
    assert.match(html, /尝试账号：alice/);
    assert.doesNotMatch(html, /已认证 Alice/);
    assert.doesNotMatch(html, /legacy-user/);
    const authenticated = render(AuditLogTable, {
      items: [
        event({
          actor_id: alice,
          actor_username: 'alice',
          status: 401,
          details: { attempted_username: 'bob' },
        }),
      ],
      users: [{ id: alice, username: 'alice', display_name: '已认证 Alice', enabled: true }],
    });
    assert.match(authenticated, /已认证 Alice/);
    assert.match(authenticated, /尝试账号/);
    assert.match(authenticated, />bob</);
    assert.doesNotMatch(authenticated, /未归属账号|尝试账号：bob/);
  });

  await test('legacy API records receive operation fallbacks and query strings never reach display', () => {
    const legacy = event({
      action: 'api.mutation',
      path: '/api/v1/assets/asset-1/archive?token=old-query#secret',
      details: {},
    });
    assert.equal(auditOperationLabel(legacy), '删除资产登记');
    const html = render(AuditLogTable, { items: [legacy] });
    assert.match(html, /POST \/api\/v1\/assets\/asset-1\/archive/);
    assert.doesNotMatch(html, /old-query|#secret|token=/);
    assert.equal(
      auditOperationLabel(event({ action: 'users.reset_password', details: null })),
      '重置用户密码',
    );
    assert.equal(
      auditOperationLabel(
        event({
          action: 'api.mutation',
          method: 'PATCH',
          path: '/api/v1/new-resource',
          details: null,
        }),
      ),
      '修改请求',
    );
  });

  await test('only scalar whitelisted details survive and credential-shaped strings are hidden defensively', () => {
    const safe = safeAuditDetails({
      request_id: 'Bearer private-value',
      target_name: 'token=private-value',
      duration_ms: -1,
      created_count: 2.5,
      skipped_count: Infinity,
      target_id: { raw: 'no' },
      enabled: true,
    });
    assert.deepEqual(safe, { request_id: '[已隐藏]', target_name: '[已隐藏]', enabled: true });
    assert.ok(
      auditDetailRows(event({ details: safe })).every(
        (row) => !row.value.includes('private-value'),
      ),
    );
  });

  await test('filters validate identities and date ordering then encode exact UTC query values', () => {
    const filters = submitAuditFilters({
      ...emptyAuditFilters(),
      actor_id: alice,
      action: 'users.update',
      outcome: 'denied',
      q: ' asset + name ',
      from_time: '2026-11-05T10:00:00+08:00',
      to_time: '2026-11-05T12:00:00+08:00',
    });
    const url = new URL(auditEventsPath(filters, 'opaque+signed/=='), 'https://fixture.invalid');
    assert.equal(url.pathname, '/audit/events');
    assert.equal(url.searchParams.get('limit'), '50');
    assert.equal(url.searchParams.get('from_time'), '2026-11-05T02:00:00.000Z');
    assert.equal(url.searchParams.get('q'), 'asset + name');
    assert.equal(url.searchParams.get('cursor'), 'opaque+signed/==');
    assert.equal(url.searchParams.get('actor_id'), alice);
    assert.equal(
      submitAuditFilters({ ...emptyAuditFilters(), actor_id: 'unattributed' }).actor_id,
      'unattributed',
    );
    assert.throws(
      () => submitAuditFilters({ ...emptyAuditFilters(), actor_id: 'not-an-account' }),
      /用户/,
    );
    assert.throws(() => submitAuditFilters({ ...emptyAuditFilters(), q: 'x'.repeat(101) }), /100/);
    assert.throws(
      () => submitAuditFilters({ ...emptyAuditFilters(), from_time: 'invalid' }),
      /时间/,
    );
    assert.throws(
      () =>
        submitAuditFilters({
          ...emptyAuditFilters(),
          from_time: '2026-11-06',
          to_time: '2026-11-05',
        }),
      /开始时间/,
    );
  });

  await test('opaque cursor history advances and returns without offsets or carrying filters into another query', () => {
    const first = firstAuditPage();
    const second = nextAuditPage(first, { items: [], next_cursor: 'signed-2', has_more: true });
    const third = nextAuditPage(second, { items: [], next_cursor: 'signed-3', has_more: true });
    assert.deepEqual(previousAuditPage(third), second);
    assert.deepEqual(previousAuditPage(second), first);
    assert.equal(nextAuditPage(first, { items: [], next_cursor: null, has_more: false }), first);
    assert.throws(
      () => nextAuditPage(second, { items: [], next_cursor: 'signed-2', has_more: true }),
      /游标/,
    );
    assert.doesNotMatch(
      auditEventsPath({ ...emptyAuditFilters(), actor_id: bob }),
      /cursor|offset/,
    );
  });

  await test('the real read client sends only GET with the same filters and sanitizes before storing page data', async () => {
    transport.setSessionCsrf('admin-session', alice);
    const client = createAuditEventsClient();
    let request;
    globalThis.fetch = async (url, options) => {
      request = { url, options };
      return response(
        [event({ details: { target_name: 'safe', password: 'not-retained' } })],
        'next',
        true,
      );
    };
    const filters = { ...emptyAuditFilters(), actor_id: bob, q: 'model' };
    const page = await client.load(filters, 'current');
    assert.equal(request.options.method, undefined);
    assert.equal(request.options.body, undefined);
    assert.equal(request.options.cache, 'no-store');
    assert.match(request.url, /actor_id=22222222/);
    assert.match(request.url, /cursor=current/);
    assert.deepEqual(page.items[0].details, { target_name: 'safe' });
    client.dispose();
  });

  await test('a newer actor/page request suppresses an older response even when fetch ignores abort', async () => {
    transport.setSessionCsrf('admin-session', alice);
    const client = createAuditEventsClient(),
      old = deferred();
    globalThis.fetch = async (url) =>
      url.includes(`actor_id=${alice}`)
        ? old.promise
        : response([event({ id: 'bob-event', actor_id: bob })]);
    const pending = client.load({ ...emptyAuditFilters(), actor_id: alice });
    const current = await client.load({ ...emptyAuditFilters(), actor_id: bob });
    old.resolve(response([event()]));
    await assert.rejects(pending, (error) => error.name === 'AbortError');
    assert.equal(current.items[0].id, 'bob-event');
    client.dispose();
  });

  await test('replacement accounts and closed audit views cannot accept pending pages or issue old followups', async () => {
    transport.setSessionCsrf('alice-session', alice);
    const client = createAuditEventsClient(),
      held = deferred();
    let calls = 0;
    globalThis.fetch = async () => {
      calls++;
      return held.promise;
    };
    const pending = client.load(emptyAuditFilters());
    transport.setSessionCsrf('bob-session', bob);
    held.resolve(response([event()]));
    await assert.rejects(pending, (error) => error.code === 'SESSION_CHANGED');
    await assert.rejects(
      client.load(emptyAuditFilters()),
      (error) => error.code === 'SESSION_CHANGED',
    );
    assert.equal(calls, 1);
    client.dispose();
    await assert.rejects(client.load(emptyAuditFilters()), (error) => error.name === 'AbortError');
  });

  await test('malformed paging receipts cannot silently enable next-page navigation', async () => {
    transport.setSessionCsrf('admin-session', alice);
    const client = createAuditEventsClient();
    globalThis.fetch = async () => response([], null, true);
    await assert.rejects(
      client.load(emptyAuditFilters()),
      (error) => error.code === 'INVALID_RESPONSE',
    );
    client.dispose();
  });
} finally {
  globalThis.fetch = originalFetch;
  await server.close();
}
