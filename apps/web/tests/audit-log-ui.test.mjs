import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { homedir, tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer, transformWithEsbuild } from 'vite';
import react from '@vitejs/plugin-react';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  try {
    ({ chromium } = require(
      path.join(
        homedir(),
        '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright',
      ),
    ));
  } catch {
    /* Optional without a browser runtime. */
  }
}
const executablePath =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  (process.platform === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    : '');
const available = Boolean(chromium && executablePath && existsSync(executablePath));
const alice = '11111111-1111-4111-8111-111111111111';
const bob = '22222222-2222-4222-8222-222222222222';
const users = [
  {
    id: alice,
    username: 'admin',
    display_name: '管理员',
    role: 'admin',
    enabled: true,
    must_change_password: false,
    created_at: '2026-11-01T00:00:00Z',
  },
  {
    id: bob,
    username: 'bob',
    display_name: '测试成员',
    role: 'operator',
    enabled: false,
    must_change_password: false,
    created_at: '2026-11-01T00:00:00Z',
  },
];
const event = (id, name, actor = alice, status = 201) => ({
  id,
  created_at: '2026-11-05T02:00:00Z',
  actor_id: actor,
  actor_username: actor === bob ? 'bob' : 'admin',
  action: 'assets.register',
  method: 'POST',
  path: '/api/v1/assets',
  status,
  remote_addr: '192.0.2.169',
  details: {
    operation_label: '登记资产',
    category: 'assets',
    target_type: 'asset',
    target_id: id,
    target_name: name,
    asset_kind: 'DATASET',
    request_id: 'request-1',
    count: 2,
    related_ids: `${alice},${bob}`,
    token: 'private-token-never-render',
    body: { password: 'private-body-never-render' },
  },
});
const result = (items, next_cursor = null) => ({
  items,
  next_cursor,
  has_more: Boolean(next_cursor),
});
const deferred = () => {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
};
const until = async (predicate) => {
  for (let i = 0; i < 500; i++) {
    if (predicate()) return;
    await new Promise((r) => setTimeout(r, 10));
  }
  assert.ok(predicate(), 'expected request was not received');
};

await test(
  'audit panel supports real filter, cursor, shortcut and account-change flows',
  { skip: !available },
  async (t) => {
    const entry = `
import React,{useState}from'react';import{createRoot}from'react-dom/client';
import{AuditLog}from'/src/AuditLog.tsx';import{UserManagement}from'/src/UserManagement.tsx';
import{AuthContext}from'/src/auth.tsx';import{setSessionCsrf}from'/src/api.ts';
import'/src/styles.css';import'/src/theme.css';import'/src/auth.css';
function App(){const[config,setConfig]=useState(window.__initial);window.__switch=user=>{setSessionCsrf('fixture-'+user.id,user.id);setConfig(old=>({...old,user}));};
const auth={user:config.user,loading:false,expired:false,can:p=>config.user.role==='admin'&&['audit.read','users.manage'].includes(p),refresh:async()=>{}};
return <AuthContext.Provider value={auth}>{config.mode==='management'?<UserManagement/>:<AuditLog users={config.users}/>}</AuthContext.Provider>;}
setSessionCsrf('fixture-'+window.__initial.user.id,window.__initial.user.id);createRoot(document.getElementById('root')).render(<App/>);`;
    const cacheDir = await mkdtemp(path.join(tmpdir(), 'vla-audit-ui-'));
    t.after(() => rm(cacheDir, { recursive: true, force: true }));
    const server = await createServer({
      root: fileURLToPath(new URL('..', import.meta.url)),
      configFile: false,
      cacheDir,
      optimizeDeps: {
        noDiscovery: true,
        include: [
          'react',
          'react-dom/client',
          'react/jsx-runtime',
          'react/jsx-dev-runtime',
          'lucide-react',
        ],
      },
      resolve: { dedupe: ['react', 'react-dom'] },
      plugins: [
        react(),
        {
          name: 'audit-ui-fixture',
          resolveId(id) {
            if (id === '/__audit__.tsx') return '\0audit-ui.tsx';
          },
          async load(id) {
            if (id === '\0audit-ui.tsx')
              return (
                await transformWithEsbuild(entry, 'audit-ui.tsx', {
                  loader: 'tsx',
                  jsx: 'automatic',
                })
              ).code;
          },
          configureServer(instance) {
            instance.middlewares.use('/__fixture__', (_request, response) => {
              response.setHeader('Content-Type', 'text/html');
              response.end(
                '<!doctype html><meta charset="utf-8"><div id="root"></div><script type="module" src="/__audit__.tsx"></script>',
              );
            });
          },
        },
      ],
      server: { host: '127.0.0.1', port: 0, hmr: false },
    });
    const contexts = [];
    let browser;
    try {
      await server.listen();
      const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
      browser = await chromium.launch({ executablePath, headless: true });
      const mount = async (handler, mode = 'audit') => {
        const context = await browser.newContext({
          viewport: { width: 1440, height: 1080 },
          timezoneId: 'Asia/Shanghai',
        });
        contexts.push(context);
        const page = await context.newPage(),
          requests = [],
          errors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        await page.addInitScript(
          (initial) => {
            window.__initial = initial;
          },
          { user: users[0], users, mode },
        );
        await page.route('**/*', async (route) => {
          const request = route.request(),
            url = new URL(request.url());
          assert.equal(url.origin, origin, 'fixture must never contact a real server');
          if (!url.pathname.startsWith('/api/v1/')) return route.continue();
          assert.equal(request.method(), 'GET', 'audit browsing must not submit mutations');
          let data;
          if (url.pathname === '/api/v1/users') data = users;
          else {
            assert.equal(url.pathname, '/api/v1/audit/events');
            const params = Object.fromEntries(url.searchParams);
            requests.push(params);
            data = await handler(params, requests.length);
          }
          try {
            await route.fulfill({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify(data),
            });
          } catch {
            /* Obsolete reads may be aborted. */
          }
        });
        await page.goto(`${origin}/__fixture__`);
        await page.locator('.audit-log-panel').waitFor();
        return {
          page,
          requests,
          errors,
          close: () => context.close(),
          row: (name) =>
            page.waitForFunction(
              (label) =>
                [...document.querySelectorAll('.audit-target > span')].some(
                  (element) => element.textContent === label,
                ),
              name,
            ),
        };
      };

      await t.test(
        'query, next/previous cursors, reset and UTC conversion follow actual controls',
        async () => {
          const fixture = await mount((params) =>
            params.q
              ? result([event('filtered', '筛选结果', bob, 403)])
              : params.cursor
                ? result([event('older', '历史页面')])
                : result([event('first', '首页记录')], 'signed+/page2=='),
          );
          const { page, requests } = fixture;
          await fixture.row('首页记录');
          assert.equal(await page.getByLabel('首页每 15 秒刷新').isChecked(), false);
          await page.getByRole('button', { name: '下一页', exact: true }).click();
          await fixture.row('历史页面');
          assert.equal(requests.at(-1).cursor, 'signed+/page2==');
          await page.getByRole('button', { name: '上一页', exact: true }).click();
          await fixture.row('首页记录');
          assert.equal(requests.at(-1).cursor, undefined);
          await page.getByLabel('操作账号', { exact: true }).selectOption(bob);
          await page.getByLabel('关键词', { exact: true }).fill('model + value');
          assert.equal(await page.getByLabel('自定义操作标识', { exact: true }).count(), 0);
          await page.getByLabel('操作类型', { exact: true }).selectOption({ label: '登记资产' });
          await page.getByLabel('结果', { exact: true }).selectOption('denied');
          await page.getByLabel('开始时间（本地时间）').fill('2026-11-05T09:30');
          await page.getByLabel('结束时间（本地时间）').fill('2026-11-05T10:30');
          await page.getByRole('button', { name: '查询', exact: true }).click();
          await fixture.row('筛选结果');
          assert.deepEqual(requests.at(-1), {
            limit: '50',
            actor_id: bob,
            q: 'model + value',
            action: 'assets.register',
            outcome: 'denied',
            from_time: '2026-11-05T01:30:00.000Z',
            to_time: '2026-11-05T02:30:00.000Z',
          });
          assert.match(await page.locator('.audit-pagination').textContent(), /第 1 页/);
          await page.getByText('查看详情', { exact: true }).click();
          assert.doesNotMatch(
            await page.locator('.audit-events-table').textContent(),
            /private-token|private-body/,
          );
          assert.match(await page.locator('.audit-entry-details').textContent(), /关联任务/);
          assert.match(await page.locator('.audit-entry-details').textContent(), /数据集/);
          const artifacts = fileURLToPath(
            new URL('../../../.local/audit-log-ui-20261105/', import.meta.url),
          );
          await mkdir(artifacts, { recursive: true });
          await page.screenshot({
            path: path.join(artifacts, 'audit-log-panel.png'),
            fullPage: true,
          });
          await page.getByLabel('操作类型', { exact: true }).selectOption('__custom__');
          await page.getByLabel('自定义操作标识', { exact: true }).fill('legacy.custom_action');
          await page.getByRole('button', { name: '查询', exact: true }).click();
          await until(() => requests.at(-1).action === 'legacy.custom_action');
          await fixture.row('筛选结果');
          await page.getByRole('button', { name: '重置筛选', exact: true }).click();
          await fixture.row('首页记录');
          assert.equal(requests.at(-1).actor_id, undefined);
          assert.equal(requests.at(-1).q, undefined);
          assert.equal(requests.at(-1).action, undefined);
          assert.equal(await page.getByLabel('自定义操作标识', { exact: true }).count(), 0);
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );

      await t.test(
        'changing actor while an old read is delayed never shows the old actor page',
        async () => {
          const held = deferred();
          const fixture = await mount((params) =>
            params.actor_id === alice
              ? held.promise
              : result([
                  event('current', params.actor_id === bob ? 'Bob 当前记录' : '全部记录', bob),
                ]),
          );
          const { page, requests } = fixture;
          await fixture.row('全部记录');
          await page.getByLabel('操作账号', { exact: true }).selectOption(alice);
          await page.getByRole('button', { name: '查询', exact: true }).click();
          await until(() => requests.some((request) => request.actor_id === alice));
          await page.getByLabel('操作账号', { exact: true }).selectOption(bob);
          await page.getByRole('button', { name: '查询', exact: true }).click();
          await fixture.row('Bob 当前记录');
          held.resolve(result([event('old', 'Alice 迟到记录')]));
          await page.waitForTimeout(50);
          assert.doesNotMatch(
            await page.locator('.audit-events-table').textContent(),
            /Alice 迟到记录/,
          );
          assert.equal(await page.getByLabel('操作账号', { exact: true }).inputValue(), bob);
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );

      await t.test(
        'account replacement clears old pages and operator/viewer mounts perform no audit reads',
        async () => {
          const held = deferred();
          const fixture = await mount((_params, count) =>
            count === 1 ? held.promise : result([event('new', '新管理员记录', bob)]),
          );
          const { page, requests } = fixture;
          await until(() => requests.length === 1);
          await page.evaluate((user) => window.__switch(user), {
            ...users[0],
            id: bob,
            username: 'other-admin',
          });
          await fixture.row('新管理员记录');
          held.resolve(result([event('old', '旧账号记录')]));
          await page.waitForTimeout(50);
          assert.doesNotMatch(
            await page.locator('.audit-events-table').textContent(),
            /旧账号记录/,
          );
          const count = requests.length;
          for (const role of ['operator', 'viewer']) {
            await page.evaluate((user) => window.__switch(user), { ...users[0], role });
            await page.waitForFunction(() => !document.querySelector('.audit-log-panel'));
          }
          assert.equal(requests.length, count);
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );

      await t.test(
        'user-row shortcut includes disabled accounts and resets cursor on repeated selection',
        async () => {
          const fixture = await mount(
            (params) =>
              result(
                [
                  event(
                    params.cursor ? 'older' : 'first',
                    params.actor_id === bob
                      ? params.cursor
                        ? 'Bob 历史记录'
                        : 'Bob 快捷记录'
                      : '全部记录',
                    bob,
                  ),
                ],
                params.cursor ? null : 'next-page',
              ),
            'management',
          );
          const { page, requests } = fixture;
          await fixture.row('全部记录');
          await page.getByRole('button', { name: '查看 bob 的操作记录', exact: true }).click();
          await fixture.row('Bob 快捷记录');
          assert.equal(requests.at(-1).actor_id, bob);
          assert.match(
            await page
              .getByLabel('操作账号', { exact: true })
              .locator('option:checked')
              .textContent(),
            /已停用/,
          );
          await page.getByRole('button', { name: '下一页', exact: true }).click();
          await fixture.row('Bob 历史记录');
          await page.getByRole('button', { name: '查看 bob 的操作记录', exact: true }).click();
          await fixture.row('Bob 快捷记录');
          assert.equal(requests.at(-1).cursor, undefined);
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );

      await t.test('automatic refresh remains opt-in and pauses on history pages', async () => {
        const fixture = await mount((params) =>
          params.cursor
            ? result([event('history', '历史页不轮询')])
            : result([event('home', '首页可刷新')], 'history-cursor'),
        );
        const { page, requests } = fixture;
        await fixture.row('首页可刷新');
        await page.clock.install();
        const initial = requests.length;
        await page.clock.fastForward(30000);
        assert.equal(requests.length, initial);
        await page.getByLabel('首页每 15 秒刷新').check();
        await until(() => requests.length > initial);
        await fixture.row('首页可刷新');
        await page.getByRole('button', { name: '下一页', exact: true }).click();
        await fixture.row('历史页不轮询');
        const historical = requests.length;
        await page.clock.fastForward(45000);
        assert.equal(requests.length, historical);
        await page.getByRole('button', { name: '刷新首页', exact: true }).click();
        await fixture.row('首页可刷新');
        assert.equal(requests.at(-1).cursor, undefined);
        assert.deepEqual(fixture.errors, []);
        await fixture.close();
      });
    } finally {
      for (const context of contexts) await context.close();
      await browser?.close();
      await server.close();
    }
  },
);
