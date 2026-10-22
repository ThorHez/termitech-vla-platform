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
const originalWindow = globalThis.window;
const originalBroadcastChannel = globalThis.BroadcastChannel;
const browser = new EventTarget();
browser.sessionStorage = { removeItem() {} };
globalThis.window = browser;

try {
  const transport = await server.ssrLoadModule('/src/api.ts');
  const { AuthContext, openSessionChannel, sessionRemainingTime } =
    await server.ssrLoadModule('/src/auth.tsx');
  const { watchHumanActivity } = await server.ssrLoadModule('/src/sessionActivity.ts');
  const { ReauthenticationDialog } = await server.ssrLoadModule('/src/ReauthenticationDialog.tsx');
  const { LoginPage, RequiredPasswordChange, PasswordChangeForm } =
    await server.ssrLoadModule('/src/App.tsx');
  const { UserManagement } = await server.ssrLoadModule('/src/UserManagement.tsx');
  const state = (permissions = [], extras = {}) => ({
    permissions,
    loading: false,
    expired: false,
    can: (permission) => permissions.includes(permission),
    refresh: async () => {},
    login: async () => {},
    logout: async () => {},
    changePassword: async () => {},
    ...extras,
  });
  const render = (component, context = state()) =>
    renderToStaticMarkup(
      React.createElement(AuthContext.Provider, { value: context }, React.createElement(component)),
    );

  await test('login provides individual credentials without public signup or shared token', () => {
    const html = render(LoginPage);
    assert.match(html, /autoComplete="username"/i);
    assert.match(html, /autoComplete="current-password"/i);
    assert.match(html, /不开放公开注册/);
    assert.match(html, /初始化管理员命令/);
    assert.doesNotMatch(html, /VLA_ADMIN_TOKEN|管理员访问令牌|name="token"/);
  });

  await test('required password change renders no platform resources and retains logout', () => {
    const html = render(
      RequiredPasswordChange,
      state([], {
        user: {
          id: 'u',
          username: 'alice',
          display_name: 'Alice',
          role: 'operator',
          must_change_password: true,
        },
      }),
    );
    assert.match(html, /更新登录密码/);
    assert.match(html, /更新密码并进入平台/);
    assert.match(html, /退出登录/);
    assert.doesNotMatch(html, /训练工作台|机器资源|训练任务|新建训练/);
  });

  await test('password forms enforce length and avoid password storage hints', () => {
    const html = render(PasswordChangeForm);
    assert.equal((html.match(/minLength="12"/gi) || []).length, 2);
    assert.doesNotMatch(html, /localStorage|sessionStorage/);
  });

  await test('user administration fails closed without users.manage permission', () => {
    const html = render(UserManagement, state(['platform.read', 'training.write']));
    assert.match(html, /没有访问权限/);
    assert.doesNotMatch(html, /新建用户|用户列表|操作审计/);
  });

  await test('admin view discloses role boundaries and separate Agent credentials', () => {
    const html = render(UserManagement, state(['users.manage', 'audit.read']));
    assert.match(html, /新建用户/);
    assert.match(html, /只读用户/);
    assert.match(html, /操作员/);
    assert.match(html, /机器 Agent 使用独立凭据/);
    assert.match(html, /操作审计/);
  });

  await test('transport uses same-origin cookies and memory CSRF without bearer credentials', async () => {
    const requests = [];
    globalThis.fetch = async (url, options) => {
      requests.push({ url, options });
      return new Response('{}', { status: 200 });
    };
    transport.setSessionCsrf('session-csrf');
    await transport.post('/runs', { example: true }, { Authorization: 'Bearer old-shared-secret' });
    await transport.api('/overview');
    assert.equal(requests[0].options.credentials, 'same-origin');
    assert.equal(requests[0].options.headers.get('X-CSRF-Token'), 'session-csrf');
    assert.equal(requests[0].options.headers.has('Authorization'), false);
    assert.equal(requests[1].options.headers.has('X-CSRF-Token'), false);
    assert.equal(requests[0].options.body, '{"example":true}');
  });

  await test('401 from business API expires the workspace but invalid login does not', async () => {
    let expired = 0;
    const unsubscribe = transport.subscribeSessionExpiry(() => {
      expired += 1;
    });
    globalThis.fetch = async () =>
      new Response('{"error":{"code":"UNAUTHORIZED","message":"no"}}', { status: 401 });
    transport.setSessionCsrf('session-1');
    await assert.rejects(transport.post('/auth/login', { username: 'wrong', password: 'wrong' }));
    await assert.rejects(transport.api('/auth/me'));
    assert.equal(expired, 0);
    await assert.rejects(transport.api('/overview'));
    assert.equal(expired, 1);
    unsubscribe();
  });

  await test('401 from stale request cannot clear a newly accepted login session', async () => {
    let release;
    let expired = 0;
    const unsubscribe = transport.subscribeSessionExpiry(() => {
      expired += 1;
    });
    globalThis.fetch = async () =>
      new Promise((resolve) => {
        release = resolve;
      });
    transport.setSessionCsrf('old-session');
    const pending = transport.api('/overview');
    transport.setSessionCsrf('new-session');
    release(new Response('{}', { status: 401 }));
    await assert.rejects(pending);
    assert.equal(expired, 0);
    unsubscribe();
  });

  await test('sensitive writes wait for password confirmation before one retry with the same payload', async () => {
    const requests = [];
    let release;
    const unsubscribe = transport.registerReauthentication(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    transport.setSessionCsrf('protected-session');
    globalThis.fetch = async (url, options) => {
      requests.push({ url, options });
      return requests.length === 1
        ? new Response('{"error":{"code":"REAUTH_REQUIRED","message":"password"}}', { status: 403 })
        : new Response('{"id":"new-user"}', { status: 201 });
    };
    const pending = transport.post('/users', { username: 'alice' });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(requests.length, 1);
    release();
    assert.deepEqual(await pending, { id: 'new-user' });
    assert.equal(requests.length, 2);
    assert.equal(requests[1].options.body, requests[0].options.body);
    assert.equal(requests[1].options.headers.get('X-CSRF-Token'), 'protected-session');
    unsubscribe();
  });

  await test('password confirmation never retries a write in a replacement login session', async () => {
    let calls = 0;
    const unsubscribe = transport.registerReauthentication(async () => {
      transport.setSessionCsrf('replacement-account');
    });
    transport.setSessionCsrf('original-account');
    globalThis.fetch = async () => {
      calls += 1;
      return new Response('{"error":{"code":"REAUTH_REQUIRED"}}', { status: 403 });
    };
    await assert.rejects(transport.post('/users', {}), (error) => error.code === 'SESSION_CHANGED');
    assert.equal(calls, 1);
    unsubscribe();
  });

  await test('cancelled, aborted and repeated reauthentication challenges never apply a write', async () => {
    let calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      return new Response('{"error":{"code":"REAUTH_REQUIRED"}}', { status: 403 });
    };
    transport.setSessionCsrf('same-account');
    let unsubscribe = transport.registerReauthentication(async () => {
      throw new transport.ApiError('REAUTH_CANCELLED', 'cancelled', 403);
    });
    await assert.rejects(
      transport.post('/users', {}),
      (error) => error.code === 'REAUTH_CANCELLED',
    );
    assert.equal(calls, 1);
    unsubscribe();
    const controller = new AbortController();
    unsubscribe = transport.registerReauthentication(async () => controller.abort());
    await assert.rejects(
      transport.api('/users', { method: 'POST', body: '{}', signal: controller.signal }),
      (error) => error.name === 'AbortError',
    );
    assert.equal(calls, 2);
    unsubscribe();
    unsubscribe = transport.registerReauthentication(async () => {});
    await assert.rejects(transport.post('/users', {}), (error) => error.code === 'REAUTH_REQUIRED');
    assert.equal(calls, 4, 'a challenge after the single retry is surfaced, without a loop');
    unsubscribe();
  });

  await test('ordinary permission denials and password failures never open a confirmation loop', async () => {
    let prompts = 0;
    const unsubscribe = transport.registerReauthentication(async () => {
      prompts += 1;
    });
    globalThis.fetch = async () => new Response('{"error":{"code":"FORBIDDEN"}}', { status: 403 });
    await assert.rejects(transport.post('/users', {}));
    globalThis.fetch = async () =>
      new Response('{"error":{"code":"REAUTH_REQUIRED"}}', { status: 403 });
    await assert.rejects(transport.post('/auth/reauth', { password: 'invalid' }));
    await assert.rejects(transport.api('/users'));
    assert.equal(prompts, 0);
    unsubscribe();
  });

  await test('only trusted human inputs send bounded activity heartbeats; idle timers do not', () => {
    let now = 0;
    let sequence = 0;
    const timers = new Map();
    const listeners = new Map();
    const target = {
      addEventListener: (name, handler) => listeners.set(name, handler),
      removeEventListener: (name) => listeners.delete(name),
    };
    const advance = (duration) => {
      now += duration;
      for (const [id, timer] of [...timers]) {
        if (timer.at <= now) {
          timers.delete(id);
          timer.callback();
        }
      }
    };
    const heartbeats = [];
    const stop = watchHumanActivity(() => heartbeats.push(now), target, {
      now: () => now,
      setTimeout: (callback, delay) => {
        const id = ++sequence;
        timers.set(id, { at: now + delay, callback });
        return id;
      },
      clearTimeout: (id) => timers.delete(id),
    });
    advance(900000);
    assert.deepEqual(heartbeats, [], 'elapsed time and background polling create no activity');
    listeners.get('pointerdown')({ isTrusted: false });
    assert.deepEqual(
      heartbeats,
      [],
      'scripts cannot extend the idle deadline through synthetic events',
    );
    listeners.get('pointerdown')({ isTrusted: true });
    assert.deepEqual(heartbeats, [900000]);
    advance(10000);
    for (let index = 0; index < 30; index += 1) listeners.get('keydown')({ isTrusted: true });
    assert.equal(heartbeats.length, 1);
    advance(20000);
    assert.deepEqual(heartbeats, [900000, 930000], 'a burst produces only one trailing heartbeat');
    advance(900000);
    assert.equal(heartbeats.length, 2, 'the heartbeat cannot keep an idle tab alive');
    listeners.get('wheel')({ isTrusted: true });
    advance(1);
    listeners.get('touchstart')({ isTrusted: true });
    stop();
    advance(30000);
    assert.equal(heartbeats.length, 3, 'logout cancels pending trailing activity');
    assert.equal(listeners.size, 0);
  });

  await test('session deadlines use server-relative time despite a skewed browser clock', () => {
    const originalNow = Date.now;
    Date.now = () => Date.parse('2099-01-01T00:00:00Z');
    try {
      assert.deepEqual(
        sessionRemainingTime({
          server_time: '2026-10-22T01:00:00Z',
          expires_at: '2026-10-22T03:00:00Z',
          idle_expires_at: '2026-10-22T01:15:00Z',
        }),
        { absolute: 7200000, idle: 900000 },
      );
    } finally {
      Date.now = originalNow;
    }
  });

  await test('reauthentication dialog requests current password without storing it', () => {
    const html = renderToStaticMarkup(
      React.createElement(ReauthenticationDialog, {
        username: 'admin',
        confirm: async () => {},
        cancel: () => {},
      }),
    );
    assert.match(html, /验证当前账号密码/);
    assert.match(html, /autoComplete="current-password"/i);
    assert.match(html, /取消操作/);
    assert.match(html, /5 分钟/);
    assert.doesNotMatch(html, /localStorage|sessionStorage|name="token"/);
  });

  await test('cross-tab session signals invalidate peers without broadcasting credentials or echoing', () => {
    const channels = new Set();
    const sent = [];
    globalThis.BroadcastChannel = class {
      constructor(name) {
        this.name = name;
        channels.add(this);
      }
      postMessage(data) {
        sent.push(data);
        for (const channel of channels) {
          if (channel !== this && channel.name === this.name) channel.onmessage?.({ data });
        }
      }
      close() {
        channels.delete(this);
      }
    };
    let firstInvalidated = 0;
    let secondInvalidated = 0;
    const first = openSessionChannel(() => {
      firstInvalidated += 1;
    });
    const second = openSessionChannel(() => {
      secondInvalidated += 1;
    });
    first.publish();
    assert.equal(firstInvalidated, 0, 'sender keeps its newly accepted session');
    assert.equal(secondInvalidated, 1, 'peer is notified immediately');
    assert.deepEqual(
      sent,
      ['session-changed'],
      'only a fixed invalidation signal crosses contexts',
    );
    second.publish();
    assert.equal(firstInvalidated, 1);
    assert.equal(secondInvalidated, 1);
    assert.equal(sent.length, 2, 'receiving a signal never publishes an echo');
    second.close();
    first.publish();
    assert.equal(secondInvalidated, 1, 'closed tabs no longer receive events');
    first.close();
    globalThis.BroadcastChannel = originalBroadcastChannel;
  });

  await test('session channel is optional in browsers without BroadcastChannel', () => {
    globalThis.BroadcastChannel = undefined;
    assert.equal(
      openSessionChannel(() => assert.fail('no channel exists')),
      undefined,
    );
    globalThis.BroadcastChannel = originalBroadcastChannel;
  });

  await test('cross-tab activity triggers read-only refresh without invalidating identity or sharing secrets', () => {
    let channel;
    globalThis.BroadcastChannel = class {
      constructor() {
        channel = this;
      }
      postMessage(data) {
        this.onmessage({ data });
      }
      close() {}
    };
    let invalidations = 0;
    let refreshes = 0;
    const subscription = openSessionChannel(
      () => {
        invalidations += 1;
      },
      () => {
        refreshes += 1;
      },
    );
    subscription.publishActivity();
    channel.onmessage({ data: { user: 'admin', csrf_token: 'untrusted' } });
    assert.equal(invalidations, 0);
    assert.equal(refreshes, 1);
    subscription.close();
    globalThis.BroadcastChannel = originalBroadcastChannel;
  });

  await test('legacy admin token is discarded without reading or reusing it', () => {
    const removed = [];
    browser.sessionStorage = { removeItem: (key) => removed.push(key) };
    transport.discardLegacyToken();
    assert.deepEqual(removed, ['termitech.admin-token']);
  });
} finally {
  globalThis.fetch = originalFetch;
  globalThis.window = originalWindow;
  globalThis.BroadcastChannel = originalBroadcastChannel;
  await server.close();
}
