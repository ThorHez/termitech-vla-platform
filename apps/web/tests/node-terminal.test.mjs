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

class Socket extends EventTarget {
  readyState = 0;
  bufferedAmount = 0;
  sent = [];
  closes = [];
  send(value) {
    this.sent.push(JSON.parse(value));
  }
  close(code, reason) {
    this.readyState = 3;
    this.closes.push({ code, reason });
  }
  open() {
    this.readyState = 1;
    this.dispatchEvent(new Event('open'));
  }
  frame(value) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(value) }));
  }
  bytes(value) {
    this.dispatchEvent(new MessageEvent('message', { data: value.buffer }));
  }
}

try {
  const { connectTerminalSocket, terminalInputChunks, terminalSize, terminalSocketUrl } =
    await server.ssrLoadModule('/src/nodeTerminalSession.ts');
  const fixture = (output) => {
    const socket = new Socket();
    const events = { ready: 0, output: [], closed: [], error: [] };
    const session = connectTerminalSocket(
      socket,
      'one-use-secret',
      { cols: 100, rows: 30 },
      {
        ready() {
          events.ready += 1;
        },
        output(bytes, done) {
          events.output.push(bytes);
          if (output) output(bytes, done);
          else done();
        },
        closed(message) {
          events.closed.push(message);
        },
        error(error) {
          events.error.push(error.message);
        },
      },
    );
    return { socket, session, events };
  };

  await test('terminal uses the current origin and keeps the ticket out of the websocket URL', () => {
    assert.equal(
      terminalSocketUrl('/api/v1/node-terminals/abc-123/ws', 'https://console.test:8443'),
      'wss://console.test:8443/api/v1/node-terminals/abc-123/ws',
    );
    assert.equal(
      terminalSocketUrl('/api/v1/node-terminals/id/ws', 'http://127.0.0.1:8000'),
      'ws://127.0.0.1:8000/api/v1/node-terminals/id/ws',
    );
    for (const path of [
      '//evil.test/api/v1/node-terminals/a/ws',
      'wss://evil.test/a',
      '/api/v1/node-terminals/a/ws?ticket=secret',
      '/api/v1/node-terminals/a/ws#secret',
      '/api/v1/node-terminals/../ws',
      '/other/path',
    ]) {
      assert.throws(() => terminalSocketUrl(path, 'https://console.test'));
    }
  });

  await test('authentication is the first frame and stdin is enabled only after ready', () => {
    const { socket, session, events } = fixture();
    try {
      session.input('should not be sent');
      assert.deepEqual(socket.sent, []);
      socket.open();
      assert.deepEqual(socket.sent, [
        { type: 'auth', ticket: 'one-use-secret', cols: 100, rows: 30 },
      ]);
      session.input('still waiting');
      assert.equal(socket.sent.length, 1);
      socket.frame({ type: 'ready' });
      session.input('echo 中文\r\x03\t\x1b[A');
      session.resize(140, 40);
      assert.equal(events.ready, 1);
      assert.deepEqual(socket.sent.slice(1), [
        { type: 'input', data: 'echo 中文\r\x03\t\x1b[A' },
        { type: 'resize', cols: 140, rows: 40 },
      ]);
      socket.frame({ type: 'pong' });
      assert.deepEqual(events.error, []);
    } finally {
      session.dispose();
    }
  });

  await test('binary output preserves ANSI controls and fragmented UTF-8 for the terminal decoder', () => {
    const { socket, session, events } = fixture();
    try {
      socket.open();
      socket.frame({ type: 'ready' });
      const encoded = new TextEncoder().encode('\x1b[31m红色\x1b[0m\r\x1b[2K');
      socket.bytes(encoded.slice(0, 7));
      socket.bytes(encoded.slice(7));
      assert.deepEqual(new Uint8Array(Buffer.concat(events.output)), encoded);
      assert.deepEqual(events.error, []);
    } finally {
      session.dispose();
    }
  });

  await test('large pastes are bounded per frame without breaking Unicode or control characters', () => {
    const input = 'a'.repeat(4095) + '🦾中文\t\r\x03'.repeat(3000);
    const chunks = [...terminalInputChunks(input)];
    assert.equal(chunks.join(''), input);
    assert.ok(chunks.every((chunk) => chunk.length <= 4096));
    assert.ok(
      chunks.every(
        (chunk) => Buffer.byteLength(JSON.stringify({ type: 'input', data: chunk })) < 32768,
      ),
    );
    assert.ok(chunks.every((chunk) => !/[\uD800-\uDBFF]$/.test(chunk)));
    assert.deepEqual(terminalSize(100000, -1), { cols: 500, rows: 2 });
    assert.deepEqual(terminalSize(NaN, Infinity), { cols: 80, rows: 24 });
  });

  await test('server errors and malformed output close the connection exactly once and stop input', () => {
    for (const frame of [
      { type: 'error', code: 'SSH_FAILED', message: 'SSH host key changed' },
      { type: 'unexpected' },
    ]) {
      const { socket, session, events } = fixture();
      socket.open();
      socket.frame(frame);
      session.input('not sent');
      session.dispose();
      socket.frame({ type: 'ready' });
      assert.equal(events.error.length, 1);
      assert.equal(events.ready, 0);
      assert.equal(socket.closes.length, 1);
      assert.equal(socket.sent.length, 1);
    }
    const { socket, session, events } = fixture();
    socket.open();
    socket.bytes(new Uint8Array([1]));
    assert.match(events.error[0], /握手/);
    session.dispose();
  });

  await test('input and unrendered output backpressure fail visibly instead of silently dropping data', () => {
    const first = fixture();
    first.socket.open();
    first.socket.frame({ type: 'ready' });
    first.socket.bufferedAmount = 1024 * 1024;
    first.session.input('echo test');
    assert.match(first.events.error[0], /输入积压/);
    assert.equal(first.socket.closes.length, 1);
    first.session.dispose();

    const second = fixture(() => {});
    second.socket.open();
    second.socket.frame({ type: 'ready' });
    for (let i = 0; i < 17; i += 1) second.socket.bytes(new Uint8Array(65536));
    assert.match(second.events.error[0], /输出超过/);
    assert.equal(second.socket.closes.length, 1);
    second.session.dispose();
  });

  await test('render callbacks release backlog and remote exit is not treated as a network failure', () => {
    const { socket, session, events } = fixture();
    socket.open();
    socket.frame({ type: 'ready' });
    for (let i = 0; i < 100; i += 1) socket.bytes(new Uint8Array(65536));
    assert.deepEqual(events.error, []);
    socket.frame({ type: 'closed', message: 'Shell 已退出' });
    assert.deepEqual(events.closed, ['Shell 已退出']);
    socket.bytes(new Uint8Array(1));
    assert.equal(events.output.length, 100);
    assert.equal(socket.closes.length, 1);
    session.dispose();
  });

  await test('disconnect disposes event listeners and never reuses the ticket or reconnects automatically', () => {
    const { socket, session, events } = fixture();
    socket.open();
    socket.frame({ type: 'ready' });
    session.dispose();
    socket.dispatchEvent(new Event('close'));
    socket.dispatchEvent(new Event('error'));
    socket.open();
    socket.frame({ type: 'ready' });
    session.input('ignored');
    session.resize(80, 24);
    assert.equal(socket.sent.length, 1);
    assert.equal(socket.closes.length, 1);
    assert.equal(events.ready, 1);
    assert.deepEqual(events.closed, []);
    assert.deepEqual(events.error, []);
  });

  await test('slow SSH handshakes are allowed 45 seconds before an explicit connection timeout', (context) => {
    context.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
    const { socket, session, events } = fixture();
    socket.open();
    context.mock.timers.tick(35000);
    assert.deepEqual(events.error, []);
    context.mock.timers.tick(10000);
    assert.match(events.error[0], /连接超时/);
    assert.equal(socket.closes.length, 1);
    session.dispose();
  });

  await test('only administrators see terminal controls and the dialog does not require an online Agent', async () => {
    const { NodeTerminal } = await server.ssrLoadModule('/src/NodeTerminal.tsx');
    const { NodePage } = await server.ssrLoadModule('/src/pages.tsx');
    const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
    const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
    const node = { id: 'machine', name: '终端测试 <A>', status: 'OFFLINE' };
    const render = (admin, child) =>
      renderToStaticMarkup(
        React.createElement(
          AuthContext.Provider,
          { value: { can: (permission) => admin && permission === 'machines.manage' } },
          child,
        ),
      );
    const props = { node, close() {} };
    const denied = render(false, React.createElement(NodeTerminal, props));
    assert.match(denied, /连接机器终端需要管理员权限/);
    assert.doesNotMatch(denied, /data-terminal-input/);
    const allowed = render(true, React.createElement(NodeTerminal, props));
    assert.match(allowed, /data-terminal-input/);
    assert.match(allowed, /终端测试 &lt;A&gt;/);
    assert.match(allowed, /Shift\+Esc/);
    assert.doesNotMatch(allowed, /Agent.*离线|只允许在线/);
    for (const admin of [true, false]) {
      const html = render(
        admin,
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(NodePage, { id: 'machine', createRun() {} }),
        ),
      );
      assert.equal(/>终端<\/button>/.test(html), admin);
    }
  });
} finally {
  await server.close();
}
