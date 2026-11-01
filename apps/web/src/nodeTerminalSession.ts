export interface TerminalSize {
  cols: number;
  rows: number;
}

export interface NodeTerminalTicket {
  id: string;
  ticket: string;
  websocket_path: string;
  target: { username: string; host: string; port: number };
  expires_at: string;
}

const maximumBufferedBytes = 1024 * 1024;

export function terminalSize(cols: number, rows: number): TerminalSize {
  return {
    cols: Math.max(2, Math.min(500, Math.floor(Number.isFinite(cols) ? cols : 80))),
    rows: Math.max(2, Math.min(200, Math.floor(Number.isFinite(rows) ? rows : 24))),
  };
}

export function terminalSocketUrl(path: string, origin: string) {
  const base = new URL(origin);
  if (!['http:', 'https:'].includes(base.protocol))
    throw new Error('终端需要 HTTP 或 HTTPS 入口。');
  if (!/^\/api\/v1\/node-terminals\/[a-zA-Z0-9-]+\/ws$/.test(path)) {
    throw new Error('管理服务返回了无效的终端连接地址。');
  }
  const url = new URL(path, base);
  url.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
}

// Keep pasted Unicode intact while staying below the server's frame-size limit.
export function* terminalInputChunks(data: string) {
  for (let start = 0; start < data.length; ) {
    let end = Math.min(start + 4096, data.length);
    const last = data.charCodeAt(end - 1);
    if (end < data.length && last >= 0xd800 && last <= 0xdbff) end -= 1;
    yield data.slice(start, end);
    start = end;
  }
}

export interface TerminalSession {
  input: (data: string) => void;
  resize: (cols: number, rows: number) => void;
  dispose: () => void;
}

export function connectTerminalSocket(
  socket: WebSocket,
  ticket: string,
  size: TerminalSize,
  hooks: {
    ready: () => void;
    output: (data: Uint8Array, rendered: () => void) => void;
    closed: (message: string) => void;
    error: (error: Error) => void;
  },
): TerminalSession {
  let finished = false;
  let ready = false;
  let pendingOutput = 0;
  let handshakeTicket = ticket;
  let heartbeat: ReturnType<typeof setInterval> | undefined;
  const encoder = new TextEncoder();
  socket.binaryType = 'arraybuffer';

  const cleanup = () => {
    clearTimeout(deadline);
    clearInterval(heartbeat);
    socket.removeEventListener('open', opened);
    socket.removeEventListener('message', message);
    socket.removeEventListener('close', closed);
    socket.removeEventListener('error', failed);
    handshakeTicket = '';
  };
  const stop = () => {
    if (finished) return;
    finished = true;
    ready = false;
    cleanup();
    try {
      socket.close(1000, 'Terminal disconnected');
    } catch {
      // Browsers can reject close() when a connection has already disappeared.
    }
  };
  const fail = (text: string) => {
    if (finished) return;
    stop();
    hooks.error(new Error(text));
  };
  const send = (value: object) => {
    if (finished || socket.readyState !== 1) return false;
    const frame = JSON.stringify(value);
    if (socket.bufferedAmount + encoder.encode(frame).byteLength > maximumBufferedBytes) {
      fail('终端输入积压过多，连接已断开。请重新连接后重试。');
      return false;
    }
    try {
      socket.send(frame);
      return true;
    } catch {
      fail('终端连接已中断，请重新连接。');
      return false;
    }
  };
  const opened = () => {
    send({ type: 'auth', ticket: handshakeTicket, ...terminalSize(size.cols, size.rows) });
    handshakeTicket = '';
  };
  const closed = () => {
    if (finished) return;
    stop();
    hooks.closed('终端连接已断开。重新连接将开启新的 Shell 会话。');
  };
  const failed = () => fail('无法连接机器终端，请检查网络或重新连接。');
  const message = (event: MessageEvent) => {
    if (finished) return;
    if (event.data instanceof ArrayBuffer) {
      if (!ready) return fail('终端连接返回了无效的握手响应。');
      const data = new Uint8Array(event.data);
      pendingOutput += data.byteLength;
      if (pendingOutput > maximumBufferedBytes) {
        return fail('终端输出超过浏览器处理速度，连接已断开。请重新连接后重试。');
      }
      hooks.output(data, () => {
        pendingOutput -= data.byteLength;
      });
      return;
    }
    try {
      if (typeof event.data !== 'string') throw new Error();
      const frame: unknown = JSON.parse(event.data);
      if (!frame || typeof frame !== 'object' || !('type' in frame)) throw new Error();
      if (frame.type === 'ready') {
        if (ready) return;
        ready = true;
        clearTimeout(deadline);
        heartbeat = setInterval(() => send({ type: 'ping' }), 20000);
        hooks.ready();
      } else if (frame.type === 'error') {
        const text = 'message' in frame && typeof frame.message === 'string' ? frame.message : '';
        const code = 'code' in frame && typeof frame.code === 'string' ? frame.code : '';
        fail(`${code ? `${code}: ` : ''}${text || '机器终端连接失败，请重新连接。'}`);
      } else if (frame.type === 'closed') {
        const text = 'message' in frame && typeof frame.message === 'string' ? frame.message : '';
        stop();
        hooks.closed(text || '远端 Shell 已退出。');
      } else if (frame.type !== 'pong') {
        throw new Error();
      }
    } catch {
      fail('管理服务返回了无效的终端消息，连接已断开。');
    }
  };
  const deadline = setTimeout(() => fail('终端连接超时，请重新连接。'), 45000);
  socket.addEventListener('open', opened);
  socket.addEventListener('message', message);
  socket.addEventListener('close', closed);
  socket.addEventListener('error', failed);
  return {
    input(data) {
      if (!ready || finished) return;
      for (const chunk of terminalInputChunks(data)) {
        if (!send({ type: 'input', data: chunk })) return;
      }
    },
    resize(cols, rows) {
      if (ready) send({ type: 'resize', ...terminalSize(cols, rows) });
    },
    dispose: stop,
  };
}
