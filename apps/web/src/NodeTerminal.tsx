import { useEffect, useRef, useState } from 'react';
import type { Terminal } from '@xterm/xterm';
import { LoaderCircle, Plug, Unplug } from 'lucide-react';
import { api } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Modal } from './components';
import type { Node } from './types';
import { connectTerminalSocket, terminalSize, terminalSocketUrl } from './nodeTerminalSession';
import type { NodeTerminalTicket, TerminalSession } from './nodeTerminalSession';
import '@xterm/xterm/css/xterm.css';
import './node-terminal.css';

export function NodeTerminal({ node, close }: { node: Node; close: () => void }) {
  const { can } = useAuth();
  if (!can('machines.manage')) {
    return (
      <Modal title="权限不足" close={close}>
        连接机器终端需要管理员权限。
      </Modal>
    );
  }
  return <NodeTerminalContent key={node.id} node={node} close={close} />;
}

function NodeTerminalContent({ node, close }: { node: Node; close: () => void }) {
  const host = useRef<HTMLDivElement>(null);
  const toolbar = useRef<HTMLDivElement>(null);
  const terminal = useRef<Terminal | undefined>(undefined);
  const session = useRef<TerminalSession | undefined>(undefined);
  const disconnect = useRef<(() => void) | undefined>(undefined);
  const [initialized, setInitialized] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [phase, setPhase] = useState<'connecting' | 'ready' | 'closed' | 'error'>('connecting');
  const [message, setMessage] = useState('正在准备终端并验证连接权限…');
  const [target, setTarget] = useState<NodeTerminalTicket['target']>();
  const [error, setError] = useState<Error>();

  useEffect(() => {
    let disposed = false;
    let cleanup: (() => void) | undefined;
    const initialize = async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([
        import('@xterm/xterm'),
        import('@xterm/addon-fit'),
      ]);
      if (disposed || !host.current) return;
      const term = new Terminal({
        cursorBlink: true,
        disableStdin: true,
        fontFamily: 'Menlo, Monaco, Consolas, "Liberation Mono", monospace',
        fontSize: 13,
        scrollback: 2000,
        allowProposedApi: false,
        theme: {
          background: '#101c27',
          foreground: '#dbe6ee',
          cursor: '#5ee0ce',
          selectionBackground: '#38787580',
          black: '#1e2d3b',
          red: '#f47777',
          green: '#61d6a7',
          yellow: '#eaca75',
          blue: '#77b7f0',
          magenta: '#c3a1e8',
          cyan: '#68d4db',
          white: '#dbe6ee',
        },
      });
      const fit = new FitAddon();
      term.loadAddon(fit);
      // Remote escape sequences cannot open links or read/write the clipboard.
      const osc8 = term.parser.registerOscHandler(8, () => true);
      const osc52 = term.parser.registerOscHandler(52, () => true);
      term.open(host.current);
      terminal.current = term;
      let fitFrame = 0;
      const resize = () => {
        cancelAnimationFrame(fitFrame);
        fitFrame = requestAnimationFrame(() => {
          if (disposed || !host.current?.isConnected) return;
          fit.fit();
          const { cols, rows } = terminalSize(term.cols, term.rows);
          if (cols !== term.cols || rows !== term.rows) term.resize(cols, rows);
        });
      };
      const observer = new ResizeObserver(resize);
      observer.observe(host.current);
      const input = term.onData((data) => session.current?.input(data));
      const dimensions = term.onResize(({ cols, rows }) => session.current?.resize(cols, rows));
      resize();
      setInitialized(true);
      cleanup = () => {
        observer.disconnect();
        cancelAnimationFrame(fitFrame);
        input.dispose();
        dimensions.dispose();
        osc8.dispose();
        osc52.dispose();
        term.dispose();
        terminal.current = undefined;
      };
    };
    void initialize().catch((cause) => {
      if (!disposed) {
        setError(cause instanceof Error ? cause : new Error('无法加载终端组件。'));
        setPhase('error');
      }
    });
    return () => {
      disposed = true;
      cleanup?.();
    };
  }, []);

  useEffect(() => {
    const term = terminal.current;
    if (!initialized || !term) return;
    let active = true;
    const abort = new AbortController();
    let connection: TerminalSession | undefined;
    term.reset();
    term.options.disableStdin = true;
    setPhase('connecting');
    setMessage('正在验证权限并建立 SSH 连接…');
    setError(undefined);
    setTarget(undefined);
    const stop = () => {
      active = false;
      abort.abort();
      connection?.dispose();
      if (session.current === connection) session.current = undefined;
      term.options.disableStdin = true;
    };
    disconnect.current = () => {
      stop();
      setPhase('closed');
      setMessage('终端已断开。重新连接将开启新的 Shell 会话。');
    };
    const connect = async () => {
      const issued = await api<NodeTerminalTicket>(`/nodes/${node.id}/terminal-sessions`, {
        method: 'POST',
        body: '{}',
        signal: AbortSignal.any([abort.signal, AbortSignal.timeout(90000)]),
      });
      if (!active) return;
      setTarget(issued.target);
      const socket = new WebSocket(
        terminalSocketUrl(issued.websocket_path, window.location.origin),
      );
      connection = connectTerminalSocket(
        socket,
        issued.ticket,
        terminalSize(term.cols, term.rows),
        {
          ready() {
            if (!active) return;
            setPhase('ready');
            setMessage('已连接');
            term.options.disableStdin = false;
            connection?.resize(term.cols, term.rows);
            term.focus();
          },
          output(data, rendered) {
            if (active) term.write(data, rendered);
            else rendered();
          },
          closed(text) {
            if (!active) return;
            term.options.disableStdin = true;
            setPhase('closed');
            setMessage(text);
          },
          error(cause) {
            if (!active) return;
            term.options.disableStdin = true;
            setError(cause);
            setPhase('error');
            setMessage('终端连接失败');
          },
        },
      );
      session.current = connection;
    };
    void connect().catch((cause) => {
      if (!active) return;
      term.options.disableStdin = true;
      setError(cause instanceof Error ? cause : new Error('无法连接机器终端。'));
      setPhase('error');
      setMessage('终端连接失败');
    });
    return () => {
      stop();
      disconnect.current = undefined;
    };
  }, [initialized, generation, node.id]);

  const connected = phase === 'ready';
  const connecting = phase === 'connecting';
  return (
    <Modal
      wide
      preserveTerminalKeys
      title="机器终端"
      subtitle={node.name}
      close={close}
      className="node-terminal-modal"
    >
      <div className="node-terminal-toolbar" ref={toolbar}>
        <div className="node-terminal-status" role="status" aria-live="polite">
          <span className={`status-dot ${connected ? 'online' : ''}`} />
          <span>{message}</span>
        </div>
        {connecting || connected ? (
          <button
            className="button outline compact"
            type="button"
            onClick={() => disconnect.current?.()}
            disabled={!initialized}
          >
            {connecting ? <LoaderCircle size={15} className="spin" /> : <Unplug size={15} />}
            断开连接
          </button>
        ) : (
          <button
            className="button outline compact"
            type="button"
            disabled={!initialized}
            onClick={() => setGeneration((value) => value + 1)}
          >
            <Plug size={15} />
            重新连接
          </button>
        )}
      </div>
      {target && (
        <p className="node-terminal-target mono">
          {target.username}@{target.host}:{target.port}
        </p>
      )}
      <ErrorNotice error={error} title="终端连接失败" />
      <div
        ref={host}
        className="node-terminal-screen"
        data-terminal-input
        onKeyDownCapture={(event) => {
          if (event.shiftKey && event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            toolbar.current?.querySelector<HTMLElement>('button:not(:disabled)')?.focus();
          }
        }}
      />
      <p className="node-terminal-hint">
        命令按已保存的 SSH 账号权限执行，root 连接即拥有 root 权限。Ctrl+C 中断命令，Tab
        补全；Shift+Esc 返回工具栏。关闭窗口会结束此 Shell 会话。
      </p>
    </Modal>
  );
}
