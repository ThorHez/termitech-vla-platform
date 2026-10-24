import { useEffect, useId, useRef, useState } from 'react';
import { LoaderCircle } from 'lucide-react';
import { api } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Field, Modal } from './components';
import { useData } from './data';
import type { Node } from './types';

export function nodeNameError(value: string) {
  const name = value.trim();
  if (!name) return '请输入机器名称。';
  if (Array.from(name).length > 200) return '机器名称最多 200 个字符。';
  if (/[\u0000-\u001f\u007f-\u009f]/.test(name)) return '机器名称不能包含控制字符。';
  return undefined;
}

export function NodeRename(props: { node: Node; close: () => void; changed?: () => void }) {
  const { can } = useAuth();
  if (!can('machines.manage'))
    return (
      <Modal title="权限不足" close={props.close}>
        重命名机器需要管理员权限。
      </Modal>
    );
  return <NodeRenameForm key={props.node.id} {...props} />;
}

function NodeRenameForm({ node, close, changed }: Parameters<typeof NodeRename>[0]) {
  const { nodes } = useData();
  const [name, setName] = useState(node.name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const input = useRef<HTMLInputElement>(null);
  const pending = useRef(false);
  const formId = useId();
  const hintId = useId();
  const validationId = useId();
  const validation = nodeNameError(name);
  const unchanged = name.trim() === node.name;
  useEffect(() => {
    input.current?.focus();
    input.current?.select();
  }, []);
  const safeClose = () => {
    if (!pending.current) close();
  };
  const submit = async () => {
    if (pending.current || validation || unchanged) return;
    pending.current = true;
    setBusy(true);
    setError(undefined);
    try {
      await api<Node>(`/nodes/${node.id}`, {
        method: 'PATCH',
        body: JSON.stringify({ name: name.trim() }),
      });
      nodes.refresh();
      changed?.();
      close();
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      pending.current = false;
      setBusy(false);
    }
  };
  return (
    <Modal
      title="重命名机器"
      close={safeClose}
      footer={
        <>
          <button className="button outline" disabled={busy} onClick={safeClose}>
            取消
          </button>
          <button
            className="button primary"
            type="submit"
            form={formId}
            disabled={busy || Boolean(validation) || unchanged}
          >
            {busy && <LoaderCircle size={15} className="spin" />}
            {busy ? '保存中…' : '保存名称'}
          </button>
        </>
      }
    >
      <ErrorNotice error={error} />
      <form
        id={formId}
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
        aria-busy={busy}
      >
        <Field label="机器名称" required>
          <input
            ref={input}
            name="name"
            value={name}
            required
            disabled={busy}
            autoComplete="off"
            aria-invalid={Boolean(validation)}
            aria-describedby={`${hintId}${validation ? ` ${validationId}` : ''}`}
            onChange={(event) => {
              setName(event.target.value);
              setError(undefined);
            }}
          />
          <span className="field-hint" id={hintId}>
            用于机器列表、训练任务和数据资产中的机器名称，最多 200 个字符。
          </span>
          {validation && (
            <span className="field-hint required" id={validationId} role="alert">
              {validation}
            </span>
          )}
        </Field>
      </form>
    </Modal>
  );
}
