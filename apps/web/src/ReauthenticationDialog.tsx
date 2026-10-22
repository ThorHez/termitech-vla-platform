import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { LoaderCircle, ShieldCheck } from 'lucide-react';
import { ErrorNotice, Field, Modal } from './components';

export function ReauthenticationDialog({
  username,
  confirm,
  cancel,
}: {
  username: string;
  confirm: (password: string) => Promise<void>;
  cancel: () => void;
}) {
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const passwordInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!busy) passwordInput.current?.focus();
  }, [busy]);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(undefined);
    try {
      await confirm(password);
      setPassword('');
    } catch (cause) {
      setPassword('');
      setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal
      title="验证当前账号密码"
      subtitle="此操作需要再次确认身份。验证有效期为 5 分钟。"
      close={cancel}
    >
      <form className="auth-form" onSubmit={submit}>
        <p>正在以 {username} 执行敏感操作，请输入该账号的登录密码。</p>
        <input
          type="text"
          name="username"
          value={username}
          autoComplete="username"
          hidden
          readOnly
        />
        <Field label="当前密码" required>
          <input
            type="password"
            ref={passwordInput}
            name="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
            maxLength={256}
            autoFocus
            disabled={busy}
          />
        </Field>
        <ErrorNotice error={error} title="密码验证失败" />
        <div className="auth-form-actions">
          <button className="button" type="button" onClick={cancel}>
            取消操作
          </button>
          <button className="button primary" type="submit" disabled={busy || !password}>
            {busy ? <LoaderCircle className="spin" size={16} /> : <ShieldCheck size={16} />}
            验证并继续
          </button>
        </div>
      </form>
    </Modal>
  );
}
