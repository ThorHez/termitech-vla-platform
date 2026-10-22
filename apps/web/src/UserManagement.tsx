import { useState } from 'react';
import { KeyRound, LoaderCircle, Plus, ShieldCheck, UserRound, Users } from 'lucide-react';
import { api, date, post, useApi } from './api';
import { roleDescriptions, roleLabels, useAuth } from './auth';
import type { User, UserRole } from './auth';
import { ErrorNotice, Field, Loading, Modal, PageHeading, Panel } from './components';

export interface AuditEntry {
  id: string;
  created_at: string;
  actor_id: string | null;
  actor_username: string | null;
  action: string;
  method: string;
  path: string;
  status: number;
  details?: Record<string, unknown> | null;
  remote_addr?: string | null;
}

export function UserManagement() {
  const { can, user: currentUser, refresh: refreshSession } = useAuth();
  const users = useApi<User[]>(can('users.manage') ? '/users' : null, 0);
  const [editor, setEditor] = useState<User | 'new'>();
  const [reset, setReset] = useState<User>();
  if (!can('users.manage')) return <PermissionDenied />;
  return (
    <>
      <PageHeading
        eyebrow="ACCESS MANAGEMENT"
        title="用户与权限"
        description="账号由管理员创建。权限适用于当前工作空间，机器 Agent 使用独立凭据。"
        actions={
          <button className="button primary" onClick={() => setEditor('new')}>
            <Plus size={16} />
            新建用户
          </button>
        }
      />
      <div className="role-grid">
        {(Object.keys(roleLabels) as UserRole[]).map((role) => (
          <div className="role-card" key={role}>
            <ShieldCheck size={19} />
            <strong>{roleLabels[role]}</strong>
            <p>{roleDescriptions[role]}</p>
          </div>
        ))}
      </div>
      <ErrorNotice error={users.error} retry={users.refresh} />
      <Panel title="用户列表" subtitle="停用、修改角色或重置密码后，原会话会失效。">
        {users.loading && !users.data ? (
          <Loading />
        ) : (
          <div className="auth-table-scroll">
            <table className="auth-table">
              <thead>
                <tr>
                  <th>用户</th>
                  <th>角色</th>
                  <th>状态</th>
                  <th>创建时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {users.data?.map((entry) => (
                  <tr key={entry.id}>
                    <td>
                      <strong>{entry.display_name || entry.username}</strong>
                      <small>
                        {entry.username}
                        {entry.id === currentUser?.id ? ' · 当前账号' : ''}
                      </small>
                    </td>
                    <td>{roleLabels[entry.role]}</td>
                    <td>
                      <span className={`badge ${entry.enabled ? 'green' : 'gray'}`}>
                        {entry.enabled ? '启用' : '停用'}
                      </span>
                      {entry.must_change_password && <small>下次登录需修改密码</small>}
                    </td>
                    <td>{date(entry.created_at)}</td>
                    <td>
                      <div className="auth-row-actions">
                        <button className="button ghost small" onClick={() => setEditor(entry)}>
                          编辑
                        </button>
                        <button className="button ghost small" onClick={() => setReset(entry)}>
                          <KeyRound size={13} />
                          重置密码
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      {editor && (
        <UserEditor
          target={editor}
          close={() => setEditor(undefined)}
          saved={async () => {
            setEditor(undefined);
            users.refresh();
            await refreshSession();
          }}
        />
      )}
      {reset && (
        <ResetPassword
          target={reset}
          close={() => setReset(undefined)}
          saved={async () => {
            setReset(undefined);
            users.refresh();
            await refreshSession();
          }}
        />
      )}
      <AuditLog />
    </>
  );
}

export function PermissionDenied() {
  return (
    <div className="auth-permission" role="alert">
      <ShieldCheck size={24} />
      <h2>没有访问权限</h2>
      <p>当前账号无法访问此功能。如需调整权限，请联系管理员。</p>
    </div>
  );
}

function UserEditor({
  target,
  close,
  saved,
}: {
  target: User | 'new';
  close: () => void;
  saved: () => Promise<void>;
}) {
  const isNew = target === 'new';
  const [username, setUsername] = useState(isNew ? '' : target.username);
  const [displayName, setDisplayName] = useState(isNew ? '' : target.display_name);
  const [role, setRole] = useState<UserRole>(isNew ? 'viewer' : target.role);
  const [enabled, setEnabled] = useState(isNew ? true : target.enabled);
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  return (
    <Modal title={isNew ? '新建用户' : `编辑用户 · ${target.username}`} close={close}>
      <form
        className="auth-form"
        onSubmit={async (event) => {
          event.preventDefault();
          setBusy(true);
          setError(undefined);
          try {
            if (isNew)
              await post('/users', {
                username: username.trim(),
                display_name: displayName.trim(),
                role,
                password,
              });
            else
              await api(`/users/${encodeURIComponent(target.id)}`, {
                method: 'PATCH',
                body: JSON.stringify({ display_name: displayName.trim(), role, enabled }),
              });
            setPassword('');
            await saved();
          } catch (cause) {
            setError(cause instanceof Error ? cause : new Error(String(cause)));
          } finally {
            setBusy(false);
          }
        }}
      >
        <Field
          label="用户名"
          required
          hint="3–80 个字母、数字或 . _ -，以字母或数字开头；忽略大小写，创建后不能修改。"
        >
          <input
            autoComplete="off"
            value={username}
            required
            minLength={3}
            maxLength={80}
            pattern="[a-zA-Z0-9][a-zA-Z0-9_.-]{2,79}"
            disabled={!isNew || busy}
            onChange={(event) => setUsername(event.target.value)}
          />
        </Field>
        <Field label="显示名称">
          <input
            maxLength={120}
            value={displayName}
            disabled={busy}
            onChange={(event) => setDisplayName(event.target.value)}
          />
        </Field>
        <Field label="角色" required hint={roleDescriptions[role]}>
          <select
            value={role}
            disabled={busy}
            onChange={(event) => setRole(event.target.value as UserRole)}
          >
            {(Object.keys(roleLabels) as UserRole[]).map((key) => (
              <option key={key} value={key}>
                {roleLabels[key]}
              </option>
            ))}
          </select>
        </Field>
        {isNew ? (
          <Field label="初始密码" required hint="至少 12 个字符；用户首次登录时必须修改。">
            <input
              type="password"
              autoComplete="new-password"
              required
              minLength={12}
              maxLength={256}
              disabled={busy}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </Field>
        ) : (
          <label className="auth-check">
            <input
              type="checkbox"
              checked={enabled}
              disabled={busy}
              onChange={(event) => setEnabled(event.target.checked)}
            />
            启用此用户
          </label>
        )}
        <ErrorNotice error={error} title="无法保存用户" />
        <div className="auth-form-actions">
          <button type="button" className="button outline" disabled={busy} onClick={close}>
            取消
          </button>
          <button className="button primary" disabled={busy}>
            {busy && <LoaderCircle className="spin" size={15} />}保存用户
          </button>
        </div>
      </form>
    </Modal>
  );
}

function ResetPassword({
  target,
  close,
  saved,
}: {
  target: User;
  close: () => void;
  saved: () => Promise<void>;
}) {
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  return (
    <Modal title={`重置密码 · ${target.username}`} close={close}>
      <form
        className="auth-form"
        onSubmit={async (event) => {
          event.preventDefault();
          if (password !== confirmation) {
            setError(new Error('两次输入的密码不一致。'));
            return;
          }
          setBusy(true);
          setError(undefined);
          try {
            await post(`/users/${encodeURIComponent(target.id)}/reset-password`, { password });
            setPassword('');
            setConfirmation('');
            await saved();
          } catch (cause) {
            setError(cause instanceof Error ? cause : new Error(String(cause)));
          } finally {
            setBusy(false);
          }
        }}
      >
        <p className="muted">重置会立即注销该用户的全部会话，下次登录必须修改密码。</p>
        <Field label="临时密码" required>
          <input
            type="password"
            autoComplete="new-password"
            minLength={12}
            maxLength={256}
            required
            disabled={busy}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </Field>
        <Field label="确认临时密码" required>
          <input
            type="password"
            autoComplete="new-password"
            minLength={12}
            maxLength={256}
            required
            disabled={busy}
            value={confirmation}
            onChange={(event) => setConfirmation(event.target.value)}
          />
        </Field>
        <ErrorNotice error={error} title="重置失败" />
        <div className="auth-form-actions">
          <button type="button" className="button outline" disabled={busy} onClick={close}>
            取消
          </button>
          <button className="button primary" disabled={busy}>
            {busy && <LoaderCircle className="spin" size={15} />}重置并注销会话
          </button>
        </div>
      </form>
    </Modal>
  );
}

function AuditLog() {
  const { can } = useAuth();
  const audit = useApi<AuditEntry[]>(can('audit.read') ? '/audit?limit=100' : null, 15000);
  if (!can('audit.read')) return null;
  return (
    <Panel
      className="audit-panel"
      title="操作审计"
      subtitle="最近 100 条记录，记录登录、用户管理与业务修改的结果。"
    >
      <ErrorNotice error={audit.error} retry={audit.refresh} />
      {audit.loading && !audit.data ? (
        <Loading />
      ) : (
        <div className="auth-table-scroll">
          <table className="auth-table">
            <thead>
              <tr>
                <th>时间</th>
                <th>用户</th>
                <th>操作</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              {audit.data?.length ? (
                audit.data.map((entry) => (
                  <tr key={entry.id}>
                    <td className="nowrap">{date(entry.created_at)}</td>
                    <td>{entry.actor_username || '未登录'}</td>
                    <td>
                      <strong>{entry.action}</strong>
                      <small>
                        <code>
                          {entry.method} {entry.path}
                        </code>
                      </small>
                    </td>
                    <td>
                      <span className={`badge ${entry.status < 400 ? 'green' : 'amber'}`}>
                        {entry.status < 400 ? '成功' : '拒绝 / 失败'} · {entry.status}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="muted">
                    暂无审计记录
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

export function AccountIdentity() {
  const { user, expiresAt } = useAuth();
  if (!user) return null;
  return (
    <div className="account-identity">
      <span className="mini-icon">
        <UserRound size={21} />
      </span>
      <div>
        <strong>{user.display_name || user.username}</strong>
        <p>
          {user.username} · {roleLabels[user.role]}
        </p>
        {expiresAt && <small>会话到期 {date(expiresAt)}</small>}
      </div>
    </div>
  );
}

export function UserManagementLink() {
  const { can } = useAuth();
  if (!can('users.manage')) return null;
  return (
    <a className="button outline" href="#/users">
      <Users size={16} />
      管理用户与权限
    </a>
  );
}
