import { useEffect, useRef, useState } from 'react';
import {
  ArrowRight,
  Bell,
  ChevronDown,
  Database,
  LayoutDashboard,
  Layers3,
  LoaderCircle,
  LogOut,
  Menu,
  RefreshCw,
  Rocket,
  Search,
  Server,
  Settings2,
  ShieldCheck,
  UserRound,
  Users,
  Terminal,
  Workflow,
} from 'lucide-react';
import { navigate, useRoute } from './api';
import { AuthProvider, roleLabels, useAuth } from './auth';
import { AccountIdentity, UserManagement, UserManagementLink } from './UserManagement';
import './auth.css';
import { Badge, ErrorNotice, Field, PageHeading, Panel } from './components';
import { DataProvider, useData } from './data';
import { AssetsPage, NodePage, NodesPage, OverviewPage, RunsPage, TemplatesPage } from './pages';
import RunDetail from './RunDetail';
import RunWizard from './RunWizard';
import PolicyDeployments from './PolicyDeployments';
import { ThemeControl, ThemeProvider, ThemeSettings } from './theme';
import type { Run } from './types';

const navigation = [
  { path: '/overview', label: '训练工作台', icon: LayoutDashboard },
  { path: '/nodes', label: '机器资源', icon: Server },
  { path: '/assets', label: '数据与模型', icon: Database },
  { path: '/templates', label: '训练模板', icon: Layers3 },
  { path: '/runs', label: '训练任务', icon: Workflow },
  { path: '/deployments', label: '真机部署', icon: Rocket },
];

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <SessionGate />
      </AuthProvider>
    </ThemeProvider>
  );
}

function SessionGate() {
  const { user, loading } = useAuth();
  if (loading && !user)
    return (
      <div className="auth-loading" role="status">
        <LoaderCircle className="spin" size={24} />
        正在验证登录状态…
      </div>
    );
  if (!user) return <LoginPage />;
  if (user.must_change_password) return <RequiredPasswordChange />;
  return (
    <DataProvider enabled key={user.id}>
      <Shell />
    </DataProvider>
  );
}

function Shell() {
  const { user, can, error: sessionError } = useAuth();
  const route = useRoute();
  const path = route.split('?')[0];
  const queryParams = new URLSearchParams(route.split('?')[1] || '');
  const [mobileNav, setMobileNav] = useState(false);
  const [wizard, setWizard] = useState<{ nodeId?: string; templateId?: string; sourceRun?: Run }>();
  const { overview, alerts, runs, refresh } = useData();
  const section =
    navigation.find((item) => path.startsWith(item.path)) ||
    (path === '/users' ? { label: '用户与权限' } : undefined);
  const contentRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    setMobileNav(false);
    contentRef.current?.scrollTo(0, 0);
    document.title = `${section?.label || '平台设置'} · Termitech VLA`;
  }, [path, section?.label]);
  const createRun = (nodeId?: string, templateId?: string) => {
    if (can('training.write')) setWizard({ nodeId, templateId });
  };
  return (
    <div className="app-shell">
      <a
        href="#main-content"
        className="skip-link"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById('main-content')?.focus();
        }}
      >
        跳转到主要内容
      </a>
      {mobileNav && (
        <button className="nav-scrim" aria-label="关闭导航" onClick={() => setMobileNav(false)} />
      )}
      <aside className={`sidebar ${mobileNav ? 'open' : ''}`}>
        <a className="brand" href="#/overview">
          <span className="brand-symbol">
            <span />
            <span />
            <span />
          </span>
          <div>
            <strong>
              TERMiTECH<span> / </span>
            </strong>
            <small>VLA TRAINING PLATFORM</small>
          </div>
        </a>
        <div className="workspace-selector">
          <div className="workspace-icon">
            <Layers3 size={17} />
          </div>
          <div>
            <strong>默认工作空间</strong>
            <span>研究与训练</span>
          </div>
          <ChevronDown size={14} />
        </div>
        <div className="nav-label">工作空间</div>
        <nav aria-label="主导航">
          {navigation.map((item) => (
            <a
              className={`nav-item ${path.startsWith(item.path) ? 'active' : ''}`}
              key={item.path}
              href={`#${item.path}`}
              aria-current={path.startsWith(item.path) ? 'page' : undefined}
            >
              <item.icon size={19} strokeWidth={1.7} />
              <span>{item.label}</span>
              {item.path === '/runs' && !!overview.data?.runs_active && (
                <b>{overview.data.runs_active}</b>
              )}
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <div className="sidebar-note-icon">
              <GitSymbol />
            </div>
            <strong>训练代码，全程可追溯</strong>
            <p>
              每次启动记录实际 commit
              <br />
              与本地修改状态。
            </p>
            <a href="#/templates">
              浏览训练模板
              <ArrowRight size={13} />
            </a>
          </div>
          {can('users.manage') && (
            <a className={`nav-item ${path === '/users' ? 'active' : ''}`} href="#/users">
              <Users size={19} strokeWidth={1.7} />
              <span>用户与权限</span>
            </a>
          )}
          <a className={`nav-item ${path === '/settings' ? 'active' : ''}`} href="#/settings">
            <Settings2 size={19} strokeWidth={1.7} />
            <span>平台设置</span>
          </a>
          <div className="sidebar-status">
            <span className={`status-dot ${overview.data && !overview.error ? 'online' : ''}`} />
            <span>
              {overview.error
                ? '管理服务连接异常'
                : overview.data
                  ? '管理服务已连接'
                  : '正在连接服务'}
            </span>
            <small>v0.1</small>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button mobile-menu"
              aria-label="打开导航"
              onClick={() => setMobileNav(true)}
            >
              <Menu size={21} />
            </button>
            <span className="breadcrumb">
              工作空间<span>/</span>
              <strong>{section?.label || '平台设置'}</strong>
            </span>
          </div>
          <div className="topbar-actions">
            <GlobalSearch />
            <span className="environment-label">
              <span />
              本地资产模式
            </span>
            <ThemeControl />
            <button
              className="icon-button"
              title="刷新数据"
              aria-label="刷新数据"
              onClick={refresh}
            >
              <RefreshCw size={17} className={overview.loading && overview.data ? 'spin' : ''} />
            </button>
            <a
              className="icon-button notifications"
              href="#/overview"
              aria-label={`当前 ${alerts.data?.length || 0} 条异常`}
              title="查看异常"
            >
              <Bell size={18} />
              {!!alerts.data?.length && <span />}
            </a>
            <a
              className="avatar"
              href="#/settings"
              title={`${user?.display_name || user?.username} · ${user ? roleLabels[user.role] : ''}`}
              aria-label="当前账号设置"
            >
              {(user?.display_name || user?.username || '?').slice(0, 2).toUpperCase()}
            </a>
          </div>
        </header>
        <div className="content-scroll" ref={contentRef}>
          <main id="main-content" tabIndex={-1} className="main-content">
            <ErrorNotice error={sessionError} title="会话验证暂时不可用" />
            {path === '/overview' ? (
              <OverviewPage createRun={() => createRun()} />
            ) : path === '/nodes' ? (
              <NodesPage />
            ) : path.startsWith('/nodes/') ? (
              <NodePage
                key={path}
                id={decodeURIComponent(path.slice('/nodes/'.length))}
                createRun={createRun}
              />
            ) : path === '/assets' ? (
              <AssetsPage
                key={queryParams.get('node') || 'all'}
                nodeFilter={queryParams.get('node') || undefined}
              />
            ) : path === '/templates' ? (
              <TemplatesPage createRun={createRun} />
            ) : path === '/runs' ? (
              <RunsPage createRun={() => createRun()} />
            ) : path.startsWith('/runs/') ? (
              <RunDetail
                key={path}
                id={decodeURIComponent(path.slice('/runs/'.length))}
                onRetry={(sourceRun) => {
                  if (can('training.write')) setWizard({ sourceRun });
                }}
              />
            ) : path === '/deployments' || path.startsWith('/deployments/') ? (
              <PolicyDeployments
                path={path}
                tab={queryParams.get('tab') || 'overview'}
                initialRun={queryParams.get('run') || undefined}
              />
            ) : path === '/settings' ? (
              <SettingsPage />
            ) : path === '/users' ? (
              <UserManagement />
            ) : (
              <>
                <PageHeading title="页面不存在" description="这个页面可能已移动，请返回工作台。" />
                <button className="button primary" onClick={() => navigate('/overview')}>
                  返回工作台
                </button>
              </>
            )}
          </main>
          <footer className="page-footer">
            <span>TERMiTECH VLA · 让训练更清晰</span>
            <span>
              {runs.data?.length ? `${runs.data.length} 项训练记录` : '数据存于本机 · 管理集中统一'}
            </span>
          </footer>
        </div>
      </div>
      {wizard && can('training.write') && (
        <RunWizard
          key={wizard.sourceRun?.id || 'new-training'}
          initialNode={wizard.nodeId}
          initialTemplate={wizard.templateId}
          sourceRun={wizard.sourceRun}
          close={() => setWizard(undefined)}
        />
      )}
    </div>
  );
}

function GitSymbol() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M7 7v10m10-10v3a5 5 0 0 1-5 5H7" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="7" cy="5" r="2" fill="#162f39" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="17" cy="5" r="2" fill="#162f39" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="7" cy="19" r="2" fill="#162f39" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function GlobalSearch() {
  const { nodes, assets, runs } = useData();
  const [query, setQuery] = useState('');
  const [focused, setFocused] = useState(false);
  const lower = query.toLowerCase();
  const results = query.trim()
    ? [
        ...(runs.data || [])
          .filter((item) => item.name.toLowerCase().includes(lower))
          .slice(0, 3)
          .map((item) => ({
            id: item.id,
            title: item.name,
            kind: '训练任务',
            path: `/runs/${item.id}`,
            icon: Workflow,
          })),
        ...(nodes.data || [])
          .filter((item) => `${item.name} ${item.hostname}`.toLowerCase().includes(lower))
          .slice(0, 3)
          .map((item) => ({
            id: item.id,
            title: item.name,
            kind: '机器',
            path: `/nodes/${item.id}`,
            icon: Server,
          })),
        ...(assets.data || [])
          .filter((item) => item.name.toLowerCase().includes(lower))
          .slice(0, 3)
          .map((item) => ({
            id: item.id,
            title: item.name,
            kind: '资产',
            path: `/assets?node=${item.node_id}`,
            icon: Database,
          })),
      ]
    : [];
  return (
    <div className="global-search">
      <Search size={15} />
      <input
        aria-label="全局搜索"
        value={query}
        placeholder="搜索训练、机器或资产"
        onFocus={() => setFocused(true)}
        onBlur={(event) => {
          if (!event.currentTarget.parentElement?.contains(event.relatedTarget)) setFocused(false);
        }}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            setFocused(false);
            event.currentTarget.blur();
          }
        }}
      />
      {focused && query && (
        <div className="search-results">
          {results.length ? (
            results.map((result) => (
              <a
                key={`${result.kind}-${result.id}`}
                href={`#${result.path}`}
                onClick={() => {
                  setQuery('');
                  setFocused(false);
                }}
              >
                <result.icon size={16} />
                <strong>{result.title}</strong>
                <span>{result.kind}</span>
              </a>
            ))
          ) : (
            <p>没有找到匹配的资源</p>
          )}
        </div>
      )}
    </div>
  );
}

export function LoginPage() {
  const { login, error: connectionError, refresh, expired } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  return (
    <div className="login-page">
      <div className="theme-auth-control">
        <ThemeControl />
      </div>
      <div className="login-card">
        <div className="auth-brand">
          <span className="mini-icon">
            <ShieldCheck size={26} />
          </span>
          <span>TERMiTECH VLA</span>
        </div>
        <div className="eyebrow">TRAINING PLATFORM</div>
        <h1>登录训练平台</h1>
        <p className="login-description">使用管理员为你分配的账号访问工作空间。</p>
        {expired && (
          <div className="inline-note" role="status">
            登录已失效，请重新登录。
          </div>
        )}
        <ErrorNotice
          error={connectionError}
          title="无法验证登录状态"
          retry={() => void refresh()}
        />
        <form
          className="auth-form"
          onSubmit={async (event) => {
            event.preventDefault();
            setBusy(true);
            setError(undefined);
            try {
              await login(username.trim(), password);
              setPassword('');
            } catch (cause) {
              setError(cause instanceof Error ? cause : new Error(String(cause)));
            } finally {
              setBusy(false);
            }
          }}
        >
          <Field label="用户名" required>
            <input
              name="username"
              autoComplete="username"
              autoCapitalize="none"
              spellCheck={false}
              autoFocus
              required
              maxLength={80}
              disabled={busy}
              value={username}
              onChange={(event) => setUsername(event.target.value)}
            />
          </Field>
          <Field label="密码" required>
            <input
              name="password"
              type="password"
              autoComplete="current-password"
              required
              maxLength={256}
              disabled={busy}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </Field>
          <ErrorNotice error={error} title="登录失败" />
          <button className="button primary login-submit" disabled={busy}>
            {busy ? <LoaderCircle size={17} className="spin" /> : <UserRound size={17} />}
            {busy ? '正在登录' : '登录'}
          </button>
        </form>
        <p className="login-help">
          不开放公开注册。首次部署请由服务维护人员使用初始化管理员命令创建账号。
        </p>
      </div>
      <div className="login-footer">TERMiTECH VLA · 让训练更清晰</div>
    </div>
  );
}

export function PasswordChangeForm({ requiredChange = false }: { requiredChange?: boolean }) {
  const { changePassword } = useAuth();
  const [current, setCurrent] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [success, setSuccess] = useState(false);
  return (
    <form
      className="auth-form"
      onSubmit={async (event) => {
        event.preventDefault();
        setSuccess(false);
        setError(undefined);
        if (password !== confirmation) {
          setError(new Error('两次输入的新密码不一致。'));
          return;
        }
        if (password === current) {
          setError(new Error('新密码必须与当前密码不同。'));
          return;
        }
        setBusy(true);
        try {
          await changePassword(current, password);
          setCurrent('');
          setPassword('');
          setConfirmation('');
          setSuccess(true);
        } catch (cause) {
          setError(cause instanceof Error ? cause : new Error(String(cause)));
        } finally {
          setBusy(false);
        }
      }}
    >
      {requiredChange && (
        <p className="muted">此账号需要先修改初始或临时密码，完成后即可进入工作空间。</p>
      )}
      <Field label="当前密码" required>
        <input
          name="current-password"
          type="password"
          autoComplete="current-password"
          maxLength={256}
          required
          disabled={busy}
          value={current}
          onChange={(event) => setCurrent(event.target.value)}
        />
      </Field>
      <Field label="新密码" required hint="至少 12 个字符，请使用独立且难以猜测的密码。">
        <input
          name="new-password"
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
      <Field label="确认新密码" required>
        <input
          name="confirm-password"
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
      <ErrorNotice error={error} title="修改密码失败" />
      {success && (
        <div className="inline-note green" role="status">
          密码已修改，其他会话已注销。
        </div>
      )}
      <div>
        <button className="button primary" disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <ShieldCheck size={16} />}
          {requiredChange ? '更新密码并进入平台' : '修改密码'}
        </button>
      </div>
    </form>
  );
}

function LogoutButton() {
  const { logout } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  return (
    <div>
      <button
        type="button"
        className="button ghost"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setError(undefined);
          try {
            await logout();
          } catch (cause) {
            setError(cause instanceof Error ? cause : new Error(String(cause)));
          } finally {
            setBusy(false);
          }
        }}
      >
        {busy ? <LoaderCircle size={16} className="spin" /> : <LogOut size={16} />}退出登录
      </button>
      <ErrorNotice error={error} title="退出失败" />
    </div>
  );
}

export function RequiredPasswordChange() {
  return (
    <div className="login-page">
      <div className="theme-auth-control">
        <ThemeControl />
      </div>
      <div className="login-card">
        <div className="auth-brand">
          <ShieldCheck size={26} />
          TERMiTECH VLA
        </div>
        <h1>更新登录密码</h1>
        <AccountIdentity />
        <PasswordChangeForm requiredChange />
        <div className="auth-forced-logout">
          <LogoutButton />
        </div>
      </div>
    </div>
  );
}

function SettingsPage() {
  const { overview } = useData();
  return (
    <>
      <PageHeading
        eyebrow="PLATFORM SETTINGS"
        title="账号与平台设置"
        description="管理界面外观、当前账号与登录密码。"
      />
      <div className="theme-settings-panel">
        <Panel title="界面外观" subtitle="选择适合当前环境的显示模式。">
          <ThemeSettings />
        </Panel>
      </div>
      <div className="settings-grid">
        <Panel title="当前账号">
          <div className="settings-form">
            <AccountIdentity />
            <UserManagementLink />
            <LogoutButton />
            <div className="connection-state">
              <span className="mini-icon">
                <Terminal size={20} />
              </span>
              <div>
                <strong>控制面 API</strong>
                <code>{window.location.origin}/api/v1</code>
              </div>
              <Badge state={overview.data && !overview.error ? 'ONLINE' : 'OFFLINE'} />
            </div>
          </div>
        </Panel>
        <Panel title="修改密码" subtitle="保存后保留当前登录，注销该账号的其他会话。">
          <div className="settings-form">
            <PasswordChangeForm />
          </div>
        </Panel>
      </div>
    </>
  );
}
