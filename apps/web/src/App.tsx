import { useEffect, useRef, useState } from 'react';
import {
  ArrowRight,
  Bell,
  ChevronDown,
  CircleHelp,
  Cpu,
  Database,
  KeyRound,
  LayoutDashboard,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  Menu,
  RefreshCw,
  Search,
  Server,
  Settings2,
  ShieldCheck,
  Terminal,
  Workflow,
} from 'lucide-react';
import { api, clearToken, getToken, navigate, saveToken, useRoute } from './api';
import { Badge, ErrorNotice, Field, PageHeading, Panel } from './components';
import { DataProvider, useData } from './data';
import { AssetsPage, NodePage, NodesPage, OverviewPage, RunsPage, TemplatesPage } from './pages';
import RunDetail from './RunDetail';
import RunWizard from './RunWizard';
import type { Overview } from './types';

const navigation = [
  { path: '/overview', label: '训练工作台', icon: LayoutDashboard },
  { path: '/nodes', label: '机器资源', icon: Server },
  { path: '/assets', label: '数据与模型', icon: Database },
  { path: '/templates', label: '训练模板', icon: Layers3 },
  { path: '/runs', label: '训练任务', icon: Workflow },
];

export default function App() {
  const [token, setToken] = useState(getToken());
  return (
    <DataProvider enabled={!!token} key={token}>
      <Shell authenticated={!!token} onAuth={() => setToken(getToken())} />
    </DataProvider>
  );
}

function Shell({ authenticated, onAuth }: { authenticated: boolean; onAuth: () => void }) {
  const route = useRoute();
  const path = route.split('?')[0];
  const queryParams = new URLSearchParams(route.split('?')[1] || '');
  const [mobileNav, setMobileNav] = useState(false);
  const [wizard, setWizard] = useState<{ nodeId?: string; templateId?: string }>();
  const { overview, alerts, runs, refresh } = useData();
  const section = navigation.find((item) => path.startsWith(item.path));
  const contentRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    setMobileNav(false);
    contentRef.current?.scrollTo(0, 0);
    document.title = `${section?.label || '平台设置'} · Termitech VLA`;
  }, [path, section?.label]);
  const createRun = (nodeId?: string, templateId?: string) => setWizard({ nodeId, templateId });
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
          <a className={`nav-item ${path === '/settings' ? 'active' : ''}`} href="#/settings">
            <Settings2 size={19} strokeWidth={1.7} />
            <span>平台设置</span>
          </a>
          <div className="sidebar-status">
            <span
              className={`status-dot ${authenticated && overview.data && !overview.error ? 'online' : ''}`}
            />
            <span>
              {authenticated
                ? overview.error
                  ? '管理服务连接异常'
                  : overview.data
                    ? '管理服务已连接'
                    : '正在连接服务'
                : '尚未连接管理服务'}
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
            <a className="avatar" href="#/settings" title="管理员设置">
              AD
            </a>
          </div>
        </header>
        <div className="content-scroll" ref={contentRef}>
          <main id="main-content" tabIndex={-1} className="main-content">
            {path === '/overview' ? (
              <OverviewPage createRun={() => createRun()} />
            ) : path === '/nodes' ? (
              <NodesPage />
            ) : path.startsWith('/nodes/') ? (
              <NodePage
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
              <RunDetail key={path} id={decodeURIComponent(path.slice('/runs/'.length))} />
            ) : path === '/settings' ? (
              <SettingsPage authenticated={authenticated} onAuth={onAuth} />
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
      {wizard && (
        <RunWizard
          initialNode={wizard.nodeId}
          initialTemplate={wizard.templateId}
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

function SettingsPage({ authenticated, onAuth }: { authenticated: boolean; onAuth: () => void }) {
  const { overview } = useData();
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [success, setSuccess] = useState(false);
  return (
    <>
      <PageHeading
        eyebrow="PLATFORM SETTINGS"
        title="平台设置"
        description="管理当前浏览器与控制面的连接。"
      />
      <div className="settings-grid">
        <Panel title="管理服务连接" subtitle="首个版本使用单项目管理员令牌">
          <form
            className="settings-form"
            onSubmit={async (event) => {
              event.preventDefault();
              const previous = getToken();
              setBusy(true);
              setError(undefined);
              setSuccess(false);
              saveToken(token);
              try {
                await api<Overview>('/overview');
                setSuccess(true);
                onAuth();
              } catch (err) {
                if (previous) saveToken(previous);
                else clearToken();
                setError(err as Error);
              } finally {
                setBusy(false);
              }
            }}
          >
            <div className="connection-state">
              <span className="mini-icon">
                <Terminal size={20} />
              </span>
              <div>
                <strong>控制面 API</strong>
                <code>{window.location.origin}/api/v1</code>
              </div>
              <Badge
                state={authenticated && overview.data && !overview.error ? 'ONLINE' : 'OFFLINE'}
              />
            </div>
            <Field
              label="管理员访问令牌"
              required
              hint="填写管理服务配置的 VLA_ADMIN_TOKEN；与 Agent 登记令牌不同。"
            >
              <div className="input-icon">
                <KeyRound size={17} />
                <input
                  autoComplete="off"
                  type="password"
                  required
                  minLength={1}
                  placeholder={authenticated ? '输入新令牌以更新连接' : '输入管理员访问令牌'}
                  value={token}
                  onChange={(event) => setToken(event.target.value)}
                />
              </div>
            </Field>
            <div className="inline-note">
              <LockKeyhole size={16} />
              <span>令牌仅保存在当前标签页会话中，通过请求头发送到同源管理服务。</span>
            </div>
            <ErrorNotice error={error} />
            {success && <div className="inline-note green">连接成功，已载入管理服务数据。</div>}
            <div className="settings-form-actions">
              <button className="button primary" disabled={busy} type="submit">
                {busy ? <LoaderCircle className="spin" size={16} /> : <ShieldCheck size={16} />}
                {authenticated ? '更新并验证连接' : '验证并连接'}
              </button>
              {authenticated && (
                <button
                  className="button ghost"
                  type="button"
                  onClick={() => {
                    clearToken();
                    onAuth();
                  }}
                >
                  <LogOut size={16} />
                  退出当前连接
                </button>
              )}
            </div>
          </form>
        </Panel>
        <div>
          <Panel title="部署说明">
            <div className="deployment-summary">
              <div>
                <Server size={19} />
                <section>
                  <strong>一个管理控制面</strong>
                  <p>提供资源目录、训练任务、调度与可视化。</p>
                </section>
              </div>
              <div>
                <Cpu size={19} />
                <section>
                  <strong>每台机器一个 Agent</strong>
                  <p>汇报资源与心跳，执行本机预检和训练。</p>
                </section>
              </div>
              <div>
                <Database size={19} />
                <section>
                  <strong>数据与模型保留本地</strong>
                  <p>仅在同一台机器上选择输入资产进行训练。</p>
                </section>
              </div>
            </div>
          </Panel>
          <div className="settings-help">
            <CircleHelp size={19} />
            <div>
              <strong>服务无法连接？</strong>
              <p>
                开发模式请确认 API 服务运行于 127.0.0.1:8000。正式部署请为 /api 配置同源反向代理。
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
