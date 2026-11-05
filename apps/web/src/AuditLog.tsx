import { Fragment, useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, RefreshCw, Search } from 'lucide-react';
import { useAuth } from './auth';
import type { User } from './auth';
import { ErrorNotice, Field, Loading, Panel } from './components';
import {
  createAuditEventsClient,
  firstAuditPage,
  nextAuditPage,
  previousAuditPage,
} from './auditLogData';
import {
  auditActionGroups,
  auditDetailRows,
  auditEventsPath,
  auditOperationLabel,
  auditOutcome,
  auditOutcomeLabels,
  auditTarget,
  auditTime,
  emptyAuditFilters,
  safeAuditPath,
  sanitizeAuditEntry,
  submitAuditFilters,
} from './auditLogPresentation';
import type { AuditEntry, AuditEventsPage, AuditFilters } from './auditLogPresentation';
import './audit-log.css';

export interface AuditActorSelection {
  actorId: string;
  revision: number;
}
export type AuditUser = Pick<User, 'id' | 'username' | 'display_name' | 'enabled'>;

export function AuditLog({
  users = [],
  actorSelection,
}: {
  users?: AuditUser[];
  actorSelection?: AuditActorSelection;
}) {
  const { user, can } = useAuth();
  if (!user?.id || user.role !== 'admin' || !can('audit.read')) return null;
  return (
    <AuditLogContent
      key={`${user.id}:${actorSelection?.actorId || ''}:${actorSelection?.revision || 0}`}
      users={users}
      initialActorId={actorSelection?.actorId || ''}
    />
  );
}

function AuditLogContent({
  users,
  initialActorId,
}: {
  users: AuditUser[];
  initialActorId: string;
}) {
  const initial = () => ({ ...emptyAuditFilters(), actor_id: initialActorId });
  const [draft, setDraft] = useState(initial);
  const [customAction, setCustomAction] = useState(false);
  const [filters, setFilters] = useState(initial);
  const [paging, setPaging] = useState(firstAuditPage);
  const [revision, setRevision] = useState(0);
  const [automatic, setAutomatic] = useState(false);
  const [filterError, setFilterError] = useState<Error>();
  const [view, setView] = useState<{
    path: string;
    revision: number;
    page?: AuditEventsPage;
    error?: Error;
    loading: boolean;
  }>();
  const client = useRef(createAuditEventsClient());
  const path = auditEventsPath(filters, paging.cursor);
  const current = view?.path === path && view.revision === revision ? view : undefined;
  const loading = !current || current.loading;
  const page = current?.page;
  useEffect(() => {
    const scope = client.current;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      setView({ path, revision, loading: true });
      try {
        const result = await scope.load(filters, paging.cursor);
        if (!active) return;
        setView({ path, revision, page: result, loading: false });
        if (automatic && !paging.cursor) timer = setTimeout(() => void load(), 15000);
      } catch (cause) {
        if (active)
          setView({
            path,
            revision,
            error: cause instanceof Error ? cause : new Error('无法读取操作记录。'),
            loading: false,
          });
      }
    };
    void load();
    return () => {
      active = false;
      scope.cancel();
      clearTimeout(timer);
    };
  }, [path, revision, automatic]);
  const query = (next: AuditFilters) => {
    try {
      const validated = submitAuditFilters(next);
      setFilterError(undefined);
      setFilters(validated);
      setPaging(firstAuditPage());
      setRevision((value) => value + 1);
    } catch (cause) {
      setFilterError(cause as Error);
    }
  };
  const refresh = () => {
    setPaging(firstAuditPage());
    setRevision((value) => value + 1);
  };
  const set = (key: keyof AuditFilters, value: string) =>
    setDraft((previous) => ({ ...previous, [key]: value }));
  return (
    <Panel
      className="audit-log-panel"
      title="操作记录"
      subtitle="查询账号和业务操作的结果。已受理表示任务进入后台流程，最终执行结果请查看对应任务。"
    >
      <form
        className="audit-filter-form"
        onSubmit={(event) => {
          event.preventDefault();
          query(draft);
        }}
      >
        <div className="audit-filter-grid">
          <Field label="操作账号">
            <select
              aria-label="操作账号"
              value={draft.actor_id}
              onChange={(event) => set('actor_id', event.target.value)}
            >
              <option value="">全部账号</option>
              <option value="unattributed">未登录 / 无归属记录</option>
              {users.map((user) => (
                <option key={user.id} value={user.id}>
                  {user.display_name || user.username} · {user.username}
                  {user.enabled ? '' : '（已停用）'}
                </option>
              ))}
            </select>
          </Field>
          <Field label="关键词">
            <input
              type="search"
              value={draft.q}
              maxLength={100}
              placeholder="搜索操作、账号或对象"
              onChange={(event) => set('q', event.target.value)}
            />
          </Field>
          <Field label="操作类型">
            <div className="audit-action-filter">
              <select
                aria-label="操作类型"
                value={customAction ? '__custom__' : draft.action}
                onChange={(event) => {
                  const custom = event.target.value === '__custom__';
                  setCustomAction(custom);
                  set('action', custom ? '' : event.target.value);
                }}
              >
                <option value="">全部操作类型</option>
                {auditActionGroups.map((group) => (
                  <optgroup key={group.label} label={group.label}>
                    {Object.entries(group.actions).map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </optgroup>
                ))}
                <option value="__custom__">自定义标识（高级）</option>
              </select>
              {customAction && (
                <input
                  aria-label="自定义操作标识"
                  value={draft.action}
                  maxLength={120}
                  required
                  placeholder="输入历史操作标识，精确匹配"
                  onChange={(event) => set('action', event.target.value)}
                />
              )}
            </div>
          </Field>
          <Field label="结果">
            <select
              aria-label="结果"
              value={draft.outcome}
              onChange={(event) => set('outcome', event.target.value)}
            >
              <option value="">全部结果</option>
              {Object.entries(auditOutcomeLabels).map(([value, label]) => (
                <option value={value} key={value}>
                  {label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="开始时间（本地时间）">
            <input
              type="datetime-local"
              value={draft.from_time}
              onChange={(event) => set('from_time', event.target.value)}
            />
          </Field>
          <Field label="结束时间（本地时间）">
            <input
              type="datetime-local"
              value={draft.to_time}
              onChange={(event) => set('to_time', event.target.value)}
            />
          </Field>
        </div>
        <div className="audit-filter-actions">
          <button className="button primary" type="submit">
            <Search size={15} />
            查询
          </button>
          <button
            className="button outline"
            type="button"
            onClick={() => {
              const empty = emptyAuditFilters();
              setDraft(empty);
              setCustomAction(false);
              setAutomatic(false);
              query(empty);
            }}
          >
            重置筛选
          </button>
          <button className="button outline" type="button" onClick={refresh}>
            <RefreshCw size={15} />
            刷新首页
          </button>
          <label className="audit-auto-refresh">
            <input
              type="checkbox"
              checked={automatic}
              disabled={Boolean(paging.cursor)}
              onChange={(event) => setAutomatic(event.target.checked)}
            />
            首页每 15 秒刷新
          </label>
          {paging.cursor && <span className="field-hint">历史页暂停自动刷新</span>}
        </div>
      </form>
      <ErrorNotice error={filterError} title="筛选条件无效" />
      <ErrorNotice error={current?.error} title="操作记录读取失败" retry={refresh} />
      {loading ? (
        <Loading label="正在读取操作记录…" />
      ) : (
        page && <AuditLogTable items={page.items} users={users} />
      )}
      <nav className="audit-pagination" aria-label="操作记录分页">
        <span>
          每页 50 条 · 第 {paging.previous.length + 1} 页
          {page ? ` · 本页 ${page.items.length} 条` : ''}
        </span>
        <div>
          <button
            className="button outline small"
            type="button"
            disabled={loading || !paging.previous.length}
            onClick={() => setPaging(previousAuditPage)}
          >
            <ChevronLeft size={15} />
            上一页
          </button>
          <button
            className="button outline small"
            type="button"
            disabled={loading || !page?.has_more || !page.next_cursor}
            onClick={() => {
              if (!page) return;
              try {
                setPaging(nextAuditPage(paging, page));
              } catch (cause) {
                setFilterError(cause as Error);
              }
            }}
          >
            下一页
            <ChevronRight size={15} />
          </button>
        </div>
      </nav>
    </Panel>
  );
}

export function AuditLogTable({ items, users = [] }: { items: AuditEntry[]; users?: AuditUser[] }) {
  return (
    <div className="audit-table-scroll">
      <table className="audit-events-table">
        <thead>
          <tr>
            <th>时间</th>
            <th>账号</th>
            <th>操作</th>
            <th>对象</th>
            <th>结果</th>
            <th>来源 IP</th>
            <th>详情</th>
          </tr>
        </thead>
        <tbody>
          {items.length ? (
            items.map((raw) => {
              const entry = sanitizeAuditEntry(raw);
              const outcome = auditOutcome(entry.status);
              const target = auditTarget(entry);
              const actor = users.find((user) => user.id === entry.actor_id);
              const actorName = entry.actor_username || actor?.username || entry.actor_id;
              const details = auditDetailRows(entry);
              const attemptedUsername = entry.details?.attempted_username || entry.actor_username;
              return (
                <tr key={entry.id}>
                  <td className="audit-time">
                    <time dateTime={entry.created_at}>{auditTime(entry.created_at)}</time>
                  </td>
                  <td>
                    {entry.actor_id ? (
                      <>
                        <strong>{actorName}</strong>
                        {actor?.display_name && <small>{actor.display_name}</small>}
                      </>
                    ) : (
                      <>
                        <strong>未归属账号</strong>
                        {attemptedUsername && <small>尝试账号：{String(attemptedUsername)}</small>}
                      </>
                    )}
                  </td>
                  <td>
                    <strong>{auditOperationLabel(entry)}</strong>
                    <small>{entry.action}</small>
                  </td>
                  <td className="audit-target">
                    <span>{target.label}</span>
                    {target.id && <code>{target.id}</code>}
                  </td>
                  <td>
                    <span className={`audit-outcome audit-outcome-${outcome}`}>
                      {auditOutcomeLabels[outcome]} · {entry.status}
                    </span>
                  </td>
                  <td className="audit-address">{entry.remote_addr || '—'}</td>
                  <td>
                    <details className="audit-entry-details">
                      <summary>查看详情</summary>
                      <dl>
                        {details.map((row) => (
                          <Fragment key={row.key}>
                            <dt>{row.label}</dt>
                            <dd>
                              {row.value}
                              {row.identifier && <small>{row.identifier}</small>}
                            </dd>
                          </Fragment>
                        ))}
                      </dl>
                      {(entry.method || entry.path) && (
                        <p>
                          <span>请求</span>
                          <code>
                            {entry.method} {safeAuditPath(entry.path)}
                          </code>
                        </p>
                      )}
                    </details>
                  </td>
                </tr>
              );
            })
          ) : (
            <tr>
              <td colSpan={7} className="audit-empty">
                没有符合条件的操作记录
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
