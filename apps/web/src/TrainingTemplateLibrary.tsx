import { useEffect, useRef, useState } from 'react';
import { BookMarked, Download, RefreshCw, Save, Trash2 } from 'lucide-react';
import { api, date } from './api';
import { ErrorNotice, Field } from './components';
import type { TrainingDraftSnapshot } from './trainingDraft';
import {
  prepareTrainingTemplate,
  type TrainingTemplateRecord,
  type TrainingTemplateSummary,
} from './trainingTemplate';
import './training-template-library.css';

type TemplateWrite = {
  name: string;
  description: string;
  content: ReturnType<typeof prepareTrainingTemplate>;
};

// Each mounted account owns its requests. Even a transport that finishes after
// abort cannot return a previous account's template to the current wizard.
export function createTrainingTemplateClient(userId: string) {
  let active = true;
  const controllers = new Set<AbortController>();
  const requireActive = () => {
    if (!active) throw new DOMException('训练模板请求已取消。', 'AbortError');
  };
  const request = async <T,>(path: string, options: RequestInit = {}): Promise<T> => {
    requireActive();
    const controller = new AbortController();
    controllers.add(controller);
    const headers = new Headers(options.headers);
    headers.set('X-Training-Template-User', userId);
    try {
      const result = await api<T>(path, {
        ...options,
        headers,
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]),
      });
      requireActive();
      return result;
    } finally {
      controllers.delete(controller);
    }
  };
  const path = (id: string) => `/training-templates/${encodeURIComponent(id)}`;
  return {
    userId,
    isActive: () => active,
    dispose() {
      active = false;
      for (const controller of controllers) controller.abort();
      controllers.clear();
    },
    list: () => request<TrainingTemplateSummary[]>('/training-templates'),
    get: (id: string) => request<TrainingTemplateRecord>(path(id)),
    create: (body: TemplateWrite) =>
      request<TrainingTemplateRecord>('/training-templates', {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    update: (id: string, body: TemplateWrite, expectedRevision: number) =>
      request<TrainingTemplateRecord>(path(id), {
        method: 'PUT',
        body: JSON.stringify({ ...body, expected_revision: expectedRevision }),
      }),
    remove: (id: string, expectedRevision: number) =>
      request<{ id: string; deleted: boolean }>(path(id), {
        method: 'DELETE',
        body: JSON.stringify({ expected_revision: expectedRevision }),
      }),
  };
}

type TemplateClient = ReturnType<typeof createTrainingTemplateClient>;
type Operation = 'save' | 'apply' | 'delete';
type LibraryState = {
  userId: string;
  rows: TrainingTemplateSummary[];
  selectedId: string;
  name: string;
  description: string;
  loading: boolean;
  operation?: Operation;
  confirmDelete: boolean;
  error?: Error;
  notice: string;
};
const initialState = (userId: string, name: string): LibraryState => ({
  userId,
  rows: [],
  selectedId: '',
  name: name.slice(0, 100),
  description: '',
  loading: true,
  confirmDelete: false,
  notice: '',
});
const summary = ({
  id,
  name,
  description,
  revision,
  created_at,
  updated_at,
}: TrainingTemplateRecord) => ({
  id,
  name,
  description,
  revision,
  created_at,
  updated_at,
});

export function TrainingTemplateLibrary({
  userId,
  snapshot,
  onApply,
  busy = false,
}: {
  userId: string;
  snapshot: TrainingDraftSnapshot;
  onApply: (record: TrainingTemplateRecord) => void | Promise<void>;
  busy?: boolean;
}) {
  const [state, setState] = useState(() => initialState(userId, snapshot.request.name));
  const client = useRef<TemplateClient | undefined>(undefined);
  const running = useRef<TemplateClient | undefined>(undefined);
  const loadSequence = useRef(0);
  const renderedUser = useRef(userId);
  renderedUser.current = userId;
  const latestName = useRef(snapshot.request.name);
  latestName.current = snapshot.request.name;
  const current = (scope: TemplateClient) =>
    scope.isActive() && client.current === scope && renderedUser.current === scope.userId;
  const update = (scope: TemplateClient, transform: (previous: LibraryState) => LibraryState) => {
    if (current(scope)) {
      setState((previous) => (previous.userId === scope.userId ? transform(previous) : previous));
    }
  };
  const load = async (scope: TemplateClient) => {
    const sequence = ++loadSequence.current;
    update(scope, (previous) => ({ ...previous, loading: true, error: undefined }));
    try {
      const rows = await scope.list();
      if (sequence !== loadSequence.current) return;
      update(scope, (previous) => ({
        ...previous,
        rows,
        selectedId: rows.some((row) => row.id === previous.selectedId) ? previous.selectedId : '',
        loading: false,
        confirmDelete: false,
      }));
    } catch (error) {
      if (sequence !== loadSequence.current) return;
      update(scope, (previous) => ({
        ...previous,
        loading: false,
        error: error instanceof Error ? error : new Error(String(error)),
      }));
    }
  };
  useEffect(() => {
    const scope = createTrainingTemplateClient(userId);
    client.current = scope;
    setState(initialState(userId, latestName.current));
    void load(scope);
    return () => {
      scope.dispose();
      if (client.current === scope) client.current = undefined;
      if (running.current === scope) running.current = undefined;
    };
  }, [userId]);

  const view = state.userId === userId ? state : initialState(userId, snapshot.request.name);
  const selected = view.rows.find((row) => row.id === view.selectedId);
  const disabled = busy || view.loading || Boolean(view.operation) || !userId;
  const persistable = Boolean(view.name.trim());
  const perform = async (
    operation: Operation,
    action: (scope: TemplateClient) => Promise<void>,
  ) => {
    const scope = client.current;
    if (!scope || !current(scope) || disabled || running.current) return;
    running.current = scope;
    update(scope, (previous) => ({ ...previous, operation, error: undefined, notice: '' }));
    try {
      await action(scope);
    } catch (error) {
      update(scope, (previous) => ({
        ...previous,
        error: error instanceof Error ? error : new Error(String(error)),
      }));
    } finally {
      if (running.current === scope) running.current = undefined;
      update(scope, (previous) => ({ ...previous, operation: undefined }));
    }
  };
  const save = () =>
    perform('save', async (scope) => {
      if (!persistable) return;
      const body = {
        name: view.name.trim(),
        description: view.description.trim(),
        content: prepareTrainingTemplate(snapshot),
      };
      const existing = view.rows.find((row) => row.name === body.name);
      const record = existing
        ? await scope.update(existing.id, body, existing.revision)
        : await scope.create(body);
      update(scope, (previous) => ({
        ...previous,
        rows: [
          summary(record),
          ...previous.rows.filter((row) => row.id !== record.id && row.name !== record.name),
        ],
        selectedId: record.id,
        name: record.name,
        description: record.description,
        confirmDelete: false,
        notice: `已${record.revision > 1 ? '更新' : '保存'}模板「${record.name}」· v${record.revision}。`,
      }));
    });
  const apply = () =>
    perform('apply', async (scope) => {
      if (!selected) return;
      const record = await scope.get(selected.id);
      if (record.id !== selected.id) throw new Error('模板响应与所选记录不一致，请刷新后重试。');
      if (!current(scope)) return;
      await onApply(record);
      update(scope, (previous) => ({
        ...previous,
        rows: previous.rows.map((row) => (row.id === record.id ? summary(record) : row)),
        name: record.name,
        description: record.description,
        confirmDelete: false,
        notice: `已套用「${record.name}」· v${record.revision}，请核对配置后重新预检。`,
      }));
    });
  const remove = () =>
    perform('delete', async (scope) => {
      if (!selected || !view.confirmDelete) return;
      await scope.remove(selected.id, selected.revision);
      update(scope, (previous) => ({
        ...previous,
        rows: previous.rows.filter((row) => row.id !== selected.id),
        selectedId: '',
        name: latestName.current.slice(0, 100),
        description: '',
        confirmDelete: false,
        notice: `已删除模板「${selected.name}」。`,
      }));
    });
  const refresh = () => {
    const scope = client.current;
    if (scope && current(scope) && !disabled && !running.current) void load(scope);
  };
  return (
    <section className="training-template-library" aria-label="我的训练模板">
      <header className="training-template-heading">
        <div>
          <h3>
            <BookMarked size={16} />
            我的训练模板
          </h3>
          <p>按当前账号保存多套配置，跨浏览器和设备恢复。套用后核对配置，再预检启动。</p>
        </div>
        <button type="button" className="button small ghost" disabled={disabled} onClick={refresh}>
          <RefreshCw size={13} className={view.loading ? 'spin' : ''} />
          刷新
        </button>
      </header>
      <ErrorNotice error={view.error} title="训练模板操作失败" />
      <div className="training-template-columns">
        <div className="training-template-catalog">
          <Field label={`已保存的模板${view.rows.length ? `（${view.rows.length}）` : ''}`}>
            <select
              value={view.selectedId}
              disabled={disabled}
              onChange={(event) => {
                const row = view.rows.find((item) => item.id === event.target.value);
                setState((previous) => ({
                  ...previous,
                  selectedId: row?.id || '',
                  name: row?.name || snapshot.request.name.slice(0, 100),
                  description: row?.description || '',
                  confirmDelete: false,
                  notice: '',
                  error: undefined,
                }));
              }}
            >
              <option value="">
                {view.loading
                  ? '正在读取个人模板…'
                  : view.rows.length
                    ? '选择模板'
                    : '暂无模板，可保存当前配置'}
              </option>
              {view.rows.map((row) => (
                <option key={row.id} value={row.id}>
                  {row.name} · v{row.revision}
                </option>
              ))}
            </select>
          </Field>
          {selected ? (
            <div className="training-template-selection">
              <strong>{selected.name}</strong>
              <span>
                v{selected.revision} · 更新于 {date(selected.updated_at)}
              </span>
              {selected.description && <p>{selected.description}</p>}
              <div className="training-template-actions">
                <button
                  type="button"
                  className="button small primary"
                  disabled={disabled}
                  onClick={() => void apply()}
                >
                  <Download size={13} />
                  {view.operation === 'apply' ? '正在套用…' : '套用模板'}
                </button>
                <button
                  type="button"
                  className="button small ghost"
                  disabled={disabled}
                  onClick={() => setState((previous) => ({ ...previous, confirmDelete: true }))}
                >
                  <Trash2 size={13} />
                  删除
                </button>
              </div>
              {view.confirmDelete && (
                <div className="training-template-delete" role="alert">
                  <p>确认删除「{selected.name}」？</p>
                  <div className="training-template-actions">
                    <button
                      type="button"
                      className="button small danger"
                      disabled={disabled}
                      onClick={() => void remove()}
                    >
                      {view.operation === 'delete' ? '正在删除…' : '确认删除'}
                    </button>
                    <button
                      type="button"
                      className="button small ghost"
                      disabled={disabled}
                      onClick={() =>
                        setState((previous) => ({ ...previous, confirmDelete: false }))
                      }
                    >
                      取消
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="field-hint">模板仅当前账号可见。同名保存会覆盖更新，使用新名称可另存模板。</p>
          )}
        </div>
        <div className="training-template-editor">
          <Field label="模板名称" required>
            <input
              value={view.name}
              maxLength={100}
              disabled={disabled}
              placeholder="例如 双臂叠衣 · π0.5"
              onChange={(event) =>
                setState((previous) => ({ ...previous, name: event.target.value }))
              }
            />
          </Field>
          <Field label="模板说明">
            <textarea
              value={view.description}
              maxLength={2000}
              rows={2}
              disabled={disabled}
              placeholder="记录适用的数据、模型与训练方式"
              onChange={(event) =>
                setState((previous) => ({ ...previous, description: event.target.value }))
              }
            />
          </Field>
          <div className="training-template-actions">
            <button
              type="button"
              className="button small"
              disabled={disabled || !persistable}
              onClick={() => void save()}
            >
              <Save size={13} />
              {view.operation === 'save' ? '正在保存…' : '保存模板'}
            </button>
          </div>
          <p className="field-hint">
            保存当前训练配置：同名覆盖更新，新名称另存。模板仍需重新预检及确认生成的 config。
          </p>
        </div>
      </div>
      {view.notice && (
        <p className="training-template-notice" role="status">
          {view.notice}
        </p>
      )}
    </section>
  );
}
