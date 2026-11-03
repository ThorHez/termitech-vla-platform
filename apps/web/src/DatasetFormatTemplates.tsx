import { useEffect, useId, useRef, useState } from 'react';
import { BookmarkPlus, LoaderCircle, RefreshCw, Trash2 } from 'lucide-react';
import { api } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Field, Modal } from './components';
import type { Asset } from './types';
import type { DatasetProfile } from './DatasetProfileForm';
import {
  datasetFormatProfile,
  datasetTemplateNameError,
  type DatasetFormatTemplate,
  type DatasetFormatTemplateSummary,
} from './datasetFormatTemplateData';
import './dataset-format-templates.css';

export function DatasetFormatTemplates({
  value,
  onApply,
  disabled = false,
  initialName = '',
  saveOnly = false,
  personalDraft = false,
}: {
  value: DatasetProfile;
  onApply?: (profile: DatasetProfile) => void;
  disabled?: boolean;
  initialName?: string;
  saveOnly?: boolean;
  personalDraft?: boolean;
}) {
  const { user, can } = useAuth();
  const owner = user?.id;
  const writable = Boolean(owner && can('dataset-templates.write'));
  const currentOwner = useRef({ owner, writable });
  currentOwner.current = { owner, writable };
  if (!writable)
    return <p className="field-hint">使用个人账户登录后，可保存和使用自己的数据集格式模板。</p>;
  return (
    <DatasetFormatTemplatesContent
      key={owner}
      owner={owner!}
      value={value}
      onApply={onApply}
      disabled={disabled}
      initialName={initialName}
      saveOnly={saveOnly}
      personalDraft={personalDraft}
      ownerCurrent={() => currentOwner.current.owner === owner && currentOwner.current.writable}
    />
  );
}

function DatasetFormatTemplatesContent({
  owner,
  value,
  onApply,
  disabled,
  initialName,
  saveOnly,
  personalDraft,
  ownerCurrent,
}: {
  owner: string;
  value: DatasetProfile;
  onApply?: (profile: DatasetProfile) => void;
  disabled: boolean;
  initialName: string;
  saveOnly: boolean;
  personalDraft: boolean;
  ownerCurrent: () => boolean;
}) {
  const [templates, setTemplates] = useState<DatasetFormatTemplateSummary[]>([]);
  const [selected, setSelected] = useState('');
  const [name, setName] = useState(initialName);
  const [busy, setBusy] = useState<'list' | 'apply' | 'save' | 'delete'>();
  const [error, setError] = useState<Error>();
  const [message, setMessage] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [revision, setRevision] = useState(0);
  const request = useRef<AbortController | undefined>(undefined);
  const mounted = useRef(true);
  const latest = useRef({ owner, value, disabled, onApply });
  latest.current = { owner, value, disabled, onApply };
  const titleId = useId();
  const chosen = templates.find((item) => item.id === selected);
  const sameName = templates.find((item) => item.name === name.trim());
  const source = datasetFormatProfile(value);
  const nameError = datasetTemplateNameError(name);
  const headers = { 'X-Dataset-Template-User': owner };
  const begin = (kind: 'list' | 'apply' | 'save' | 'delete') => {
    request.current?.abort();
    const abort = new AbortController();
    request.current = abort;
    setBusy(kind);
    setError(undefined);
    setMessage('');
    return abort;
  };
  const active = (abort: AbortController) =>
    mounted.current &&
    !abort.signal.aborted &&
    request.current === abort &&
    latest.current.owner === owner &&
    ownerCurrent();
  const signal = (abort: AbortController) =>
    AbortSignal.any([abort.signal, AbortSignal.timeout(20000)]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      request.current?.abort();
    };
  }, []);
  useEffect(() => {
    const abort = begin('list');
    void api<DatasetFormatTemplateSummary[]>('/dataset-format-templates', {
      headers,
      signal: signal(abort),
    })
      .then((items) => {
        if (active(abort)) setTemplates(items);
      })
      .catch((cause) => {
        if (active(abort)) setError(cause instanceof Error ? cause : new Error(String(cause)));
      })
      .finally(() => {
        if (active(abort)) setBusy(undefined);
      });
    return () => abort.abort();
  }, [owner, revision]);
  useEffect(() => {
    if (disabled && busy && busy !== 'list') {
      request.current?.abort();
      setBusy(undefined);
    }
  }, [disabled]);
  const save = async () => {
    if (busy || disabled || !source || nameError) return;
    const abort = begin('save');
    const snapshot = structuredClone(source);
    try {
      const saved = await api<DatasetFormatTemplate>('/dataset-format-templates', {
        method: 'POST',
        headers,
        body: JSON.stringify({ name: name.trim(), profile: snapshot }),
        signal: signal(abort),
      });
      if (!active(abort)) return;
      setTemplates((items) => [saved, ...items.filter((item) => item.id !== saved.id)]);
      setSelected(saved.id);
      setName(saved.name);
      setConfirmDelete(false);
      setMessage(
        personalDraft
          ? `已保存个人模板“${saved.name}”。`
          : `已保存“${saved.name}”，可在登记其他数据集时套用。`,
      );
    } catch (cause) {
      if (active(abort)) setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      if (active(abort)) setBusy(undefined);
    }
  };
  const apply = async () => {
    if (busy || disabled || !chosen || !onApply) return;
    const abort = begin('apply');
    const before = JSON.stringify(value);
    try {
      const saved = await api<DatasetFormatTemplate>(
        `/dataset-format-templates/${encodeURIComponent(chosen.id)}`,
        {
          headers,
          signal: signal(abort),
        },
      );
      if (!active(abort)) return;
      const profile = datasetFormatProfile(saved.profile);
      if (!profile) throw new Error('模板的数据特征无效，请重新保存模板。');
      if (latest.current.disabled || JSON.stringify(latest.current.value) !== before)
        throw new Error('读取模板期间配置已变更，请重新点击套用。');
      latest.current.onApply?.(profile);
      setTemplates((items) => items.map((item) => (item.id === saved.id ? saved : item)));
      setName(saved.name);
      setConfirmDelete(false);
      setMessage(
        personalDraft
          ? `已载入“${saved.name}”到个人格式草稿；公共数据集保持原样。`
          : `已套用“${saved.name}”。名称、机器、路径、版本和 norm_stats 绑定保持原样；请核对任务描述。`,
      );
    } catch (cause) {
      if (active(abort)) setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      if (active(abort)) setBusy(undefined);
    }
  };
  const remove = async () => {
    if (busy || disabled || !chosen) return;
    const abort = begin('delete');
    try {
      await api(`/dataset-format-templates/${encodeURIComponent(chosen.id)}`, {
        method: 'DELETE',
        headers,
        body: JSON.stringify({ expected_revision: chosen.revision }),
        signal: signal(abort),
      });
      if (!active(abort)) return;
      setTemplates((items) => items.filter((item) => item.id !== chosen.id));
      setSelected('');
      setConfirmDelete(false);
      setMessage('模板已删除，已登记的数据集保持原样。');
    } catch (cause) {
      if (active(abort)) setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      if (active(abort)) setBusy(undefined);
    }
  };
  return (
    <section
      className="dataset-format-templates"
      aria-labelledby={titleId}
      aria-busy={Boolean(busy)}
    >
      <h3 id={titleId}>数据集格式模板</h3>
      <p className="field-hint">
        {personalDraft
          ? '选择模板查看完整格式，或编辑个人草稿并另存副本。模板仅当前账户可见。'
          : '保存机器人、视觉和状态/动作字段标签，下次登记直接套用。模板仅当前账户可见。'}
      </p>
      {!saveOnly && (
        <div className="dataset-format-template-select">
          <Field label="选择格式模板">
            <select
              aria-label="选择格式模板"
              value={selected}
              disabled={disabled || Boolean(busy)}
              onChange={(event) => {
                setSelected(event.target.value);
                setConfirmDelete(false);
                setMessage('');
                const item = templates.find((entry) => entry.id === event.target.value);
                if (item) setName(item.name);
              }}
            >
              <option value="">{busy === 'list' ? '正在读取模板…' : '选择自己的格式模板'}</option>
              {templates.map((item) => (
                <option value={item.id} key={item.id}>
                  {item.name} · v{item.revision}
                </option>
              ))}
            </select>
          </Field>
          <button
            type="button"
            className="button outline small"
            disabled={disabled || Boolean(busy) || !chosen || !onApply}
            onClick={() => void apply()}
          >
            {busy === 'apply' && <LoaderCircle size={14} className="spin" />}
            {personalDraft ? '载入个人草稿' : '套用模板'}
          </button>
          <button
            type="button"
            className="icon-button"
            title="重新读取格式模板"
            aria-label="重新读取格式模板"
            disabled={disabled || Boolean(busy)}
            onClick={() => setRevision((value) => value + 1)}
          >
            <RefreshCw size={15} />
          </button>
          <button
            type="button"
            className="icon-button"
            title="删除所选模板"
            aria-label="删除所选模板"
            disabled={disabled || Boolean(busy) || !chosen}
            onClick={() => setConfirmDelete(true)}
          >
            <Trash2 size={15} />
          </button>
        </div>
      )}
      <div className="dataset-format-template-save">
        <Field label="格式模板名称">
          <input
            maxLength={100}
            value={name}
            placeholder="例如 ARX5 双臂 Joint · 三视角"
            disabled={disabled || Boolean(busy)}
            onChange={(event) => {
              setName(event.target.value);
              setMessage('');
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                void save();
              }
            }}
          />
        </Field>
        <button
          type="button"
          className="button outline small"
          disabled={disabled || Boolean(busy) || Boolean(nameError) || !source}
          onClick={() => void save()}
        >
          {busy === 'save' ? (
            <LoaderCircle size={14} className="spin" />
          ) : (
            <BookmarkPlus size={14} />
          )}
          {sameName ? '覆盖同名模板' : '保存当前格式为模板'}
        </button>
      </div>
      {!source && <p className="field-hint">填写完整数据集特征后即可保存格式模板。</p>}
      {name.trim() && nameError && (
        <p className="field-hint required" role="status">
          {nameError}
        </p>
      )}
      {sameName && (
        <p className="field-hint">保存将更新当前账户的同名模板；已登记的数据集不受影响。</p>
      )}
      {confirmDelete && chosen && (
        <div className="dataset-format-template-confirm" role="alert">
          <p>删除模板“{chosen.name}”？已登记的数据集不受影响。</p>
          <button
            type="button"
            className="button outline small"
            disabled={Boolean(busy)}
            onClick={() => setConfirmDelete(false)}
          >
            取消删除模板
          </button>
          <button
            type="button"
            className="button danger small"
            disabled={Boolean(busy)}
            onClick={() => void remove()}
          >
            确认删除模板
          </button>
        </div>
      )}
      {message && (
        <p className="dataset-format-template-message" role="status">
          {message}
        </p>
      )}
      <ErrorNotice error={error} title="格式模板操作失败" />
    </section>
  );
}

export function DatasetFormatTemplateDialog({ asset, close }: { asset: Asset; close: () => void }) {
  const profile = datasetFormatProfile(asset.metadata?.dataset_profile);
  const { can } = useAuth();
  return (
    <Modal
      title="保存数据集格式模板"
      subtitle={asset.name}
      close={close}
      footer={
        <button type="button" className="button outline" onClick={close}>
          关闭
        </button>
      }
    >
      {can('dataset-templates.write') && asset.kind === 'DATASET' && profile ? (
        <DatasetFormatTemplates value={profile} initialName={`${asset.name} 格式`} saveOnly />
      ) : (
        <p className="field-hint">需要个人格式模板权限和完整的数据集特征，才能保存格式模板。</p>
      )}
    </Modal>
  );
}
