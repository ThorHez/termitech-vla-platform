import {
  decodeTrainingDraft,
  encodeTrainingDraft,
  type TrainingDraftSnapshot,
} from './trainingDraft';

export interface LocalTrainingDraft {
  user_id: string;
  template_id?: string;
  revision: number;
  content: Record<string, unknown>;
  writer_id?: string;
}

export interface StoredTrainingDraft {
  value: LocalTrainingDraft;
  key: string;
  raw: string;
}

type DraftStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

const legacyPendingKey = (userId: string) => `termitech.training-draft.pending.v1:${userId}`;

export const trainingDraftPendingKey = (userId: string, templateId: string) =>
  `termitech.training-draft.pending.v2:${encodeURIComponent(userId)}:${encodeURIComponent(templateId)}`;

export const trainingDraftPath = (templateId: string) =>
  `/training-draft?template_id=${encodeURIComponent(templateId)}`;

export function requireTrainingDraftScope(value: { template_id?: string }, templateId: string) {
  if (value.template_id !== templateId)
    throw new Error('管理服务尚未支持当前模板的独立草稿，请更新管理服务后重试。');
}

export function decodeScopedTrainingDraft(content: unknown, templateId: string) {
  const snapshot = decodeTrainingDraft(content);
  return snapshot?.request.template_id === templateId ? snapshot : undefined;
}

export function encodeScopedTrainingDraft(snapshot: TrainingDraftSnapshot, templateId: string) {
  if (snapshot.request.template_id !== templateId)
    throw new Error('训练模板已改变，配置未写入其他模板的草稿。');
  return encodeTrainingDraft(snapshot);
}

export function readTrainingDraftPending(
  storage: DraftStorage,
  userId: string,
  templateId: string,
): StoredTrainingDraft | undefined {
  const scopedKey = trainingDraftPendingKey(userId, templateId);
  for (const key of [scopedKey, legacyPendingKey(userId)]) {
    const raw = storage.getItem(key);
    if (raw === null) continue;
    try {
      const value = JSON.parse(raw) as LocalTrainingDraft;
      if (
        value?.user_id === userId &&
        (key === scopedKey
          ? value.template_id === templateId
          : value.template_id === undefined || value.template_id === templateId) &&
        Number.isSafeInteger(value.revision) &&
        value.revision >= 0 &&
        decodeScopedTrainingDraft(value.content, templateId)
      )
        return { value, key, raw };
    } catch {
      // An invalid checkpoint is never a source of form inputs.
    }
    // A newer scoped checkpoint takes precedence, even if it is incompatible.
    if (key === scopedKey) return undefined;
  }
  return undefined;
}

export function writeTrainingDraftPending(
  storage: DraftStorage,
  userId: string,
  templateId: string,
  value: Omit<LocalTrainingDraft, 'user_id' | 'template_id'>,
  adopted?: StoredTrainingDraft,
) {
  if (!decodeScopedTrainingDraft(value.content, templateId))
    throw new Error('训练草稿与当前模板不一致，未保存。');
  storage.setItem(
    trainingDraftPendingKey(userId, templateId),
    JSON.stringify({ ...value, user_id: userId, template_id: templateId }),
  );
  // Only remove the exact legacy checkpoint that was adopted, after the scoped
  // copy is durable. A different tab may have replaced the shared legacy key.
  if (adopted?.key === legacyPendingKey(userId) && storage.getItem(adopted.key) === adopted.raw)
    storage.removeItem(adopted.key);
}

export function clearTrainingDraftPending(
  storage: DraftStorage,
  userId: string,
  templateId: string,
  writerId: string,
  adopted?: StoredTrainingDraft,
) {
  const key = trainingDraftPendingKey(userId, templateId);
  const raw = storage.getItem(key);
  if (raw !== null) {
    try {
      const value = JSON.parse(raw) as LocalTrainingDraft;
      if (
        value.user_id === userId &&
        value.template_id === templateId &&
        (value.writer_id === writerId || (adopted?.key === key && adopted.raw === raw))
      )
        storage.removeItem(key);
    } catch {
      // Do not remove a checkpoint whose ownership cannot be established.
    }
  }
  if (adopted?.key === legacyPendingKey(userId) && storage.getItem(adopted.key) === adopted.raw)
    storage.removeItem(adopted.key);
}
