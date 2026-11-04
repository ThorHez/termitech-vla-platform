import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from './api';
import type { TrainingDraftSnapshot } from './trainingDraft';
import {
  clearTrainingDraftPending,
  decodeScopedTrainingDraft,
  encodeScopedTrainingDraft,
  readTrainingDraftPending,
  requireTrainingDraftScope,
  trainingDraftPath,
  writeTrainingDraftPending,
  type StoredTrainingDraft,
} from './trainingDraftScope';

interface DraftResponse {
  template_id?: string;
  content: Record<string, unknown> | null;
  revision: number;
  updated_at: string | null;
}
type DraftStatus = 'loading' | 'idle' | 'pending' | 'saving' | 'saved' | 'error' | 'conflict';

const signature = (content: unknown) => JSON.stringify(content);
const message = (error: unknown) =>
  error instanceof Error ? error.message : '训练草稿暂未保存，请重试。';

export function useTrainingDraft(userId: string, templateId: string) {
  const [loaded, setLoaded] = useState(false);
  const [initialSnapshot, setInitialSnapshot] = useState<TrainingDraftSnapshot>();
  const [generation, setGeneration] = useState(0);
  const [status, setStatus] = useState<DraftStatus>('loading');
  const [error, setError] = useState<string>();
  const [updatedAt, setUpdatedAt] = useState<string>();
  const [hasContent, setHasContent] = useState(false);
  const snapshotRef = useRef<TrainingDraftSnapshot | undefined>(undefined);
  const revision = useRef(0);
  const committed = useRef<string | undefined>(undefined);
  const dirty = useRef(false);
  const knownServer = useRef(false);
  const scopeVerified = useRef(false);
  const conflict = useRef(false);
  const active = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const saving = useRef<Promise<void> | undefined>(undefined);
  const changing = useRef(false);
  const writerId = useRef(crypto.randomUUID());
  const adoptedPending = useRef<StoredTrainingDraft | undefined>(undefined);
  const flushRef = useRef<() => Promise<void>>(async () => {});
  const ownerHeaders = useCallback(() => ({ 'X-Training-Draft-User': userId }), [userId]);
  const stopTimer = useCallback(() => {
    clearTimeout(timer.current);
    timer.current = undefined;
  }, []);
  const clearLocal = useCallback(() => {
    try {
      clearTrainingDraftPending(
        localStorage,
        userId,
        templateId,
        writerId.current,
        adoptedPending.current,
      );
    } catch {
      // Saving to the server remains available when browser storage is disabled.
    }
  }, [userId, templateId]);
  const checkpoint = useCallback(() => {
    if (!dirty.current || !snapshotRef.current) return;
    const content = encodeScopedTrainingDraft(snapshotRef.current, templateId);
    try {
      writeTrainingDraftPending(
        localStorage,
        userId,
        templateId,
        {
          revision: revision.current,
          writer_id: writerId.current,
          content,
        },
        adoptedPending.current,
      );
    } catch {
      // The visible server save state remains authoritative.
    }
    return content;
  }, [userId, templateId]);
  const readLocal = useCallback((): StoredTrainingDraft | undefined => {
    try {
      return readTrainingDraftPending(localStorage, userId, templateId);
    } catch {
      // Malformed or incompatible checkpoints are never applied to the form.
    }
    return undefined;
  }, [userId, templateId]);
  const request = useCallback(
    async (options: RequestInit = {}) => {
      if ((options.method || 'GET').toUpperCase() !== 'GET' && !scopeVerified.current)
        throw new Error('请先成功读取当前模板的独立草稿，再保存或清空。');
      const result = await api<DraftResponse>(trainingDraftPath(templateId), {
        ...options,
        headers: { ...ownerHeaders(), ...options.headers },
        signal: options.signal
          ? AbortSignal.any([options.signal, AbortSignal.timeout(5000)])
          : AbortSignal.timeout(5000),
      });
      scopeVerified.current = false;
      requireTrainingDraftScope(result, templateId);
      scopeVerified.current = true;
      return result;
    },
    [ownerHeaders, templateId],
  );
  const schedule = useCallback(() => {
    stopTimer();
    if (dirty.current && knownServer.current && !conflict.current && !changing.current) {
      timer.current = setTimeout(() => void flushRef.current().catch(() => {}), 700);
    }
  }, [stopTimer]);
  const flush = useCallback(async () => {
    stopTimer();
    if (changing.current) return;
    while (saving.current) await saving.current;
    if (changing.current) return;
    if (!active.current || !dirty.current || !snapshotRef.current) return;
    const content = checkpoint();
    if (!content) return;
    if (!knownServer.current || conflict.current)
      throw new Error('草稿尚未同步到服务端，当前内容已保留在此浏览器。');
    const sent = signature(content);
    if (sent === committed.current) {
      dirty.current = false;
      clearLocal();
      setStatus('saved');
      return;
    }
    setStatus('saving');
    setError(undefined);
    const operation = (async () => {
      try {
        const result = await request({
          method: 'PUT',
          body: JSON.stringify({ expected_revision: revision.current, content }),
        });
        revision.current = result.revision;
        committed.current = sent;
        if (!active.current) return;
        setUpdatedAt(result.updated_at || undefined);
        setHasContent(true);
        if (
          snapshotRef.current &&
          signature(encodeScopedTrainingDraft(snapshotRef.current, templateId)) === sent
        ) {
          dirty.current = false;
          clearLocal();
          setStatus('saved');
        } else {
          checkpoint();
          setStatus('pending');
        }
      } catch (cause) {
        if (!active.current) return;
        if (cause instanceof ApiError && cause.code === 'TRAINING_DRAFT_CONFLICT') {
          conflict.current = true;
          setStatus('conflict');
          setError('其他设备已更新这份草稿。当前填写仍保留，尚未覆盖服务端内容。');
        } else {
          setStatus('error');
          setError(message(cause));
        }
        throw cause;
      }
    })();
    saving.current = operation;
    try {
      await operation;
    } finally {
      saving.current = undefined;
    }
    if (dirty.current && !conflict.current) schedule();
  }, [checkpoint, clearLocal, request, schedule, stopTimer, templateId]);
  flushRef.current = flush;

  useEffect(() => {
    active.current = true;
    const controller = new AbortController();
    const pending = readLocal();
    const local = pending?.value;
    adoptedPending.current = pending;
    void request({ signal: controller.signal })
      .then((result) => {
        if (!active.current || controller.signal.aborted) return;
        revision.current = result.revision;
        knownServer.current = true;
        committed.current = signature(result.content);
        setUpdatedAt(result.updated_at || undefined);
        const server = result.content
          ? decodeScopedTrainingDraft(result.content, templateId)
          : undefined;
        if (result.content && !server) {
          conflict.current = true;
          setStatus('conflict');
          setError('服务端草稿格式暂不兼容，原草稿已保留；可清空后重新配置。');
        } else if (local && signature(local.content) !== signature(result.content)) {
          setInitialSnapshot(decodeScopedTrainingDraft(local.content, templateId));
          snapshotRef.current = decodeScopedTrainingDraft(local.content, templateId);
          dirty.current = true;
          if (local.revision === result.revision) {
            setStatus('pending');
            schedule();
          } else {
            conflict.current = true;
            setStatus('conflict');
            setError('已恢复此浏览器尚未同步的配置；其他设备也更新了草稿，请选择保留哪份内容。');
          }
        } else {
          clearLocal();
          setInitialSnapshot(server);
          setStatus(server ? 'saved' : 'idle');
        }
        setHasContent(!!result.content || !!local);
        setLoaded(true);
      })
      .catch((cause) => {
        if (controller.signal.aborted || !active.current) return;
        const restored = local ? decodeScopedTrainingDraft(local.content, templateId) : undefined;
        revision.current = local?.revision || 0;
        setInitialSnapshot(restored);
        snapshotRef.current = restored;
        dirty.current = !!restored;
        setHasContent(!!restored);
        setError(message(cause));
        setStatus('error');
        setLoaded(true);
      });
    const onPageHide = () => {
      try {
        checkpoint();
      } catch {
        // Size/format failures are shown while the form is open.
      }
    };
    window.addEventListener('pagehide', onPageHide);
    return () => {
      active.current = false;
      controller.abort();
      stopTimer();
      window.removeEventListener('pagehide', onPageHide);
    };
  }, [checkpoint, clearLocal, readLocal, request, schedule, stopTimer, templateId]);

  const markEdited = useCallback(() => {
    dirty.current = true;
    setStatus((previous) =>
      previous === 'conflict' || previous === 'error' ? previous : 'pending',
    );
    schedule();
  }, [schedule]);
  const changed = useCallback(() => {
    if (!dirty.current) return;
    try {
      checkpoint();
      schedule();
    } catch (cause) {
      setError(message(cause));
      setStatus('error');
    }
  }, [checkpoint, schedule]);
  const clear = useCallback(async () => {
    stopTimer();
    changing.current = true;
    try {
      if (saving.current) await saving.current.catch(() => {});
      const result = await request({
        method: 'DELETE',
        body: JSON.stringify({ expected_revision: revision.current }),
      });
      revision.current = result.revision;
      committed.current = signature(null);
      dirty.current = false;
      conflict.current = false;
      knownServer.current = true;
      snapshotRef.current = undefined;
      clearLocal();
      setInitialSnapshot(undefined);
      setHasContent(false);
      setError(undefined);
      setStatus('idle');
      setGeneration((value) => value + 1);
    } catch (cause) {
      if (cause instanceof ApiError && cause.code === 'TRAINING_DRAFT_CONFLICT') {
        conflict.current = true;
        setStatus('conflict');
        setError('其他设备已更新草稿，本次清空未执行。请选择保留哪份内容。');
      }
      throw cause;
    } finally {
      changing.current = false;
    }
  }, [clearLocal, request, stopTimer]);
  const reload = useCallback(async () => {
    stopTimer();
    changing.current = true;
    try {
      if (saving.current) await saving.current.catch(() => {});
      const result = await request();
      const restored = result.content
        ? decodeScopedTrainingDraft(result.content, templateId)
        : undefined;
      if (result.content && !restored) throw new Error('服务端草稿格式暂不兼容。');
      revision.current = result.revision;
      committed.current = signature(result.content);
      knownServer.current = true;
      dirty.current = false;
      conflict.current = false;
      clearLocal();
      setInitialSnapshot(restored);
      setHasContent(!!restored);
      setUpdatedAt(result.updated_at || undefined);
      setError(undefined);
      setStatus(restored ? 'saved' : 'idle');
      setGeneration((value) => value + 1);
    } finally {
      changing.current = false;
    }
  }, [clearLocal, request, stopTimer, templateId]);
  const saveCurrent = useCallback(async () => {
    stopTimer();
    if (saving.current) await saving.current.catch(() => {});
    const result = await request();
    revision.current = result.revision;
    committed.current = signature(result.content);
    knownServer.current = true;
    conflict.current = false;
    dirty.current = true;
    await flushRef.current();
  }, [request, stopTimer]);
  const replace = useCallback(
    async (snapshot: TrainingDraftSnapshot) => {
      const content = encodeScopedTrainingDraft(snapshot, templateId);
      // Save pending edits first; replacement is atomic and respects other devices.
      await flushRef.current();
      stopTimer();
      if (!active.current || !knownServer.current || conflict.current)
        throw new Error('请先同步当前草稿或处理版本冲突，再套用训练模板。');
      changing.current = true;
      try {
        const result = await request({
          method: 'PUT',
          body: JSON.stringify({ expected_revision: revision.current, content }),
        });
        if (!active.current) return;
        const restored = decodeScopedTrainingDraft(result.content, templateId);
        if (!restored) throw new Error('套用后的训练草稿格式无效。');
        revision.current = result.revision;
        committed.current = signature(result.content);
        dirty.current = false;
        conflict.current = false;
        knownServer.current = true;
        snapshotRef.current = restored;
        clearLocal();
        setInitialSnapshot(restored);
        setHasContent(true);
        setUpdatedAt(result.updated_at || undefined);
        setError(undefined);
        setStatus('saved');
        setGeneration((value) => value + 1);
      } catch (cause) {
        if (cause instanceof ApiError && cause.code === 'TRAINING_DRAFT_CONFLICT') {
          conflict.current = true;
          setStatus('conflict');
          setError('其他设备已更新草稿，模板未覆盖新内容；请先处理草稿冲突。');
        }
        throw cause;
      } finally {
        changing.current = false;
      }
    },
    [clearLocal, request, stopTimer, templateId],
  );
  const retry = useCallback(async () => {
    if (knownServer.current && scopeVerified.current && !conflict.current)
      return flushRef.current();
    const result = await request();
    if (!dirty.current) return reload();
    knownServer.current = true;
    if (
      result.content &&
      snapshotRef.current &&
      signature(result.content) !==
        signature(encodeScopedTrainingDraft(snapshotRef.current, templateId))
    ) {
      conflict.current = true;
      setStatus('conflict');
      setError('服务端已有另一份草稿，请选择加载服务端内容，或明确保留当前填写。');
      return;
    }
    revision.current = result.revision;
    committed.current = signature(result.content);
    conflict.current = false;
    return flushRef.current();
  }, [reload, request, templateId]);
  const finished = useCallback(async () => {
    stopTimer();
    changing.current = true;
    dirty.current = false;
    if (saving.current) await saving.current.catch(() => {});
    try {
      await request({
        method: 'DELETE',
        body: JSON.stringify({ expected_revision: revision.current }),
      });
    } finally {
      // A newer draft on another device is never deleted after this run starts.
      clearLocal();
    }
  }, [clearLocal, request, stopTimer]);
  const hasCheckpoint = useCallback(() => {
    if (!snapshotRef.current) return true;
    try {
      const local = readLocal();
      return (
        !!local &&
        signature(local.value.content) ===
          signature(encodeScopedTrainingDraft(snapshotRef.current, templateId))
      );
    } catch {
      return false;
    }
  }, [readLocal, templateId]);
  return {
    loaded,
    initialSnapshot,
    generation,
    status,
    error,
    updatedAt,
    hasContent,
    snapshotRef,
    markEdited,
    changed,
    flush,
    clear,
    reload,
    saveCurrent,
    replace,
    retry,
    finished,
    hasCheckpoint,
  };
}

export type TrainingDraftSession = ReturnType<typeof useTrainingDraft>;
