import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from './api';
import {
  decodeTrainingDraft,
  encodeTrainingDraft,
  type TrainingDraftSnapshot,
} from './trainingDraft';

interface DraftResponse {
  content: Record<string, unknown> | null;
  revision: number;
  updated_at: string | null;
}
interface LocalPending {
  user_id: string;
  revision: number;
  content: Record<string, unknown>;
  writer_id?: string;
}
type DraftStatus = 'loading' | 'idle' | 'pending' | 'saving' | 'saved' | 'error' | 'conflict';

// Only an unsynchronised form checkpoint is kept locally. Server ownership is
// always determined by the authenticated account, never by this storage key.
const pendingKey = (userId: string) => `termitech.training-draft.pending.v1:${userId}`;
const signature = (content: unknown) => JSON.stringify(content);
const message = (error: unknown) =>
  error instanceof Error ? error.message : '训练草稿暂未保存，请重试。';

export function useTrainingDraft(userId: string) {
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
  const conflict = useRef(false);
  const active = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const saving = useRef<Promise<void> | undefined>(undefined);
  const changing = useRef(false);
  const writerId = useRef(crypto.randomUUID());
  const adoptedPending = useRef<LocalPending | undefined>(undefined);
  const flushRef = useRef<() => Promise<void>>(async () => {});
  const ownerHeaders = useCallback(() => ({ 'X-Training-Draft-User': userId }), [userId]);
  const stopTimer = useCallback(() => {
    clearTimeout(timer.current);
    timer.current = undefined;
  }, []);
  const clearLocal = useCallback(() => {
    try {
      const raw = localStorage.getItem(pendingKey(userId));
      if (!raw) return;
      const current = JSON.parse(raw) as LocalPending;
      if (
        current.writer_id !== writerId.current &&
        signature(current) !== signature(adoptedPending.current)
      )
        return;
      localStorage.removeItem(pendingKey(userId));
    } catch {
      // Saving to the server remains available when browser storage is disabled.
    }
  }, [userId]);
  const checkpoint = useCallback(() => {
    if (!dirty.current || !snapshotRef.current) return;
    const content = encodeTrainingDraft(snapshotRef.current);
    try {
      localStorage.setItem(
        pendingKey(userId),
        JSON.stringify({
          user_id: userId,
          revision: revision.current,
          writer_id: writerId.current,
          content,
        }),
      );
    } catch {
      // The visible server save state remains authoritative.
    }
    return content;
  }, [userId]);
  const readLocal = useCallback((): LocalPending | undefined => {
    try {
      const value = JSON.parse(localStorage.getItem(pendingKey(userId)) || 'null') as LocalPending;
      if (
        value?.user_id === userId &&
        Number.isSafeInteger(value.revision) &&
        value.revision >= 0 &&
        decodeTrainingDraft(value.content)
      )
        return value;
    } catch {
      // Malformed or incompatible checkpoints are never applied to the form.
    }
    return undefined;
  }, [userId]);
  const request = useCallback(
    (options: RequestInit = {}) =>
      api<DraftResponse>('/training-draft', {
        ...options,
        headers: { ...ownerHeaders(), ...options.headers },
        signal: options.signal
          ? AbortSignal.any([options.signal, AbortSignal.timeout(5000)])
          : AbortSignal.timeout(5000),
      }),
    [ownerHeaders],
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
        if (snapshotRef.current && signature(encodeTrainingDraft(snapshotRef.current)) === sent) {
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
  }, [checkpoint, clearLocal, request, schedule, stopTimer]);
  flushRef.current = flush;

  useEffect(() => {
    active.current = true;
    const controller = new AbortController();
    const local = readLocal();
    adoptedPending.current = local;
    void request({ signal: controller.signal })
      .then((result) => {
        if (!active.current || controller.signal.aborted) return;
        revision.current = result.revision;
        knownServer.current = true;
        committed.current = signature(result.content);
        setUpdatedAt(result.updated_at || undefined);
        const server = result.content ? decodeTrainingDraft(result.content) : undefined;
        if (result.content && !server) {
          conflict.current = true;
          setStatus('conflict');
          setError('服务端草稿格式暂不兼容，原草稿已保留；可清空后重新配置。');
        } else if (local && signature(local.content) !== signature(result.content)) {
          setInitialSnapshot(decodeTrainingDraft(local.content));
          snapshotRef.current = decodeTrainingDraft(local.content);
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
        const restored = local ? decodeTrainingDraft(local.content) : undefined;
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
  }, [checkpoint, clearLocal, readLocal, request, schedule, stopTimer]);

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
      const restored = result.content ? decodeTrainingDraft(result.content) : undefined;
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
  }, [clearLocal, request, stopTimer]);
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
      // Save pending edits first; replacement is atomic and respects other devices.
      await flushRef.current();
      stopTimer();
      if (!active.current || !knownServer.current || conflict.current)
        throw new Error('请先同步当前草稿或处理版本冲突，再套用训练模板。');
      changing.current = true;
      try {
        const content = encodeTrainingDraft(snapshot);
        const result = await request({
          method: 'PUT',
          body: JSON.stringify({ expected_revision: revision.current, content }),
        });
        if (!active.current) return;
        const restored = decodeTrainingDraft(result.content);
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
    [clearLocal, request, stopTimer],
  );
  const retry = useCallback(async () => {
    if (knownServer.current && !conflict.current) return flushRef.current();
    const result = await request();
    if (!dirty.current) return reload();
    knownServer.current = true;
    if (
      result.content &&
      snapshotRef.current &&
      signature(result.content) !== signature(encodeTrainingDraft(snapshotRef.current))
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
  }, [reload, request]);
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
        !!local && signature(local.content) === signature(encodeTrainingDraft(snapshotRef.current))
      );
    } catch {
      return false;
    }
  }, [readLocal]);
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
