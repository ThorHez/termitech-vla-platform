import { useEffect, useId, useRef, useState } from 'react';
import { BookOpen, Check, Code2, LoaderCircle, RotateCcw, Save } from 'lucide-react';
import { api, ApiError, post } from './api';
import { useAuth } from './auth';
import { Badge, ErrorNotice, Field } from './components';
import { recipeDatasetSlots } from './datasetBindingHelpers';
import { assetManagedOverridePaths, isAssetManagedField } from './recipeFieldPolicy';
import type { RecipeEditorDraft } from './trainingDraft';
import type {
  Node,
  Profile,
  RecipeDetailQuery,
  RecipeDetail,
  RecipeDatasetSlot,
  RecipeField,
  RecipeOverrides,
  RecipeLiteral,
  RecipeWriteback,
  RecipeSource,
  SavedRecipe,
} from './types';

const editable: Record<
  string,
  {
    label: string;
    integer?: boolean;
    min?: number;
    max?: number;
    exclusiveMax?: boolean;
    exclusiveMin?: boolean;
  }
> = {
  num_train_steps: { label: '训练步数', integer: true, min: 1, max: 1000000 },
  batch_size: { label: '批次大小', integer: true, min: 1, max: 4096 },
  'lr_schedule.peak_lr': { label: '峰值学习率', min: 0.00000001, max: 0.1 },
  'lr_schedule.warmup_steps': { label: '学习率预热步数', integer: true, min: 0, max: 10000000 },
  'lr_schedule.decay_steps': { label: '学习率衰减步数', integer: true, min: 1, max: 100000000 },
  'lr_schedule.decay_lr': { label: '衰减结束学习率', min: 0, max: 0.1 },
  ema_decay: { label: 'EMA 衰减', min: 0, max: 1, exclusiveMax: true },
  fsdp_devices: { label: 'FSDP 分片数', integer: true, min: 1, max: 4096 },
  num_workers: { label: '数据加载进程数', integer: true, min: 0, max: 128 },
  log_interval: { label: '日志间隔', integer: true, min: 1, max: 1000000 },
  save_interval: { label: '检查点保存间隔', integer: true, min: 1, max: 1000000 },
  keep_period: { label: '检查点保留周期', integer: true, min: 1, max: 10000000 },
  seed: { label: '随机种子', integer: true, min: 0, max: 4294967295 },
  'optimizer.clip_gradient_norm': { label: '梯度裁剪阈值', min: 0, max: 1000, exclusiveMin: true },
};
const commonKeys: Record<string, string> = {
  num_train_steps: 'steps',
  batch_size: 'batch_size',
  'lr_schedule.peak_lr': 'learning_rate',
  num_workers: 'num_workers',
  save_interval: 'save_interval',
  keep_period: 'keep_period',
  log_interval: 'log_interval',
  'lr_schedule.warmup_steps': 'warmup_steps',
  'lr_schedule.decay_steps': 'decay_steps',
  'lr_schedule.decay_lr': 'decay_lr',
  'optimizer.clip_gradient_norm': 'clip_gradient_norm',
  ema_decay: 'ema_decay',
  fsdp_devices: 'fsdp_devices',
};
const warningMessages: Record<string, string> = {
  DYNAMIC_REGISTRY_UNSUPPORTED: '部分配置由 Python 动态生成，无法静态解析。',
  DISCOVERY_LIMIT_REACHED: '配置读取达到扫描上限，部分内容未展开。',
  INVALID_RECIPE_NAME: '已忽略不符合命名规则的配方。',
  DETAIL_FIELDS_TRUNCATED: '字段较多，仅展示读取范围内的配置。',
  DETAIL_EXPRESSION_TRUNCATED: '部分较长的 Python 表达式已截断。',
  UNRESOLVED_DEFAULTS: '部分默认值无法静态解析，将在训练环境中计算。',
};

function showValue(field?: RecipeField) {
  if (!field) return '未能静态读取';
  if (field.kind === 'expression') return field.expression || '运行时计算';
  if (field.value === null) return 'None';
  if (typeof field.value === 'boolean') return field.value ? 'True' : 'False';
  if (typeof field.value === 'string') return field.value;
  return JSON.stringify(field.value) ?? field.expression ?? '未能静态读取';
}
function sourceIdentity(source: RecipeSource) {
  return JSON.stringify([
    source.path,
    source.sha256,
    source.commit_sha,
    source.changes_digest,
    source.code_dir,
    source.dirty,
  ]);
}

const protectedFields = new Set([
  'name',
  'code_dir',
  'python',
  'entrypoint',
  'output_dir',
  'output_root',
  'checkpoint_base_dir',
  'exp_name',
  'resume',
  'overwrite',
  'wandb_enabled',
]);
function literalValue(value: unknown, depth = 0): value is RecipeLiteral {
  if (depth > 4) return false;
  if (value === null || typeof value === 'boolean') return true;
  if (typeof value === 'number')
    return Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value));
  if (typeof value === 'string')
    return value.length <= 1024 && !/[\u0000-\u0008\u000b-\u001f]/.test(value);
  return (
    Array.isArray(value) &&
    value.length <= 64 &&
    value.every((item) => literalValue(item, depth + 1))
  );
}
function sameLiteralType(value: RecipeLiteral, original: unknown): boolean {
  if (original === null) return true;
  if (Array.isArray(original)) {
    if (!Array.isArray(value)) return false;
    if (!original.length) return true;
    if (value.length === original.length)
      return value.every((item, index) => sameLiteralType(item, original[index]));
    return [...original, ...value].every((item) => sameLiteralType(item, original[0]));
  }
  return (
    typeof value === typeof original &&
    (typeof original !== 'number' ||
      !Number.isInteger(original) ||
      (typeof value === 'number' && Number.isInteger(value)))
  );
}
function literalText(value: unknown) {
  return typeof value === 'string' ? value : (JSON.stringify(value) ?? '');
}
function displayLiteral(value: unknown) {
  return value === null ? 'None' : typeof value === 'string' ? value : JSON.stringify(value);
}
function literalEditable(field: RecipeField, fields: RecipeField[]) {
  if (isAssetManagedField(field.path)) return false;
  if (editable[field.path]) return true;
  if (
    field.kind !== 'literal' ||
    !literalValue(field.value) ||
    JSON.stringify(field.value).length > 8192
  )
    return false;
  if (!/^(data|model|optimizer|lr_schedule)\./.test(field.path)) return false;
  if (
    !/^[A-Za-z][A-Za-z0-9_]*(?:(?:\.[A-Za-z][A-Za-z0-9_]*)|(?:\[(?:0|[1-9][0-9]?)\]))*$/.test(
      field.path,
    )
  )
    return false;
  const tokens = field.path.match(/[A-Za-z][A-Za-z0-9_]*|[0-9]+/g) || [];
  if (
    tokens.length > 16 ||
    tokens.some(
      (token) =>
        protectedFields.has(token) ||
        token.startsWith('_') ||
        (/^[0-9]+$/.test(token) && Number(token) >= 64),
    )
  )
    return false;
  return !fields.some(
    (parent) =>
      parent.path !== field.path &&
      (field.path.startsWith(`${parent.path}.`) || field.path.startsWith(`${parent.path}[`)) &&
      parent.kind === 'literal' &&
      Array.isArray(parent.value) &&
      literalValue(parent.value),
  );
}
function suggestedConfigName(base: string) {
  const suffix = `_custom_${Date.now().toString(36)}_${globalThis.crypto?.randomUUID?.().slice(0, 8) || Math.random().toString(36).slice(2, 10)}`;
  return `${base.slice(0, 255 - suffix.length)}${suffix}`;
}
interface PendingRecipeSave {
  saved: SavedRecipe;
  configName: string;
  mode: 'new' | 'revision';
  job?: RecipeWriteback;
}
const pendingRecipeWrites = new Map<string, PendingRecipeSave>();
const activeRecipeWrites = new Map<string, Promise<RecipeWriteback>>();
const activeRecipeSaves = new Map<string, Promise<RecipeWriteback>>();
const pendingStorageKey = (key: string) => `termitech.recipe-writeback.${key}`;
function readPending(key: string) {
  if (pendingRecipeWrites.has(key)) return pendingRecipeWrites.get(key);
  try {
    const raw = sessionStorage.getItem(pendingStorageKey(key));
    if (raw) {
      const value = JSON.parse(raw) as PendingRecipeSave;
      if (value.saved?.latest_revision?.id && typeof value.configName === 'string') {
        pendingRecipeWrites.set(key, value);
        return value;
      }
    }
  } catch {
    /* Storage may be unavailable; in-memory recovery still works. */
  }
}
function rememberPending(key: string, value?: PendingRecipeSave) {
  if (value) pendingRecipeWrites.set(key, value);
  else pendingRecipeWrites.delete(key);
  try {
    if (value) sessionStorage.setItem(pendingStorageKey(key), JSON.stringify(value));
    else sessionStorage.removeItem(pendingStorageKey(key));
  } catch {
    /* The current tab retains the pending operation in memory. */
  }
}
function writebackError(job: RecipeWriteback) {
  const code = job.error?.code || 'RECIPE_WRITEBACK_FAILED';
  const suffix = ['RECIPE_CHECKOUT_BUSY', 'RECIPE_WRITEBACK_PENDING', 'RECIPE_WRITE_BUSY'].includes(
    code,
  )
    ? '请等该代码目录中的训练或写入结束，或使用独立代码副本。'
    : ['RECIPE_SOURCE_CHANGED', 'RECIPE_WRITE_SOURCE_CHANGED'].includes(code)
      ? '请重新读取当前代码，再保存新版本。'
      : ['GIT_WORKTREE_DIRTY', 'RECIPE_WRITE_SOURCE_DIRTY', 'RECIPE_WRITE_GIT_BUSY'].includes(code)
        ? '请先处理训练代码目录已有的 Git 修改，再重新保存。'
        : code === 'RECIPE_WRITE_DETAIL_UNAVAILABLE'
          ? `源码已经提交，请在代码配方中选择 ${job.config_name} 重新读取，完成平台登记。`
          : '';
  return new ApiError(code, `${job.error?.message || '配置写入或 Git 提交失败。'}${suffix}`, 0);
}
async function finishWriteback(key: string, pending: PendingRecipeSave, request = true) {
  const active = activeRecipeWrites.get(key);
  if (active) return active;
  const operation = (async () => {
    if (request && !pending.job) {
      pending.job = await post<RecipeWriteback>(
        `/recipe-revisions/${encodeURIComponent(pending.saved.latest_revision.id)}/writeback`,
        { config_name: pending.configName },
      );
      rememberPending(key, pending);
    }
    if (!pending.job) throw new Error('平台草稿已保存，请继续写入配置并提交。');
    while (pending.job.status === 'PENDING') {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      pending.job = await api<RecipeWriteback>(
        `/recipe-writebacks/${encodeURIComponent(pending.job.id)}`,
      );
      rememberPending(key, pending);
    }
    return pending.job;
  })();
  activeRecipeWrites.set(key, operation);
  try {
    return await operation;
  } finally {
    activeRecipeWrites.delete(key);
  }
}

export function RecipeDetails({
  node,
  profile,
  recipe,
  saved,
  onSaved,
  onDefaults,
  onDatasets,
  onResolved,
  onDirtyChange,
  onLoadingChange,
  restoredEditorDraft,
  onEditorDraftChange,
}: {
  node?: Node;
  profile?: Profile;
  recipe: string;
  saved?: SavedRecipe;
  onSaved: (saved: SavedRecipe) => void;
  onDefaults: (defaults: Record<string, number>, recipe: string, revisionId?: string) => void;
  onDatasets: (slots: RecipeDatasetSlot[], recipe: string, revisionId?: string) => void;
  onResolved?: (detail: RecipeDetail, recipe: string, revisionId?: string) => void;
  onDirtyChange: (dirty: boolean) => void;
  onLoadingChange: (loading: boolean) => void;
  restoredEditorDraft?: RecipeEditorDraft;
  onEditorDraftChange?: (draft: RecipeEditorDraft | undefined) => void;
}) {
  const { can } = useAuth();
  const mayEdit = can('recipes.write');
  const mayWriteback = can('recipes.writeback');
  const mayQuery = can('training.write');
  const id = useId();
  const [query, setQuery] = useState<{ scope: string; value: RecipeDetailQuery }>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error>();
  const [revision, setRevision] = useState(0);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [reviewedAssetOverrides, setReviewedAssetOverrides] = useState(false);
  const [reviewedAssetValues, setReviewedAssetValues] = useState<RecipeOverrides>({});
  const [showSave, setShowSave] = useState<'new' | 'revision'>();
  const [saveName, setSaveName] = useState('');
  const [configName, setConfigName] = useState('');
  const [pendingSave, setPendingSave] = useState<PendingRecipeSave>();
  const [writeNotice, setWriteNotice] = useState<{ configName: string; sha: string }>();
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<Error>();
  const [useCurrentSource, setUseCurrentSource] = useState(false);
  const defaultsRef = useRef(onDefaults);
  const datasetsRef = useRef(onDatasets);
  const resolvedRef = useRef(onResolved);
  const dirtyRef = useRef(onDirtyChange);
  const loadingRef = useRef(onLoadingChange);
  const editorChangeRef = useRef(onEditorDraftChange);
  const ignoredEditorToken = useRef<string | undefined>(undefined);
  const hydratedEditor = useRef<{ scope: string; token: string; sourceSha256: string } | undefined>(undefined);
  const publishedEditor = useRef<{ scope: string; token: string } | undefined>(undefined);
  const editorPublishReady = useRef(false);
  defaultsRef.current = onDefaults;
  datasetsRef.current = onDatasets;
  resolvedRef.current = onResolved;
  dirtyRef.current = onDirtyChange;
  loadingRef.current = onLoadingChange;
  editorChangeRef.current = onEditorDraftChange;
  const scope = JSON.stringify([
    node?.id,
    profile?.id,
    profile?.code_dir,
    profile?.python,
    profile?.adapter,
    recipe,
    saved?.latest_revision.id,
  ]);
  const scopeRef = useRef(scope);
  const mountedRef = useRef(true);
  const saveTriggerRef = useRef<HTMLButtonElement | null>(null);
  const pendingSaves = useRef(new Set<string>());
  const writeKey = JSON.stringify([
    node?.id,
    profile?.id,
    profile?.code_dir,
    profile?.python,
    profile?.adapter,
    recipe,
    saved?.id || 'code',
  ]);
  const writeKeyRef = useRef(writeKey);
  writeKeyRef.current = writeKey;
  const onSavedRef = useRef(onSaved);
  onSavedRef.current = onSaved;
  scopeRef.current = scope;
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  const current = query?.scope === scope ? query.value : undefined;
  const detail =
    !useCurrentSource && saved?.latest_revision.detail
      ? saved.latest_revision.detail
      : current?.detail;
  const restoredEditorToken = restoredEditorDraft ? JSON.stringify(restoredEditorDraft) : undefined;
  const cachedEditor =
    restoredEditorDraft?.scope === scope && restoredEditorToken !== ignoredEditorToken.current
      ? restoredEditorDraft
      : undefined;
  const hasCachedEditor =
    !!cachedEditor && (Object.keys(cachedEditor.fields).length > 0 || !!cachedEditor.saveMode);
  const cachedEditorSourceVerified =
    !!cachedEditor &&
    !!current?.detail &&
    !loading &&
    !!detail &&
    cachedEditor.sourceSha256 === current.detail.source.sha256 &&
    cachedEditor.sourceSha256 === detail.source.sha256;
  const cachedEditorConflict =
    hasCachedEditor && !!current?.detail && !loading && !cachedEditorSourceVerified;
  const cachedEditorBlocked =
    hasCachedEditor &&
    (!cachedEditorSourceVerified ||
      (hydratedEditor.current?.token !== restoredEditorToken &&
        publishedEditor.current?.token !== restoredEditorToken));
  const savedOverrides = saved?.latest_revision.definition.overrides || {};
  const savedAssetOverrides = assetManagedOverridePaths(savedOverrides);
  const draftAssetOverrides = assetManagedOverridePaths(
    Object.fromEntries(
      Object.entries(draft).map(([path, text]) => {
        if (typeof detail?.fields.find((field) => field.path === path)?.value === 'string')
          return [path, text];
        try {
          return [path, JSON.parse(text)];
        } catch {
          return [path, text];
        }
      }),
    ),
  );
  const pendingAssetOverrides = assetManagedOverridePaths(
    pendingSave?.saved.latest_revision.definition.overrides || {},
  );
  const baseOverrides = reviewedAssetOverrides
    ? Object.fromEntries(
        Object.entries(savedOverrides).filter(([path]) => !savedAssetOverrides.includes(path)),
      )
    : savedOverrides;
  const assetOverridePaths = [
    ...new Set([
      ...savedAssetOverrides,
      ...Object.keys(reviewedAssetValues),
      ...draftAssetOverrides,
      ...pendingAssetOverrides,
    ]),
  ];
  const requiresAssetReview =
    (!reviewedAssetOverrides && !pendingSave && savedAssetOverrides.length > 0) ||
    draftAssetOverrides.length > 0 ||
    pendingAssetOverrides.length > 0;
  const hasDraft = Object.keys(draft).length > 0 || Object.keys(reviewedAssetValues).length > 0;

  useEffect(() => {
    ignoredEditorToken.current = undefined;
    hydratedEditor.current = undefined;
    publishedEditor.current = undefined;
    editorPublishReady.current = false;
    setDraft({});
    setReviewedAssetOverrides(false);
    setReviewedAssetValues({});
    setShowSave(undefined);
    setSaveName('');
    setConfigName('');
    setDescription('');
    setSaveError(undefined);
    setUseCurrentSource(false);
    if (writeNotice && writeNotice.configName !== recipe) setWriteNotice(undefined);
    setSaving(false);
    const pending = mayWriteback ? readPending(writeKey) : undefined;
    setPendingSave(pending);
    if (pending) {
      setShowSave(pending.mode);
      setSaveName(pending.saved.name);
      setDescription(pending.saved.description);
      setConfigName(pending.configName);
      if (pending.job?.status === 'FAILED') setSaveError(writebackError(pending.job));
    }
    dirtyRef.current(!!pending || hasCachedEditor);
  }, [scope, mayWriteback]);

  const forgetEditorDraft = () => {
    ignoredEditorToken.current = restoredEditorToken;
    publishedEditor.current = { scope, token: `empty:${scope}` };
    editorChangeRef.current?.(undefined);
  };

  const acceptWriteback = (job: RecipeWriteback, key: string, expectedScope: string) => {
    if (!mountedRef.current || scopeRef.current !== expectedScope || writeKeyRef.current !== key)
      return;
    const pending = readPending(key);
    setPendingSave(pending);
    if (job.status === 'FAILED') {
      setSaveError(writebackError(job));
      return;
    }
    if (job.status !== 'SUCCEEDED' || !job.recipe || !job.commit_sha)
      throw new Error('配置写入结果不完整，请重新查询提交结果。');
    rememberPending(key);
    setPendingSave(undefined);
    setWriteNotice({ configName: job.config_name, sha: job.commit_sha });
    setDraft({});
    setShowSave(undefined);
    forgetEditorDraft();
    dirtyRef.current(false);
    onSavedRef.current(job.recipe);
  };

  useEffect(() => {
    if (!mayWriteback) return;
    const pending = readPending(writeKey);
    const active = activeRecipeSaves.get(writeKey);
    if (!active && (!pending?.job || pending.job.status === 'FAILED')) return;
    let live = true;
    setSaving(true);
    void (active || finishWriteback(writeKey, pending!, false))
      .then((job) => {
        if (live) acceptWriteback(job, writeKey, scope);
      })
      .catch((err) => {
        if (live && mountedRef.current && scopeRef.current === scope) setSaveError(err as Error);
      })
      .finally(() => {
        if (live && mountedRef.current && scopeRef.current === scope) setSaving(false);
      });
    return () => {
      live = false;
    };
  }, [scope, mayWriteback]);

  useEffect(() => {
    setQuery(undefined);
    setError(undefined);
    if (!mayQuery) {
      setLoading(false);
      return;
    }
    if (
      !node ||
      !profile ||
      profile.node_id !== node.id ||
      !/^[A-Za-z0-9_-]{1,255}$/.test(recipe)
    ) {
      setLoading(false);
      return;
    }
    if (node.status !== 'ONLINE') {
      setLoading(false);
      setError(new Error('机器离线，暂时无法读取配方详情。'));
      return;
    }
    let live = true;
    let timedOut = false;
    let timer: ReturnType<typeof setTimeout>;
    let deadline: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setLoading(true);
    const debounce = setTimeout(async () => {
      deadline = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, 15000);
      try {
        let response = await api<RecipeDetailQuery>(
          `/nodes/${encodeURIComponent(node.id)}/recipe-details`,
          {
            method: 'POST',
            body: JSON.stringify({ profile_id: profile.id, recipe }),
            signal: controller.signal,
          },
        );
        while (response.status === 'PENDING') {
          await new Promise<void>((resolve, reject) => {
            const abort = () => {
              clearTimeout(timer);
              reject(new DOMException('Aborted', 'AbortError'));
            };
            if (controller.signal.aborted) return abort();
            controller.signal.addEventListener('abort', abort, { once: true });
            timer = setTimeout(() => {
              controller.signal.removeEventListener('abort', abort);
              resolve();
            }, 500);
          });
          response = await api<RecipeDetailQuery>(
            `/nodes/${encodeURIComponent(node.id)}/recipe-details/${encodeURIComponent(response.id)}`,
            { signal: controller.signal },
          );
        }
        if (response.status === 'FAILED' || !response.detail)
          throw new ApiError(
            response.error?.code || 'RECIPE_DETAIL_FAILED',
            response.error?.message || '无法读取配方详情。',
            0,
          );
        if (
          response.node_id !== node.id ||
          response.profile_id !== profile.id ||
          response.recipe !== recipe
        )
          throw new Error('配方详情与当前选择不一致，请重新读取。');
        if (live) setQuery({ scope, value: response });
      } catch (err) {
        if (live)
          setError(timedOut ? new Error('配方详情读取超过 15 秒，请重试。') : (err as Error));
      } finally {
        clearTimeout(deadline);
        if (live) setLoading(false);
      }
    }, 500);
    return () => {
      live = false;
      clearTimeout(debounce);
      clearTimeout(timer);
      clearTimeout(deadline);
      controller.abort();
    };
  }, [scope, node?.status, revision, mayQuery]);

  useEffect(() => {
    if (!current?.detail || loading || !detail || saving || pendingSave || cachedEditorConflict) {
      editorPublishReady.current = false;
      return;
    }
    if (
      cachedEditor &&
      restoredEditorToken &&
      cachedEditorSourceVerified &&
      publishedEditor.current?.token !== restoredEditorToken &&
      (hydratedEditor.current?.scope !== scope ||
        hydratedEditor.current.token !== restoredEditorToken ||
        hydratedEditor.current.sourceSha256 !== current.detail.source.sha256)
    ) {
      editorPublishReady.current = false;
      hydratedEditor.current = {
        scope,
        token: restoredEditorToken,
        sourceSha256: current.detail.source.sha256,
      };
      setDraft({ ...cachedEditor.fields });
      setShowSave(cachedEditor.saveMode);
      setSaveName(cachedEditor.saveName || '');
      setConfigName(cachedEditor.configName || '');
      setDescription(cachedEditor.description || '');
      // Source verification and asset-override review always happen anew.
      setReviewedAssetOverrides(false);
      setReviewedAssetValues({});
      return;
    }
    editorPublishReady.current = true;
  }, [
    scope,
    current?.detail,
    detail,
    loading,
    saving,
    pendingSave,
    restoredEditorToken,
    cachedEditorConflict,
    cachedEditorSourceVerified,
    draft,
  ]);

  useEffect(() => {
    if (
      !editorChangeRef.current ||
      !editorPublishReady.current ||
      !current?.detail ||
      !detail ||
      loading ||
      saving ||
      pendingSave ||
      cachedEditorBlocked
    )
      return;
    const hasText = Object.keys(draft).length > 0 || !!showSave;
    // Reading a different recipe must not erase its still-cached editor.
    if (
      !hasText &&
      restoredEditorDraft?.scope !== scope &&
      publishedEditor.current?.scope !== scope
    )
      return;
    const value: RecipeEditorDraft | undefined = hasText
      ? {
          scope,
          sourceSha256: detail.source.sha256,
          fields: { ...draft },
          ...(showSave ? { saveMode: showSave, saveName, configName, description } : {}),
        }
      : undefined;
    const token = value ? JSON.stringify(value) : `empty:${scope}`;
    if (publishedEditor.current?.scope === scope && publishedEditor.current.token === token) return;
    publishedEditor.current = { scope, token };
    editorChangeRef.current(value);
  }, [
    scope,
    current?.detail,
    detail,
    loading,
    saving,
    pendingSave,
    cachedEditorBlocked,
    restoredEditorToken,
    draft,
    showSave,
    saveName,
    configName,
    description,
  ]);

  useEffect(() => {
    if (!detail) return;
    const values: Record<string, number> = {};
    for (const [path, parameter] of Object.entries(commonKeys)) {
      const field = detail.fields.find((entry) => entry.path === path);
      const value = Object.hasOwn(baseOverrides, path)
        ? baseOverrides[path]
        : field?.kind === 'literal'
          ? field.value
          : undefined;
      if (typeof value === 'number' && Number.isFinite(value)) values[parameter] = value;
    }
    defaultsRef.current(values, recipe, saved?.latest_revision.id);
    datasetsRef.current(recipeDatasetSlots(detail), recipe, saved?.latest_revision.id);
    resolvedRef.current?.(
      {
        ...detail,
        fields: detail.fields.map((field) =>
          Object.hasOwn(baseOverrides, field.path) &&
          !savedAssetOverrides.includes(field.path) &&
          !isAssetManagedField(field.path)
            ? { ...field, kind: 'literal', value: baseOverrides[field.path] }
            : field,
        ),
      },
      recipe,
      saved?.latest_revision.id,
    );
  }, [detail, saved?.latest_revision.id, reviewedAssetOverrides]);

  useEffect(() => {
    dirtyRef.current(hasDraft || hasCachedEditor || requiresAssetReview || !!pendingSave || saving);
  }, [hasDraft, hasCachedEditor, requiresAssetReview, pendingSave, saving]);
  useEffect(() => {
    loadingRef.current(loading || saving);
    return () => {
      loadingRef.current(false);
    };
  }, [loading, saving]);

  const fieldMap = new Map(detail?.fields.map((field) => [field.path, field]));
  const effectiveValue = (path: string) =>
    Object.hasOwn(baseOverrides, path)
      ? baseOverrides[path]
      : fieldMap.get(path)?.kind === 'literal'
        ? fieldMap.get(path)?.value
        : undefined;
  const overrides: RecipeOverrides = { ...baseOverrides };
  const invalid: string[] = [];
  for (const [path, text] of Object.entries(draft)) {
    const rule = editable[path];
    try {
      const field = fieldMap.get(path);
      let value: RecipeLiteral;
      if (rule) {
        const number = text.trim() === '' ? NaN : Number(text);
        if (
          !Number.isFinite(number) ||
          (rule.integer && !Number.isInteger(number)) ||
          (rule.min !== undefined &&
            (rule.exclusiveMin ? number <= rule.min : number < rule.min)) ||
          (rule.max !== undefined && (rule.exclusiveMax ? number >= rule.max : number > rule.max))
        )
          throw new Error();
        value = number;
      } else {
        if (!field || !literalEditable(field, detail?.fields || [])) throw new Error();
        value = typeof field.value === 'string' ? text : JSON.parse(text);
        if (
          !literalValue(value) ||
          !(path === 'data.weights'
            ? value === null ||
              (Array.isArray(value) &&
                value.length > 0 &&
                value.every(
                  (weight) => typeof weight === 'number' && Number.isFinite(weight) && weight > 0,
                ))
            : sameLiteralType(value, field.value)) ||
          JSON.stringify(value).length > 8192
        )
          throw new Error();
      }
      overrides[path] = value;
    } catch {
      invalid.push(rule?.label || path);
    }
  }
  const changedPaths = [...new Set([...Object.keys(draft), ...Object.keys(reviewedAssetValues)])];
  const multiDataset = detail?.fields.some(
    (field) =>
      field.path.startsWith('data.datasets') ||
      (field.path === 'data' && field.expression.includes('MultiDataConfigFactory')),
  );
  const sourceChanged =
    !!saved &&
    !!current?.detail &&
    sourceIdentity(saved.latest_revision.definition.source) !==
      sourceIdentity(current.detail.source);
  const canSave =
    mayEdit &&
    !!current?.detail &&
    (!sourceChanged || useCurrentSource) &&
    !requiresAssetReview &&
    !cachedEditorBlocked;
  const save = async () => {
    if (
      !mayEdit ||
      !node ||
      !profile ||
      !showSave ||
      pendingSaves.current.has(scope) ||
      activeRecipeSaves.has(writeKey)
    )
      return;
    const pending = mayWriteback ? readPending(writeKey) : undefined;
    if (requiresAssetReview && (!pending?.job || pending.job.status !== 'PENDING')) {
      setSaveError(new Error('请先核对并移除配方中的资产覆盖，再到“资产与参数”选择训练资产。'));
      return;
    }
    if (!pending && (!current || !detail || !canSave)) return;
    if (!pending && invalid.length) {
      setSaveError(new Error(`请检查${invalid.join('、')}的类型、格式或数值范围。`));
      return;
    }
    if (!pending && showSave === 'new' && !saveName.trim()) {
      setSaveError(new Error('请填写新配方名称。'));
      return;
    }
    if (mayWriteback && !/^[A-Za-z0-9_-]{1,255}$/.test(pending?.configName || configName)) {
      setSaveError(new Error('config_name 只能包含 1–255 个英文字母、数字、下划线或连字符。'));
      return;
    }
    setSaving(true);
    pendingSaves.current.add(scope);
    setSaveError(undefined);
    setWriteNotice(undefined);
    if (!mayWriteback && current) {
      try {
        const body = { detail_query_id: current.id, overrides, description: description.trim() };
        const result =
          showSave === 'revision' && saved
            ? await post<SavedRecipe>(`/recipes/${encodeURIComponent(saved.id)}/revisions`, body)
            : await post<SavedRecipe>('/recipes', {
                ...body,
                node_id: node.id,
                profile_id: profile.id,
                name: saveName.trim(),
              });
        if (mountedRef.current && scopeRef.current === scope) {
          setDraft({});
          setShowSave(undefined);
          forgetEditorDraft();
          dirtyRef.current(false);
          onSavedRef.current(result);
        }
      } catch (err) {
        if (mountedRef.current && scopeRef.current === scope) setSaveError(err as Error);
      } finally {
        pendingSaves.current.delete(scope);
        if (mountedRef.current && scopeRef.current === scope) setSaving(false);
      }
      return;
    }
    const operationPromise = (async () => {
      let operation = pending;
      if (!operation && current) {
        const body = { detail_query_id: current.id, overrides, description: description.trim() };
        const result =
          showSave === 'revision' && saved
            ? await post<SavedRecipe>(`/recipes/${encodeURIComponent(saved.id)}/revisions`, body)
            : await post<SavedRecipe>('/recipes', {
                ...body,
                node_id: node.id,
                profile_id: profile.id,
                name: saveName.trim(),
              });
        operation = { saved: result, configName, mode: showSave };
        rememberPending(writeKey, operation);
        if (mountedRef.current && scopeRef.current === scope) setPendingSave(operation);
      }
      if (!operation) throw new Error('请重新读取配方配置。');
      return finishWriteback(writeKey, operation);
    })();
    activeRecipeSaves.set(writeKey, operationPromise);
    try {
      acceptWriteback(await operationPromise, writeKey, scope);
    } catch (err) {
      if (mountedRef.current && scopeRef.current === scope) {
        setPendingSave(readPending(writeKey));
        setSaveError(
          err instanceof ApiError
            ? writebackError({ error: { code: err.code, message: err.message } } as RecipeWriteback)
            : (err as Error),
        );
      }
    } finally {
      activeRecipeSaves.delete(writeKey);
      pendingSaves.current.delete(scope);
      if (mountedRef.current && scopeRef.current === scope) setSaving(false);
    }
  };

  const closeSave = () => {
    if (saving || pendingSave) return;
    setShowSave(undefined);
    setSaveError(undefined);
    saveTriggerRef.current?.focus();
  };
  const reviewAssetOverrides = () => {
    if (!mayEdit) return;
    if (saving || pendingSave?.job?.status === 'PENDING' || pendingSave?.job?.commit_sha) return;
    const previousOverrides = {
      ...savedOverrides,
      ...pendingSave?.saved.latest_revision.definition.overrides,
      ...Object.fromEntries(
        Object.entries(draft).filter(([path]) => draftAssetOverrides.includes(path)),
      ),
    };
    const previousAssetPaths = assetManagedOverridePaths(previousOverrides);
    setReviewedAssetValues(
      Object.fromEntries(
        Object.entries(previousOverrides).filter(
          ([path]) => previousAssetPaths.includes(path) || draftAssetOverrides.includes(path),
        ),
      ),
    );
    const next = { ...draft };
    if (pendingSave) {
      for (const [path, value] of Object.entries(
        pendingSave.saved.latest_revision.definition.overrides,
      )) {
        if (!pendingAssetOverrides.includes(path)) next[path] = literalText(value);
      }
      // The immutable platform draft is retained; only its unsent/failed local retry is closed.
      rememberPending(writeKey);
      setPendingSave(undefined);
    }
    for (const path of [...previousAssetPaths, ...draftAssetOverrides]) delete next[path];
    setDraft(next);
    setReviewedAssetOverrides(true);
    setShowSave(saved ? 'revision' : 'new');
    setSaveName(pendingSave?.saved.name || (saved ? `${saved.name} 副本` : `${recipe} 自定义`));
    setConfigName(suggestedConfigName(recipe));
    setDescription(pendingSave?.saved.description || saved?.description || '');
    setSaveError(undefined);
  };
  const renderField = (field: RecipeField, view: 'group' | 'all') => {
    const rule = editable[field.path];
    const assetManaged =
      isAssetManagedField(field.path) ||
      savedAssetOverrides.includes(field.path) ||
      draftAssetOverrides.includes(field.path);
    const canEdit = mayEdit && !assetManaged && literalEditable(field, detail?.fields || []);
    const overridden = Object.hasOwn(baseOverrides, field.path);
    const value = effectiveValue(field.path);
    return (
      <div className="recipe-field-row" key={field.path}>
        <div className="recipe-field-label">
          <label htmlFor={canEdit ? `${id}-${view}-${field.path}` : undefined}>
            {rule?.label || field.path}
          </label>
          <code>{field.path}</code>
          <small>
            {assetManaged ? '由训练资产绑定' : field.origin === 'default' ? '类默认值' : '配方定义'}
            {!assetManaged &&
              (field.kind === 'expression'
                ? rule
                  ? ' · Python 表达式（可填写数值覆盖）'
                  : ' · Python 表达式（只读）'
                : canEdit
                  ? ' · 可编辑'
                  : ' · 只读')}
          </small>
        </div>
        <div className="recipe-field-value">
          {assetManaged ? (
            <>
              <p className="recipe-asset-binding">
                {field.path === 'data.weights'
                  ? '混采权重在“资产与参数”的数据绑定中设置。'
                  : '在“资产与参数”中选择本次训练使用的资产。'}
              </p>
              <details className="recipe-asset-default">
                <summary>查看代码默认值（仅供参考）</summary>
                <code>{showValue(field)}</code>
              </details>
            </>
          ) : (
            <code>{showValue(field)}</code>
          )}
          {overridden && (
            <span className={assetManaged ? 'recipe-detail-warning' : 'recipe-field-saved'}>
              {assetManaged ? '旧版本的资产覆盖（需移除）：' : '保存的覆盖值：'}
              {displayLiteral(baseOverrides[field.path])}
            </span>
          )}
          {canEdit &&
            (() => {
              const originalText = literalText(value);
              const text = Object.hasOwn(draft, field.path) ? draft[field.path] : originalText;
              const update = (nextText: string) =>
                setDraft((previous) => {
                  const next = { ...previous };
                  if (nextText === originalText) delete next[field.path];
                  else next[field.path] = nextText;
                  return next;
                });
              const inputProps = {
                id: `${id}-${view}-${field.path}`,
                disabled: saving || !!pendingSave || cachedEditorBlocked,
                'aria-label': `修改${rule?.label || field.path}`,
                'aria-invalid':
                  Object.hasOwn(draft, field.path) && invalid.includes(rule?.label || field.path),
                value: text,
                onChange: (event: { target: { value: string } }) => update(event.target.value),
              };
              return (
                <div className="recipe-field-edit">
                  {!rule && typeof field.value === 'boolean' ? (
                    <select {...inputProps}>
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  ) : !rule && (Array.isArray(field.value) || field.value === null) ? (
                    <textarea
                      {...inputProps}
                      rows={2}
                      spellCheck={false}
                      placeholder="JSON，例如 [1, 2, 3] 或 null"
                    />
                  ) : (
                    <input
                      {...inputProps}
                      type={rule || typeof field.value === 'number' ? 'number' : 'text'}
                      inputMode={
                        rule?.integer
                          ? 'numeric'
                          : rule || typeof field.value === 'number'
                            ? 'decimal'
                            : 'text'
                      }
                      step={rule?.integer ? 1 : 'any'}
                      min={Object.hasOwn(draft, field.path) ? rule?.min : undefined}
                      max={Object.hasOwn(draft, field.path) ? rule?.max : undefined}
                      placeholder={
                        field.kind === 'expression' && !overridden
                          ? '填写数值覆盖表达式'
                          : undefined
                      }
                    />
                  )}
                  {Object.hasOwn(draft, field.path) && (
                    <button
                      type="button"
                      className="text-link"
                      disabled={saving || !!pendingSave}
                      onClick={() => update(originalText)}
                    >
                      <RotateCcw size={12} />
                      还原
                    </button>
                  )}
                </div>
              );
            })()}
        </div>
      </div>
    );
  };
  if (!recipe) return <p className="field-hint">选定配方后读取详细配置，可保存为自己的配方。</p>;
  return (
    <section
      className="recipe-details"
      aria-label="配方详细配置"
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return;
        if (event.key === 'Escape' && showSave) {
          event.preventDefault();
          event.stopPropagation();
          closeSave();
        }
        if (event.key === 'Enter' && event.target instanceof HTMLInputElement) {
          event.preventDefault();
          if (showSave && !saving) void save();
        }
      }}
    >
      <div className="recipe-details-heading">
        <div>
          <BookOpen size={17} />
          <h3>配方详情</h3>
          <code>{recipe}</code>
        </div>
        {saved && (
          <Badge
            state={saved.latest_revision.validated ? 'READY' : 'PLANNED'}
            label={`${saved.name} · v${saved.latest_revision.revision} · ${saved.latest_revision.validated ? '已有验收记录' : '预检时验证'}`}
          />
        )}
      </div>
      {writeNotice && (
        <div className="inline-note" role="status">
          <Check size={14} />
          <span>
            已写入配方配置文件：<code>{writeNotice.configName}</code>；Git 提交{' '}
            <code title={writeNotice.sha}>{writeNotice.sha.slice(0, 12)}</code>
            。新版本已选中，预检时验证配置。
          </span>
        </div>
      )}
      {loading && (
        <div className="recipe-detail-loading" role="status">
          <LoaderCircle size={15} className="spin" />
          正在读取代码定义与默认配置…
        </div>
      )}
      {hasCachedEditor && !cachedEditorSourceVerified && (
        <div className="inline-note amber" role={cachedEditorConflict ? 'alert' : 'status'}>
          <div>
            <strong>
              {cachedEditorConflict ? '缓存的配方修改与当前源码不一致' : '配方修改尚未恢复'}
            </strong>
            <p>
              {cachedEditorConflict
                ? '源码摘要已变化，旧修改保留在训练草稿中，但不会应用或用于训练。请放弃旧修改后按当前代码重新编辑。'
                : '需要重新读取配方并核对源码摘要后，才能恢复这些编辑文本；资产覆盖的核对状态不会恢复。'}
            </p>
          </div>
          <button
            type="button"
            className="button outline small"
            disabled={saving || !!pendingSave}
            onClick={() => {
              forgetEditorDraft();
              setDraft({});
              setShowSave(undefined);
              setSaveName('');
              setConfigName('');
              setDescription('');
              setReviewedAssetOverrides(false);
              setReviewedAssetValues({});
            }}
          >
            放弃旧配方修改
          </button>
        </div>
      )}
      <ErrorNotice error={error} retry={() => setRevision((value) => value + 1)} />
      {!mayQuery && !detail && (
        <p className="field-hint">
          只读账号可查看已保存的配方快照。读取机器上的代码配方需要操作员或管理员权限。
        </p>
      )}
      {detail && (
        <>
          <p className="recipe-detail-caption">
            {mayWriteback
              ? '可修改训练超参数，以及模型、数据处理和优化器中的静态配置值。保存会在来源配置文件中新增独立配方，并自动执行一次 Git 提交。'
              : mayEdit
                ? '可修改训练超参数，以及模型、数据处理和优化器中的静态配置值。保存将创建平台配方版本；写入训练代码需要管理员权限。'
                : '当前为只读配方快照，可查看配置和来源。编辑配方需要操作员或管理员权限。'}
          </p>
          <div className="inline-note" role="note">
            代码中的资产字段仅为默认值；本次训练以“资产与参数”中选择的数据集、归一化参数、初始模型及混采权重为准。
          </div>
          {assetOverridePaths.length > 0 && (
            <div className="recipe-asset-review inline-note amber" role="alert">
              <div>
                <strong>
                  {requiresAssetReview
                    ? '此版本含有旧的资产覆盖，需要先核对'
                    : '已确认移除旧的资产覆盖，尚未保存'}
                </strong>
                <p>
                  {requiresAssetReview
                    ? '这些字段与训练资产绑定冲突，不能直接保存或用于新训练。请移除覆盖后保存新版本，再到“资产与参数”选择资产；旧版本和已有训练记录保留。'
                    : '下方差异会列出移除项。保存新版本后，再为本次训练明确选择资产。'}
                </p>
                <ul>
                  {assetOverridePaths.map((path) => (
                    <li key={path}>
                      <code>{path}</code>
                    </li>
                  ))}
                </ul>
                {requiresAssetReview && pendingSave?.job?.commit_sha && (
                  <p>
                    此草稿已有 Git 提交回执，请先按下方任务结果重新读取对应代码；提交记录会保留。
                  </p>
                )}
              </div>
              {mayEdit && requiresAssetReview && (
                <button
                  type="button"
                  className="button outline small"
                  disabled={
                    saving ||
                    cachedEditorBlocked ||
                    pendingSave?.job?.status === 'PENDING' ||
                    !!pendingSave?.job?.commit_sha
                  }
                  onClick={reviewAssetOverrides}
                >
                  {pendingSave ? '保留旧草稿并核对移除项' : '核对移除项并准备新版本'}
                </button>
              )}
            </div>
          )}
          <dl className="recipe-source-info">
            <div>
              <dt>来源</dt>
              <dd>
                <code>
                  {detail.source.code_dir}/{detail.source.path}:{detail.source.line}
                </code>
              </dd>
            </div>
            <div>
              <dt>代码版本</dt>
              <dd>
                <code title={detail.source.commit_sha || undefined}>
                  {detail.source.commit_sha?.slice(0, 12) || '未读取到 Git SHA'}
                </code>
                {detail.source.dirty && <span>有本地修改</span>}
              </dd>
            </div>
            <div>
              <dt>文件摘要</dt>
              <dd>
                <code title={detail.source.sha256}>SHA256 {detail.source.sha256.slice(0, 16)}</code>
              </dd>
            </div>
          </dl>
          {sourceChanged && (
            <div className="inline-note amber">
              <span>
                {useCurrentSource
                  ? '已切换到当前代码。保存会绑定新代码快照，原配方版本保留。'
                  : '当前代码与该配方保存时不同，下面展示原版本的配置。使用新代码保存前请重新读取。'}
              </span>
              {mayQuery && !useCurrentSource && (
                <button
                  type="button"
                  className="text-link"
                  onClick={() => {
                    setUseCurrentSource(true);
                    setDraft({});
                    setReviewedAssetOverrides(false);
                    setReviewedAssetValues({});
                  }}
                >
                  以当前代码重新读取
                </button>
              )}
            </div>
          )}
          {multiDataset && (
            <div className="inline-note amber">
              此配方包含多个数据集。Agent
              将按下一步确认的数据列表，逐组检查数据集、归一化参数与混采权重。
            </div>
          )}
          <p className="field-hint">
            {mayEdit
              ? '训练与数据处理中的数字、布尔、字符串和列表可编辑。列表使用 JSON 格式；Python 构造器和动态表达式只读，请在训练代码中修改。运行输出目录由训练任务管理。'
              : '训练与数据处理配置只读。运行输出目录由训练任务管理。'}
          </p>
          <div className="recipe-field-groups">
            {[
              {
                title: '训练与检查点',
                match: (path: string) =>
                  [
                    'num_train_steps',
                    'batch_size',
                    'ema_decay',
                    'num_workers',
                    'fsdp_devices',
                    'log_interval',
                    'save_interval',
                    'keep_period',
                    'seed',
                  ].includes(path),
              },
              {
                title: '优化器与学习率',
                match: (path: string) =>
                  path.startsWith('optimizer') || path.startsWith('lr_schedule'),
              },
              { title: '模型', match: (path: string) => path.startsWith('model') },
              {
                title: '数据与权重',
                match: (path: string) =>
                  path.startsWith('data') ||
                  path.startsWith('weight_loader') ||
                  path.startsWith('assets'),
              },
            ].map((group) => {
              const fields = detail.fields.filter((field) => group.match(field.path));
              if (!fields.length) return null;
              return (
                <details
                  className="recipe-field-group"
                  key={group.title}
                  open={group.title === '训练与检查点' || group.title === '优化器与学习率'}
                >
                  <summary>
                    {group.title}
                    <span>{fields.length} 项</span>
                  </summary>
                  <div className="recipe-fields">
                    {fields.map((field) => renderField(field, 'group'))}
                  </div>
                </details>
              );
            })}
            <details className="recipe-field-group">
              <summary>
                全部字段<span>{detail.fields.length} 项</span>
              </summary>
              <div className="recipe-fields" aria-label="全部字段参数">
                {detail.fields.map((field) => renderField(field, 'all'))}
              </div>
            </details>
          </div>
          {mayEdit && Object.keys(editable).some((path) => !fieldMap.has(path)) && (
            <details className="recipe-field-group">
              <summary>
                其他可覆盖参数<span>原值尚未展开</span>
              </summary>
              <p className="field-hint">
                这些字段可能来自 Python
                表达式或默认对象。可明确填写数值覆盖，具体兼容性仍需训练预检。
              </p>
              <div className="form-grid recipe-missing-fields">
                {Object.entries(editable)
                  .filter(([path]) => !fieldMap.has(path))
                  .map(([path, rule]) => (
                    <Field key={path} label={rule.label} hint={path}>
                      <input
                        type="number"
                        inputMode={rule.integer ? 'numeric' : 'decimal'}
                        step={rule.integer ? 1 : 'any'}
                        min={Object.hasOwn(draft, path) ? rule.min : undefined}
                        max={Object.hasOwn(draft, path) ? rule.max : undefined}
                        placeholder="填写数值覆盖；留空沿用原定义"
                        aria-label={`修改${rule.label}`}
                        value={
                          Object.hasOwn(draft, path)
                            ? draft[path]
                            : Object.hasOwn(baseOverrides, path)
                              ? String(baseOverrides[path])
                              : ''
                        }
                        onChange={(event) => {
                          const text = event.target.value;
                          setDraft((previous) => {
                            const next = { ...previous };
                            if (
                              text ===
                              (Object.hasOwn(baseOverrides, path)
                                ? String(baseOverrides[path])
                                : '')
                            )
                              delete next[path];
                            else next[path] = text;
                            return next;
                          });
                        }}
                        disabled={saving || !!pendingSave || cachedEditorBlocked}
                      />
                    </Field>
                  ))}
              </div>
            </details>
          )}
          {detail.warnings.map((warning) =>
            warningMessages[warning] ? (
              <p className="recipe-detail-warning" key={warning}>
                {warningMessages[warning]}
              </p>
            ) : null,
          )}
          <details className="recipe-source-code">
            <summary>
              <Code2 size={14} />
              查看 Python 定义
            </summary>
            <pre>
              <code>{detail.source_code}</code>
            </pre>
          </details>
          <p className="field-hint">
            {multiDataset
              ? '下一步可配置多组数据集、各自归一化参数与混采权重。'
              : '数据、初始权重、归一化统计和输出目录由本次训练选择。'}
            {mayWriteback
              ? '步数、批次与学习率可在下一步临时调整。这里的修改需要先保存，保存后会写入所选训练代码目录的来源配置文件，并提交到本地 Git 仓库。'
              : mayEdit
                ? '步数、批次与学习率可在下一步临时调整。这里的修改需要先保存到平台配方库，训练时使用选中的固定版本。'
                : '配置与来源按此配方保存时的版本展示。'}
          </p>
          {hasDraft && (
            <div className="recipe-pending-edits" role="status">
              <strong>
                {invalid.length
                  ? `请检查：${invalid.join('、')}`
                  : `${changedPaths.length} 项修改尚未保存`}
              </strong>
              <button
                type="button"
                className="text-link"
                disabled={saving || !!pendingSave || cachedEditorBlocked}
                onClick={() => {
                  setDraft({});
                  setReviewedAssetOverrides(false);
                  setReviewedAssetValues({});
                }}
              >
                <RotateCcw size={12} />
                还原全部修改
              </button>
            </div>
          )}
          {mayEdit && (
            <div className="recipe-save-actions">
              <button
                type="button"
                className="button outline small"
                disabled={saving || !!pendingSave || !canSave}
                onClick={(event) => {
                  saveTriggerRef.current = event.currentTarget;
                  setShowSave('new');
                  setSaveName(saved ? `${saved.name} 副本` : `${recipe} 自定义`);
                  setConfigName(suggestedConfigName(recipe));
                  setDescription(saved?.description || '');
                  setSaveError(undefined);
                }}
              >
                <Save size={14} />
                另存为新配方
              </button>
              {saved && (
                <button
                  type="button"
                  className="button outline small"
                  disabled={saving || !!pendingSave || !canSave || !hasDraft}
                  onClick={(event) => {
                    saveTriggerRef.current = event.currentTarget;
                    setShowSave('revision');
                    setConfigName(suggestedConfigName(recipe));
                    setDescription(saved.description);
                    setSaveError(undefined);
                  }}
                >
                  <Save size={14} />
                  保存新版本
                </button>
              )}
            </div>
          )}
          {mayEdit && showSave && (
            <section
              className="recipe-save-panel"
              aria-label={showSave === 'new' ? '另存为新配方' : '保存新版本'}
            >
              <h4>{showSave === 'new' ? '另存为新配方' : `${saved?.name} · 新版本`}</h4>
              {showSave === 'new' && (
                <Field label="平台显示名称" required>
                  <input
                    autoFocus
                    maxLength={160}
                    value={saveName}
                    onChange={(event) => setSaveName(event.target.value)}
                    placeholder="例如 抓取实验-低学习率"
                    disabled={saving || !!pendingSave || cachedEditorBlocked}
                  />
                </Field>
              )}
              {mayWriteback && (
                <Field
                  label="代码配方名称（config_name）"
                  required
                  hint="用于服务器直接启动训练；仅限英文字母、数字、下划线和连字符，必须与现有配方不同。"
                >
                  <input
                    maxLength={255}
                    value={configName}
                    onChange={(event) => setConfigName(event.target.value)}
                    placeholder="例如 pi05_umi_custom_v2"
                    disabled={saving || !!pendingSave || cachedEditorBlocked}
                    aria-invalid={!!configName && !/^[A-Za-z0-9_-]{1,255}$/.test(configName)}
                  />
                </Field>
              )}
              <Field label="说明">
                <textarea
                  maxLength={2000}
                  rows={2}
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  disabled={saving || !!pendingSave || cachedEditorBlocked}
                />
              </Field>
              <div className="recipe-save-diff">
                <strong>将保存的覆盖参数</strong>
                {Object.keys(overrides).length ? (
                  <ul>
                    {Object.entries(overrides).map(([path, value]) => (
                      <li key={path}>
                        <code>{path}</code>
                        <span>
                          {showValue(fieldMap.get(path))} → <b>{displayLiteral(value)}</b>
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p>保持原配方配置，保存为独立的配方版本。</p>
                )}
                {Object.keys(reviewedAssetValues).length > 0 && (
                  <div>
                    <strong>将移除的资产覆盖</strong>
                    <ul>
                      {Object.entries(reviewedAssetValues).map(([path, value]) => (
                        <li key={path}>
                          <code>{path}</code>
                          <span>{displayLiteral(value)} → 由本次训练资产绑定</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              <p className="field-hint">
                {mayWriteback
                  ? '保存会先记录平台草稿，再由机器 Agent 向来源配置文件新增配方并自动执行 Git commit；提交成功后才选中可训练的新版本。存在进行中的训练或源码版本冲突时会拒绝写入。'
                  : '保存到平台配方库并选中新版本，训练预检会验证其配置。此操作不修改训练机器上的配方源码文件。'}
              </p>
              {pendingSave && (
                <div className="inline-note amber" role="status">
                  <span>
                    平台草稿已保存：{pendingSave.saved.name} · v
                    {pendingSave.saved.latest_revision.revision}。
                    {saving
                      ? '正在等待配置写入与 Git 提交结果…'
                      : pendingSave.job?.status === 'FAILED'
                        ? pendingSave.job.commit_sha
                          ? '源码已经提交，但平台版本登记尚未完成。'
                          : '写入或提交失败，尚未选中此草稿。'
                        : '配置写入与提交结果尚未确认，可继续查询；不会重复创建草稿。'}
                    {pendingSave.job && <code>任务 {pendingSave.job.id}</code>}
                    {pendingSave.job?.commit_sha && (
                      <span>
                        config_name：<code>{pendingSave.job.config_name}</code>；Git 提交{' '}
                        <code title={pendingSave.job.commit_sha}>
                          {pendingSave.job.commit_sha.slice(0, 12)}
                        </code>
                        。请在代码配方中选择此名称重新读取。
                      </span>
                    )}
                  </span>
                </div>
              )}
              <ErrorNotice error={saveError} />
              {pendingSave &&
                !saving &&
                (pendingSave.job?.status === 'FAILED' || (!pendingSave.job && saveError)) && (
                  <p className="field-hint">
                    平台草稿已保留。也可以先处理错误，再重新读取配置并保存新版本。
                    <button
                      type="button"
                      className="text-link"
                      disabled={saving}
                      onClick={() => {
                        rememberPending(writeKey);
                        setPendingSave(undefined);
                        setShowSave(undefined);
                        setSaveError(undefined);
                        setDraft({});
                        setUseCurrentSource(true);
                        setRevision((value) => value + 1);
                        dirtyRef.current(false);
                      }}
                    >
                      保留草稿并重新读取配置
                    </button>
                  </p>
                )}
              <div className="recipe-save-actions">
                <button
                  type="button"
                  className="button outline small"
                  disabled={saving || !!pendingSave}
                  onClick={closeSave}
                >
                  取消
                </button>
                <button
                  type="button"
                  className="button primary small"
                  disabled={
                    saving ||
                    (requiresAssetReview && pendingSave?.job?.status !== 'PENDING') ||
                    pendingSave?.job?.status === 'FAILED' ||
                    (!pendingSave &&
                      (!!invalid.length ||
                        (showSave === 'new' && !saveName.trim()) ||
                        (mayWriteback && !/^[A-Za-z0-9_-]{1,255}$/.test(configName))))
                  }
                  onClick={() => void save()}
                >
                  {saving ? <LoaderCircle size={14} className="spin" /> : <Check size={14} />}
                  {pendingSave
                    ? pendingSave.job
                      ? '继续查询写入与提交结果'
                      : '继续写入并提交'
                    : mayWriteback
                      ? '保存并写入 config'
                      : '保存到平台配方库'}
                </button>
              </div>
            </section>
          )}
        </>
      )}
    </section>
  );
}
