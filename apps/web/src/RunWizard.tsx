import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Circle,
  Cpu,
  Database,
  GitCommitHorizontal,
  LoaderCircle,
  Play,
  RotateCcw,
  Server,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import { ApiError, navigate, post, useApi } from './api';
import { submitTrainingRun } from './runSubmission';
import { useAuth } from './auth';
import { CodeVersion, ErrorNotice, Field, Modal } from './components';
import { inputRoles, requiredInputs, useData } from './data';
import { RecipeLibrary } from './RecipeLibrary';
import { NormStatsFields } from './NormStatsFields';
import { NormDatasetEditor } from './NormDatasetEditor';
import { fillRunRepresentationDefaults } from './ComposedDatasetProcessing';
import {
  buildNormPreviewRequest,
  inferNormActionHorizon,
  newNormDataset,
  normAdvancedRecipe,
  normDatasetErrors,
  normDatasetsForRequest,
} from './normStats';
import { DatasetBindings, DatasetBindingSummary } from './DatasetBindings';
import { newDatasetBindings } from './datasetBindingHelpers';
import { GeneratedConfigReview } from './GeneratedConfigReview';
import { PreflightChecks, sharedNormIncompatibilityDetail } from './PreflightChecks';
import { PreflightDatasetValidation } from './PreflightDatasetValidation';
import {
  DATASET_BASE_VALIDATION_REQUIRED,
  preflightValidationAsset,
} from './datasetValidationHelpers';
import { useTrainingDraft, type TrainingDraftSession } from './useTrainingDraft';
import type { ImageLayoutEdits } from './imageLayout';
import { TrainingTemplateLibrary } from './TrainingTemplateLibrary';
import { retryTrainingSnapshot, sameTrainingConfiguration } from './runRetry';
import { TrainingGpuSelector } from './TrainingGpuSelector';
import { TrainingNormalizationField } from './TrainingNormalizationField';
import {
  DEFAULT_TRAINING_NORMALIZATION_METHOD,
  trainingNormalizationLabel,
  withTrainingNormalizationMethod,
} from './trainingNormalization';
import { requestedGpuError, requestedGpuSummary } from './trainingGpuSelection';
import { trainingGpuPolicySummary } from './trainingGpuPolicy';
import {
  applyTrainingTemplate,
  TRAINING_TEMPLATE_PROCESSING_MARKER,
  type TrainingTemplateRecord,
} from './trainingTemplate';
import type { RecipeEditorDraft, TrainingDraftSnapshot } from './trainingDraft';
import {
  automaticConfigErrors,
  configurationConfirmed,
  DEFAULT_AUTO_CONFIG,
  inferDatasetActionHorizon,
} from './trainingConfiguration';
import {
  actionHorizonDraftParameters,
  defaultActionHorizon,
  restoreActionHorizonMode,
} from './actionHorizonDefaults';
import {
  editTrainingParameter,
  optionalTrainingParameter,
  trainingParameterDefaults,
  trainingParameterErrors,
  trainingParameterGroups,
  trainingParameterHint,
} from './trainingParameters';
import {
  initializeVisibleTrainingParameters,
  recipeNumericParameterDefaults,
  recipeParameterCompatibility,
  unsupportedTrainingParameters,
  type TrainingParameterCompatibility,
} from './trainingParameterInitialization';
import {
  TRAINING_RESOURCE_FIELDS,
  applyResourceRecommendations,
  automaticResourceRecommendation,
  isTrainingResourceField,
  recommendTrainingResources,
  recommendNormResources,
  normResourceRecommendationHint,
  normResourceRecommendationTracking,
  resourceRecommendationTracking,
  setResourceRecommendationMode,
  trainingResourceRecommendationHint,
} from './trainingResourceRecommendations';
import './training-wizard.css';
import { PathAutocomplete, validTrainingEntrypoint } from './PathAutocomplete';
import { recapRecipeConfiguration, stageParameterLabels, trainingStageLabels } from './recap';
import type {
  ParameterSpec,
  Preview,
  RecipeDatasetSlot,
  RecipeDetail,
  RecipeRevision,
  Run,
  RunRequest,
  Template,
  TrainingStage,
} from './types';

function preflightHint(error?: Error) {
  if (error instanceof ApiError) {
    if (error.code === 'PROFILE_NOT_VALIDATED')
      return '当前服务仍使用旧的配方验收限制，请更新中台和这台机器的 Agent 后重新预检。';
    if (error.code === 'PREFLIGHT_EXPIRED') return '本次结果已过期，请重新预检。';
    if (['NODE_OFFLINE', 'NODE_UNAVAILABLE', 'NODE_NOT_ONLINE'].includes(error.code))
      return '请确认所选机器的 Agent 已在线，再重新预检。';
  }
  return error ? '请根据错误提示处理后重试。' : undefined;
}

function specs(template?: Pick<Template, 'parameters'>): Record<string, ParameterSpec> {
  if (Array.isArray(template?.parameters))
    return Object.fromEntries(
      template.parameters.filter((value) => value.name).map((value) => [value.name!, value]),
    );
  return Object.fromEntries(
    Object.entries(template?.parameters || {}).map(([key, value]) => [
      key,
      typeof value === 'number' ? { default: value, type: 'number' } : value,
    ]),
  );
}
interface RunWizardProps {
  close: () => void;
  initialNode?: string;
  initialTemplate?: string;
  sourceRun?: Run;
}

export default function RunWizard(props: RunWizardProps) {
  const { can, user } = useAuth();
  if (!can('training.write'))
    return (
      <Modal title="权限不足" close={props.close}>
        创建训练需要操作员或管理员权限。
      </Modal>
    );
  if (!user?.id)
    return (
      <Modal title="新建训练" close={props.close}>
        正在读取当前用户身份…
      </Modal>
    );
  return (
    <TrainingDraftWorkspace
      key={JSON.stringify([user.id, props.initialTemplate, props.sourceRun?.id])}
      {...props}
      userId={user.id}
    />
  );
}

function TrainingDraftWorkspace(props: RunWizardProps & { userId: string }) {
  const [selection, setSelection] = useState(() => ({
    templateId: props.sourceRun?.template_id || props.initialTemplate || 'openpi-sft',
    sourceRun: props.sourceRun,
    personalTemplate: undefined as TrainingTemplateRecord | undefined,
  }));
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  return (
    <TrainingDraftWizard
      key={selection.templateId}
      {...props}
      initialTemplate={selection.templateId}
      sourceRun={selection.sourceRun}
      personalTemplate={selection.personalTemplate}
      selectTemplate={(templateId, personalTemplate) => {
        if (mounted.current) setSelection({ templateId, sourceRun: undefined, personalTemplate });
      }}
    />
  );
}

function TrainingDraftWizard(
  props: RunWizardProps & {
    userId: string;
    initialTemplate: string;
    personalTemplate?: TrainingTemplateRecord;
    selectTemplate: (templateId: string, personalTemplate?: TrainingTemplateRecord) => void;
  },
) {
  const data = useData();
  const draft = useTrainingDraft(props.userId, props.initialTemplate);
  const [templateNotice, setTemplateNotice] = useState<{ name: string; issues: string[] }>();
  const consumedSourceRun = useRef<string | undefined>(undefined);
  const sourceImportInFlight = useRef(false);
  const [sourceImport, setSourceImport] = useState<{
    state: 'confirm' | 'loading' | 'error';
    snapshot?: TrainingDraftSnapshot;
    previous?: TrainingDraftSnapshot;
    error?: Error;
  }>();
  const [sourceNotice, setSourceNotice] = useState<string>();
  const consumedPersonalTemplate = useRef(false);
  const [personalTemplateAction, setPersonalTemplateAction] = useState<{
    state: 'loading' | 'error';
    record: TrainingTemplateRecord;
    error?: Error;
  }>();
  const savedRevisionId = draft.initialSnapshot?.request.recipe_revision_id;
  const resources = [data.nodes, data.profiles, data.assets, data.templates];
  const resourceError = resources.find((resource) => resource.error)?.error;
  const ready = resources.every((resource) => resource.data !== undefined);
  const loadPersonalTemplate = useCallback(
    async (record: TrainingTemplateRecord) => {
      setPersonalTemplateAction({ state: 'loading', record });
      try {
        const applied = applyTrainingTemplate(record, {
          assets: data.assets.data || [],
          profiles: data.profiles.data || [],
          nodes: data.nodes.data || [],
        });
        applied.snapshot.request.name = `${record.name} 训练`;
        await draft.replace(applied.snapshot);
        setTemplateNotice({ name: record.name, issues: applied.issues });
        setPersonalTemplateAction(undefined);
      } catch (cause) {
        setPersonalTemplateAction({
          state: 'error',
          record,
          error: cause instanceof Error ? cause : new Error(String(cause)),
        });
        throw cause;
      }
    },
    [draft.replace, data.assets.data, data.profiles.data, data.nodes.data],
  );
  useEffect(() => {
    if (!props.personalTemplate || !draft.loaded || !ready || consumedPersonalTemplate.current)
      return;
    consumedPersonalTemplate.current = true;
    void loadPersonalTemplate(props.personalTemplate).catch(() => {});
  }, [props.personalTemplate, draft.loaded, ready, loadPersonalTemplate]);
  const loadSource = useCallback(
    async (snapshot: TrainingDraftSnapshot) => {
      if (sourceImportInFlight.current) return;
      sourceImportInFlight.current = true;
      setSourceImport({ state: 'loading', snapshot });
      try {
        await draft.replace(snapshot);
        setSourceNotice(props.sourceRun?.name);
        setSourceImport(undefined);
      } catch (cause) {
        setSourceImport({
          state: 'error',
          snapshot,
          error: cause instanceof Error ? cause : new Error(String(cause)),
        });
      } finally {
        sourceImportInFlight.current = false;
      }
    },
    [draft.replace, props.sourceRun?.name],
  );
  useEffect(() => {
    const source = props.sourceRun;
    if (!source || !draft.loaded || !ready || consumedSourceRun.current === source.id) return;
    consumedSourceRun.current = source.id;
    try {
      const snapshot = retryTrainingSnapshot(source);
      const previous = draft.snapshotRef.current || draft.initialSnapshot;
      if (previous && !sameTrainingConfiguration(previous, snapshot))
        setSourceImport({ state: 'confirm', snapshot, previous });
      else void loadSource(snapshot);
    } catch (cause) {
      setSourceImport({
        state: 'error',
        error: cause instanceof Error ? cause : new Error(String(cause)),
      });
    }
  }, [props.sourceRun?.id, draft.loaded, ready, draft.initialSnapshot, loadSource]);
  if (!draft.loaded || !ready)
    return (
      <Modal title="新建训练" close={props.close}>
        {resourceError ? (
          <ErrorNotice error={resourceError} retry={data.refresh} />
        ) : (
          <p className="training-draft-loading">
            <LoaderCircle size={16} className="spin" />
            正在恢复训练草稿与资源…
          </p>
        )}
      </Modal>
    );
  if (props.sourceRun && (consumedSourceRun.current !== props.sourceRun.id || sourceImport)) {
    const importing = sourceImport?.state === 'loading' || !sourceImport;
    return (
      <Modal
        title={sourceImport?.state === 'confirm' ? '确认载入原训练配置' : '载入原训练配置'}
        subtitle={`来源：${props.sourceRun.name}`}
        close={() => {
          if (!importing) props.close();
        }}
        footer={
          sourceImport?.state === 'confirm' ? (
            <>
              <button type="button" className="button outline" onClick={props.close}>
                取消
              </button>
              <button
                type="button"
                className="button primary"
                id="load-source-run-draft"
                onClick={() => {
                  if (sourceImport.snapshot) void loadSource(sourceImport.snapshot);
                }}
              >
                载入并替换草稿
              </button>
            </>
          ) : sourceImport?.state === 'error' ? (
            <button type="button" className="button outline" onClick={props.close}>
              关闭，保留原草稿
            </button>
          ) : undefined
        }
      >
        {importing ? (
          <p className="training-draft-loading">
            <LoaderCircle size={16} className="spin" />
            正在载入配置并保存到当前训练模板的草稿…
          </p>
        ) : sourceImport?.state === 'error' ? (
          <ErrorNotice
            title="原配置未载入"
            error={sourceImport.error}
            hint="本页仍保留原草稿，未自动重试覆盖。请核对服务端草稿或先处理版本冲突。"
          />
        ) : (
          <>
            <p>当前训练模板已有未完成的草稿。载入下面的配置会替换它，请确认后继续。</p>
            <dl className="detail-list">
              <div>
                <dt>将被替换的草稿</dt>
                <dd>{sourceImport.previous?.request.name || '未命名训练'}</dd>
              </div>
              <div>
                <dt>即将载入</dt>
                <dd>{sourceImport.snapshot?.request.name}</dd>
              </div>
              <div>
                <dt>计算机器</dt>
                <dd>
                  {data.nodes.data?.find(
                    (node) => node.id === sourceImport.snapshot?.request.node_id,
                  )?.name || sourceImport.snapshot?.request.node_id}
                </dd>
              </div>
              <div>
                <dt>初始模型</dt>
                <dd>
                  {data.assets.data?.find(
                    (asset) => asset.id === sourceImport.snapshot?.request.inputs.model,
                  )?.name ||
                    sourceImport.snapshot?.request.inputs.model ||
                    '无'}
                </dd>
              </div>
              <div>
                <dt>数据集与资源</dt>
                <dd>
                  {sourceImport.snapshot?.request.norm_datasets?.length ||
                    sourceImport.snapshot?.request.dataset_bindings?.length ||
                    (sourceImport.snapshot?.request.inputs.dataset ? 1 : 0)}{' '}
                  个数据集 · {sourceImport.snapshot?.request.gpu_count} 张 GPU
                </dd>
              </div>
            </dl>
            <p className="field-hint">
              载入后从原初始模型重新训练，需重新预检、核对 config 并确认启动。原训练记录保持不变。
            </p>
          </>
        )}
      </Modal>
    );
  }
  if ((props.personalTemplate && !consumedPersonalTemplate.current) || personalTemplateAction)
    return (
      <Modal
        title={personalTemplateAction?.record.name || '载入模板'}
        close={props.close}
        footer={
          personalTemplateAction?.state === 'error' ? (
            <>
              <button type="button" className="button outline" onClick={props.close}>
                关闭
              </button>
              <button
                type="button"
                className="button primary"
                onClick={() =>
                  void loadPersonalTemplate(personalTemplateAction.record).catch(() => {})
                }
              >
                重试载入模板
              </button>
            </>
          ) : undefined
        }
      >
        {personalTemplateAction?.state === 'error' ? (
          <ErrorNotice
            title="模板未载入"
            error={personalTemplateAction.error}
            hint="该训练方式的草稿已保留。关闭后重新打开，可先处理草稿同步问题或版本冲突。"
          />
        ) : (
          <p className="training-draft-loading">
            <LoaderCircle size={16} className="spin" />
            正在载入所选模板与数据集配置…
          </p>
        )}
      </Modal>
    );
  return (
    <RestoredRunWizard
      key={`${draft.generation}:${savedRevisionId || ''}`}
      {...props}
      draftSession={draft}
      initialDraft={draft.initialSnapshot}
      userId={props.userId}
      templateNotice={templateNotice}
      sourceRunNotice={sourceNotice}
      onSelectTemplate={async (templateId) => {
        if (!templateId || templateId === props.initialTemplate) return;
        await draft.flush();
        props.selectTemplate(templateId);
      }}
      onApplyTemplate={async (record) => {
        const applied = applyTrainingTemplate(record, {
          nodes: data.nodes.data || [],
          profiles: data.profiles.data || [],
          assets: data.assets.data || [],
        });
        if (applied.snapshot.request.template_id === props.initialTemplate)
          await loadPersonalTemplate(record);
        else {
          await draft.flush();
          props.selectTemplate(applied.snapshot.request.template_id, record);
        }
      }}
    />
  );
}

// Each restored configuration owns its recipe request. A previous revision's
// result or error must not initialise a newly applied template's recipe editor.
function RestoredRunWizard(props: Parameters<typeof RunWizardContent>[0]) {
  const revisionId = props.initialDraft?.request.recipe_revision_id;
  const revision = useApi<RecipeRevision>(
    revisionId ? `/recipe-revisions/${encodeURIComponent(revisionId)}` : null,
    0,
  );
  if (revisionId && !revision.data && !revision.error)
    return (
      <Modal title="新建训练" close={props.close}>
        <p className="training-draft-loading">
          <LoaderCircle size={16} className="spin" />
          正在恢复配方版本…
        </p>
      </Modal>
    );
  return (
    <RunWizardContent
      {...props}
      initialSavedRevision={revision.data?.id === revisionId ? revision.data : undefined}
    />
  );
}

function RunWizardContent({
  close,
  initialNode,
  initialTemplate,
  initialDraft,
  initialSavedRevision,
  draftSession,
  userId,
  templateNotice,
  onApplyTemplate,
  onSelectTemplate,
  sourceRunNotice,
}: RunWizardProps & {
  initialDraft?: TrainingDraftSnapshot;
  initialSavedRevision?: RecipeRevision;
  draftSession: TrainingDraftSession;
  userId: string;
  templateNotice?: { name: string; issues: string[] };
  onApplyTemplate: (record: TrainingTemplateRecord) => Promise<void>;
  onSelectTemplate: (templateId: string) => Promise<void>;
  sourceRunNotice?: string;
}) {
  const { nodes, assets, profiles, templates, refresh } = useData();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const initialAdvancedRecipe = initialDraft
    ? normAdvancedRecipe(initialDraft.request, initialDraft.advancedRecipe)
    : false;
  const [step, setStep] = useState(initialDraft?.step || 0);
  const [advancedRecipe, setAdvancedRecipe] = useState(initialAdvancedRecipe);
  const [confirmedConfigSha256, setConfirmedConfigSha256] = useState<string>();
  const [request, setRequest] = useState<RunRequest>(() =>
    fillRunRepresentationDefaults(
      (initialDraft?.request
        ? initialDraft.request.template_id === 'openpi-norm'
          ? {
              ...initialDraft.request,
              inputs: {},
              auto_config: initialAdvancedRecipe
                ? undefined
                : {
                    ...(initialDraft.request.auto_config || DEFAULT_AUTO_CONFIG),
                    finetune_mode: 'full',
                  },
              norm_datasets: normDatasetsForRequest(initialDraft.request).map((row) =>
                !initialAdvancedRecipe && !row.processor
                  ? {
                      ...newNormDataset(
                        true,
                        assets.data?.find((asset) => asset.id === row.dataset_asset_id),
                      ),
                      dataset_asset_id: row.dataset_asset_id,
                    }
                  : row,
              ),
            }
          : initialDraft.request
        : undefined) || {
        name: '',
        node_id: initialNode || '',
        profile_id: '',
        template_id: initialTemplate || 'openpi-sft',
        inputs: {},
        ...(initialTemplate === 'openpi-norm' ? { norm_datasets: [newNormDataset(true)] } : {}),
        parameters: {},
        gpu_count: initialTemplate === 'openpi-norm' ? 0 : 1,
        auto_config: initialTemplate === 'smoke' ? undefined : { ...DEFAULT_AUTO_CONFIG },
        ...(['smoke', 'openpi-norm'].includes(initialTemplate || '')
          ? {}
          : { normalization_method: DEFAULT_TRAINING_NORMALIZATION_METHOD }),
      },
      assets.data || [],
      new Set(initialDraft?.editedParameters),
    ),
  );
  const [previewId, setPreviewId] = useState('');
  const [createdPreview, setCreatedPreview] = useState<Preview>();
  const [busy, setBusy] = useState(false);
  const [datasetValidationPending, setDatasetValidationPending] = useState(false);
  const [error, setError] = useState<Error>();
  const idempotencyKey = useRef(crypto.randomUUID());
  const recipeSelection = useRef<string | undefined>(undefined);
  const editedParameters = useRef(
    (initialDraft?.request.template_id ?? initialTemplate) === 'openpi-norm'
      ? normResourceRecommendationTracking(initialDraft?.editedParameters)
      : new Set<string>(initialDraft?.editedParameters),
  );
  const defaultedParameterKeys = useRef(new Set<string>());
  const defaultFillNeedsSave = useRef(
    !!initialDraft &&
      (request.norm_datasets?.some(
        (row, index) => row !== initialDraft.request.norm_datasets?.[index],
      ) ||
        request.dataset_bindings?.some(
          (row, index) => row !== initialDraft.request.dataset_bindings?.[index],
        )),
  );
  const initializedParameterScope = useRef<string | undefined>(undefined);
  const currentStep = useRef(step);
  currentStep.current = step;
  const [resolvedParameterSource, setResolvedParameterSource] = useState<{
    scope: string;
    defaults: Record<string, number>;
    compatibility: TrainingParameterCompatibility;
  }>();
  const resolvedParameterSourceRef = useRef(resolvedParameterSource);
  const [imageLayoutEditVersion, setImageLayoutEditVersion] = useState(0);
  const imageLayoutEdits: ImageLayoutEdits = {
    has: (marker) => editedParameters.current.has(marker),
    set: (marker, manual) => {
      if (editedParameters.current.has(marker) === manual) return;
      if (manual) editedParameters.current.add(marker);
      else editedParameters.current.delete(marker);
      setImageLayoutEditVersion((value) => value + 1);
      draftSession.markEdited();
    },
  };
  const [actionHorizonMode, setActionHorizonMode] = useState(() =>
    restoreActionHorizonMode(initialDraft),
  );
  const editedStageParameters = useRef(new Set<string>(initialDraft?.editedStageParameters));
  const entrypointEdited = useRef(initialDraft?.entrypointEdited || false);
  const [selectedRecipeRevision, setSelectedRecipeRevision] = useState<RecipeRevision | undefined>(
    initialSavedRevision,
  );
  const [recipeDirty, setRecipeDirty] = useState(
    !!Object.keys(initialDraft?.recipeEditorDraft?.fields || {}).length,
  );
  const [recipeReading, setRecipeReading] = useState(false);
  const [recipeDefaults, setRecipeDefaults] = useState<Record<string, number>>(
    initialDraft?.recipeDefaults || {},
  );
  const [recipeDatasets, setRecipeDatasets] = useState<RecipeDatasetSlot[]>(
    initialDraft?.recipeDatasets || [],
  );
  const [processorValidation, setProcessorValidation] = useState<{
    scope: string;
    valid: boolean;
  }>();
  const [stageRecipeDefaults, setStageRecipeDefaults] = useState<Record<string, number>>(
    initialDraft?.stageRecipeDefaults || {},
  );
  const [stageRecipeInfo, setStageRecipeInfo] = useState<
    ReturnType<typeof recapRecipeConfiguration> | undefined
  >(initialDraft?.stageRecipeInfo);
  const [recipeEditorDraft, setRecipeEditorDraft] = useState<RecipeEditorDraft | undefined>(
    initialDraft?.recipeEditorDraft,
  );
  const [draftActionError, setDraftActionError] = useState<Error>();
  const [confirmClearDraft, setConfirmClearDraft] = useState(false);
  const [draftActionBusy, setDraftActionBusy] = useState(false);
  const [unsavedClose, setUnsavedClose] = useState(false);
  const initialDefaultsScope = useRef(
    initialDraft ? `${initialDraft.request.template_id}:${initialAdvancedRecipe}` : undefined,
  );
  const initialSelectionInstalled = useRef(false);
  const initialProfile = useRef(initialDraft?.request.profile_id);
  const datasetDefaultsScope = useRef('');
  const requestRef = useRef(request);
  requestRef.current = request;
  const previewResource = useApi<Preview>(previewId ? `/run-previews/${previewId}` : null, 1800);
  const preview = previewResource.data?.id === previewId ? previewResource.data : createdPreview;
  const sharedNormDetail = sharedNormIncompatibilityDetail(preview?.error?.message);
  const datasetValidationRequired =
    preview?.status === 'FAILED' && preview.error?.code === DATASET_BASE_VALIDATION_REQUIRED;
  const validationRequest = datasetValidationRequired ? preview.request : undefined;
  const validationAsset = datasetValidationRequired
    ? preflightValidationAsset(preview, assets.data || [])
    : undefined;
  const previewReadError = previewId ? previewResource.error : undefined;
  const displayedError = error || (step === 2 ? previewReadError : undefined);
  const requestFailed = !previewId && !!error && !busy;
  const previewExpired = preview?.status === 'EXPIRED';
  const preflightFailed = requestFailed || preview?.status === 'FAILED' || previewExpired;
  const progressUnavailable = !!previewReadError && preview?.status === 'PENDING';
  const automaticPreflight = preview?.code_snapshot?.automatic_preflight;
  const frozenRecipe =
    preview?.request?.recipe_definition || preview?.code_snapshot?.recipe_definition;
  const template = templates.data?.find((item) => item.id === request.template_id);
  const isRecap = template?.id === 'recap';
  const isNorm = template?.id === 'openpi-norm';
  const hasLocalTraining = template?.id === 'openpi-sft' || isRecap;
  const requiresConfigConfirmation = hasLocalTraining || isNorm;
  const normRows = normDatasetsForRequest(request);
  const automaticMode = hasLocalTraining && !advancedRecipe;
  const automaticNorm = isNorm && !advancedRecipe;
  const automaticConfiguration = automaticMode || automaticNorm;
  const generatedConfig = automaticPreflight?.plan.generated_config;
  const configConfirmed =
    !requiresConfigConfirmation || configurationConfirmed(generatedConfig, confirmedConfigSha256);
  useEffect(() => {
    setConfirmedConfigSha256(undefined);
  }, [request, previewId, generatedConfig?.sha256]);
  const trainingStage = isRecap
    ? template?.stages?.find((stage) => stage.id === request.training_stage)
    : undefined;
  const node = nodes.data?.find((item) => item.id === request.node_id);
  const resourceRecommendations = isNorm
    ? recommendNormResources(node?.metrics)
    : recommendTrainingResources(request.gpu_count, node?.metrics);
  const parameterScope = JSON.stringify([
    request.node_id,
    request.profile_id,
    request.template_id,
    advancedRecipe,
    request.recipe,
    request.recipe_revision_id,
    request.training_stage,
    request.auto_config?.finetune_mode,
  ]);
  const parameterCompatibility =
    advancedRecipe && resolvedParameterSource?.scope === parameterScope
      ? resolvedParameterSource.compatibility
      : {};
  const unusedParameters = unsupportedTrainingParameters(parameterCompatibility);
  const emaDefaultDisabled = automaticMode
    ? request.auto_config?.finetune_mode === 'lora'
    : parameterCompatibility.emaDisabled === true;
  const parameterContextRef = useRef({ parameterScope, metrics: node?.metrics });
  parameterContextRef.current = { parameterScope, metrics: node?.metrics };
  const compatibleProfiles = useMemo(
    () =>
      profiles.data?.filter(
        (profile) =>
          profile.node_id === request.node_id &&
          profile.adapter === (template?.runtime_adapter || request.template_id) &&
          (!profile.execution_mode || profile.execution_mode === 'native'),
      ) || [],
    [profiles.data, request.node_id, request.template_id, template?.runtime_adapter],
  );
  const parameterSpecs = specs(template);
  const parameterSpecsRef = useRef(parameterSpecs);
  parameterSpecsRef.current = parameterSpecs;
  const parameterGroups = trainingParameterGroups(parameterSpecs);
  const parameterIssues = [
    ...trainingParameterErrors(
      request.parameters,
      parameterSpecs,
      request.gpu_count,
      advancedRecipe ? recipeDefaults : {},
    ),
    ...unusedParameters
      .filter((key) => Object.hasOwn(request.parameters, key))
      .map((key) => ({
        key,
        message: `此配方不使用${parameterSpecs[key]?.label || key}，请清除此项后重新预检。`,
      })),
  ];
  const parameterBlocker =
    step === 1 ? parameterIssues.map((issue) => issue.message).join(' ') : '';
  const gpuSelectionError = requestedGpuError(
    request,
    node,
    template?.min_gpus ?? (template?.id === 'smoke' ? 0 : 1),
    userId,
  );
  const stageParameterSpecs = specs(trainingStage);
  const inputRequirements = requiredInputs(trainingStage || template);
  const selectedProfile = compatibleProfiles.find((profile) => profile.id === request.profile_id);
  if (initialDraft && !initialSelectionInstalled.current) {
    recipeSelection.current = `${request.node_id}\0${hasLocalTraining ? 'local' : request.template_id}\0${request.profile_id}\0${selectedProfile?.code_dir}`;
    initialSelectionInstalled.current = true;
  }
  const snapshot: TrainingDraftSnapshot = {
    request,
    step: step === 0 ? 0 : 1,
    advancedRecipe,
    editedParameters: actionHorizonDraftParameters(editedParameters.current, actionHorizonMode),
    editedStageParameters: [...editedStageParameters.current],
    entrypointEdited: entrypointEdited.current,
    recipeDatasets,
    recipeDefaults,
    stageRecipeDefaults,
    stageRecipeInfo,
    recipeEditorDraft,
  };
  draftSession.snapshotRef.current = snapshot;
  useEffect(() => {
    if (!defaultFillNeedsSave.current) return;
    defaultFillNeedsSave.current = false;
    draftSession.markEdited();
  }, [request, draftSession.markEdited]);
  useEffect(() => {
    draftSession.changed();
  }, [
    request,
    actionHorizonMode,
    imageLayoutEditVersion,
    step,
    advancedRecipe,
    recipeDatasets,
    recipeDefaults,
    stageRecipeDefaults,
    stageRecipeInfo,
    recipeEditorDraft,
    draftSession.changed,
  ]);
  const processorValidationScope = JSON.stringify([
    step,
    request.node_id,
    request.profile_id,
    selectedProfile?.code_dir,
    selectedProfile?.python,
    request.training_stage || 'sft',
    request.dataset_bindings,
    request.norm_datasets,
    request.recipe,
    request.stage_config,
    request.auto_config,
  ]);
  const processorsValid =
    processorValidation?.scope === processorValidationScope && processorValidation.valid;
  const missingProcessors = automaticConfiguration
    ? (isNorm ? normRows : request.dataset_bindings || []).flatMap((binding, index) =>
        binding.processor ? [] : [`数据集 ${index + 1}`],
      )
    : [];
  const dataProcessingBlocker =
    step === 1 && (hasLocalTraining || isNorm) && !processorsValid
      ? isNorm
        ? '请返回数据配置，核对每个数据集的统计处理方式。'
        : missingProcessors.length
          ? `${missingProcessors.slice(0, 3).join('、')}${missingProcessors.length > 3 ? `等 ${missingProcessors.length} 组数据` : ''}尚未选择数据处理器。自动组装需要为每个数据集明确配置处理方式。`
          : '数据处理配置错误，修正后才能生成 config 并预检。请检查各数据集列出的错误。'
      : undefined;
  const mixedDropoutUnresolved =
    !!stageRecipeInfo?.mixedDropout &&
    (typeof request.stage_config?.acp_dropout !== 'number' ||
      !Number.isFinite(request.stage_config.acp_dropout) ||
      request.stage_config.acp_dropout < 0 ||
      request.stage_config.acp_dropout > 1);
  const editDataProcessing = () => {
    draftSession.markEdited();
    setStep(0);
    setPreviewId('');
    setCreatedPreview(undefined);
    setError(undefined);
    setConfirmedConfigSha256(undefined);
  };
  const setProcessorsValid = useCallback(
    (valid: boolean) => {
      setProcessorValidation((previous) =>
        previous?.scope === processorValidationScope && previous.valid === valid
          ? previous
          : { scope: processorValidationScope, valid },
      );
    },
    [processorValidationScope],
  );
  const localAssets = useMemo(
    () =>
      assets.data?.filter(
        (asset) =>
          asset.node_id === request.node_id &&
          !['ARCHIVED', 'MISSING', 'INVALID', 'UNAVAILABLE'].includes(
            asset.status || asset.state || '',
          ),
      ) || [],
    [assets.data, request.node_id],
  );
  const actionHorizonInference = useMemo(
    () =>
      isNorm
        ? inferNormActionHorizon(normRows, localAssets)
        : inferDatasetActionHorizon(request.dataset_bindings || [], localAssets),
    [isNorm, request.norm_datasets, request.dataset_bindings, localAssets],
  );
  const actionHorizonErrors = automaticConfiguration
    ? [
        ...actionHorizonInference.errors,
        ...(!isNorm &&
        actionHorizonInference.recommendedHorizon !== undefined &&
        request.auto_config?.action_horizon !== actionHorizonInference.recommendedHorizon
          ? [
              `数据集已存储 ${actionHorizonInference.recommendedHorizon} 步动作，动作预测长度必须与其一致。当前处理器不会自动裁剪或补齐已有动作块。`,
            ]
          : []),
      ]
    : [];
  const actionHorizonBlocker = step === 1 ? actionHorizonErrors.join(' ') : '';
  const processingBlocker = actionHorizonBlocker || dataProcessingBlocker;
  useEffect(() => {
    if (
      !automaticConfiguration ||
      actionHorizonMode !== 'automatic' ||
      step >= 2 ||
      busy ||
      draftActionBusy
    )
      return;
    setRequest((value) => {
      if (!value.auto_config) return value;
      const inferred = isNorm
        ? inferNormActionHorizon(normDatasetsForRequest(value), localAssets)
        : inferDatasetActionHorizon(value.dataset_bindings || [], localAssets);
      const horizon = defaultActionHorizon(
        value.auto_config.action_horizon,
        actionHorizonMode,
        inferred.recommendedHorizon,
        inferred.errors,
      );
      return Object.is(horizon, value.auto_config.action_horizon)
        ? value
        : { ...value, auto_config: { ...value.auto_config, action_horizon: horizon } };
    });
  }, [
    automaticConfiguration,
    isNorm,
    actionHorizonMode,
    step,
    busy,
    draftActionBusy,
    request.norm_datasets,
    request.dataset_bindings,
    localAssets,
  ]);
  const validatedRecipes = Array.isArray(selectedProfile?.metadata?.validated_recipes)
    ? [
        ...new Set(
          selectedProfile.metadata.validated_recipes
            .filter((value): value is string => typeof value === 'string' && Boolean(value.trim()))
            .map((value) => value.trim()),
        ),
      ]
    : [];
  const minGpus = template?.min_gpus ?? (template?.id === 'smoke' ? 0 : 1);
  useEffect(() => {
    if (!template) return;
    const modeScope = `${request.template_id}:${advancedRecipe}`;
    if (initialDefaultsScope.current === modeScope) return;
    initialDefaultsScope.current = modeScope;
    setActionHorizonMode('automatic');
    const local = ['openpi-sft', 'recap'].includes(request.template_id);
    if (local && !advancedRecipe)
      TRAINING_RESOURCE_FIELDS.forEach((key) =>
        setResourceRecommendationMode(editedParameters.current, key, true),
      );
    if (isNorm)
      editedParameters.current = normResourceRecommendationTracking(editedParameters.current);
    setRequest((value) => ({
      ...value,
      parameters: local
        ? value.parameters
        : {
            ...trainingParameterDefaults(specs(template)),
            ...(isNorm && Object.hasOwn(value.parameters, 'num_workers')
              ? { num_workers: value.parameters.num_workers }
              : {}),
          },
      gpu_count: isNorm
        ? 0
        : value.requested_gpu_uuids !== undefined
          ? value.requested_gpu_uuids.length
          : template?.id === 'smoke'
            ? 0
            : Math.max(template?.min_gpus ?? 1, value.gpu_count),
      auto_config:
        (local || isNorm) && !advancedRecipe
          ? isNorm
            ? { ...(value.auto_config || DEFAULT_AUTO_CONFIG), finetune_mode: 'full' }
            : value.auto_config || { ...DEFAULT_AUTO_CONFIG }
          : undefined,
      normalization_method: local ? value.normalization_method : undefined,
      requested_gpu_uuids: isNorm ? undefined : value.requested_gpu_uuids,
      training_gpu_policy: local ? value.training_gpu_policy : undefined,
      inputs: local ? value.inputs : {},
      dataset_bindings: local ? value.dataset_bindings || newDatasetBindings([]) : undefined,
      norm_datasets: isNorm ? value.norm_datasets || [newNormDataset(!advancedRecipe)] : undefined,
      training_stage: request.template_id === 'recap' ? value.training_stage : undefined,
      stage_config: request.template_id === 'recap' ? value.stage_config : undefined,
      recipe: template?.id === 'smoke' ? 'smoke' : advancedRecipe ? value.recipe : undefined,
      recipe_revision_id: advancedRecipe ? value.recipe_revision_id : undefined,
      entrypoint: undefined,
    }));
    // Reapply defaults only when the method or assembly mode changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.template_id, advancedRecipe, template?.id]);
  useEffect(() => {
    if (!hasLocalTraining || step >= 2 || initializedParameterScope.current === parameterScope)
      return;
    initializedParameterScope.current = parameterScope;
    setRequest((value) => {
      const resolvedSource = resolvedParameterSourceRef.current;
      const source =
        advancedRecipe && resolvedSource?.scope === parameterScope ? resolvedSource : undefined;
      const initialized = initializeVisibleTrainingParameters(
        value.parameters,
        parameterSpecsRef.current,
        {
          recipeDefaults: source?.defaults,
          compatibility: source?.compatibility,
          resources: recommendTrainingResources(value.gpu_count, node?.metrics),
          lora: !advancedRecipe && value.auto_config?.finetune_mode === 'lora',
          defaultedKeys: defaultedParameterKeys.current,
          edited: editedParameters.current,
          phase: 'initial',
        },
      );
      defaultedParameterKeys.current = initialized.defaultedKeys;
      editedParameters.current = initialized.edited;
      if (initialized.parameters !== value.parameters) defaultFillNeedsSave.current = true;
      return initialized.parameters === value.parameters
        ? value
        : { ...value, parameters: initialized.parameters };
    });
    // Initialization boundaries deliberately exclude live CPU refresh and edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasLocalTraining, parameterScope, step]);
  useEffect(() => {
    // A generated preview is frozen. Resume resource following only after the
    // user returns to configuration, without changing its approval underneath.
    if ((!hasLocalTraining && !isNorm) || step >= 2 || busy || draftActionBusy) return;
    const next = applyResourceRecommendations(
      request.parameters,
      editedParameters.current,
      resourceRecommendations,
    );
    if (next === request.parameters) return;
    draftSession.markEdited();
    setRequest((value) => {
      const parameters = applyResourceRecommendations(
        value.parameters,
        editedParameters.current,
        isNorm
          ? recommendNormResources(node?.metrics)
          : recommendTrainingResources(value.gpu_count, node?.metrics),
      );
      return parameters === value.parameters ? value : { ...value, parameters };
    });
    // Include current values so a late schema/default read cannot overwrite an
    // automatic value without being corrected on the next render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    hasLocalTraining,
    isNorm,
    advancedRecipe,
    step,
    busy,
    draftActionBusy,
    request.gpu_count,
    request.parameters.fsdp_devices,
    request.parameters.num_workers,
    node?.id,
    node?.metrics?.cpu_count,
    node?.metrics?.available_cpu_count,
  ]);
  useEffect(() => {
    setRequest((value) =>
      initialProfile.current && value.profile_id === initialProfile.current
        ? value
        : compatibleProfiles.some((profile) => profile.id === value.profile_id)
          ? value
          : { ...value, profile_id: compatibleProfiles[0]?.id || '' },
    );
  }, [compatibleProfiles]);
  useEffect(() => {
    const selection = `${request.node_id}\0${hasLocalTraining ? 'local' : request.template_id}\0${request.profile_id}\0${selectedProfile?.code_dir}`;
    if (!selectedProfile) {
      if (initialProfile.current) return;
      recipeSelection.current = undefined;
      setRequest((value) =>
        value.recipe === undefined && value.entrypoint === undefined
          ? value
          : { ...value, recipe: undefined, entrypoint: undefined },
      );
      return;
    }
    if (recipeSelection.current === selection) return;
    recipeSelection.current = selection;
    editedParameters.current = resourceRecommendationTracking(editedParameters.current);
    editedStageParameters.current.clear();
    entrypointEdited.current = false;
    datasetDefaultsScope.current = '';
    setRecipeDatasets([]);
    setRecipeDefaults({});
    setStageRecipeDefaults({});
    setStageRecipeInfo(undefined);
    setRequest((value) => ({
      ...value,
      recipe:
        request.template_id === 'smoke'
          ? 'smoke'
          : advancedRecipe
            ? selectedProfile.recipe?.trim() ||
              (validatedRecipes.length === 1 ? validatedRecipes[0] : '')
            : undefined,
      recipe_revision_id: undefined,
      entrypoint: undefined,
      inputs: {},
      dataset_bindings: hasLocalTraining ? newDatasetBindings([]) : undefined,
      norm_datasets: isNorm ? [newNormDataset(!advancedRecipe)] : undefined,
      parameters: value.parameters,
    }));
    setSelectedRecipeRevision(undefined);
    setRecipeDirty(false);
    // Only selection changes choose a default; background refreshes preserve user input.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.node_id, request.template_id, request.profile_id, selectedProfile]);
  useEffect(() => {
    if (!trainingStage) return;
    setRequest((value) =>
      value.stage_config !== undefined
        ? value
        : {
            ...value,
            stage_config: trainingParameterDefaults(specs(trainingStage)),
          },
    );
  }, [trainingStage]);
  useEffect(() => {
    if (!hasLocalTraining || automaticMode || entrypointEdited.current) return;
    const multiple =
      (request.dataset_bindings?.length || 0) > 1 || recipeDatasets.some((slot) => slot.multiple);
    const entrypoint = isRecap
      ? multiple
        ? trainingStage?.default_multi_entrypoint
        : trainingStage?.default_entrypoint
      : multiple
        ? 'scripts/train_multi_dataset.py'
        : 'scripts/train.py';
    if (entrypoint)
      setRequest((value) => (value.entrypoint === entrypoint ? value : { ...value, entrypoint }));
  }, [
    hasLocalTraining,
    automaticMode,
    isRecap,
    trainingStage,
    recipeDatasets,
    request.dataset_bindings?.length,
  ]);
  const update = <K extends keyof RunRequest>(key: K, value: RunRequest[K]) => {
    draftSession.markEdited();
    if (key === 'node_id' || key === 'profile_id' || key === 'template_id')
      initialProfile.current = undefined;
    setConfirmedConfigSha256(undefined);
    setPreviewId('');
    setCreatedPreview(undefined);
    setRequest((previous) => {
      const next = {
        ...previous,
        [key]: value,
        ...(key === 'node_id' && value !== previous.node_id
          ? {
              requested_gpu_uuids: undefined,
              training_gpu_policy: undefined,
              gpu_count: Math.max(minGpus, previous.gpu_count),
            }
          : {}),
      };
      return key === 'normalization_method'
        ? withTrainingNormalizationMethod(next, value as RunRequest['normalization_method'])
        : next;
    });
  };
  const updateGpuSelection = (
    selection: Pick<RunRequest, 'gpu_count' | 'requested_gpu_uuids' | 'training_gpu_policy'>,
  ) => {
    draftSession.markEdited();
    setConfirmedConfigSha256(undefined);
    setPreviewId('');
    setCreatedPreview(undefined);
    setRequest((previous) => ({ ...previous, ...selection }));
  };
  const toggleRecipeMode = (enabled: boolean) => {
    draftSession.markEdited();
    setAdvancedRecipe(enabled);
    setActionHorizonMode('automatic');
    entrypointEdited.current = false;
    datasetDefaultsScope.current = '';
    setRecipeDirty(false);
    setRecipeReading(false);
    setSelectedRecipeRevision(undefined);
    setRecipeDatasets([]);
    setRecipeDefaults({});
    setStageRecipeDefaults({});
    setStageRecipeInfo(undefined);
    if (isNorm) editedParameters.current = resourceRecommendationTracking(editedParameters.current);
    else editedParameters.current.clear();
    editedStageParameters.current.clear();
    setRequest((value) => ({
      ...value,
      recipe: enabled ? selectedProfile?.recipe?.trim() || '' : undefined,
      recipe_revision_id: undefined,
      entrypoint: undefined,
      auto_config: enabled ? undefined : { ...DEFAULT_AUTO_CONFIG },
      ...(isNorm
        ? {
            norm_datasets: normDatasetsForRequest(value).map((row) =>
              !enabled && !row.processor
                ? {
                    ...newNormDataset(
                      true,
                      localAssets.find((asset) => asset.id === row.dataset_asset_id),
                    ),
                    dataset_asset_id: row.dataset_asset_id,
                  }
                : row,
            ),
          }
        : {}),
    }));
  };
  const applyRecipeDefaults = useCallback(
    (defaults: Record<string, number>, recipe: string, revisionId?: string) => {
      const current = requestRef.current;
      if (current.recipe !== recipe || current.recipe_revision_id !== revisionId) return;
      setRecipeDefaults(defaults);
    },
    [],
  );
  const applyStageRecipeDefaults = useCallback(
    (detail: RecipeDetail, recipe: string, revisionId?: string) => {
      const current = requestRef.current;
      if (current.recipe !== recipe || current.recipe_revision_id !== revisionId) return;
      const compatibility = recipeParameterCompatibility(detail);
      const defaults = recipeNumericParameterDefaults(detail);
      const context = parameterContextRef.current;
      const source = { scope: context.parameterScope, defaults, compatibility };
      resolvedParameterSourceRef.current = source;
      setResolvedParameterSource(source);
      if (currentStep.current < 2)
        setRequest((value) => {
          const initialized = initializeVisibleTrainingParameters(
            value.parameters,
            parameterSpecsRef.current,
            {
              recipeDefaults: defaults,
              compatibility,
              resources: recommendTrainingResources(value.gpu_count, context.metrics),
              defaultedKeys: defaultedParameterKeys.current,
              edited: editedParameters.current,
              phase: 'resolved',
            },
          );
          defaultedParameterKeys.current = initialized.defaultedKeys;
          editedParameters.current = initialized.edited;
          if (initialized.parameters !== value.parameters) defaultFillNeedsSave.current = true;
          return initialized.parameters === value.parameters
            ? value
            : { ...value, parameters: initialized.parameters };
        });
      if (
        current.template_id !== 'recap' ||
        current.training_stage !== request.training_stage ||
        current.recipe !== recipe ||
        current.recipe_revision_id !== revisionId ||
        !current.training_stage
      )
        return;
      const configuration = recapRecipeConfiguration(detail, current.training_stage);
      setStageRecipeInfo(configuration);
      setStageRecipeDefaults(configuration.values);
      setRequest((value) => ({
        ...value,
        stage_config: {
          ...value.stage_config,
          ...Object.fromEntries(
            Object.entries(configuration.values).filter(
              ([key]) => !editedStageParameters.current.has(key),
            ),
          ),
          ...(configuration.mixedDropout && !editedStageParameters.current.has('acp_dropout')
            ? { acp_dropout: NaN }
            : {}),
        },
      }));
    },
    [request.training_stage],
  );
  const applyDatasetDefaults = useCallback(
    (slots: RecipeDatasetSlot[], recipe: string, revisionId?: string) => {
      const current = requestRef.current;
      if (current.recipe !== recipe || current.recipe_revision_id !== revisionId) return;
      const scope = JSON.stringify([
        current.node_id,
        current.profile_id,
        current.training_stage,
        recipe,
        revisionId,
      ]);
      if (datasetDefaultsScope.current === scope) return;
      datasetDefaultsScope.current = scope;
      setRecipeDatasets(slots);
      setRequest((value) => {
        if (value.recipe !== recipe || value.recipe_revision_id !== revisionId) return value;
        return {
          ...value,
          inputs: value.inputs.model ? { model: value.inputs.model } : {},
          dataset_bindings: value.dataset_bindings?.some((binding) => binding.dataset_asset_id)
            ? value.dataset_bindings
            : newDatasetBindings(slots),
        };
      });
    },
    [],
  );
  const selectRecipe = (recipe: string, revision?: RecipeRevision) => {
    draftSession.markEdited();
    setRecipeDirty(false);
    setSelectedRecipeRevision(revision);
    if (recipe !== request.recipe || revision?.id !== request.recipe_revision_id) {
      editedParameters.current.clear();
      datasetDefaultsScope.current = '';
      setRecipeDatasets([]);
      setRecipeDefaults({});
      setStageRecipeDefaults({});
      setStageRecipeInfo(undefined);
      editedStageParameters.current.clear();
      setRequest((value) => ({
        ...value,
        recipe,
        recipe_revision_id: revision?.id,
        parameters: value.parameters,
        dataset_bindings: value.dataset_bindings,
        stage_config:
          request.template_id === 'recap'
            ? trainingParameterDefaults(stageParameterSpecs)
            : undefined,
      }));
    }
  };
  const selectStage = (stage: TrainingStage | undefined) => {
    draftSession.markEdited();
    editedParameters.current = resourceRecommendationTracking(editedParameters.current);
    editedStageParameters.current.clear();
    entrypointEdited.current = false;
    datasetDefaultsScope.current = '';
    setRecipeDefaults({});
    setRecipeDatasets([]);
    setStageRecipeDefaults({});
    setStageRecipeInfo(undefined);
    setSelectedRecipeRevision(undefined);
    setRecipeDirty(false);
    setRecipeReading(false);
    setPreviewId('');
    setCreatedPreview(undefined);
    setError(undefined);
    setRequest((value) => ({
      ...value,
      training_stage: stage,
      stage_config: undefined,
      recipe: advancedRecipe ? '' : undefined,
      recipe_revision_id: undefined,
      entrypoint: undefined,
      auto_config: value.auto_config
        ? {
            ...value.auto_config,
            finetune_mode: stage === 'value' ? 'full' : value.auto_config.finetune_mode,
          }
        : undefined,
      parameters: value.parameters,
    }));
  };
  const beginPreview = async () => {
    setBusy(true);
    setError(undefined);
    setPreviewId('');
    setCreatedPreview(undefined);
    setConfirmedConfigSha256(undefined);
    idempotencyKey.current = crypto.randomUUID();
    try {
      const result = await post<Preview>(
        '/run-previews',
        isNorm
          ? buildNormPreviewRequest(request)
          : {
              name: request.name,
              node_id: request.node_id,
              profile_id: request.profile_id,
              template_id: request.template_id,
              inputs: request.inputs,
              dataset_bindings: hasLocalTraining ? request.dataset_bindings : undefined,
              training_stage: isRecap ? request.training_stage : undefined,
              stage_config: isRecap ? request.stage_config : undefined,
              parameters: request.parameters,
              gpu_count: request.gpu_count,
              requested_gpu_uuids: request.requested_gpu_uuids,
              training_gpu_policy: hasLocalTraining ? request.training_gpu_policy : undefined,
              auto_config: automaticMode ? request.auto_config : undefined,
              ...(hasLocalTraining && request.normalization_method
                ? { normalization_method: request.normalization_method }
                : {}),
              recipe_revision_id: advancedRecipe ? request.recipe_revision_id : undefined,
              recipe: automaticMode ? undefined : request.recipe?.trim(),
              entrypoint:
                hasLocalTraining && advancedRecipe ? request.entrypoint?.trim() : undefined,
            },
      );
      setCreatedPreview(result);
      setPreviewId(result.id);
    } catch (err) {
      setError(err as Error);
    } finally {
      setBusy(false);
    }
  };
  const start = async () => {
    if (draftActionBusy || preview?.status !== 'READY' || !configConfirmed) return;
    setBusy(true);
    setError(undefined);
    try {
      await submitTrainingRun({
        preflightId: preview.id,
        confirmedConfigSha256: requiresConfigConfirmation ? confirmedConfigSha256 : undefined,
        idempotencyKey: idempotencyKey.current,
        finishDraft: draftSession.finished,
        isActive: () => mounted.current,
        onSubmitted: (run) => {
          refresh();
          close();
          navigate(`/runs/${run.id}`);
        },
      });
    } catch (err) {
      if (mounted.current) setError(err as Error);
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  const draftAction = async (operation: () => Promise<void>) => {
    setDraftActionBusy(true);
    setDraftActionError(undefined);
    try {
      await operation();
      setConfirmClearDraft(false);
    } catch (cause) {
      setDraftActionError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      setDraftActionBusy(false);
    }
  };
  const draftStatusText = {
    loading: '正在读取草稿…',
    idle: '填写后自动保存到当前模板草稿',
    pending: '有修改，正在等待自动保存…',
    saving: '正在保存草稿…',
    saved: `草稿已保存${draftSession.updatedAt ? ` · ${new Date(draftSession.updatedAt).toLocaleTimeString()}` : ''}`,
    error: '尚未同步到服务端，当前修改会尽可能暂存在此浏览器',
    conflict: '草稿版本冲突，当前填写尚未覆盖服务端',
  }[draftSession.status];
  return (
    <Modal
      title={isNorm ? '计算 norm_stats' : '新建训练'}
      className="training-run-modal"
      subtitle={
        isNorm
          ? '选择本机数据集和处理方式，核对统计配置后提交计算任务。'
          : '选择数据和初始模型，确认训练方式与参数，预检后核对 config 再启动。'
      }
      close={() => {
        if (busy || draftActionBusy) return;
        void draftSession
          .flush()
          .then(close)
          .catch((cause) => {
            if (draftSession.hasCheckpoint()) close();
            else {
              setDraftActionError(
                new Error(`草稿未能保存，请重试。${cause instanceof Error ? cause.message : ''}`),
              );
              setUnsavedClose(true);
            }
          });
      }}
      wide
      footer={
        <>
          {parameterBlocker && (
            <div className="wizard-processing-blocker" id="training-parameter-blocker" role="alert">
              <XCircle size={16} aria-hidden="true" />
              <span>{parameterBlocker}</span>
              <button
                type="button"
                className="button outline small"
                onClick={() =>
                  document.getElementById(`training-parameter-${parameterIssues[0]?.key}`)?.focus()
                }
              >
                检查训练参数
              </button>
            </div>
          )}
          {step === 1 && gpuSelectionError && (
            <div className="wizard-processing-blocker" id="training-gpu-blocker" role="alert">
              <XCircle size={16} aria-hidden="true" />
              <span>{gpuSelectionError}</span>
            </div>
          )}
          {processingBlocker && (
            <div
              className="wizard-processing-blocker"
              id="training-processing-blocker"
              role="alert"
            >
              <XCircle size={16} aria-hidden="true" />
              <span>{processingBlocker}</span>
              <button
                type="button"
                className="button outline small"
                disabled={busy}
                onClick={() => {
                  if (
                    actionHorizonBlocker &&
                    actionHorizonInference.recommendedHorizon !== undefined
                  )
                    document.getElementById('training-action-horizon')?.focus();
                  else editDataProcessing();
                }}
              >
                {actionHorizonBlocker && actionHorizonInference.recommendedHorizon !== undefined
                  ? '检查预测长度'
                  : '返回数据配置'}
              </button>
            </div>
          )}
          <div className="wizard-footer-note">
            <ShieldCheck size={14} />
            <span>
              {step === 2 ? '代码版本由 Agent 读取并绑定本次训练' : '数据与模型保持在所选机器本地'}
            </span>
          </div>
          {step > 0 && (
            <button
              className="button outline"
              disabled={busy || draftActionBusy}
              onClick={() => {
                draftSession.markEdited();
                setStep(step - 1);
                setPreviewId('');
                setCreatedPreview(undefined);
                setError(undefined);
                setConfirmedConfigSha256(undefined);
              }}
            >
              <ArrowLeft size={15} />
              上一步
            </button>
          )}
          {step < 2 ? (
            <button
              className="button primary"
              type="submit"
              form="run-wizard-form"
              formNoValidate={step === 0}
              aria-describedby={
                [
                  processingBlocker && 'training-processing-blocker',
                  parameterBlocker && 'training-parameter-blocker',
                  gpuSelectionError && 'training-gpu-blocker',
                ]
                  .filter(Boolean)
                  .join(' ') || undefined
              }
              disabled={
                busy ||
                draftActionBusy ||
                (step === 1 && recipeReading) ||
                !!actionHorizonBlocker ||
                !!parameterBlocker ||
                (step === 1 && !!gpuSelectionError) ||
                (step === 1 && (hasLocalTraining || isNorm) && !processorsValid)
              }
            >
              {step === 1 ? (recipeReading ? '读取配方配置…' : '生成 config 并预检') : '下一步'}
              <ArrowRight size={15} />
            </button>
          ) : (
            <button
              className="button primary"
              disabled={busy || draftActionBusy || preview?.status !== 'READY' || !configConfirmed}
              onClick={() => void start()}
            >
              {busy ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}
              {isNorm ? '确认开始计算' : '确认启动训练'}
            </button>
          )}
        </>
      }
    >
      <div className="wizard-steps">
        {[
          { title: isNorm ? '本机数据集' : '数据和初始模型', icon: Database },
          { title: isNorm ? '统计配置与参数' : '训练方式与参数', icon: Cpu },
          { title: '预检与 config 确认', icon: GitCommitHorizontal },
        ].map((item, index) => (
          <div
            key={item.title}
            className={`wizard-step ${index === step ? 'current' : index < step ? 'done' : ''}`}
          >
            <span>{index < step ? <Check size={15} /> : index + 1}</span>
            <strong>{item.title}</strong>
          </div>
        ))}
      </div>
      {sourceRunNotice && (
        <div className="inline-note" role="status">
          已载入“{sourceRunNotice}
          ”的训练设置。默认使用原初始模型创建新训练，请核对资源、重新预检并确认 config。
        </div>
      )}
      <section className="training-draft-status" aria-label="训练配置草稿">
        <div className="training-draft-summary" role="status">
          <strong>{draftStatusText}</strong>
          <span>
            {initialDraft
              ? '已恢复当前模板的未完成配置；请重新核对资源并预检。'
              : '每个训练模板独立保存草稿，可在同一账号跨浏览器、跨设备恢复。'}
          </span>
        </div>
        <div className="training-draft-actions">
          <button
            type="button"
            className="button outline small"
            disabled={busy || draftActionBusy || draftSession.status === 'saving'}
            onClick={() =>
              void draftAction(async () => {
                draftSession.markEdited();
                await draftSession.retry();
              })
            }
          >
            {draftSession.status === 'error' ? '重试保存草稿' : '立即保存草稿'}
          </button>
          <button
            type="button"
            className="button outline small"
            disabled={busy || draftActionBusy || draftSession.status === 'saving'}
            onClick={() => setConfirmClearDraft(true)}
          >
            清空当前模板草稿
          </button>
        </div>
        {draftSession.error && (
          <p className="training-draft-error" role="alert">
            {draftSession.error}
          </p>
        )}
        {draftSession.status === 'conflict' && (
          <div className="training-draft-actions training-draft-choice">
            <button
              type="button"
              className="button outline small"
              disabled={draftActionBusy}
              onClick={() => void draftAction(draftSession.reload)}
            >
              加载服务端草稿（替换当前填写）
            </button>
            <button
              type="button"
              className="button outline small"
              disabled={draftActionBusy}
              onClick={() => void draftAction(draftSession.saveCurrent)}
            >
              保留当前填写并覆盖服务端草稿
            </button>
          </div>
        )}
        {confirmClearDraft && (
          <div className="training-draft-choice" role="alert">
            <span>清空会移除当前训练模板的草稿，并重新开始当前配置。</span>
            <button
              type="button"
              className="button outline small"
              disabled={draftActionBusy}
              onClick={() => void draftAction(draftSession.clear)}
            >
              确认清空并重新配置
            </button>
            <button
              type="button"
              className="button outline small"
              disabled={draftActionBusy}
              onClick={() => setConfirmClearDraft(false)}
            >
              保留草稿
            </button>
          </div>
        )}
        <ErrorNotice error={draftActionError} title="草稿操作未完成" />
        {unsavedClose && (
          <button type="button" className="button outline small" onClick={close}>
            放弃未保存修改并关闭
          </button>
        )}
      </section>
      <TrainingTemplateLibrary
        userId={userId}
        snapshot={snapshot}
        busy={busy || draftActionBusy}
        onApply={async (record) => {
          setDraftActionBusy(true);
          try {
            await onApplyTemplate(record);
          } finally {
            setDraftActionBusy(false);
          }
        }}
      />
      {templateNotice && editedParameters.current.has(TRAINING_TEMPLATE_PROCESSING_MARKER) && (
        <div className="inline-note" role="status">
          <strong>已套用模板：{templateNotice.name}</strong>
          <span>
            选择本次数据集和归一化参数后，可继续预检；已有字段组合、模型和训练参数已保留。
          </span>
          {templateNotice.issues.map((issue, index) => (
            <span key={index}>{issue}</span>
          ))}
        </div>
      )}
      <ErrorNotice
        error={displayedError}
        title={
          step === 2
            ? error
              ? preview
                ? '训练提交失败'
                : '预检请求失败'
              : '预检状态读取失败'
            : '请检查训练配置'
        }
        hint={
          step === 2
            ? error && preview
              ? '请根据错误提示处理，并确认任务状态后再提交。'
              : preflightHint(displayedError)
            : undefined
        }
        retry={step === 2 && !error && previewReadError ? previewResource.refresh : undefined}
      />
      {step < 2 && (
        <form
          id="run-wizard-form"
          inert={draftActionBusy}
          onChangeCapture={draftSession.markEdited}
          onSubmit={(event) => {
            event.preventDefault();
            if (!request.name.trim() || request.name.trim().length > 160) {
              setError(new Error('请填写 1 到 160 个字符的训练名称。'));
              return;
            }
            if (!selectedProfile || node?.status !== 'ONLINE' || template?.status !== 'AVAILABLE') {
              setError(new Error('请选择在线机器和匹配的本机运行环境。'));
              return;
            }
            if (isNorm && normDatasetErrors(normRows, automaticNorm).length) {
              setError(new Error(normDatasetErrors(normRows, automaticNorm).join(' ')));
              return;
            }
            if (
              hasLocalTraining &&
              (!request.inputs.model ||
                !request.dataset_bindings?.length ||
                request.dataset_bindings.some(
                  (binding) =>
                    !binding.dataset_asset_id ||
                    !binding.norm_asset_id ||
                    !Number.isFinite(binding.weight) ||
                    binding.weight <= 0,
                ))
            ) {
              setError(
                new Error('请选择初始模型，为每个数据集选择归一化参数，并填写大于 0 的混采权重。'),
              );
              return;
            }
            if (step === 1) {
              if (gpuSelectionError) {
                setError(new Error(gpuSelectionError));
                return;
              }
              if (parameterIssues.length) {
                setError(new Error(parameterIssues.map((issue) => issue.message).join(' ')));
                return;
              }
              if (actionHorizonErrors.length) {
                setError(new Error(actionHorizonErrors.join(' ')));
                return;
              }
              if (isRecap && !trainingStage) {
                setError(new Error('请选择本次 ReCap 训练阶段。'));
                return;
              }
              if ((hasLocalTraining || isNorm) && !processorsValid) {
                setError(
                  new Error(
                    isNorm
                      ? '请完成每个数据集的统计处理配置。'
                      : automaticMode
                        ? '请完成独立数据处理配置；自动组装不使用旧版配方数据项。'
                        : '请完成数据处理配置，或明确选择沿用代码配方数据项。',
                  ),
                );
                return;
              }
              if (
                automaticConfiguration &&
                (!request.auto_config || automaticConfigErrors(request.auto_config).length)
              ) {
                setError(
                  new Error(
                    request.auto_config
                      ? automaticConfigErrors(request.auto_config).join(' ')
                      : '缺少自动组装配置。',
                  ),
                );
                return;
              }
              if (advancedRecipe && (!request.recipe?.trim() || recipeDirty || recipeReading)) {
                setError(
                  new Error(
                    recipeDirty
                      ? '请先保存配方修改或还原修改。'
                      : recipeReading
                        ? '正在读取配方配置，完成后再继续。'
                        : isNorm
                          ? '请选择统计使用的代码配方。'
                          : '请在高级设置中选择代码配方。',
                  ),
                );
                return;
              }
              if (
                advancedRecipe &&
                hasLocalTraining &&
                !validTrainingEntrypoint(request.entrypoint?.trim() || '')
              ) {
                setError(new Error('请选择代码项目内的相对 .py 训练入口文件。'));
                return;
              }
            }
            setError(undefined);
            draftSession.markEdited();
            setStep(step + 1);
            if (step === 1) void beginPreview();
          }}
        >
          {step === 0 ? (
            <div
              className={`training-data-layout${hasLocalTraining || isNorm ? ' with-datasets' : ''}`}
            >
              <section className="training-data-resources" aria-label="训练与运行资源">
                <h3>{isNorm ? '数据与运行环境' : '训练与运行资源'}</h3>
                <Field
                  label={isNorm ? '统计任务名称' : '训练名称'}
                  required
                  hint="使用清晰的名称，便于查找任务结果。"
                >
                  <input
                    required
                    autoFocus
                    maxLength={160}
                    placeholder={isNorm ? '例如 umi-norm-stats' : '例如 umi-pi05-sft'}
                    value={request.name}
                    onChange={(event) => update('name', event.target.value)}
                  />
                </Field>
                <Field label="计算机器" required>
                  <select
                    required
                    value={request.node_id}
                    onChange={(event) => update('node_id', event.target.value)}
                  >
                    <option value="">选择在线机器</option>
                    {nodes.data?.map((item) => (
                      <option key={item.id} value={item.id} disabled={item.status !== 'ONLINE'}>
                        {item.name} · {item.gpus?.length || 0} GPU
                        {item.status !== 'ONLINE' ? ' · 离线' : ''}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field
                  label="本机运行环境"
                  required
                  hint="环境由管理员配置，平台根据模板过滤兼容的环境。"
                >
                  <select
                    required
                    value={request.profile_id}
                    onChange={(event) => update('profile_id', event.target.value)}
                  >
                    <option value="">
                      {request.node_id
                        ? compatibleProfiles.length
                          ? '选择环境'
                          : '此机器没有匹配的运行环境'
                        : '请先选择机器'}
                    </option>
                    {compatibleProfiles.map((profile) => (
                      <option value={profile.id} key={profile.id}>
                        {profile.name}
                      </option>
                    ))}
                  </select>
                </Field>
                {selectedProfile && (
                  <details className="training-runtime-details">
                    <summary>查看运行环境路径与执行方式</summary>
                    <dl className="detail-list inset">
                      <div>
                        <dt>代码目录</dt>
                        <dd>
                          <code>{selectedProfile.code_dir}</code>
                        </dd>
                      </div>
                      <div>
                        <dt>Python</dt>
                        <dd>
                          <code>{selectedProfile.python}</code>
                        </dd>
                      </div>
                      <div>
                        <dt>执行方式</dt>
                        <dd>{selectedProfile.execution_mode || 'native'}</dd>
                      </div>
                    </dl>
                  </details>
                )}
                {isNorm ? (
                  <NormStatsFields
                    showAutomaticFields={false}
                    node={node}
                    profile={selectedProfile}
                    recipe={request.recipe || ''}
                    onChange={(recipe) => update('recipe', recipe)}
                    advanced={advancedRecipe}
                    onAdvancedChange={toggleRecipeMode}
                    autoConfig={request.auto_config}
                    onAutoConfigChange={(config) => update('auto_config', config)}
                    horizonInference={actionHorizonInference}
                    horizonMode={actionHorizonMode}
                    onHorizonModeChange={setActionHorizonMode}
                  />
                ) : inputRequirements.length ? (
                  inputRequirements
                    .filter((role) => !hasLocalTraining || role === 'model')
                    .map((role) => {
                      const kind =
                        role === 'dataset' ? 'DATASET' : role === 'model' ? 'MODEL' : 'NORM';
                      const choices =
                        assets.data?.filter(
                          (asset) =>
                            asset.node_id === request.node_id &&
                            asset.kind === kind &&
                            !['ARCHIVED', 'MISSING', 'INVALID', 'UNAVAILABLE'].includes(
                              asset.status || asset.state || '',
                            ),
                        ) || [];
                      return (
                        <Field
                          key={role}
                          label={
                            isRecap && role === 'model'
                              ? request.training_stage === 'value'
                                ? '初始骨干 / 价值模型权重'
                                : '初始策略权重'
                              : inputRoles[role] || role
                          }
                          required
                          hint={
                            choices.length
                              ? '只显示当前机器的资产，实际可用性将在预检时验证。'
                              : '当前机器没有可选资产，请先在「数据与模型」中登记。'
                          }
                        >
                          <select
                            required
                            value={request.inputs[role] || ''}
                            onChange={(event) =>
                              update('inputs', { ...request.inputs, [role]: event.target.value })
                            }
                          >
                            <option value="">选择{inputRoles[role] || role}</option>
                            {choices.map((asset) => (
                              <option key={asset.id} value={asset.id}>
                                {asset.name} · {asset.version}
                              </option>
                            ))}
                          </select>
                        </Field>
                      );
                    })
                ) : (
                  <div className="inline-note">
                    当前模板使用合成数据进行链路测试，无需选择本地数据资产。
                  </div>
                )}
                <details className="training-advanced-settings">
                  <summary>其他任务</summary>
                  <label className="training-mode-choice">
                    <input
                      type="checkbox"
                      checked={request.template_id === 'smoke'}
                      onChange={(event) =>
                        void draftAction(() =>
                          onSelectTemplate(event.target.checked ? 'smoke' : 'openpi-sft'),
                        )
                      }
                    />
                    <span>仅进行平台链路测试（使用合成数据，不产生 VLA 模型）</span>
                  </label>
                </details>
              </section>
              {isNorm && (
                <section className="training-data-datasets" aria-label="统计数据与处理方式">
                  <NormDatasetEditor
                    rows={normRows}
                    imageLayoutEdits={imageLayoutEdits}
                    assets={localAssets}
                    node={node}
                    profile={selectedProfile}
                    recipe={request.recipe}
                    automatic={automaticNorm}
                    preserveProcessingOnDatasetChange={editedParameters.current.has(
                      TRAINING_TEMPLATE_PROCESSING_MARKER,
                    )}
                    onValidityChange={setProcessorsValid}
                    onChange={(rows) => update('norm_datasets', rows)}
                  />
                </section>
              )}
              {hasLocalTraining && (
                <section className="training-data-datasets" aria-label="训练数据与处理方式">
                  <DatasetBindings
                    normalizationOverride={request.normalization_method}
                    imageLayoutEdits={imageLayoutEdits}
                    showProcessing
                    preserveProcessingOnDatasetChange={editedParameters.current.has(
                      TRAINING_TEMPLATE_PROCESSING_MARKER,
                    )}
                    bindings={request.dataset_bindings || []}
                    slots={advancedRecipe ? recipeDatasets : []}
                    automatic={automaticMode}
                    assets={localAssets}
                    node={node}
                    profile={selectedProfile}
                    trainingStage={request.training_stage || 'sft'}
                    onValidityChange={setProcessorsValid}
                    managedParameters={
                      isRecap &&
                      request.training_stage === 'offline_rl' &&
                      Object.hasOwn(request.stage_config || {}, 'acp_dropout')
                        ? { acp_dropout: request.stage_config!.acp_dropout }
                        : {}
                    }
                    onChange={(bindings) => {
                      setPreviewId('');
                      setCreatedPreview(undefined);
                      setError(undefined);
                      update('dataset_bindings', bindings);
                    }}
                  />
                </section>
              )}
            </div>
          ) : (
            <>
              <div className="selection-summary">
                <Server size={16} />
                <strong>{node?.name}</strong>
                <span>→</span>
                <span>{selectedProfile?.name}</span>
              </div>
              <Field label="训练方式" required>
                <select
                  required
                  value={request.template_id}
                  onChange={(event) => void draftAction(() => onSelectTemplate(event.target.value))}
                >
                  <option value="">选择模板</option>
                  {templates.data?.map((item) => (
                    <option
                      key={item.id}
                      value={item.id}
                      disabled={
                        item.status !== 'AVAILABLE' ||
                        (request.template_id !== 'smoke' && item.id === 'smoke')
                      }
                    >
                      {item.name}
                      {item.status !== 'AVAILABLE' ? ' · 待接入' : ''}
                    </option>
                  ))}
                </select>
              </Field>
              {isNorm && (
                <>
                  <NormStatsFields
                    node={node}
                    profile={selectedProfile}
                    recipe={request.recipe || ''}
                    onChange={(recipe) => update('recipe', recipe)}
                    advanced={advancedRecipe}
                    onAdvancedChange={toggleRecipeMode}
                    autoConfig={request.auto_config}
                    onAutoConfigChange={(config) => update('auto_config', config)}
                    horizonInference={actionHorizonInference}
                    horizonMode={actionHorizonMode}
                    onHorizonModeChange={setActionHorizonMode}
                  />
                  <NormDatasetEditor
                    rows={normRows}
                    imageLayoutEdits={imageLayoutEdits}
                    assets={localAssets}
                    node={node}
                    profile={selectedProfile}
                    recipe={request.recipe}
                    automatic={automaticNorm}
                    readOnly
                    onValidityChange={setProcessorsValid}
                    onChange={(rows) => update('norm_datasets', rows)}
                  />
                </>
              )}
              {isRecap && (
                <section className="recap-stage-selection" aria-label="ReCap 训练阶段">
                  <Field
                    label="训练阶段"
                    required
                    hint="先选择训练价值函数，或使用已完成优势标注的数据训练策略；两个阶段分别创建任务。"
                  >
                    <select
                      required
                      value={request.training_stage || ''}
                      onChange={(event) =>
                        selectStage((event.target.value || undefined) as TrainingStage | undefined)
                      }
                    >
                      <option value="">选择训练阶段</option>
                      {template.stages?.map((stage) => (
                        <option key={stage.id} value={stage.id}>
                          {stage.name}
                        </option>
                      ))}
                      {request.profile_id && !selectedProfile && (
                        <option value={request.profile_id}>
                          草稿中的运行环境暂不可用，请重新选择
                        </option>
                      )}
                    </select>
                  </Field>
                  <div className="recap-flow" aria-label="ReCap 阶段流程">
                    <span className={request.training_stage === 'value' ? 'selected' : ''}>
                      1. 价值函数训练
                    </span>
                    <ArrowRight size={14} />
                    <span>派生数据与优势标注</span>
                    <ArrowRight size={14} />
                    <span className={request.training_stage === 'offline_rl' ? 'selected' : ''}>
                      2. 离线强化学习
                    </span>
                  </div>
                  <p className="field-hint">
                    价值模型 checkpoint
                    用于后续数据标注。完成标注后，登记派生数据集再创建离线强化学习任务；本次训练不会自动改写原始数据集。
                  </p>
                  {trainingStage && (
                    <div className="inline-note amber">{trainingStage.description}</div>
                  )}
                </section>
              )}
              {automaticMode && request.auto_config && (
                <section aria-label="自动组装模型配置">
                  <p className="field-hint">
                    自动组装适用标准
                    Pi0/Pi05。多数据混采需要可核验的状态布局，建议使用资产组件组合配置；MEM、Hybrid
                    等特殊模型请使用高级代码配方。
                  </p>
                  <div className="form-section-title">
                    <Cpu size={16} />
                    <h3>模型与微调方式</h3>
                    <span>平台自动组装 config</span>
                  </div>
                  <div className="form-grid">
                    <Field label="模型结构" required hint="需要与所选初始权重匹配。">
                      <select
                        value={request.auto_config.model_type}
                        onChange={(event) =>
                          update('auto_config', {
                            ...request.auto_config!,
                            model_type: event.target.value as 'pi0' | 'pi05',
                          })
                        }
                      >
                        <option value="pi05">π0.5</option>
                        <option value="pi0">π0</option>
                      </select>
                    </Field>
                    <Field
                      label="微调方式"
                      required
                      hint={
                        request.training_stage === 'value'
                          ? '自动组装价值训练使用全量更新；仅训练价值头可使用高级代码配方。不支持 LoRA。'
                          : '全量微调或 LoRA；冻结规则写入生成的 config。'
                      }
                    >
                      <select
                        value={request.auto_config.finetune_mode}
                        onChange={(event) =>
                          update('auto_config', {
                            ...request.auto_config!,
                            finetune_mode: event.target.value as 'full' | 'lora',
                          })
                        }
                      >
                        <option value="full">
                          {request.training_stage === 'value' ? '价值模型全量训练' : '全量微调'}
                        </option>
                        <option value="lora" disabled={request.training_stage === 'value'}>
                          LoRA 微调
                        </option>
                      </select>
                    </Field>
                    <Field
                      label="动作预测长度"
                      required
                      hint={
                        actionHorizonInference.errors.length
                          ? '请先解决已登记动作字段或序列长度的错误，再确定预测长度。'
                          : actionHorizonInference.recommendedHorizon !== undefined
                            ? `已登记动作形状包含 ${actionHorizonInference.recommendedHorizon} 步序列，可自动填入；手动修改仍需与已有动作块一致。`
                            : actionHorizonInference.sources.some(
                                  (source) => source.kind === 'single-step',
                                )
                              ? '已登记为单步动作，由加载器按此长度组装；训练长度默认 50，可手动修改。'
                              : '尚未识别到已打包的动作序列；预测长度默认 50，可手动填写，远端预检会核验。'
                      }
                    >
                      <input
                        id="training-action-horizon"
                        required
                        type="number"
                        min={1}
                        max={1000}
                        step={1}
                        value={
                          Number.isFinite(request.auto_config.action_horizon)
                            ? request.auto_config.action_horizon
                            : ''
                        }
                        aria-invalid={actionHorizonErrors.length > 0 || undefined}
                        aria-describedby="training-action-horizon-source"
                        onChange={(event) => {
                          setActionHorizonMode('manual');
                          update('auto_config', {
                            ...request.auto_config!,
                            action_horizon:
                              event.target.value === '' ? NaN : Number(event.target.value),
                          });
                        }}
                      />
                      <div id="training-action-horizon-source" className="field-hint">
                        {actionHorizonMode === 'automatic'
                          ? actionHorizonInference.errors.length
                            ? '自动填入暂不可用'
                            : '自动默认值'
                          : '手动设置 · 自动识别不会覆盖'}
                        {actionHorizonInference.sources
                          .filter((source) => source.shape)
                          .map((source, index) => (
                            <div key={index}>
                              {source.assetName} · {source.actionKey} [{source.shape!.join(', ')}]
                            </div>
                          ))}
                      </div>
                      {actionHorizonMode === 'manual' && !actionHorizonInference.errors.length && (
                        <button
                          type="button"
                          className="button outline small"
                          onClick={() => {
                            setActionHorizonMode('automatic');
                            update('auto_config', {
                              ...request.auto_config!,
                              action_horizon:
                                actionHorizonInference.recommendedHorizon ??
                                DEFAULT_AUTO_CONFIG.action_horizon,
                            });
                          }}
                        >
                          {actionHorizonInference.recommendedHorizon !== undefined
                            ? `使用数据集长度 ${actionHorizonInference.recommendedHorizon}`
                            : '恢复训练默认值 50'}
                        </button>
                      )}
                      {actionHorizonErrors.map((message, index) => (
                        <div className="inline-note red" role="alert" key={index}>
                          {message}
                        </div>
                      ))}
                      {actionHorizonInference.warnings.map((message, index) => (
                        <div className="field-hint" key={index}>
                          {message}
                        </div>
                      ))}
                    </Field>
                    <Field
                      label="模型动作维度"
                      required
                      hint="应与初始化模型匹配；数据原始维度来自登记配置，平台检查填充兼容性。"
                    >
                      <input
                        required
                        type="number"
                        min={1}
                        max={4096}
                        step={1}
                        value={
                          Number.isFinite(request.auto_config.action_dim)
                            ? request.auto_config.action_dim
                            : ''
                        }
                        onChange={(event) =>
                          update('auto_config', {
                            ...request.auto_config!,
                            action_dim:
                              event.target.value === '' ? NaN : Number(event.target.value),
                          })
                        }
                      />
                    </Field>
                  </div>
                </section>
              )}
              {hasLocalTraining && (
                <details className="training-advanced-settings" open={advancedRecipe || undefined}>
                  <summary>高级设置：代码配方与训练入口</summary>
                  <label className="training-mode-choice">
                    <input
                      type="checkbox"
                      checked={advancedRecipe}
                      onChange={(event) => toggleRecipeMode(event.target.checked)}
                    />
                    <span>使用已有代码配方或自定义配方</span>
                  </label>
                  {!advancedRecipe && (
                    <p className="field-hint">
                      常规训练由平台根据数据、模型、训练方式和参数组装
                      config。启用后可复用项目内的特殊训练配置。
                    </p>
                  )}
                  {advancedRecipe && (
                    <>
                      <RecipeLibrary
                        key={`${request.node_id}\0${request.profile_id}\0${request.training_stage}`}
                        recipe={request.recipe || ''}
                        initialRevision={selectedRecipeRevision}
                        onSelect={selectRecipe}
                        onDefaults={applyRecipeDefaults}
                        onDatasets={applyDatasetDefaults}
                        onResolved={applyStageRecipeDefaults}
                        onDirtyChange={setRecipeDirty}
                        onLoadingChange={setRecipeReading}
                        restoredEditorDraft={recipeEditorDraft}
                        onEditorDraftChange={(next) => {
                          setRecipeEditorDraft(next);
                          draftSession.markEdited();
                        }}
                        node={node}
                        profile={selectedProfile}
                      />
                      <PathAutocomplete
                        key={`${request.node_id}\0${request.profile_id}\0${selectedProfile?.code_dir}`}
                        label="训练脚本入口文件"
                        placeholder="选择项目内脚本，例如 scripts/train_multi_dataset.py"
                        value={request.entrypoint || ''}
                        onChange={(value) => {
                          entrypointEdited.current = true;
                          update('entrypoint', value);
                        }}
                        node={node}
                        profile={selectedProfile}
                        projectRelative
                      />
                    </>
                  )}
                </details>
              )}
              {hasLocalTraining && (
                <details
                  className="training-advanced-settings"
                  open={isRecap || advancedRecipe || !processorsValid || undefined}
                >
                  <summary>
                    核对数据与处理配置（{request.dataset_bindings?.length || 0} 个数据集）
                  </summary>
                  <DatasetBindings
                    normalizationOverride={request.normalization_method}
                    imageLayoutEdits={imageLayoutEdits}
                    preserveProcessingOnDatasetChange={editedParameters.current.has(
                      TRAINING_TEMPLATE_PROCESSING_MARKER,
                    )}
                    readOnly
                    onEditProcessing={editDataProcessing}
                    bindings={request.dataset_bindings || []}
                    slots={advancedRecipe ? recipeDatasets : []}
                    automatic={automaticMode}
                    assets={localAssets}
                    node={node}
                    profile={selectedProfile}
                    trainingStage={request.training_stage || 'sft'}
                    onValidityChange={setProcessorsValid}
                    managedParameters={
                      isRecap &&
                      request.training_stage === 'offline_rl' &&
                      Object.hasOwn(request.stage_config || {}, 'acp_dropout')
                        ? { acp_dropout: request.stage_config!.acp_dropout }
                        : {}
                    }
                    onChange={(bindings) => {
                      setPreviewId('');
                      setCreatedPreview(undefined);
                      setError(undefined);
                      update('dataset_bindings', bindings);
                    }}
                  />
                </details>
              )}
              {isRecap && trainingStage && (
                <>
                  <div className="form-section-title">
                    <Cpu size={16} />
                    <h3>{trainingStage.name}参数</h3>
                    <span>仅用于当前阶段</span>
                  </div>
                  {stageRecipeInfo?.mixedDropout && (
                    <div
                      className={`inline-note ${mixedDropoutUnresolved ? 'red training-validation-error' : 'amber'}`}
                      role={mixedDropoutUnresolved ? 'alert' : 'status'}
                    >
                      配方中各数据项的 ACP 条件丢弃概率不一致，请明确填写本次统一覆盖值。
                    </div>
                  )}
                  <div className="form-grid parameters">
                    {Object.entries(stageParameterSpecs).map(([key, spec]) => (
                      <Field
                        key={key}
                        label={spec.label || stageParameterLabels[key] || key}
                        required
                        hint={
                          Object.hasOwn(stageRecipeDefaults, key)
                            ? `配方值：${stageRecipeDefaults[key]}${editedStageParameters.current.has(key) ? '；当前为本次覆盖' : ''}`
                            : '本次阶段参数，Agent 将在预检时核对。'
                        }
                      >
                        <input
                          type="number"
                          required
                          min={spec.min ?? spec.minimum}
                          max={spec.max ?? spec.maximum}
                          step={spec.type === 'integer' ? 1 : 'any'}
                          value={
                            Number.isFinite(request.stage_config?.[key])
                              ? request.stage_config?.[key]
                              : ''
                          }
                          onChange={(event) => {
                            editedStageParameters.current.add(key);
                            update('stage_config', {
                              ...request.stage_config,
                              [key]: event.target.value === '' ? NaN : Number(event.target.value),
                            });
                          }}
                        />
                      </Field>
                    ))}
                  </div>
                  {advancedRecipe && request.training_stage === 'value' && (
                    <dl className="detail-list inset">
                      <div>
                        <dt>训练参数范围</dt>
                        <dd>
                          由价值配方的冻结规则决定；Agent 在预检时检查。
                          {!stageRecipeInfo?.valueHead.length && (
                            <span className="field-hint">
                              可在上一步的配方详情查看模型与冻结规则。
                            </span>
                          )}
                        </dd>
                      </div>
                      {stageRecipeInfo?.valueHead.map((field) => (
                        <div key={field.path}>
                          <dt>
                            <code>{field.path}</code>
                          </dt>
                          <dd>
                            <code>{field.value}</code>
                          </dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </>
              )}
              <div className="form-section-title">
                <Cpu size={16} />
                <h3>{isNorm ? '统计参数' : '训练参数'}</h3>
                <span>参数已填入初始值，可直接修改后生成 config</span>
              </div>
              <div className="training-parameter-sections">
                {hasLocalTraining && (
                  <section className="training-parameter-section" aria-label="训练归一化方式">
                    <div className="form-grid">
                      <TrainingNormalizationField
                        value={request.normalization_method}
                        onChange={(method) => update('normalization_method', method)}
                      />
                    </div>
                  </section>
                )}
                {parameterGroups.map((group) => (
                  <section
                    className="training-parameter-section"
                    key={group.id}
                    aria-label={group.title}
                  >
                    <h4>{isNorm ? '数据加载' : group.title}</h4>
                    <div className="form-grid parameters">
                      {group.entries.map(([key, spec]) => (
                        <Field
                          key={key}
                          label={spec.label || key}
                          required={
                            !unusedParameters.includes(key) && !optionalTrainingParameter(spec)
                          }
                          hint={
                            isNorm
                              ? key === 'batch_size'
                                ? '每批读取的样本数；任务始终完整遍历数据集。'
                                : normResourceRecommendationHint(resourceRecommendations)
                              : isTrainingResourceField(key)
                                ? trainingResourceRecommendationHint(key, resourceRecommendations)
                                : trainingParameterHint(
                                    key,
                                    spec,
                                    advancedRecipe,
                                    recipeDefaults,
                                    editedParameters.current.has(key) &&
                                      Object.hasOwn(request.parameters, key),
                                  )
                          }
                        >
                          {unusedParameters.includes(key) ? (
                            <div
                              className="training-parameter-unused"
                              id={`training-parameter-unused-${key}`}
                            >
                              <span>此配方不使用</span>
                              {Object.hasOwn(request.parameters, key) && (
                                <>
                                  <code>已保存值：{String(request.parameters[key])}</code>
                                  <button
                                    type="button"
                                    className="button outline small"
                                    id={`training-clear-unsupported-${key}`}
                                    onClick={() => {
                                      defaultedParameterKeys.current.delete(key);
                                      editedParameters.current.add(key);
                                      const parameters = { ...request.parameters };
                                      delete parameters[key];
                                      update('parameters', parameters);
                                    }}
                                  >
                                    清除此项
                                  </button>
                                </>
                              )}
                            </div>
                          ) : (
                            <>
                              {key === 'ema_decay' && emaDefaultDisabled && (
                                <div className="training-ema-choice">
                                  <input
                                    type="checkbox"
                                    id="training-ema-enabled"
                                    aria-label="开启 EMA"
                                    checked={Object.hasOwn(request.parameters, 'ema_decay')}
                                    onChange={(event) => {
                                      defaultedParameterKeys.current.delete('ema_decay');
                                      editedParameters.current.add('ema_decay');
                                      const parameters = { ...request.parameters };
                                      if (event.target.checked) parameters.ema_decay = 0.99;
                                      else delete parameters.ema_decay;
                                      update('parameters', parameters);
                                    }}
                                  />
                                  <span>
                                    {Object.hasOwn(request.parameters, 'ema_decay')
                                      ? '已手动开启 EMA'
                                      : 'EMA 默认关闭，可手动开启'}
                                  </span>
                                </div>
                              )}
                              {!(
                                key === 'ema_decay' &&
                                emaDefaultDisabled &&
                                !Object.hasOwn(request.parameters, 'ema_decay')
                              ) && (
                                <input
                                  id={`training-parameter-${key}`}
                                  type="number"
                                  required={!optionalTrainingParameter(spec)}
                                  min={spec.min ?? spec.minimum}
                                  max={spec.max ?? spec.maximum}
                                  step={spec.type === 'integer' ? 1 : 'any'}
                                  value={
                                    Number.isFinite(request.parameters[key])
                                      ? request.parameters[key]
                                      : ''
                                  }
                                  aria-invalid={
                                    parameterIssues.some((issue) => issue.key === key) || undefined
                                  }
                                  aria-describedby={
                                    parameterIssues.some((issue) => issue.key === key)
                                      ? `training-parameter-error-${key}`
                                      : undefined
                                  }
                                  onChange={(event) => {
                                    defaultedParameterKeys.current.delete(key);
                                    editedParameters.current.add(key);
                                    if (isTrainingResourceField(key))
                                      setResourceRecommendationMode(
                                        editedParameters.current,
                                        key,
                                        false,
                                      );
                                    update(
                                      'parameters',
                                      editTrainingParameter(
                                        request.parameters,
                                        key,
                                        event.target.value,
                                        spec,
                                      ),
                                    );
                                  }}
                                />
                              )}
                            </>
                          )}
                          {isTrainingResourceField(key) && (!isNorm || key === 'num_workers') && (
                            <div
                              className="training-resource-recommendation"
                              id={`training-resource-recommendation-${key}`}
                            >
                              <span>
                                {resourceRecommendations[key] === undefined
                                  ? isNorm
                                    ? '等待机器 CPU 信息'
                                    : '填写有效 GPU 数量后显示推荐'
                                  : `推荐值：${resourceRecommendations[key]}`}
                                {' · '}
                                {automaticResourceRecommendation(editedParameters.current, key)
                                  ? '自动跟随'
                                  : Object.hasOwn(request.parameters, key)
                                    ? '手动保留'
                                    : '空值沿用配置'}
                              </span>
                              {!isNorm && (
                                <button
                                  type="button"
                                  className="button outline small"
                                  id={`training-use-recommendation-${key}`}
                                  aria-label={`使用${spec.label || key}推荐值`}
                                  disabled={
                                    busy ||
                                    draftActionBusy ||
                                    resourceRecommendations[key] === undefined ||
                                    automaticResourceRecommendation(editedParameters.current, key)
                                  }
                                  onClick={() => {
                                    setResourceRecommendationMode(
                                      editedParameters.current,
                                      key,
                                      true,
                                    );
                                    update(
                                      'parameters',
                                      applyResourceRecommendations(
                                        request.parameters,
                                        editedParameters.current,
                                        resourceRecommendations,
                                      ),
                                    );
                                  }}
                                >
                                  {automaticResourceRecommendation(editedParameters.current, key)
                                    ? '已使用推荐'
                                    : '使用推荐值'}
                                </button>
                              )}
                            </div>
                          )}
                          {parameterIssues.some((issue) => issue.key === key) && (
                            <span
                              className="training-parameter-error"
                              id={`training-parameter-error-${key}`}
                            >
                              {parameterIssues
                                .filter((issue) => issue.key === key)
                                .map((issue) => issue.message)
                                .join(' ')}
                            </span>
                          )}
                        </Field>
                      ))}
                      {group.id === 'resources' && !isNorm && (
                        <TrainingGpuSelector
                          request={request}
                          node={node}
                          minimum={minGpus}
                          allowShared={hasLocalTraining}
                          onChange={updateGpuSelection}
                        />
                      )}
                    </div>
                  </section>
                ))}
              </div>
            </>
          )}
        </form>
      )}
      {step === 2 && (
        <div className="preflight-content">
          <div
            className={`preflight-heading ${preview?.status === 'READY' ? 'ready' : preflightFailed || progressUnavailable ? 'failed' : ''}`}
          >
            {preview?.status === 'READY' ? (
              <CheckCircle2 size={28} />
            ) : preflightFailed || progressUnavailable ? (
              <XCircle size={28} />
            ) : busy || preview?.status === 'PENDING' ? (
              <LoaderCircle size={28} className="spin" />
            ) : (
              <Circle size={28} />
            )}
            <div>
              <h3>
                {preview?.status === 'READY'
                  ? isNorm
                    ? '统计预检通过，请检查 config 并确认'
                    : hasLocalTraining
                      ? '预检通过，请检查 config 并确认'
                      : '平台链路测试预检通过'
                  : requestFailed
                    ? '预检请求未完成'
                    : previewExpired
                      ? '预检结果已过期'
                      : preview?.status === 'FAILED'
                        ? '预检未通过'
                        : progressUnavailable
                          ? '暂时无法读取预检进度'
                          : busy || preview?.status === 'PENDING'
                            ? isNorm
                              ? 'Agent 正在检查统计配方与本机数据'
                              : 'Agent 正在自动检查配方与本机资产'
                            : '尚未发起预检'}
              </h3>
              <p>
                {preview?.status === 'READY'
                  ? isNorm
                    ? '已检查统计配方和本机数据。请核对下方生成的 config，勾选确认后开始计算 norm_stats。'
                    : '已检查配置与本机资产，尚未执行训练。请核对下方生成的 config，勾选确认后启动。'
                  : requestFailed
                    ? '尚未取得预检记录，请根据错误提示修正后重新预检。'
                    : previewExpired
                      ? '请重新预检，获取当前代码、资产与运行环境的检查结果。'
                      : preview?.status === 'FAILED'
                        ? '请按失败项修正配置、资产或运行环境，然后重新预检。'
                        : progressUnavailable
                          ? '可重试读取进度；当前显示的结果可能尚未更新。'
                          : isNorm
                            ? '自动检查统计配方、数据处理、真实样本、缓存权限、输出空间与统计脚本入口。'
                            : '自动检查配方、数据集、模型、归一化参数、输出空间、GPU 与实际执行代码。'}
              </p>
            </div>
          </div>
          {sharedNormDetail && (
            <div className="error-notice" role="alert">
              <XCircle size={18} aria-hidden="true" />
              <div>
                <strong title={preview?.error?.code}>这些数据集无法共用归一化参数</strong>
                <p>{sharedNormDetail}</p>
              </div>
            </div>
          )}
          {datasetValidationRequired && (
            <div className="error-notice">
              <XCircle size={18} aria-hidden="true" />
              <div>
                <strong role="alert" title={DATASET_BASE_VALIDATION_REQUIRED}>
                  数据集基础校验缺失或已失效
                </strong>
                {validationAsset && (
                  <p>数据集「{validationAsset.name}」需要在本次训练环境刷新校验。</p>
                )}
                {validationAsset && validationRequest ? (
                  <PreflightDatasetValidation
                    key={`${preview.id}:${validationAsset.id}:${validationRequest.profile_id}`}
                    asset={validationAsset}
                    node={nodes.data?.find((item) => item.id === validationRequest.node_id)}
                    profiles={profiles.data || []}
                    profileId={validationRequest.profile_id}
                    changed={assets.refresh}
                    onPendingChange={setDatasetValidationPending}
                  />
                ) : (
                  <p>
                    {preview.error?.message?.replace(/^[A-Z_]+:\s*/u, '') ||
                      '请在数据集详情完成基础校验，再重新预检。'}
                  </p>
                )}
              </div>
            </div>
          )}
          {preview?.checks?.length ? (
            <PreflightChecks
              checks={preview.checks}
              summarizedSharedNormError={sharedNormDetail ? preview.error?.message : undefined}
              showDatasetValidationNotice={datasetValidationRequired}
            />
          ) : (
            preview?.status === 'PENDING' &&
            !progressUnavailable && (
              <div className="preflight-wait">
                等待机器回传预检结果。机器在线时会自动继续；可关闭弹窗后重新发起。
              </div>
            )
          )}
          {preview?.error && !sharedNormDetail && !datasetValidationRequired && (
            <div className="inline-note red">
              <strong>{preview.error.code}</strong>
              <span>{preview.error.message}</span>
            </div>
          )}
          {automaticPreflight?.scope === 'configuration_and_local_assets' && (
            <dl className="detail-list inset">
              <div>
                <dt>自动预检范围</dt>
                <dd>
                  {isNorm
                    ? '检查配方和本机数据；提交后才执行完整统计。'
                    : '配方配置与本机资产检查；训练效果需启动后观察。'}
                </dd>
              </div>
              <div>
                <dt>{isNorm ? '统计输入来源' : '训练输入来源'}</dt>
                <dd>
                  {automaticPreflight.plan.input_authority === 'registered_assets'
                    ? '本次选择的已登记资产，已核对实际路径与资产身份。'
                    : '缺少资产绑定证明，请更新 Agent 后重新预检。'}
                </dd>
              </div>
              <div>
                <dt>{isNorm ? '实际统计入口' : '实际训练入口'}</dt>
                <dd>
                  <code>{automaticPreflight.plan.entrypoint}</code>
                </dd>
              </div>
              <div>
                <dt>实际数据集</dt>
                <dd>
                  {automaticPreflight.plan.datasets.length} 个
                  {automaticPreflight.plan.datasets.map((dataset, index) => (
                    <div className="preflight-dataset" key={`${dataset.path}-${index}`}>
                      <strong>数据集 {index + 1}</strong>
                      <code className="block">{dataset.path}</code>
                      {dataset.norm_path && (
                        <>
                          <span>归一化参数</span>
                          <code className="block">{dataset.norm_path}</code>
                        </>
                      )}
                    </div>
                  ))}
                </dd>
              </div>
              {automaticPreflight.plan.model && (
                <div>
                  <dt>实际初始化模型</dt>
                  <dd>
                    <code>{automaticPreflight.plan.model.path}</code>
                  </dd>
                </div>
              )}
            </dl>
          )}
          {preview?.code_snapshot && (
            <CodeVersion snapshot={preview.code_snapshot} title="本次训练代码基线" />
          )}
          <dl className="detail-list inset">
            <div>
              <dt>训练名称</dt>
              <dd>{request.name}</dd>
            </div>
            <div>
              <dt>机器 / 模板</dt>
              <dd>
                {node?.name} / {template?.name}
              </dd>
            </div>
            {isRecap && (
              <>
                <div>
                  <dt>训练阶段</dt>
                  <dd>
                    {request.training_stage
                      ? trainingStageLabels[request.training_stage]
                      : '未选择'}
                  </dd>
                </div>
                {Object.entries(preview?.request?.stage_config || request.stage_config || {}).map(
                  ([key, value]) => (
                    <div key={key}>
                      <dt>{stageParameterLabels[key] || key}</dt>
                      <dd>{value}</dd>
                    </div>
                  ),
                )}
              </>
            )}
            <div>
              <dt>{preview?.request?.auto_config ? '配置来源' : '训练配方'}</dt>
              <dd>
                {frozenRecipe && (
                  <span className="recipe-frozen-label">
                    {frozenRecipe.name} · v{frozenRecipe.revision}
                  </span>
                )}
                <code>
                  {preview?.request?.auto_config
                    ? '平台自动组装'
                    : preview?.request?.recipe ||
                      preview?.code_snapshot?.recipe ||
                      request.recipe?.trim() ||
                      '未选择'}
                </code>
              </dd>
            </div>
            {hasLocalTraining && (
              <div>
                <dt>训练归一化方式</dt>
                <dd>
                  {trainingNormalizationLabel((preview?.request || request).normalization_method)}
                </dd>
              </div>
            )}
            {frozenRecipe && (
              <div>
                <dt>配方版本摘要</dt>
                <dd>
                  <code title={frozenRecipe.digest}>{frozenRecipe.digest.slice(0, 16)}</code>
                </dd>
              </div>
            )}
            <div>
              <dt>申请 GPU</dt>
              <dd>
                {request.gpu_count} 张 · {requestedGpuSummary(request.requested_gpu_uuids, node)}
                <br />
                {trainingGpuPolicySummary((preview?.request || request).training_gpu_policy)}
              </dd>
            </div>
            {hasLocalTraining && (
              <>
                <div>
                  <dt>代码项目基地址</dt>
                  <dd>
                    <code>{preview?.code_snapshot?.code_dir || selectedProfile?.code_dir}</code>
                  </dd>
                </div>
                <div>
                  <dt>所选训练入口</dt>
                  <dd>
                    <code>
                      {preview?.request?.entrypoint ||
                        preview?.code_snapshot?.requested_entrypoint ||
                        request.entrypoint ||
                        '由训练方式和数据集数量自动选择'}
                    </code>
                  </dd>
                </div>
              </>
            )}
          </dl>
          {!!request.dataset_bindings?.length && (
            <div className="preflight-dataset-bindings">
              <h3>本次数据集绑定</h3>
              <DatasetBindingSummary
                automaticSharedNormalization
                normalizationOverride={(preview?.request || request).normalization_method}
                bindings={preview?.request?.dataset_bindings || request.dataset_bindings}
                assets={assets.data || []}
                profiles={preview?.request?.dataset_profiles}
              />
            </div>
          )}
          {isNorm && (
            <NormDatasetEditor
              rows={preview?.request?.norm_datasets || normRows}
              assets={assets.data || []}
              node={node}
              profile={selectedProfile}
              recipe={request.recipe}
              automatic={!!(preview?.request || request).auto_config}
              readOnly
              onChange={() => {}}
            />
          )}
          {requiresConfigConfirmation && preview?.status === 'READY' && (
            <GeneratedConfigReview
              purpose={isNorm ? 'norm' : 'training'}
              config={generatedConfig}
              confirmedSha256={confirmedConfigSha256}
              onConfirm={setConfirmedConfigSha256}
            />
          )}
          {(!preview || ['FAILED', 'EXPIRED', 'READY'].includes(preview.status)) && !busy && (
            <button
              className="button outline"
              disabled={datasetValidationRequired && datasetValidationPending}
              onClick={() => void beginPreview()}
            >
              <RotateCcw size={15} />
              重新预检
            </button>
          )}
        </div>
      )}
    </Modal>
  );
}
