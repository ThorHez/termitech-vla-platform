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
import { useAuth } from './auth';
import { CodeVersion, ErrorNotice, Field, Modal } from './components';
import { inputRoles, requiredInputs, useData } from './data';
import { RecipeLibrary } from './RecipeLibrary';
import { DatasetBindings, DatasetBindingSummary } from './DatasetBindings';
import { newDatasetBindings } from './datasetBindingHelpers';
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

const preflightCheckLabels: Record<string, string> = {
  runtime: '训练运行环境',
  local_paths: '本机资产路径',
  code_version: '实际执行代码版本',
  adapter: '训练模板与适配器',
  recipe_configuration: '配方配置解析',
  data_contract: '数据集与数据协议',
  norm_contract: '归一化参数',
  model_checkpoint: '模型检查点',
  training_entrypoint: '训练入口',
  training_cache: '训练缓存与数据加载权限',
  output_storage: '输出目录与磁盘空间',
  gpu_capacity: 'GPU 数量与配置',
  validation: '预检验证',
};

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
export default function RunWizard(props: Parameters<typeof RunWizardContent>[0]) {
  const { can } = useAuth();
  if (!can('training.write'))
    return (
      <Modal title="权限不足" close={props.close}>
        创建训练需要操作员或管理员权限。
      </Modal>
    );
  return <RunWizardContent {...props} />;
}

function RunWizardContent({
  close,
  initialNode,
  initialTemplate,
}: {
  close: () => void;
  initialNode?: string;
  initialTemplate?: string;
}) {
  const { nodes, assets, profiles, templates, refresh } = useData();
  const [step, setStep] = useState(0);
  const [request, setRequest] = useState<RunRequest>({
    name: '',
    node_id: initialNode || '',
    profile_id: '',
    template_id:
      initialTemplate ||
      templates.data?.find((item) => item.id === 'openpi-sft')?.id ||
      templates.data?.find((item) => item.status === 'AVAILABLE')?.id ||
      '',
    inputs: {},
    parameters: {},
    gpu_count: 1,
  });
  const [previewId, setPreviewId] = useState('');
  const [createdPreview, setCreatedPreview] = useState<Preview>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const idempotencyKey = useRef(crypto.randomUUID());
  const recipeSelection = useRef<string | undefined>(undefined);
  const editedParameters = useRef(new Set<string>());
  const editedStageParameters = useRef(new Set<string>());
  const entrypointEdited = useRef(false);
  const [selectedRecipeRevision, setSelectedRecipeRevision] = useState<RecipeRevision>();
  const [recipeDirty, setRecipeDirty] = useState(false);
  const [recipeReading, setRecipeReading] = useState(false);
  const [recipeDefaults, setRecipeDefaults] = useState<Record<string, number>>({});
  const [recipeDatasets, setRecipeDatasets] = useState<RecipeDatasetSlot[]>([]);
  const [processorsValid, setProcessorsValid] = useState(true);
  const [stageRecipeDefaults, setStageRecipeDefaults] = useState<Record<string, number>>({});
  const [stageRecipeInfo, setStageRecipeInfo] =
    useState<ReturnType<typeof recapRecipeConfiguration>>();
  const datasetDefaultsScope = useRef('');
  const requestRef = useRef(request);
  requestRef.current = request;
  const previewResource = useApi<Preview>(previewId ? `/run-previews/${previewId}` : null, 1800);
  const preview = previewResource.data?.id === previewId ? previewResource.data : createdPreview;
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
  const hasLocalTraining = template?.id === 'openpi-sft' || isRecap;
  const trainingStage = isRecap
    ? template?.stages?.find((stage) => stage.id === request.training_stage)
    : undefined;
  const node = nodes.data?.find((item) => item.id === request.node_id);
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
  const stageParameterSpecs = specs(trainingStage);
  const inputRequirements = requiredInputs(trainingStage || template);
  const selectedProfile = compatibleProfiles.find((profile) => profile.id === request.profile_id);
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
    const rules = specs(template);
    setRequest((value) => ({
      ...value,
      parameters:
        template?.id === 'openpi-sft' || template?.id === 'recap'
          ? {}
          : Object.fromEntries(
              Object.entries(rules).map(([key, rule]) => [key, rule.default ?? 1]),
            ),
      gpu_count: template?.min_gpus ?? (template?.id === 'smoke' ? 0 : 1),
      inputs: {},
      dataset_bindings: undefined,
      training_stage: undefined,
      stage_config: undefined,
    }));
    // A change to template metadata does not reset an in-progress form.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.template_id]);
  useEffect(() => {
    setRequest((value) =>
      compatibleProfiles.some((profile) => profile.id === value.profile_id)
        ? value
        : { ...value, profile_id: compatibleProfiles[0]?.id || '' },
    );
  }, [compatibleProfiles]);
  useEffect(() => {
    const selection = `${request.node_id}\0${request.template_id}\0${request.training_stage}\0${request.profile_id}\0${selectedProfile?.code_dir}`;
    if (!selectedProfile) {
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
    editedParameters.current.clear();
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
          : request.template_id === 'recap'
            ? ''
            : selectedProfile.recipe?.trim() ||
              (validatedRecipes.length === 1 ? validatedRecipes[0] : ''),
      recipe_revision_id: undefined,
      entrypoint: undefined,
      dataset_bindings: undefined,
      parameters: ['openpi-sft', 'recap'].includes(request.template_id) ? {} : value.parameters,
    }));
    setSelectedRecipeRevision(undefined);
    setRecipeDirty(false);
    // Only selection changes choose a default; background refreshes preserve user input.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    request.node_id,
    request.template_id,
    request.training_stage,
    request.profile_id,
    selectedProfile,
  ]);
  useEffect(() => {
    if (!trainingStage) return;
    setRequest((value) =>
      value.stage_config !== undefined
        ? value
        : {
            ...value,
            stage_config: Object.fromEntries(
              Object.entries(specs(trainingStage)).map(([key, rule]) => [key, rule.default ?? 1]),
            ),
          },
    );
  }, [trainingStage]);
  useEffect(() => {
    if (!isRecap || !trainingStage || entrypointEdited.current) return;
    const multiple = recipeDatasets.some((slot) => slot.multiple);
    const entrypoint = multiple
      ? trainingStage.default_multi_entrypoint
      : trainingStage.default_entrypoint;
    if (entrypoint) setRequest((value) => ({ ...value, entrypoint }));
  }, [isRecap, trainingStage, recipeDatasets]);
  const update = <K extends keyof RunRequest>(key: K, value: RunRequest[K]) =>
    setRequest((previous) => ({ ...previous, [key]: value }));
  const applyRecipeDefaults = useCallback(
    (defaults: Record<string, number>, recipe: string, revisionId?: string) => {
      const current = requestRef.current;
      if (current.recipe !== recipe || current.recipe_revision_id !== revisionId) return;
      setRecipeDefaults(defaults);
      setRequest((value) => ({
        ...value,
        parameters: {
          ...value.parameters,
          ...Object.fromEntries(
            Object.entries(defaults).filter(([key]) => !editedParameters.current.has(key)),
          ),
        },
      }));
    },
    [],
  );
  const applyStageRecipeDefaults = useCallback(
    (detail: RecipeDetail, recipe: string, revisionId?: string) => {
      const current = requestRef.current;
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
          dataset_bindings: newDatasetBindings(slots),
        };
      });
    },
    [],
  );
  const selectRecipe = (recipe: string, revision?: RecipeRevision) => {
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
        parameters: {},
        dataset_bindings: undefined,
        stage_config:
          request.template_id === 'recap'
            ? Object.fromEntries(
                Object.entries(stageParameterSpecs).map(([key, rule]) => [key, rule.default ?? 1]),
              )
            : undefined,
      }));
    }
  };
  const selectStage = (stage: TrainingStage | undefined) => {
    editedParameters.current.clear();
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
      recipe: '',
      recipe_revision_id: undefined,
      entrypoint: undefined,
      inputs: {},
      dataset_bindings: undefined,
      parameters: {},
    }));
  };
  const beginPreview = async () => {
    setBusy(true);
    setError(undefined);
    setPreviewId('');
    setCreatedPreview(undefined);
    idempotencyKey.current = crypto.randomUUID();
    try {
      const result = await post<Preview>('/run-previews', {
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
        recipe_revision_id: request.recipe_revision_id,
        recipe: request.recipe?.trim(),
        entrypoint: hasLocalTraining ? request.entrypoint?.trim() : undefined,
      });
      setCreatedPreview(result);
      setPreviewId(result.id);
    } catch (err) {
      setError(err as Error);
    } finally {
      setBusy(false);
    }
  };
  const start = async () => {
    if (preview?.status !== 'READY') return;
    setBusy(true);
    setError(undefined);
    try {
      const run = await post<Run>(
        '/runs',
        { preflight_id: preview.id },
        { 'Idempotency-Key': idempotencyKey.current },
      );
      refresh();
      close();
      navigate(`/runs/${run.id}`);
    } catch (err) {
      setError(err as Error);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal
      title="新建训练"
      subtitle="选择本机资源，预检通过后启动。"
      close={() => {
        if (!busy) close();
      }}
      wide
      footer={
        <>
          <div className="wizard-footer-note">
            <ShieldCheck size={14} />
            <span>
              {step === 2 ? '代码版本由 Agent 读取并绑定本次训练' : '数据与模型保持在所选机器本地'}
            </span>
          </div>
          {step > 0 && (
            <button
              className="button outline"
              disabled={busy}
              onClick={() => {
                setStep(step - 1);
                setPreviewId('');
                setCreatedPreview(undefined);
                setError(undefined);
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
              disabled={
                busy ||
                (step === 0 && recipeReading) ||
                (step === 1 && hasLocalTraining && !processorsValid)
              }
            >
              {step === 1 ? '开始预检' : recipeReading ? '读取配方配置…' : '下一步'}
              <ArrowRight size={15} />
            </button>
          ) : (
            <button
              className="button primary"
              disabled={busy || preview?.status !== 'READY'}
              onClick={() => void start()}
            >
              {busy ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}确认启动训练
            </button>
          )}
        </>
      }
    >
      <div className="wizard-steps">
        {[
          { title: '机器与模板', icon: Server },
          { title: '资产与参数', icon: Database },
          { title: '预检与确认', icon: GitCommitHorizontal },
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
          onSubmit={(event) => {
            event.preventDefault();
            if (
              step === 1 &&
              hasLocalTraining &&
              (!request.dataset_bindings?.length ||
                request.dataset_bindings.some(
                  (binding) =>
                    !binding.dataset_asset_id ||
                    !binding.norm_asset_id ||
                    !Number.isFinite(binding.weight) ||
                    binding.weight <= 0,
                ))
            ) {
              setError(new Error('请为每个数据集选择归一化参数，并填写大于 0 的混采权重。'));
              return;
            }
            if (step === 1 && hasLocalTraining && !processorsValid) {
              setError(
                new Error(
                  '请等待数据处理器目录读取完成，并填写所选处理器的必填参数；无法读取目录时可明确选择沿用当前配方数据项。',
                ),
              );
              return;
            }
            if (
              step === 0 &&
              (!selectedProfile || node?.status !== 'ONLINE' || template?.status !== 'AVAILABLE')
            ) {
              setError(new Error('请选择在线机器、可用模板和匹配的本机运行环境。'));
              return;
            }
            if (step === 0 && !request.recipe?.trim()) {
              setError(new Error('请填写本次训练使用的配方名称。'));
              return;
            }
            if (isRecap && !trainingStage) {
              setError(new Error('请选择本次 ReCap 训练阶段。'));
              return;
            }
            if (step === 0 && recipeDirty) {
              setError(new Error('配方详情中有未保存的修改，请先保存为配方版本，或还原修改。'));
              return;
            }
            if (step === 0 && recipeReading) {
              setError(new Error('正在读取配方配置，完成后再继续。'));
              return;
            }
            if (
              step === 1 &&
              hasLocalTraining &&
              !validTrainingEntrypoint(request.entrypoint?.trim() || '')
            ) {
              setError(
                new Error('请选择代码项目内的 .py 训练入口文件，使用相对路径，不能包含 ..。'),
              );
              return;
            }
            setError(undefined);
            if (step === 0 && hasLocalTraining && !request.dataset_bindings) {
              update('dataset_bindings', [
                { dataset_asset_id: '', norm_asset_id: '', recipe_index: 0, weight: 1 },
              ]);
            }
            setStep(step + 1);
            if (step === 1) void beginPreview();
          }}
        >
          {step === 0 ? (
            <>
              <Field label="训练名称" required hint="使用清晰的名称，便于查找训练结果。">
                <input
                  required
                  autoFocus
                  maxLength={160}
                  placeholder="例如 shellgame-sft-1015"
                  value={request.name}
                  onChange={(event) => update('name', event.target.value)}
                />
              </Field>
              <div className="form-grid">
                <Field label="计算机器" required>
                  <select
                    required
                    value={request.node_id}
                    onChange={(event) => {
                      update('node_id', event.target.value);
                      update('inputs', {});
                    }}
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
                <Field label="训练模板" required>
                  <select
                    required
                    value={request.template_id}
                    onChange={(event) => update('template_id', event.target.value)}
                  >
                    <option value="">选择模板</option>
                    {templates.data?.map((item) => (
                      <option key={item.id} value={item.id} disabled={item.status !== 'AVAILABLE'}>
                        {item.name}
                        {item.status !== 'AVAILABLE' ? ' · 待接入' : ''}
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
              {template && (
                <div className={`template-note ${template.id === 'smoke' ? 'amber' : ''}`}>
                  <ShieldCheck size={17} />
                  <p>{template.description}</p>
                </div>
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
              {(!isRecap || trainingStage) && (
                <>
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
                  )}
                  {selectedProfile && template?.id !== 'smoke' && (
                    <>
                      {isRecap && trainingStage?.recipe_prefixes?.length && (
                        <p className="field-hint">
                          本阶段配方名称参考：{trainingStage.recipe_prefixes.join('、')}。Agent
                          将核对配方实际模型与数据配置。
                        </p>
                      )}
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
                        node={node}
                        profile={selectedProfile}
                      />
                    </>
                  )}
                </>
              )}
              {selectedProfile && template?.id === 'smoke' && (
                <div className="inline-note">平台链路测试自动使用 smoke 配方，无需填写。</div>
              )}
            </>
          ) : (
            <>
              <div className="selection-summary">
                <Server size={16} />
                <strong>{node?.name}</strong>
                <span>→</span>
                <span>{template?.name}</span>
                {trainingStage && (
                  <>
                    <span>→</span>
                    <strong>{trainingStage.name}</strong>
                  </>
                )}
              </div>
              {isRecap && (
                <div className="inline-note amber">
                  {request.training_stage === 'value'
                    ? '价值函数训练：数据需包含 value_target 等价值监督字段，选择初始骨干或价值模型权重。'
                    : '离线强化学习：使用已完成优势标注的数据集（如 is_positive），并选择初始策略权重。价值模型 checkpoint 用于前序标注，不替代策略权重。'}
                </div>
              )}
              {inputRequirements.length ? (
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
              {hasLocalTraining && (
                <>
                  <DatasetBindings
                    bindings={request.dataset_bindings || []}
                    slots={recipeDatasets}
                    assets={localAssets}
                    node={node}
                    profile={selectedProfile}
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
                  <Field label="代码项目基地址" hint="训练脚本入口固定在所选运行环境的代码项目内。">
                    <input
                      className="mono"
                      readOnly
                      value={selectedProfile?.code_dir || ''}
                      aria-label="代码项目基地址"
                    />
                  </Field>
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
              {isRecap && trainingStage && (
                <>
                  <div className="form-section-title">
                    <Cpu size={16} />
                    <h3>{trainingStage.name}参数</h3>
                    <span>仅用于当前阶段</span>
                  </div>
                  {stageRecipeInfo?.mixedDropout && (
                    <div className="inline-note amber">
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
                  {request.training_stage === 'value' && (
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
                <h3>训练参数</h3>
                <span>
                  {hasLocalTraining
                    ? '已解析的配方值自动填入；未解析项请填写本次值'
                    : '模板已填入推荐默认值'}
                </span>
              </div>
              <div className="form-grid parameters">
                {Object.entries(parameterSpecs).map(([key, spec]) => (
                  <Field
                    key={key}
                    label={spec.label || key}
                    required
                    hint={
                      hasLocalTraining
                        ? Object.hasOwn(recipeDefaults, key)
                          ? `配方值：${recipeDefaults[key]}${editedParameters.current.has(key) ? '；当前为本次覆盖' : ''}`
                          : '无法静态读取，请明确填写本次训练值。'
                        : undefined
                    }
                  >
                    <input
                      type="number"
                      required
                      min={spec.min ?? spec.minimum}
                      max={spec.max ?? spec.maximum}
                      step={spec.type === 'integer' ? 1 : 'any'}
                      value={
                        Number.isFinite(request.parameters[key]) ? request.parameters[key] : ''
                      }
                      onChange={(event) => {
                        editedParameters.current.add(key);
                        update('parameters', {
                          ...request.parameters,
                          [key]: event.target.value === '' ? NaN : Number(event.target.value),
                        });
                      }}
                    />
                  </Field>
                ))}
                <Field
                  label="申请 GPU 数量"
                  hint={minGpus === 0 ? '测试模板支持 0 GPU。' : '资源不足时将排队等待。'}
                >
                  <input
                    type="number"
                    required
                    min={minGpus}
                    max={Math.max(minGpus, node?.gpus?.length || 0)}
                    step={1}
                    value={request.gpu_count}
                    onChange={(event) => update('gpu_count', Number(event.target.value))}
                  />
                </Field>
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
                  ? 'Agent 自动预检通过，可以提交训练'
                  : requestFailed
                    ? '预检请求未完成'
                    : previewExpired
                      ? '预检结果已过期'
                      : preview?.status === 'FAILED'
                        ? '预检未通过'
                        : progressUnavailable
                          ? '暂时无法读取预检进度'
                          : busy || preview?.status === 'PENDING'
                            ? 'Agent 正在自动检查配方与本机资产'
                            : '尚未发起预检'}
              </h3>
              <p>
                {preview?.status === 'READY'
                  ? '已检查配方配置与本机资产，尚未执行训练。启动前会再次核对代码与资源。'
                  : requestFailed
                    ? '尚未取得预检记录，请根据错误提示修正后重新预检。'
                    : previewExpired
                      ? '请重新预检，获取当前代码、资产与运行环境的检查结果。'
                      : preview?.status === 'FAILED'
                        ? '请按失败项修正配置、资产或运行环境，然后重新预检。'
                        : progressUnavailable
                          ? '可重试读取进度；当前显示的结果可能尚未更新。'
                          : '自动检查配方、数据集、模型、归一化参数、输出空间、GPU 与实际执行代码。'}
              </p>
            </div>
          </div>
          {preview?.checks?.length ? (
            <div className="preflight-checks">
              {preview.checks.map((check, index) => (
                <div className={check.ok ? 'ok' : 'bad'} key={`${check.name}-${index}`}>
                  {check.ok ? <CheckCircle2 size={17} /> : <XCircle size={17} />}
                  <strong>{preflightCheckLabels[check.name] || check.name}</strong>
                  <span>{check.message}</span>
                </div>
              ))}
            </div>
          ) : (
            preview?.status === 'PENDING' &&
            !progressUnavailable && (
              <div className="preflight-wait">
                等待机器回传预检结果。机器在线时会自动继续；可关闭弹窗后重新发起。
              </div>
            )
          )}
          {preview?.error && (
            <div className="inline-note red">
              <strong>{preview.error.code}</strong>
              <span>{preview.error.message}</span>
            </div>
          )}
          {automaticPreflight?.scope === 'configuration_and_local_assets' && (
            <dl className="detail-list inset">
              <div>
                <dt>自动预检范围</dt>
                <dd>配方配置与本机资产检查；训练效果需启动后观察。</dd>
              </div>
              <div>
                <dt>训练输入来源</dt>
                <dd>
                  {automaticPreflight.plan.input_authority === 'registered_assets'
                    ? '本次选择的已登记资产，已核对实际路径与资产身份。'
                    : '缺少资产绑定证明，请更新 Agent 后重新预检。'}
                </dd>
              </div>
              <div>
                <dt>实际训练入口</dt>
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
              <dt>训练配方</dt>
              <dd>
                {frozenRecipe && (
                  <span className="recipe-frozen-label">
                    {frozenRecipe.name} · v{frozenRecipe.revision}
                  </span>
                )}
                <code>
                  {preview?.request?.recipe ||
                    preview?.code_snapshot?.recipe ||
                    request.recipe?.trim() ||
                    '未选择'}
                </code>
              </dd>
            </div>
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
              <dd>{request.gpu_count} 张</dd>
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
                        request.entrypoint}
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
                bindings={preview?.request?.dataset_bindings || request.dataset_bindings}
                assets={assets.data || []}
              />
            </div>
          )}
          {(!preview || ['FAILED', 'EXPIRED'].includes(preview.status)) && !busy && (
            <button className="button outline" onClick={() => void beginPreview()}>
              <RotateCcw size={15} />
              重新预检
            </button>
          )}
        </div>
      )}
    </Modal>
  );
}
