import { useEffect, useRef, useState } from 'react';
import { LoaderCircle, RefreshCw, XCircle } from 'lucide-react';
import { emptyDatasetProfile, readDatasetProfile } from './DatasetProfileForm';
import { ComposedProcessingEditor } from './ComposedProcessingEditor';
import { imageLayoutErrors, imageLayoutManualMarker, inferImageLayout, type ImageFeature, type ImageLayoutEdits } from './imageLayout';
import { useDatasetImageMetadata } from './useDatasetImageMetadata';
import {
  composedProcessingErrors,
  composedProcessingSummary,
  normalizeComposedProcessing,
  trainingCompositionDraft,
  trainingCompositionProfileErrors,
  type ComposedProcessingSpec,
  type ProcessingProfileSeed,
} from './composedProcessing';
import type {
  Asset,
  DataProcessorEntry,
  DatasetBinding,
  Node,
  RecipeDatasetSlot,
  RecipeLiteral,
} from './types';

export function compositionToken(value: unknown): string {
  const canonical = (entry: unknown): unknown =>
    Array.isArray(entry)
      ? entry.map(canonical)
      : entry && typeof entry === 'object'
        ? Object.fromEntries(
            Object.entries(entry)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([key, child]) => [key, canonical(child)]),
          )
        : entry;
  return JSON.stringify(canonical(value));
}

export function composedChoice(spec: ComposedProcessingSpec) {
  return {
    name: 'ComposableDataConfig',
    parameters: { composition: normalizeComposedProcessing(spec) as unknown as RecipeLiteral },
  };
}

export function trainingProfileSeed(dataset?: Asset): ProcessingProfileSeed {
  return readDatasetProfile(dataset?.metadata?.dataset_profile) || emptyDatasetProfile();
}

function editableDraft(value: unknown): value is ComposedProcessingSpec {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const spec = value as ComposedProcessingSpec;
  return (
    spec.schema_version === 1 &&
    typeof spec.robot_model === 'string' &&
    (spec.coordinate_frame == null || typeof spec.coordinate_frame === 'string') &&
    typeof spec.action_key === 'string' &&
    (spec.prompt_key == null || typeof spec.prompt_key === 'string') &&
    Array.isArray(spec.state_fields) &&
    spec.state_fields.every(
      (row) => row && typeof row.key === 'string' && typeof row.width === 'number',
    ) &&
    Array.isArray(spec.action_layout) &&
    spec.action_layout.every((row) => row && typeof row.width === 'number') &&
    Array.isArray(spec.cameras) &&
    spec.cameras.every((row) => row && typeof row.key === 'string') &&
    (spec.gripper_range == null ||
      (Array.isArray(spec.gripper_range) && spec.gripper_range.length === 2)) &&
    Array.isArray(spec.aux_keys)
  );
}

export function compositionBindingState({
  binding,
  dataset,
  entries,
  slots,
  index,
  stage,
  imageFeatures,
}: {
  binding: DatasetBinding;
  dataset?: Asset;
  entries: DataProcessorEntry[];
  slots: RecipeDatasetSlot[];
  index: number;
  stage: 'sft' | 'value' | 'offline_rl';
  imageFeatures?: readonly ImageFeature[];
}) {
  const raw =
    binding.processor?.name === 'ComposableDataConfig'
      ? binding.processor.parameters.composition
      : undefined;
  const errors = composedProcessingErrors(raw);
  const registered = readDatasetProfile(dataset?.metadata?.dataset_profile);
  const spec = editableDraft(raw) ? normalizeComposedProcessing(raw) : undefined;
  if (spec && registered && !errors.length)
    errors.push(...trainingCompositionProfileErrors(registered, spec));
  if (spec) errors.push(...spec.cameras.flatMap((camera) => imageLayoutErrors(camera, imageFeatures)));
  if (!dataset) errors.push('请先选择本组训练数据集。');
  const slot =
    slots.find((entry) => entry.recipe_index === binding.recipe_index) || slots[index] || slots[0];
  const modelFamily = entries.find((entry) => entry.name === slot?.processing)?.dataset_contract
    ?.model_family;
  if (modelFamily === 'memory' || modelFamily === 'hybrid')
    errors.push('当前 MEM/Hybrid 配方使用专用变换，标准组件组合不可替代该处理器。');
  const factory = entries.find((entry) => entry.name === 'ComposableDataConfig');
  if (!factory || factory.composition_schema_version !== 1 || !factory.available || !factory.source)
    errors.push(
      factory?.reason ||
        '尚未确认当前训练环境支持版本 1 的 ComposableDataConfig，请读取处理器目录后继续预检。',
    );
  if (spec && stage === 'value' && !spec.aux_keys.includes('value_target'))
    errors.push('价值训练需要在本次处理方案中声明真实 value_target 辅助字段。');
  if (spec && stage === 'offline_rl' && !spec.aux_keys.includes('is_positive'))
    errors.push('离线强化学习需要在本次处理方案中声明真实 is_positive 辅助字段。');
  const expected = spec ? composedChoice(spec) : undefined;
  const matches =
    !!expected &&
    binding.processor?.name === expected.name &&
    Object.keys(binding.processor.parameters).length === 1 &&
    compositionToken(spec) ===
      compositionToken(expected.parameters.composition);
  if (expected && !matches)
    errors.push('处理器参数与当前组合方案不一致，请返回数据配置重新保存处理方案。');
  return {
    spec,
    expected,
    matches,
    errors: [...new Set(errors)],
    modelFamily,
    valid: !errors.length && matches,
  };
}

export function ComposedDatasetProcessing(props: {
  binding: DatasetBinding;
  dataset?: Asset;
  node?: Node;
  imageLayoutEdits?: ImageLayoutEdits;
  entries: DataProcessorEntry[];
  slots: RecipeDatasetSlot[];
  index: number;
  stage: 'sft' | 'value' | 'offline_rl';
  readOnly?: boolean;
  validationToken?: string;
  onChange: (binding: DatasetBinding) => void;
  onValidityChange?: (valid: boolean) => void;
}) {
  const images = useDatasetImageMetadata(props.node, props.dataset);
  const features = images.inspection?.features;
  const state = compositionBindingState({ ...props, imageFeatures: features });
  const profile = { ...trainingProfileSeed(props.dataset), image_features: features };
  const manualLayouts = useRef(new Set<string>());
  const [layoutNotice, setLayoutNotice] = useState<string[]>([]);
  const layoutKey = (key: string, kind: string) => imageLayoutManualMarker([images.scope, props.index, key.trim(), kind]);
  const validity = useRef(props.onValidityChange);
  validity.current = props.onValidityChange;
  useEffect(() => {
    validity.current?.(state.valid && !images.loading);
  }, [state.valid, images.loading, props.validationToken]);
  const update = (spec: ComposedProcessingSpec) => {
    const { recipe_index: _recipe, processor: _processor, ...assets } = props.binding;
    props.onChange({ ...assets, processor: composedChoice(spec) });
  };
  const change = useRef(update);
  change.current = update;
  const cameraSignature = JSON.stringify(state.spec?.cameras);
  useEffect(() => {
    if (props.readOnly || !state.spec || !features) return;
    const changes: string[] = [];
    const cameras = state.spec.cameras.map((camera) => {
      const key = layoutKey(camera.key, camera.kind);
      const evidence = inferImageLayout(features, camera.key, camera.kind);
      if (props.imageLayoutEdits?.has(key) || manualLayouts.current.has(key)) return camera;
      if (!evidence.layout || evidence.layout === camera.layout) return camera;
      changes.push(`${camera.key} [${evidence.shape?.join(' × ')}] → ${evidence.layout.toUpperCase()}`);
      return { ...camera, layout: evidence.layout };
    });
    if (changes.length) {
      setLayoutNotice(changes);
      change.current({ ...state.spec, cameras });
    }
  }, [images.scope, features, cameraSignature, props.readOnly]);
  const legacy = (props.dataset?.metadata?.dataset_profile as { processing?: unknown } | undefined)
    ?.processing;
  const canImport = !composedProcessingErrors(legacy).length;
  return (
    <div className="dataset-processing composed-training-selection">
      <div className="inline-note" role="status">
        <div>
          <strong>
            {images.loading ? '正在读取图像字段并识别布局…' : '图像布局按当前数据集元信息识别'}
          </strong>
          {images.loading && <LoaderCircle size={14} className="spin" />}
          {layoutNotice.length > 0 && (
            <p>已按当前数据集元信息校准：{layoutNotice.join('；')}。当前手工调整会保留。</p>
          )}
          {images.error && <p>{images.error} 可重新读取，或手动核对图像布局。</p>}
          {images.inspection && (
            <p>来源：meta/info.json · {images.inspection.source.info_sha256.slice(0, 12)}；仅填写布局，不自动选择历史帧。</p>
          )}
          {!props.readOnly && <button
            type="button"
            className="button outline small"
            disabled={images.loading || props.node?.status !== 'ONLINE'}
            onClick={() => {
              manualLayouts.current.clear();
              for (const camera of state.spec?.cameras || [])
                props.imageLayoutEdits?.set(layoutKey(camera.key, camera.kind), false);
              setLayoutNotice([]);
              images.refresh();
            }}
          >
            <RefreshCw size={13} />重新识别图像布局
          </button>}
        </div>
      </div>
      <div
        className={`inline-note ${state.valid ? '' : 'red training-validation-error'}`}
        role={state.valid ? 'status' : 'alert'}
      >
        {!state.valid && <XCircle size={16} aria-hidden="true" />}
        <div>
          <strong>{state.valid ? '本次训练组合方案已配置' : '数据处理配置错误，修正后才能预检'}</strong>
          <p>方案随本次训练保存；数据资产的字段标签保持原样。</p>
          {state.spec && !composedProcessingErrors(state.spec).length && (
            <p>{composedProcessingSummary(state.spec)}</p>
          )}
          {state.errors.length > 0 && (
            <ul className="training-validation-errors">
              {state.errors.map((error) => <li key={error}>{error}</li>)}
            </ul>
          )}
        </div>
      </div>
      {!props.readOnly && (
        <>
          <div className="composed-training-actions">
            <button
              type="button"
              className="button outline small"
              onClick={() => update(trainingCompositionDraft(profile))}
            >
              从登记字段标签重建草稿
            </button>
            {canImport && (
              <button
                type="button"
                className="button outline small"
                onClick={() => update(legacy as ComposedProcessingSpec)}
              >
                导入资产中的历史组合方案
              </button>
            )}
          </div>
          {state.spec ? (
            <ComposedProcessingEditor
              value={state.spec}
              profile={profile}
              onChange={update}
              onImageLayoutEdit={(key, kind) => {
                const marker = layoutKey(key, kind);
                manualLayouts.current.add(marker);
                props.imageLayoutEdits?.set(marker, true);
              }}
            />
          ) : (
            <p className="field-hint">
              选择数据集后，可从字段标签创建可编辑的组合草稿；未知字段宽度需要确认，编码声明与任务字段可选填。
            </p>
          )}
        </>
      )}
      {props.readOnly && state.spec && (
        <div className="composed-summary-lines">
          <p>
            状态：
            {state.spec.state_fields.map((field) => `${field.key} [${field.width}]`).join(' → ')}
          </p>
          <p>
            动作：{state.spec.action_key} ·{' '}
            {state.spec.action_layout
              .map((block) => `${block.arm}/${block.kind} [${block.width}]`)
              .join(' → ')}
          </p>
          <p>
            视觉：{state.spec.cameras.map((camera) => `${camera.key} → ${camera.slot}`).join('；')}
          </p>
        </div>
      )}
    </div>
  );
}
