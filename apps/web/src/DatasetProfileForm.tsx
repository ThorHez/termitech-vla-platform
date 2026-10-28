import { useEffect, useId, useRef, useState } from 'react';
import { LoaderCircle, Plus, Trash2 } from 'lucide-react';
import { api } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Field, Modal } from './components';
import { useData } from './data';
import type { Asset } from './types';
import { DatasetFieldLabelsEditor } from './DatasetFieldLabelsEditor';
import {
  fieldAnnotationErrors,
  normalizedFieldAnnotations,
  type DatasetFieldAnnotation,
} from './datasetFieldAnnotations';
import { RobotTypeField } from './RobotTypeField';
import { DatasetInspection, useDatasetInspectionDraft } from './DatasetInspection';
import {
  composedProcessingErrors,
  normalizeComposedProcessing,
  type ComposedProcessingSpec,
} from './composedProcessing';
import './dataset-profile.css';

export interface DatasetCamera {
  name: string;
  key: string;
  kind: 'rgb' | 'depth';
}

export type { DatasetFieldAnnotation } from './datasetFieldAnnotations';

export interface DatasetProfile {
  schema_version: 1;
  format: 'lerobot' | 'rlds' | 'other';
  robot_type: string;
  arm_mode: 'single' | 'bimanual';
  action_representation: 'eef' | 'joint' | 'unknown';
  cameras: DatasetCamera[];
  task_description: string;
  action_dim?: number;
  state_dim?: number;
  fps?: number;
  action_mode: 'absolute' | 'delta' | 'unknown';
  rotation_representation: 'axis_angle' | 'quaternion' | 'rotation_6d' | 'none' | 'unknown';
  feature_keys?: string[];
  field_annotations?: DatasetFieldAnnotation[];
  processing?: ComposedProcessingSpec;
}

export function emptyDatasetProfile(): DatasetProfile {
  return {
    schema_version: 1,
    format: 'lerobot',
    robot_type: '',
    arm_mode: 'single',
    action_representation: 'unknown',
    cameras: [{ name: '', key: '', kind: 'rgb' }],
    task_description: '',
    action_mode: 'unknown',
    rotation_representation: 'unknown',
    feature_keys: [],
    field_annotations: [],
  };
}

function textError(value: unknown, maximum: number, multiline = false) {
  return (
    typeof value !== 'string' ||
    !value.trim() ||
    Array.from(value.trim()).length > maximum ||
    (multiline ? /[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/ : /[\u0000-\u001f\u007f-\u009f]/).test(
      value,
    )
  );
}

export function datasetProfileErrors(profile: DatasetProfile): string[] {
  const errors: string[] = [];
  if (profile.schema_version !== 1) errors.push('数据特征版本不支持，请重新填写。');
  if (!['lerobot', 'rlds', 'other'].includes(profile.format)) errors.push('请选择数据格式。');
  if (textError(profile.robot_type, 100)) errors.push('请填写机器人类型，最多 100 个字符。');
  if (!['single', 'bimanual'].includes(profile.arm_mode)) errors.push('请选择单臂或双臂。');
  if (!['eef', 'joint', 'unknown'].includes(profile.action_representation))
    errors.push('请选择动作表示。');
  if (textError(profile.task_description, 2000, true))
    errors.push('请填写任务描述，最多 2000 个字符。');
  if (!['absolute', 'delta', 'unknown'].includes(profile.action_mode))
    errors.push('动作模式无效。');
  if (
    !['axis_angle', 'quaternion', 'rotation_6d', 'none', 'unknown'].includes(
      profile.rotation_representation,
    )
  )
    errors.push('旋转表示无效。');
  for (const field of ['action_dim', 'state_dim'] as const) {
    const value = profile[field];
    if (value !== undefined && (!Number.isInteger(value) || value < 1 || value > 512))
      errors.push(`${field === 'action_dim' ? '动作' : '状态'}维度需要是 1 到 512 的整数。`);
  }
  if (
    profile.fps !== undefined &&
    (!Number.isFinite(profile.fps) || profile.fps <= 0 || profile.fps > 1000)
  )
    errors.push('采样率需要大于 0 且不超过 1000。');
  if (!Array.isArray(profile.cameras) || profile.cameras.length < 1 || profile.cameras.length > 16)
    errors.push('请登记 1 到 16 个视觉输入。');
  else {
    for (const [index, camera] of profile.cameras.entries()) {
      if (
        !camera ||
        textError(camera.name, 100) ||
        textError(camera.key, 256) ||
        !['rgb', 'depth'].includes(camera.kind)
      )
        errors.push(`请完整填写视觉输入 ${index + 1} 的名称、真实字段名和图像类型。`);
    }
    if (
      new Set(profile.cameras.map((camera) => camera?.name?.trim())).size !== profile.cameras.length
    )
      errors.push('视觉输入名称不能重复。');
    if (
      new Set(profile.cameras.map((camera) => camera?.key?.trim())).size !== profile.cameras.length
    )
      errors.push('视觉字段名不能重复。');
  }
  const features = profile.feature_keys;
  if (
    features !== undefined &&
    (!Array.isArray(features) ||
      features.length > 256 ||
      features.some((key) => textError(key, 256)))
  )
    errors.push('数据字段最多 256 项，每项最多 256 个字符。');
  else if (features && new Set(features.map((key) => key.trim())).size !== features.length)
    errors.push('数据字段名不能重复。');
  errors.push(...fieldAnnotationErrors(profile.field_annotations));
  if (profile.processing !== undefined) {
    const compositionErrors = composedProcessingErrors(profile.processing);
    errors.push(...compositionErrors);
    if (profile.format !== 'lerobot') errors.push('组件组合目前只支持 LeRobot 数据。');
  }
  return errors;
}

export function normalizedDatasetProfile(profile: DatasetProfile): DatasetProfile {
  return {
    ...profile,
    robot_type: profile.robot_type.trim(),
    task_description: profile.task_description.trim(),
    cameras: profile.cameras.map((camera) => ({
      ...camera,
      name: camera.name.trim(),
      key: camera.key.trim(),
    })),
    feature_keys: (profile.feature_keys || []).map((key) => key.trim()),
    ...(profile.field_annotations
      ? { field_annotations: normalizedFieldAnnotations(profile.field_annotations) }
      : {}),
    ...(profile.processing ? { processing: normalizeComposedProcessing(profile.processing) } : {}),
  };
}

export function readDatasetProfile(value: unknown): DatasetProfile | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const allowed = new Set([
    'schema_version',
    'format',
    'robot_type',
    'arm_mode',
    'action_representation',
    'cameras',
    'task_description',
    'action_dim',
    'state_dim',
    'fps',
    'action_mode',
    'rotation_representation',
    'feature_keys',
    'field_annotations',
    'processing',
  ]);
  if (Object.keys(value).some((key) => !allowed.has(key))) return undefined;
  const profile = {
    action_mode: 'unknown',
    rotation_representation: 'unknown',
    feature_keys: [],
    ...value,
  } as unknown as DatasetProfile;
  try {
    if (
      profile.cameras.some(
        (camera) =>
          !camera || Object.keys(camera).some((key) => !['name', 'key', 'kind'].includes(key)),
      )
    )
      return undefined;
    return datasetProfileErrors(profile).length ? undefined : normalizedDatasetProfile(profile);
  } catch {
    return undefined;
  }
}

export function datasetProfileSummary(profile: DatasetProfile): string {
  return `${profile.robot_type} · ${profile.arm_mode === 'single' ? '单臂' : '双臂'} · ${profile.action_representation === 'unknown' ? '表示待确认' : profile.action_representation === 'eef' ? 'EEF 末端' : 'Joint 关节'} · ${profile.cameras.length} 个视觉输入`;
}

export function applyComposedProcessing(
  profile: DatasetProfile,
  processing: ComposedProcessingSpec,
): DatasetProfile {
  const previous = profile.processing;
  const sum = (rows: { width: number }[]) =>
    rows.every((row) => Number.isInteger(row.width) && row.width > 0)
      ? rows.reduce((total, row) => total + row.width, 0)
      : undefined;
  const cameras = (rows: ComposedProcessingSpec['cameras']) =>
    JSON.stringify(rows.map((camera) => [camera.key.trim(), camera.kind]).sort());
  const cameraSetChanged = !previous || cameras(previous.cameras) !== cameras(processing.cameras);
  const changed = <K extends keyof ComposedProcessingSpec>(key: K) =>
    !previous || previous[key] !== processing[key];
  return {
    ...profile,
    processing,
    // Only this edit's semantic changes may update the shared description.
    // Unrelated edits must retain a draft made in the basic fields above.
    robot_type:
      !previous || previous.robot_model.trim() !== processing.robot_model.trim()
        ? processing.robot_model
        : profile.robot_type,
    arm_mode: changed('arm_mode') ? processing.arm_mode : profile.arm_mode,
    action_representation: changed('action_representation')
      ? processing.action_representation
      : profile.action_representation,
    action_mode:
      changed('action_mode') && (processing.action_mode === 'absolute' || processing.action_mode === 'delta')
        ? processing.action_mode
        : profile.action_mode,
    rotation_representation:
      changed('rotation_representation') &&
      (processing.rotation_representation === 'axis_angle' ||
        processing.rotation_representation === 'quaternion' ||
        processing.rotation_representation === 'rotation_6d' ||
        processing.rotation_representation === 'none')
        ? processing.rotation_representation
        : profile.rotation_representation,
    action_dim:
      !previous || sum(previous.action_layout) !== sum(processing.action_layout)
        ? sum(processing.action_layout)
        : profile.action_dim,
    state_dim:
      !previous || sum(previous.state_fields) !== sum(processing.state_fields)
        ? sum(processing.state_fields)
        : profile.state_dim,
    cameras: cameraSetChanged
      ? processing.cameras.map((camera, index) => {
          const oldCamera = previous?.cameras.find((item) => item.slot === camera.slot);
          return {
            name:
              profile.cameras.find((item) => item.key.trim() === camera.key.trim())?.name ||
              profile.cameras.find((item) => item.key.trim() === oldCamera?.key.trim())?.name ||
              profile.cameras[index]?.name ||
              `视觉输入 ${index + 1}`,
            key: camera.key,
            kind: camera.kind,
          };
        })
      : profile.cameras,
  };
}

export function DatasetProfileForm({
  value,
  onChange,
  onFieldEdit,
  disabled = false,
  compact = false,
  showTaskDescription = true,
}: {
  value: DatasetProfile;
  onChange: (value: DatasetProfile) => void;
  onFieldEdit?: (path: string) => void;
  disabled?: boolean;
  compact?: boolean;
  showTaskDescription?: boolean;
}) {
  const set = <K extends keyof DatasetProfile>(key: K, next: DatasetProfile[K]) => {
    if (key !== 'processing') onFieldEdit?.(key);
    onChange({ ...value, [key]: next });
  };
  const camera = (index: number, next: Partial<DatasetCamera>) =>
    set(
      'cameras',
      value.cameras.map((item, position) => (position === index ? { ...item, ...next } : item)),
    );
  return (
    <fieldset className={`dataset-profile-form${compact ? ' compact' : ''}`} disabled={disabled}>
      <legend
        title={compact ? '默认值均可修改，请按数据集真实字段填写，训练预检会再次核对。' : undefined}
      >
        数据集特征
      </legend>
      <p className="field-hint">
        {compact
          ? '默认值均可修改，请按数据集真实字段填写，训练预检会再次核对。'
          : '登记一次，训练时用于匹配数据处理器。请按数据集的真实字段填写，预检会再次核对。'}
      </p>
      <div className="form-grid dataset-profile-basic">
        <RobotTypeField
          value={value.robot_type}
          onChange={(next) => set('robot_type', next)}
          disabled={disabled}
          hint={compact ? '' : undefined}
        />
        <Field label="机械臂配置" required>
          <select
            value={value.arm_mode}
            onChange={(event) => set('arm_mode', event.target.value as DatasetProfile['arm_mode'])}
          >
            <option value="single">单臂</option>
            <option value="bimanual">双臂</option>
          </select>
        </Field>
        <Field label="数据格式" required>
          <select
            value={value.format}
            onChange={(event) => set('format', event.target.value as DatasetProfile['format'])}
          >
            <option value="lerobot">LeRobot</option>
            <option value="rlds">RLDS</option>
            <option value="other">其他 / 自定义</option>
          </select>
        </Field>
        <Field label="采样频率（Hz）">
          <input
            type="number"
            min={0.001}
            max={1000}
            step="any"
            value={value.fps ?? ''}
            onChange={(event) =>
              set('fps', event.target.value === '' ? undefined : Number(event.target.value))
            }
          />
        </Field>
      </div>
      <div className="dataset-profile-cameras">
        <div className="dataset-profile-section-title">
          <strong>视觉输入</strong>
          <span>{value.cameras.length} 个</span>
          <button
            type="button"
            className="button outline small"
            disabled={disabled || value.cameras.length >= 16}
            onClick={() => set('cameras', [...value.cameras, { name: '', key: '', kind: 'rgb' }])}
          >
            <Plus size={14} />
            添加视角
          </button>
        </div>
        <p className="field-hint">
          {compact
            ? '每行登记一个 RGB 或深度字段；同一相机的 RGB 与深度分开填写。'
            : '分别登记 RGB 或深度字段；同一相机的 RGB 与深度算两个输入。'}
        </p>
        {value.cameras.map((item, index) => (
          <div className="dataset-profile-camera" key={index}>
            <div className="dataset-profile-section-title">
              <strong>视觉输入 {index + 1}</strong>
              <button
                type="button"
                className="icon-button"
                aria-label={`移除视觉输入 ${index + 1}`}
                title="移除视觉输入"
                disabled={disabled || value.cameras.length <= 1}
                onClick={() =>
                  set(
                    'cameras',
                    value.cameras.filter((_, position) => position !== index),
                  )
                }
              >
                <Trash2 size={15} />
              </button>
            </div>
            <div className="form-grid">
              <Field label={`视角名称 ${index + 1}`} required>
                <input
                  required
                  maxLength={100}
                  value={item.name}
                  placeholder="例如 左腕、右腕、头部"
                  onChange={(event) => camera(index, { name: event.target.value })}
                />
              </Field>
              <Field label={`图像类型 ${index + 1}`}>
                <select
                  value={item.kind}
                  onChange={(event) =>
                    camera(index, { kind: event.target.value as DatasetCamera['kind'] })
                  }
                >
                  <option value="rgb">RGB 彩色</option>
                  <option value="depth">深度</option>
                </select>
              </Field>
            </div>
            <Field
              label={`数据字段名 ${index + 1}`}
              required
              hint={
                compact ? undefined : '填写数据文件中的完整键名；相机数量相同，字段映射也可能不同。'
              }
            >
              <input
                required
                maxLength={256}
                value={item.key}
                placeholder="例如 observation.images.left_wrist"
                title="填写数据文件中的完整键名；相机数量相同，字段映射也可能不同。"
                onChange={(event) => camera(index, { key: event.target.value })}
              />
            </Field>
          </div>
        ))}
      </div>
      {showTaskDescription && (
        <Field label="任务描述" required hint="描述机器人执行的任务，可填写多行。">
          <textarea
            required
            maxLength={2000}
            rows={compact ? 2 : 3}
            value={value.task_description}
            placeholder="例如 双臂抓取衣物并折叠，放到桌面指定区域。"
            onChange={(event) => set('task_description', event.target.value)}
          />
        </Field>
      )}
      <DatasetFieldLabelsEditor
        value={value.field_annotations}
        featureKeys={value.feature_keys}
        disabled={disabled}
        onFieldEdit={onFieldEdit}
        onChange={(field_annotations) => onChange({ ...value, field_annotations })}
      />
      {value.processing && (
        <details className="dataset-profile-legacy-processing">
          <summary>历史处理方案</summary>
          <p className="field-hint">
            旧方案已保留，可在新建训练的数据集卡片中导入和修改；当前登记只保存原始字段标签。
          </p>
        </details>
      )}
    </fieldset>
  );
}

export function DatasetProfileEditor(props: { asset: Asset; close: () => void }) {
  const { can } = useAuth();
  if (!can('assets.write'))
    return (
      <Modal title="权限不足" close={props.close}>
        编辑数据特征需要操作员或管理员权限。
      </Modal>
    );
  return <DatasetProfileEditorContent key={props.asset.id} {...props} />;
}

function DatasetProfileEditorContent({ asset, close }: { asset: Asset; close: () => void }) {
  const { assets, nodes } = useData();
  const [value, setValue] = useState(
    () => readDatasetProfile(asset.metadata?.dataset_profile) || emptyDatasetProfile(),
  );
  const [busy, setBusy] = useState(false);
  const inspectionDraft = useDatasetInspectionDraft(
    value,
    setValue,
    !!readDatasetProfile(asset.metadata?.dataset_profile),
  );
  const [error, setError] = useState<Error>();
  const errorArea = useRef<HTMLDivElement>(null);
  const pending = useRef(false);
  const formId = useId();
  const errors = datasetProfileErrors(value);
  useEffect(() => {
    if (error) errorArea.current?.scrollIntoView({ block: 'nearest' });
  }, [error]);
  const safeClose = () => {
    if (!pending.current) close();
  };
  return (
    <Modal
      title="数据集特征"
      subtitle={asset.name}
      close={safeClose}
      footer={
        <>
          <button className="button outline" disabled={busy} onClick={safeClose}>
            取消
          </button>
          <button
            type="submit"
            form={formId}
            className="button primary"
            disabled={busy || errors.length > 0}
          >
            {busy && <LoaderCircle size={15} className="spin" />}
            {busy ? '保存中…' : '保存数据特征'}
          </button>
        </>
      }
    >
      <p className="field-hint">用于后续新建训练的处理器匹配；已创建训练保留原配置。</p>
      <code className="dataset-profile-path">{asset.path}</code>
      <div ref={errorArea}>
        <ErrorNotice error={error} title="数据特征保存失败" />
      </div>
      <form
        id={formId}
        aria-busy={busy}
        onSubmit={async (event) => {
          event.preventDefault();
          if (pending.current || errors.length) return;
          pending.current = true;
          setBusy(true);
          setError(undefined);
          try {
            await api<Asset>(`/assets/${encodeURIComponent(asset.id)}/dataset-profile`, {
              method: 'PATCH',
              body: JSON.stringify({ dataset_profile: normalizedDatasetProfile(value) }),
            });
            assets.refresh();
            close();
          } catch (cause) {
            setError(cause instanceof Error ? cause : new Error(String(cause)));
          } finally {
            pending.current = false;
            setBusy(false);
          }
        }}
      >
        <DatasetInspection
          node={nodes.data?.find((node) => node.id === asset.node_id)}
          path={asset.path}
          value={value}
          draft={inspectionDraft}
          disabled={busy}
        />
        <DatasetProfileForm
          value={value}
          onChange={inspectionDraft.manualChange}
          onFieldEdit={inspectionDraft.touch}
          disabled={busy}
        />
        {errors.length > 0 && (
          <p className="field-hint required" role="status">
            {errors[0]}
          </p>
        )}
      </form>
    </Modal>
  );
}
