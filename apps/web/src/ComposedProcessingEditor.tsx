import { ArrowDown, ArrowUp, Plus, Trash2, XCircle } from 'lucide-react';
import { useState } from 'react';
import { Field } from './components';
import { RobotTypeField } from './RobotTypeField';
import { inferImageLayout } from './imageLayout';
import {
  composedProcessingErrors,
  composedProcessingSummary,
  draftComposedProcessing,
  inferSourceActionLayout,
  umiV4ComposedPreset,
  updateComposedProcessingFromProfile,
  VISUAL_SLOTS,
  type ActionBlock,
  type ActionKind,
  type ComposedCamera,
  type ComposedProcessingSpec,
  type ProcessingProfileSeed,
  type StateField,
} from './composedProcessing';
import './composed-processing.css';

export type { ComposedProcessingSpec } from './composedProcessing';
export {
  composedProcessingErrors,
  normalizeComposedProcessing,
  draftComposedProcessing,
  composedProcessingSummary,
} from './composedProcessing';

const kinds = {
  position: '位置',
  rotation: '旋转',
  joint: '关节',
  gripper: '夹爪',
  other: '其他状态',
};
const arms = { left: '左臂', right: '右臂', shared: '共享状态' };
const slotLabels = {
  base_0_rgb: '主视角 · base_0_rgb',
  left_wrist_0_rgb: '左腕 · left_wrist_0_rgb',
  right_wrist_0_rgb: '右腕 · right_wrist_0_rgb',
};
const number = (value: string) => (value === '' ? NaN : Number(value));
const shown = (value: number) =>
  Number.isFinite(value) && value !== 0 ? value : Number.isFinite(value) ? 0 : '';
function move<T>(rows: T[], index: number, offset: number): T[] {
  const next = [...rows],
    target = index + offset;
  if (target < 0 || target >= rows.length) return rows;
  [next[index], next[target]] = [next[target], next[index]];
  return next;
}

export function ComposedProcessingEditor({
  value,
  profile,
  onChange,
  onFieldEdit,
  onImageLayoutEdit,
  disabled = false,
}: {
  value: ComposedProcessingSpec;
  profile: ProcessingProfileSeed;
  onChange: (value: ComposedProcessingSpec) => void;
  onFieldEdit?: (path: string) => void;
  onImageLayoutEdit?: (key: string, kind: ComposedCamera['kind']) => void;
  disabled?: boolean;
}) {
  const set = <K extends keyof ComposedProcessingSpec>(key: K, next: ComposedProcessingSpec[K]) => {
    onFieldEdit?.('processing.' + key);
    onChange({ ...value, [key]: next });
  };
  const state = (index: number, next: Partial<StateField>) =>
    set(
      'state_fields',
      value.state_fields.map((field, row) => (row === index ? { ...field, ...next } : field)),
    );
  const action = (index: number, next: Partial<ActionBlock>) =>
    set(
      'action_layout',
      value.action_layout.map((block, row) => (row === index ? { ...block, ...next } : block)),
    );
  const camera = (index: number, next: Partial<ComposedCamera>) =>
    set(
      'cameras',
      value.cameras.map((field, row) => (row === index ? { ...field, ...next } : field)),
    );
  const errors = composedProcessingErrors(value);
  const sourceAction = inferSourceActionLayout(profile, value.action_key);
  const actionWidth = value.action_layout.reduce(
    (sum, block) => sum + (Number.isFinite(block.width) ? block.width : 0),
    0,
  );
  const actionWidthMismatch =
    sourceAction.width !== undefined && sourceAction.width !== actionWidth;
  const [advanced, setAdvanced] = useState(() => errors.length > 0);
  return (
    <fieldset className="composed-processing-editor" disabled={disabled}>
      <legend>组件组合处理</legend>
      <p className="field-hint">
        将真实数据字段组合成标准 Pi0/Pi05 输入。型号只作说明，不会推测关节自由度、执行 IK
        或转换动作单位/坐标系。MEM、Hybrid 继续使用专用处理器。
      </p>
      <div className="composed-presets">
        <button
          type="button"
          className="button outline small"
          onClick={() => {
            onFieldEdit?.('processing');
            setAdvanced(true);
            onChange(umiV4ComposedPreset(profile, 'single'));
          }}
        >
          填入 UMI V4 单臂结构样例
        </button>
        <button
          type="button"
          className="button outline small"
          onClick={() => {
            onFieldEdit?.('processing');
            setAdvanced(true);
            onChange(umiV4ComposedPreset(profile, 'bimanual'));
          }}
        >
          填入 UMI V4 双臂结构样例
        </button>
        <button
          type="button"
          className="button outline small"
          onClick={() => {
            onFieldEdit?.('processing');
            setAdvanced(true);
            onChange(updateComposedProcessingFromProfile(profile, value));
          }}
        >
          用基础描述更新组合草稿
        </button>
        <button
          type="button"
          className="button ghost small"
          onClick={() => {
            onFieldEdit?.('processing');
            setAdvanced(true);
            onChange(draftComposedProcessing(profile));
          }}
        >
          从当前登记信息重建空草稿
        </button>
      </div>
      <p className="field-hint">
        样例会替换状态顺序与动作布局；保留当前相机字段。robot0/robot1
        按样例映射左/右，请核对实际机械臂、字段、历史长度、编码和单位。
      </p>
      <p className="field-hint">
        更新基础描述只同步型号、臂数、动作声明和相机字段，保留状态/动作分块与已有视觉映射；不匹配的宽度或编码仍须手动核对。
      </p>
      <div className="form-grid">
        <RobotTypeField
          label="组合机器人型号"
          hint="可选择其他型号或自定义；不会根据型号预填关节宽度。"
          value={value.robot_model}
          onChange={(next) => set('robot_model', next)}
          disabled={disabled}
        />
        <Field label="组合机械臂配置" required>
          <select
            value={value.arm_mode}
            onChange={(event) =>
              set('arm_mode', event.target.value as ComposedProcessingSpec['arm_mode'])
            }
          >
            <option value="single">单臂</option>
            <option value="bimanual">双臂</option>
          </select>
        </Field>
        <Field label="状态表示" required>
          <select
            value={value.state_representation}
            onChange={(event) =>
              set(
                'state_representation',
                event.target.value as ComposedProcessingSpec['state_representation'],
              )
            }
          >
            <option value="eef">EEF · 末端状态</option>
            <option value="joint">Joint · 关节状态</option>
          </select>
        </Field>
        <Field label="组合动作表示" required>
          <select
            value={value.action_representation}
            onChange={(event) =>
              set(
                'action_representation',
                event.target.value as ComposedProcessingSpec['action_representation'],
              )
            }
          >
            <option value="eef">EEF · 末端动作</option>
            <option value="joint">Joint · 关节动作</option>
          </select>
        </Field>
      </div>
      <section aria-label="状态字段拼接">
        <div className="composed-section-title">
          <strong>状态字段拼接顺序</strong>
          <span>
            每个时刻{' '}
            {value.state_fields.reduce(
              (sum, field) => sum + (Number.isFinite(field.width) ? field.width : 0),
              0,
            )}{' '}
            维
          </span>
          <button
            type="button"
            className="button outline small"
            disabled={disabled || value.state_fields.length >= 32}
            onClick={() =>
              set('state_fields', [
                ...value.state_fields,
                { key: '', width: 0, arm: 'shared', kind: 'other', normalize: true },
              ])
            }
          >
            <Plus size={14} />
            添加状态字段
          </button>
        </div>
        <p className="field-hint">
          按表格顺序拼接；宽度是该字段最后一维的真实长度，不能从机器人名称猜测。共享状态和相对位姿字段可独立添加。
        </p>
        {value.state_fields.map((field, index) => (
          <div className="composed-row" key={index}>
            <div className="composed-row-title">
              <strong>状态字段 {index + 1}</strong>
              <div>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`上移状态字段 ${index + 1}`}
                  disabled={disabled || index === 0}
                  onClick={() => set('state_fields', move(value.state_fields, index, -1))}
                >
                  <ArrowUp size={14} />
                </button>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`下移状态字段 ${index + 1}`}
                  disabled={disabled || index === value.state_fields.length - 1}
                  onClick={() => set('state_fields', move(value.state_fields, index, 1))}
                >
                  <ArrowDown size={14} />
                </button>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`移除状态字段 ${index + 1}`}
                  disabled={disabled || value.state_fields.length === 1}
                  onClick={() =>
                    set(
                      'state_fields',
                      value.state_fields.filter((_, row) => row !== index),
                    )
                  }
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
            <Field label={`状态真实字段 ${index + 1}`} required>
              <input
                required
                maxLength={256}
                value={field.key}
                placeholder="填写数据文件中的实际键名"
                onChange={(event) => state(index, { key: event.target.value })}
              />
            </Field>
            <div className="composed-grid-three">
              <Field label={`状态宽度 ${index + 1}`} required>
                <input
                  required
                  type="number"
                  min={1}
                  max={128}
                  step={1}
                  value={field.width || ''}
                  onChange={(event) => state(index, { width: number(event.target.value) })}
                />
              </Field>
              <Field label={`状态归属 ${index + 1}`}>
                <select
                  value={field.arm}
                  onChange={(event) =>
                    state(index, { arm: event.target.value as StateField['arm'] })
                  }
                >
                  {Object.entries(arms).map(([name, label]) => (
                    <option value={name} key={name}>
                      {label}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`状态类型 ${index + 1}`}>
                <select
                  value={field.kind}
                  onChange={(event) =>
                    state(index, { kind: event.target.value as StateField['kind'] })
                  }
                >
                  {Object.entries(kinds).map(([name, label]) => (
                    <option value={name} key={name}>
                      {label}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={field.normalize}
                onChange={(event) => state(index, { normalize: event.target.checked })}
              />
              归一化状态字段 {index + 1}
            </label>
          </div>
        ))}
      </section>
      <section aria-label="动作分块布局">
        <div className="composed-section-title">
          <strong>动作向量分块顺序</strong>
          <span>当前分块合计 {actionWidth} 维</span>
          <button
            type="button"
            className="button outline small"
            disabled={disabled || value.action_layout.length >= 16}
            onClick={() =>
              set('action_layout', [
                ...value.action_layout,
                {
                  arm: 'left',
                  kind: value.action_representation === 'joint' ? 'joint' : 'position',
                  width: 0,
                  normalize: true,
                },
              ])
            }
          >
            <Plus size={14} />
            添加动作分块
          </button>
        </div>
        <Field label="动作真实字段" required>
          <input
            required
            maxLength={256}
            value={value.action_key}
            onChange={(event) => set('action_key', event.target.value)}
          />
        </Field>
        <div className="composed-action-source">
          <Field
            label="数据集动作总宽度"
            hint={
              sourceAction.width !== undefined
                ? `依据登记字段形状 [${sourceAction.shape?.join(', ')}] 的末轴读取。`
                : '选择已登记形状的动作字段后自动读取。'
            }
          >
            <input readOnly value={sourceAction.width ?? ''} placeholder="待读取" />
          </Field>
          <button
            type="button"
            className="button outline small"
            disabled={disabled || !sourceAction.blocks?.length}
            title="依据明确的坐标名称或字段标签替换当前动作分块；其余配置保留。"
            onClick={() => {
              if (sourceAction.blocks?.length)
                set(
                  'action_layout',
                  sourceAction.blocks.map((block) => ({ ...block })),
                );
            }}
          >
            从数据集填入动作分块
          </button>
        </div>
        {sourceAction.shape?.length === 2 && sourceAction.width !== undefined && (
          <p className="field-hint">
            每条样本已有 {sourceAction.shape[0]} 步动作，每步 {sourceAction.width}{' '}
            维。这里配置的是每步的分块宽度。
          </p>
        )}
        {sourceAction.reason && <p className="field-hint">{sourceAction.reason}</p>}
        {sourceAction.width !== undefined && (
          <p
            className={`composed-action-width-status${actionWidthMismatch ? ' mismatch' : ''}`}
            role={actionWidthMismatch ? 'alert' : 'status'}
          >
            {actionWidthMismatch
              ? `当前分块合计 ${actionWidth} 维，与原始动作总宽度 ${sourceAction.width} 维不一致，请调整各分块宽度。`
              : `当前分块合计 ${actionWidth} 维，与原始动作总宽度一致。`}
          </p>
        )}
        <p className="field-hint">
          分块从动作向量第 0
          维依次排列。总宽度来自原始动作形状；分块有明确的坐标名称或字段标签时自动填入，无法识别时手动配置。双臂须分别声明左右臂。
        </p>
        {value.action_layout.map((block, index) => (
          <div className="composed-row" key={index}>
            <div className="composed-row-title">
              <strong>动作分块 {index + 1}</strong>
              <div>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`上移动作分块 ${index + 1}`}
                  disabled={disabled || index === 0}
                  onClick={() => set('action_layout', move(value.action_layout, index, -1))}
                >
                  <ArrowUp size={14} />
                </button>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`下移动作分块 ${index + 1}`}
                  disabled={disabled || index === value.action_layout.length - 1}
                  onClick={() => set('action_layout', move(value.action_layout, index, 1))}
                >
                  <ArrowDown size={14} />
                </button>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`移除动作分块 ${index + 1}`}
                  disabled={disabled || value.action_layout.length === 1}
                  onClick={() =>
                    set(
                      'action_layout',
                      value.action_layout.filter((_, row) => row !== index),
                    )
                  }
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
            <div className="composed-grid-three">
              <Field label={`动作归属 ${index + 1}`}>
                <select
                  value={block.arm}
                  onChange={(event) =>
                    action(index, { arm: event.target.value as ActionBlock['arm'] })
                  }
                >
                  <option value="left">左臂</option>
                  <option value="right">右臂</option>
                </select>
              </Field>
              <Field label={`动作类型 ${index + 1}`}>
                <select
                  value={block.kind}
                  onChange={(event) => action(index, { kind: event.target.value as ActionKind })}
                >
                  {Object.entries(kinds)
                    .filter(([name]) => name !== 'other')
                    .map(([name, label]) => (
                      <option value={name} key={name}>
                        {label}
                      </option>
                    ))}
                </select>
              </Field>
              <Field
                label={`动作宽度 ${index + 1}`}
                required
                hint={
                  value.action_representation === 'eef' && block.kind === 'position'
                    ? 'EEF 位置分块必须为 3 维。'
                    : value.action_representation === 'eef' && block.kind === 'rotation'
                      ? '旋转分块宽度需与所选旋转编码一致。'
                      : '当前分块占用的维数。'
                }
              >
                <input
                  required
                  type="number"
                  min={1}
                  max={Math.min(128, sourceAction.width ?? 128)}
                  step={1}
                  value={block.width || ''}
                  onChange={(event) => action(index, { width: number(event.target.value) })}
                />
              </Field>
            </div>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={block.normalize}
                onChange={(event) => action(index, { normalize: event.target.checked })}
              />
              归一化动作分块 {index + 1}
            </label>
          </div>
        ))}
      </section>
      <section aria-label="模型视觉槽位">
        <div className="composed-section-title">
          <strong>视觉输入与模型槽位</strong>
          <span>{value.cameras.length} 路 / 最多 3 路</span>
          <button
            type="button"
            className="button outline small"
            disabled={disabled || value.cameras.length >= 3}
            onClick={() =>
              set('cameras', [
                ...value.cameras,
                {
                  key: '',
                  slot:
                    VISUAL_SLOTS.find(
                      (slot) => !value.cameras.some((camera) => camera.slot === slot),
                    ) || 'base_0_rgb',
                  kind: 'rgb',
                  layout: '',
                  time_index: null,
                  depth_min: 0,
                  depth_max: 1,
                },
              ])
            }
          >
            <Plus size={14} />
            添加组合视觉输入
          </button>
        </div>
        <p className="field-hint">
          每个模型槽位最多绑定一个真实字段。深度仅按显式显示范围映射为图像，不代表生成新的视角或转换测量单位。
        </p>
        {value.cameras.map((item, index) => (
          <div className="composed-row" key={index}>
            <div className="composed-row-title">
              <strong>组合视觉输入 {index + 1}</strong>
              <button
                type="button"
                className="icon-button"
                aria-label={`移除组合视觉输入 ${index + 1}`}
                disabled={disabled || value.cameras.length === 1}
                onClick={() =>
                  set(
                    'cameras',
                    value.cameras.filter((_, row) => row !== index),
                  )
                }
              >
                <Trash2 size={14} />
              </button>
            </div>
            <Field label={`组合视觉字段 ${index + 1}`} required>
              <input
                required
                maxLength={256}
                value={item.key}
              onChange={(event) => {
                const key = event.target.value;
                const inferred = inferImageLayout(profile.image_features, key, item.kind);
                camera(index, { key, layout: inferred.layout || '' });
              }}
              />
            </Field>
            <div className="form-grid">
              <Field label={`模型视觉槽位 ${index + 1}`}>
                <select
                  value={item.slot}
                  onChange={(event) =>
                    camera(index, { slot: event.target.value as ComposedCamera['slot'] })
                  }
                >
                  {VISUAL_SLOTS.map((slot) => (
                    <option value={slot} key={slot}>
                      {slotLabels[slot]}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`组合图像类型 ${index + 1}`}>
                <select
                  value={item.kind}
                  onChange={(event) =>
                    camera(index, { kind: event.target.value as ComposedCamera['kind'] })
                  }
                >
                  <option value="rgb">RGB 彩色</option>
                  <option value="depth">深度</option>
                </select>
              </Field>
              <Field
                label={`图像布局 ${index + 1}`}
                required
                hint={(() => {
                  const inferred = inferImageLayout(profile.image_features, item.key, item.kind);
                  return inferred.layout
                    ? `原始形状 [${inferred.shape?.join(' × ')}]，识别为 ${inferred.layout.toUpperCase()}。`
                    : inferred.reason;
                })()}
              >
                <select
                  required
                  value={item.layout}
                  onChange={(event) => {
                    onImageLayoutEdit?.(item.key, item.kind);
                    camera(index, { layout: event.target.value as ComposedCamera['layout'] });
                  }}
                >
                  <option value="" disabled>待读取或手动选择布局</option>
                  <option value="chw">CHW · 通道/高/宽</option>
                  <option value="hwc">HWC · 高/宽/通道</option>
                </select>
              </Field>
              <Field
                label={`图像时间索引 ${index + 1}`}
                hint="静态图像留空；已有时间维时显式选择帧。"
              >
                <input
                  type="number"
                  min={-64}
                  max={63}
                  step={1}
                  value={item.time_index ?? ''}
                  onChange={(event) =>
                    camera(index, {
                      time_index: event.target.value === '' ? null : Number(event.target.value),
                    })
                  }
                />
              </Field>
              {item.kind === 'depth' && (
                <>
                  <Field label={`原始深度最小值 ${index + 1}`} required>
                    <input
                      type="number"
                      required
                      step="any"
                      value={shown(item.depth_min)}
                      onChange={(event) => camera(index, { depth_min: number(event.target.value) })}
                    />
                  </Field>
                  <Field label={`原始深度最大值 ${index + 1}`} required>
                    <input
                      type="number"
                      required
                      step="any"
                      value={shown(item.depth_max)}
                      onChange={(event) => camera(index, { depth_max: number(event.target.value) })}
                    />
                  </Field>
                </>
              )}
            </div>
          </div>
        ))}
      </section>
      <details
        className="composed-advanced"
        open={advanced}
        onToggle={(event) => setAdvanced(event.currentTarget.open)}
      >
        <summary>编码、归一化与阶段辅助字段</summary>
        <p className="field-hint">
          编码、单位和坐标系都是对现有数据的声明，不执行
          IK、轴角/四元数变换或坐标变换。历史长度表示数据已打包的历史维，不会自动从时间戳生成历史。
          单位和坐标系可选填；夹爪范围仅在启用额外校验时填写。
        </p>
        <div className="form-grid">
          <Field label="已打包状态历史长度" required>
            <input
              required
              type="number"
              min={1}
              max={64}
              step={1}
              value={shown(value.state_history)}
              onChange={(event) => set('state_history', number(event.target.value))}
            />
          </Field>
          <Field label="组合归一化方式">
            <select
              value={value.normalization}
              onChange={(event) =>
                set('normalization', event.target.value as ComposedProcessingSpec['normalization'])
              }
            >
              <option value="quantile">分位数（Quantile）</option>
              <option value="zscore">均值方差（Z-score）</option>
            </select>
          </Field>
          <Field label="组合动作模式（可选）">
            <select
              value={value.action_mode}
              onChange={(event) =>
                set('action_mode', event.target.value as ComposedProcessingSpec['action_mode'])
              }
            >
              <option value="">不指定（沿用原始数据）</option>
              <option value="absolute">绝对量</option>
              <option value="delta">增量</option>
            </select>
          </Field>
          <Field label="组合动作旋转编码（可选）">
            <select
              value={value.rotation_representation}
              onChange={(event) =>
                set(
                  'rotation_representation',
                  event.target.value as ComposedProcessingSpec['rotation_representation'],
                )
              }
            >
              <option value="">不指定（沿用原始数据）</option>
              <option value="axis_angle">轴角 · 3 维</option>
              <option value="quaternion">四元数 · 4 维</option>
              <option value="rotation_6d">6D 旋转 · 6 维</option>
              <option value="none">无独立旋转（Joint）</option>
            </select>
          </Field>
          <Field label="位置单位（可选）">
            <select
              value={value.position_unit}
              onChange={(event) =>
                set('position_unit', event.target.value as ComposedProcessingSpec['position_unit'])
              }
            >
              <option value="">不填写</option>
              <option value="m">米（m）</option>
              <option value="mm">毫米（mm）</option>
            </select>
          </Field>
          <Field label="关节单位（可选）">
            <select
              value={value.joint_unit}
              onChange={(event) =>
                set('joint_unit', event.target.value as ComposedProcessingSpec['joint_unit'])
              }
            >
              <option value="">不填写</option>
              <option value="rad">弧度（rad）</option>
              <option value="deg">角度（deg）</option>
            </select>
          </Field>
          <Field label="动作坐标系（可选）">
            <input
              maxLength={100}
              value={value.coordinate_frame}
              placeholder="可留空，填写实际坐标系定义"
              onChange={(event) => set('coordinate_frame', event.target.value)}
            />
          </Field>
          <Field label="任务文本真实字段（可选）">
            <input
              maxLength={256}
              value={value.prompt_key}
              placeholder="自动读取 task / prompt，无文本时使用空提示"
              onChange={(event) => set('prompt_key', event.target.value)}
            />
          </Field>
        </div>
        <p className="field-hint">
          动作模式和旋转编码留空时沿用原始数值，不做编码转换；填写后核对声明。
          任务字段留空时优先读取 task、其次 prompt；都没有文本时使用空提示，不使用语言条件。
        </p>
        <label className="checkbox-label composed-gripper-toggle">
          <input
            type="checkbox"
            checked={value.gripper_range != null}
            onChange={(event) =>
              set('gripper_range', event.target.checked ? [NaN, NaN] : null)
            }
          />
          启用夹爪范围校验（可选）
        </label>
        <p className="field-hint">
          启用后检查夹爪数值是否越界；不做缩放。未启用时仍检查数据为有限数值。
        </p>
        {value.gripper_range != null && (
          <div className="form-grid">
            <Field label="夹爪原始最小值" required>
              <input
                required
                type="number"
                step="any"
                value={shown(value.gripper_range[0])}
                onChange={(event) =>
                  set('gripper_range', [number(event.target.value), value.gripper_range![1]])
                }
              />
            </Field>
            <Field label="夹爪原始最大值" required>
              <input
                required
                type="number"
                step="any"
                value={shown(value.gripper_range[1])}
                onChange={(event) =>
                  set('gripper_range', [value.gripper_range![0], number(event.target.value)])
                }
              />
            </Field>
          </div>
        )}
        <div className="composed-aux">
          <strong>已有阶段监督字段</strong>
          {(['value_target', 'is_positive'] as const).map((key) => (
            <label className="checkbox-label" key={key}>
              <input
                type="checkbox"
                checked={value.aux_keys.includes(key)}
                onChange={(event) =>
                  set(
                    'aux_keys',
                    event.target.checked
                      ? [...value.aux_keys, key]
                      : value.aux_keys.filter((entry) => entry !== key),
                  )
                }
              />
              {key}
            </label>
          ))}
        </div>
        <p className="field-hint">
          仅勾选数据中真实存在的字段。训练阶段由本次任务选择，不写进组合方案。
        </p>
      </details>
      {errors.length ? (
        <div className="inline-note red training-validation-error" role="alert">
          <XCircle size={16} aria-hidden="true" />
          <div>
            <strong>组合方案配置错误</strong>
            <ul>
              {errors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          </div>
        </div>
      ) : (
        <p className="field-hint">组合摘要：{composedProcessingSummary(value)}</p>
      )}
    </fieldset>
  );
}
