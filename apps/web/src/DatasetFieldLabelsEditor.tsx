import { Plus, Trash2 } from 'lucide-react';
import { sourceActionHorizon, type DatasetFieldAnnotation } from './datasetFieldAnnotations';
import './dataset-field-annotations.css';

export function DatasetFieldLabelsEditor({
  value = [],
  featureKeys = [],
  disabled = false,
  onChange,
  onFieldEdit,
}: {
  value?: DatasetFieldAnnotation[];
  featureKeys?: string[];
  disabled?: boolean;
  onChange: (value: DatasetFieldAnnotation[]) => void;
  onFieldEdit?: (path: string) => void;
}) {
  const update = <K extends keyof DatasetFieldAnnotation>(
    index: number,
    key: K,
    next: DatasetFieldAnnotation[K],
  ) => {
    onFieldEdit?.(`field_annotations[${index}].${key}`);
    onChange(value.map((row, position) => (position === index ? { ...row, [key]: next } : row)));
  };
  return (
    <section className="dataset-field-annotations" aria-label="状态与动作字段标签">
      <div className="dataset-profile-section-title">
        <strong>状态与动作字段标签</strong>
        <span>{value.length} 项</span>
        <button
          type="button"
          className="button outline small"
          disabled={disabled || value.length >= 256}
          onClick={() => {
            onFieldEdit?.('field_annotations');
            onChange([
              ...value,
              {
                key: '',
                role: 'state',
                representation: 'unknown',
                value_mode: 'unknown',
                component: 'other',
                arm: 'unknown',
              },
            ]);
          }}
        >
          <Plus size={13} />
          添加字段标签
        </button>
      </div>
      <p className="field-hint">
        这里只描述原始数据；无法确认的语义保留“未知”。拼接和归一化在新建训练中配置。
      </p>
      {!value.length ? (
        <p className="field-hint">读取数据集后自动填入状态与动作字段，也可手动添加。</p>
      ) : (
        <div className="dataset-field-labels-table">
          <table>
            <thead>
              <tr>
                <th>真实字段 / 原始形状</th>
                <th>用途</th>
                <th>表示</th>
                <th>数值方式</th>
                <th title="数据中每条样本已有的动作步数，不是训练预测步数">Horizon</th>
                <th>分量</th>
                <th>归属</th>
                <th>
                  <span className="sr-only">移除</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {value.map((row, index) => (
                <tr key={index}>
                  <td>
                    {featureKeys.length ? (
                      <select
                        aria-label={`标签真实字段 ${index + 1}`}
                        value={row.key}
                        disabled={disabled}
                        onChange={(event) => {
                          onFieldEdit?.('field_annotations');
                          const {
                            shape: _shape,
                            dtype: _dtype,
                            names: _names,
                            horizon: _horizon,
                            ...labels
                          } = row;
                          onChange(
                            value.map((item, position) =>
                              position === index ? { ...labels, key: event.target.value } : item,
                            ),
                          );
                        }}
                      >
                        <option value="">选择真实字段</option>
                        {[...new Set([...featureKeys, ...(row.key ? [row.key] : [])])].map(
                          (key) => (
                            <option key={key} value={key}>
                              {key}
                            </option>
                          ),
                        )}
                      </select>
                    ) : (
                      <input
                        aria-label={`标签真实字段 ${index + 1}`}
                        value={row.key}
                        maxLength={256}
                        disabled={disabled}
                        onChange={(event) => {
                          onFieldEdit?.('field_annotations');
                          const {
                            shape: _shape,
                            dtype: _dtype,
                            names: _names,
                            horizon: _horizon,
                            ...labels
                          } = row;
                          onChange(
                            value.map((item, position) =>
                              position === index ? { ...labels, key: event.target.value } : item,
                            ),
                          );
                        }}
                      />
                    )}
                    <span className="field-hint">
                      {row.dtype || '形状待读取'}
                      {row.shape ? ` [${row.shape.join(', ')}]` : ''}
                    </span>
                  </td>
                  <td>
                    <select
                      aria-label={`字段用途 ${index + 1}`}
                      value={row.role}
                      disabled={disabled}
                      onChange={(event) => {
                        const role = event.target.value as DatasetFieldAnnotation['role'];
                        onFieldEdit?.(`field_annotations[${index}].role`);
                        onFieldEdit?.(`field_annotations[${index}].horizon`);
                        onChange(
                          value.map((item, position) => {
                            if (position !== index) return item;
                            const { horizon: _horizon, ...labels } = item;
                            const horizon =
                              role === 'action' ? sourceActionHorizon(item.shape) : undefined;
                            return {
                              ...labels,
                              role,
                              ...(horizon !== undefined ? { horizon } : {}),
                            };
                          }),
                        );
                      }}
                    >
                      <option value="state">状态</option>
                      <option value="action">动作</option>
                    </select>
                  </td>
                  <td>
                    <select
                      aria-label={`字段表示 ${index + 1}`}
                      value={row.representation}
                      disabled={disabled}
                      onChange={(event) => {
                        const representation = event.target
                          .value as DatasetFieldAnnotation['representation'];
                        const component =
                          representation === 'gripper'
                            ? 'gripper'
                            : row.representation === 'gripper' && row.component === 'gripper'
                              ? 'other'
                              : row.component;
                        onFieldEdit?.(`field_annotations[${index}].representation`);
                        if (component !== row.component)
                          onFieldEdit?.(`field_annotations[${index}].component`);
                        onChange(
                          value.map((item, position) =>
                            position === index ? { ...item, representation, component } : item,
                          ),
                        );
                      }}
                    >
                      <option value="unknown">未知</option>
                      <option value="eef">EEF</option>
                      <option value="joint">Joint</option>
                      <option value="gripper">夹爪开度</option>
                    </select>
                  </td>
                  <td>
                    <select
                      aria-label={`字段数值方式 ${index + 1}`}
                      value={row.value_mode}
                      disabled={disabled}
                      onChange={(event) =>
                        update(
                          index,
                          'value_mode',
                          event.target.value as DatasetFieldAnnotation['value_mode'],
                        )
                      }
                    >
                      <option value="unknown">未知</option>
                      <option value="absolute">绝对</option>
                      <option value="delta">增量</option>
                    </select>
                  </td>
                  <td>
                    {row.role === 'action' ? (
                      <input
                        type="number"
                        min={1}
                        max={2 ** 31 - 1}
                        step={1}
                        aria-label={`动作 Horizon ${index + 1}`}
                        title="数据中每条样本已有的动作序列长度；训练预测步数在训练配置中设置。"
                        placeholder="待读取"
                        value={row.horizon ?? sourceActionHorizon(row.shape) ?? ''}
                        disabled={disabled}
                        onChange={(event) => {
                          onFieldEdit?.(`field_annotations[${index}].horizon`);
                          const horizon =
                            event.target.value === '' ? undefined : Number(event.target.value);
                          onChange(
                            value.map((item, position) => {
                              if (position !== index) return item;
                              const { horizon: _horizon, ...labels } = item;
                              return { ...labels, ...(horizon !== undefined ? { horizon } : {}) };
                            }),
                          );
                        }}
                      />
                    ) : (
                      <span aria-label={`状态字段 ${index + 1} 无动作 Horizon`}>—</span>
                    )}
                  </td>
                  <td>
                    <select
                      aria-label={`字段分量 ${index + 1}`}
                      value={row.component}
                      disabled={disabled}
                      onChange={(event) => {
                        const component = event.target.value as DatasetFieldAnnotation['component'];
                        const representation =
                          row.representation === 'gripper' && component !== 'gripper'
                            ? 'unknown'
                            : row.representation;
                        onFieldEdit?.(`field_annotations[${index}].component`);
                        if (representation !== row.representation)
                          onFieldEdit?.(`field_annotations[${index}].representation`);
                        onChange(
                          value.map((item, position) =>
                            position === index ? { ...item, component, representation } : item,
                          ),
                        );
                      }}
                    >
                      <option value="other">其他</option>
                      <option value="position">位置</option>
                      <option value="rotation">旋转</option>
                      <option value="joint">关节</option>
                      <option value="gripper">夹爪</option>
                    </select>
                  </td>
                  <td>
                    <select
                      aria-label={`字段归属 ${index + 1}`}
                      value={row.arm}
                      disabled={disabled}
                      onChange={(event) =>
                        update(index, 'arm', event.target.value as DatasetFieldAnnotation['arm'])
                      }
                    >
                      <option value="unknown">未知</option>
                      <option value="left">左臂</option>
                      <option value="right">右臂</option>
                      <option value="shared">共享</option>
                    </select>
                  </td>
                  <td>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label={`移除字段标签 ${index + 1}`}
                      disabled={disabled}
                      onClick={() => {
                        onFieldEdit?.('field_annotations');
                        onChange(value.filter((_, position) => position !== index));
                      }}
                    >
                      <Trash2 size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
