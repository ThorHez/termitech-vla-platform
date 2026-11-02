import { Field } from './components';
import { TRAINING_NORMALIZATION_LABELS } from './trainingNormalization';
import type { TrainingNormalizationMethod } from './types';

export function TrainingNormalizationField({
  value,
  onChange,
}: {
  value?: TrainingNormalizationMethod;
  onChange: (value?: TrainingNormalizationMethod) => void;
}) {
  return (
    <Field
      label="训练归一化方式"
      hint="统一用于本次全部数据集的状态和动作；推理反变换跟随训练。切换公式后需重新预检并确认 config。"
    >
      <select
        value={value || ''}
        onChange={(event) =>
          onChange((event.target.value || undefined) as TrainingNormalizationMethod | undefined)
        }
      >
        <option value="">沿用配方 / 旧配置</option>
        {Object.entries(TRAINING_NORMALIZATION_LABELS).map(([method, label]) => (
          <option value={method} key={method}>
            {label}
          </option>
        ))}
      </select>
    </Field>
  );
}
