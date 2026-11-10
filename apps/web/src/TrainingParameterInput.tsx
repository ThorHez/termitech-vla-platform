import { useState, type ComponentProps } from 'react';
import {
  formatLearningRate,
  isLearningRateParameter,
  parseLearningRate,
} from './trainingParameters';

type Props = Omit<ComponentProps<'input'>, 'type' | 'value' | 'onChange'> & {
  parameter: string;
  value: number | undefined;
  onChange: (text: string) => void;
};

export function TrainingParameterInput({ parameter, value, onChange, onBlur, ...props }: Props) {
  const scientific = isLearningRateParameter(parameter);
  const [draft, setDraft] = useState<{ text: string; value: number | undefined }>();
  // Keep incomplete exponents visible while the numeric model marks them invalid.
  const currentDraft = draft && Object.is(draft.value, value) ? draft : undefined;
  if (draft && !currentDraft) setDraft(undefined);
  const display = scientific
    ? (currentDraft?.text ?? formatLearningRate(value))
    : Number.isFinite(value)
      ? value
      : '';

  return (
    <input
      {...props}
      type={scientific ? 'text' : 'number'}
      spellCheck={scientific ? false : props.spellCheck}
      value={display}
      onChange={(event) => {
        const text = event.currentTarget.value;
        if (scientific) setDraft({ text, value: parseLearningRate(text, props.required ?? false) });
        onChange(text);
      }}
      onBlur={(event) => {
        if (scientific && (Number.isFinite(value) || value === undefined)) setDraft(undefined);
        onBlur?.(event);
      }}
    />
  );
}
