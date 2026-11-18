import { EditableChoiceField } from './EditableChoiceField';

const choices = [
  'ARX5',
  'UMI',
  'ALOHA',
  'ROKAE',
  'AGIBOT G2',
  'Franka',
  'UR5',
  'Piper',
  'SO-100',
  'SO-101',
  '仿真机器人',
];

export function RobotTypeField({
  value,
  onChange,
  disabled = false,
  label = '机器人类型',
  hint = '默认值可直接修改；展开可选择其他类型，也可输入自定义机器人型号。',
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  label?: string;
  hint?: string;
}) {
  return (
    <EditableChoiceField
      choices={choices}
      value={value}
      onChange={onChange}
      disabled={disabled}
      label={label}
      hint={hint}
      placeholder="例如 ARX5、UMI、ALOHA"
      maxLength={100}
      caseSensitive={false}
    />
  );
}
