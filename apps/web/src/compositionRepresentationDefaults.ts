import type { ComposedProcessingSpec, ProcessingProfileSeed } from './composedProcessing';
import { imageLayoutManualMarker } from './imageLayout';

export type CompositionRepresentation = 'eef' | 'joint' | 'composite';
export type RepresentationField = 'state_representation' | 'action_representation';
export interface RepresentationEvidence {
  value?: CompositionRepresentation;
  reason: string;
}
const label = (value: CompositionRepresentation) =>
  value === 'eef' ? 'EEF' : value === 'joint' ? 'Joint' : '组合向量（多分量）';
const supported = (
  value: unknown,
  purpose: 'training' | 'norm',
): value is CompositionRepresentation =>
  value === 'eef' || value === 'joint' || (purpose === 'norm' && value === 'composite');
const sameKeys = (first: string[], second: string[]) =>
  first.length === second.length &&
  [...first].sort().every((key, index) => key === [...second].sort()[index]);
type Fact = { value: CompositionRepresentation; source: string };
function resolve(
  facts: Fact[],
  unresolved: string[],
  covered: boolean,
  missing: string,
): RepresentationEvidence {
  if (new Set(facts.map((fact) => fact.value)).size > 1)
    return {
      reason: `表示声明冲突：${facts.map((fact) => `${fact.source}为 ${label(fact.value)}`).join('；')}。请核对后选择。`,
    };
  if (unresolved.length && !covered)
    return { reason: `${unresolved.join('；')}。没有覆盖当前选择的明确处理语义，需手动确认。` };
  if (!facts.length) return { reason: missing };
  return {
    value: facts[0].value,
    reason: `依据${facts.map((fact) => fact.source).join('、')}：${label(facts[0].value)}。`,
  };
}

/** Source labels and matching saved semantics are independent for state/action. */
export function inferCompositionRepresentations(
  profile: ProcessingProfileSeed,
  selection: Pick<ComposedProcessingSpec, 'state_fields' | 'action_key'>,
  purpose: 'training' | 'norm' = 'training',
): Record<RepresentationField, RepresentationEvidence> {
  const annotations = profile.field_annotations || [];
  const stateKeys = selection.state_fields.map((field) => field.key.trim()).filter(Boolean);
  const saved = profile.processing;
  const stateFacts: Fact[] = [];
  const unknownStates: string[] = [];
  for (const key of stateKeys) {
    const labels = annotations.filter(
      (field) => field.role === 'state' && field.key.trim() === key,
    );
    const field = labels.length === 1 ? labels[0] : undefined;
    if (field && (field.component === 'gripper' || field.representation === 'gripper')) continue;
    if (field && supported(field.representation, purpose))
      stateFacts.push({ value: field.representation, source: `状态字段 ${key} 的登记标签` });
    else
      unknownStates.push(
        `状态字段 ${key} ${field ? `仅登记了 ${field.representation === 'composite' ? '组合向量' : '未知'} 表示` : '缺少唯一的表示标签'}`,
      );
  }
  const savedStates =
    !!saved &&
    sameKeys(
      stateKeys,
      saved.state_fields.map((field) => field.key.trim()),
    ) &&
    stateKeys.length > 0 &&
    supported(saved.state_representation, purpose);
  if (savedStates)
    stateFacts.push({ value: saved!.state_representation, source: '相同状态字段的已保存组合方案' });
  const state = stateKeys.length
    ? resolve(
        stateFacts,
        unknownStates,
        savedStates,
        '所选状态字段未声明 EEF/Joint 表示，不能从动作表示、维数或机器人型号推断。',
      )
    : { reason: '请先选择状态真实字段，再读取对应的表示标签。' };

  const actionKey = selection.action_key.trim();
  const actions = annotations.filter((field) => field.role === 'action');
  const selectedActions = actions.filter((field) => field.key.trim() === actionKey);
  const selected = selectedActions.length === 1 ? selectedActions[0] : undefined;
  const actionFacts: Fact[] = [];
  const unknownActions: string[] = [];
  if (selected && supported(selected.representation, purpose))
    actionFacts.push({
      value: selected.representation,
      source: `动作字段 ${actionKey} 的登记标签`,
    });
  else if (actionKey)
    unknownActions.push(
      `动作字段 ${actionKey} ${selected ? `仅登记了 ${selected.representation === 'composite' ? '组合向量' : '未知'} 表示` : '缺少唯一的表示标签'}`,
    );
  const savedAction =
    !!saved &&
    saved.action_key.trim() === actionKey &&
    !!actionKey &&
    supported(saved.action_representation, purpose);
  if (savedAction)
    actionFacts.push({
      value: saved!.action_representation,
      source: `动作字段 ${actionKey} 的已保存组合方案`,
    });
  const globalAction =
    supported(profile.action_representation, purpose) &&
    !!actionKey &&
    ((actions.length === 1 && selectedActions.length === 1) ||
      (!actions.length &&
        (['action', 'actions'].includes(actionKey) || saved?.action_key.trim() === actionKey)));
  if (globalAction)
    actionFacts.push({
      value: profile.action_representation as CompositionRepresentation,
      source: '数据集登记的动作表示',
    });
  const action = !actionKey
    ? { reason: '请先选择动作真实字段，再读取对应的表示标签。' }
    : resolve(
        actionFacts,
        unknownActions,
        savedAction || globalAction,
        `动作字段 ${actionKey} 没有明确表示，不能从共享归属、维数或机器人型号推断。`,
      );
  return { state_representation: state, action_representation: action };
}

export function fillMissingCompositionRepresentations(
  spec: ComposedProcessingSpec,
  profile: ProcessingProfileSeed,
  purpose: 'training' | 'norm' = 'training',
  manual: ReadonlySet<RepresentationField> = new Set(),
): ComposedProcessingSpec {
  const inferred = inferCompositionRepresentations(profile, spec, purpose);
  let result = spec;
  for (const field of ['state_representation', 'action_representation'] as const) {
    const current: unknown = spec[field];
    if (
      !manual.has(field) &&
      (current == null || current === '' || current === 'unknown') &&
      inferred[field].value
    ) {
      if (result === spec) result = { ...spec };
      result[field] = inferred[field].value!;
    }
  }
  return result;
}

export function representationManualMarker(
  scope: readonly unknown[],
  field: RepresentationField,
): string {
  const plain = `composition_representation.manual.${JSON.stringify([...scope, field])}`;
  return plain.length <= 256
    ? plain
    : `composition_representation.manual.${imageLayoutManualMarker([...scope, field]).split(':')[1]}`;
}
