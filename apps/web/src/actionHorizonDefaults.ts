import type { TrainingDraftSnapshot } from './trainingDraft';
import { DEFAULT_AUTO_CONFIG } from './trainingConfiguration';

export type ActionHorizonMode = 'automatic' | 'manual';
const MANUAL = 'auto_config.action_horizon';
const AUTOMATIC = `${MANUAL}:automatic`;

export function restoreActionHorizonMode(draft?: TrainingDraftSnapshot): ActionHorizonMode {
  if (draft?.editedParameters.includes(MANUAL)) return 'manual';
  if (draft?.editedParameters.includes(AUTOMATIC)) return 'automatic';
  // Older drafts did not record provenance. Preserve their saved model choice.
  return draft?.request.auto_config ? 'manual' : 'automatic';
}

export function actionHorizonDraftParameters(
  parameters: Iterable<string>,
  mode: ActionHorizonMode,
) {
  return [
    ...Array.from(parameters).filter((key) => key !== MANUAL && key !== AUTOMATIC),
    mode === 'automatic' ? AUTOMATIC : MANUAL,
  ];
}

export function defaultActionHorizon(
  current: number,
  mode: ActionHorizonMode,
  recommended: number | undefined,
  errors: readonly string[],
) {
  if (mode === 'manual' || errors.length) return current;
  return recommended ?? DEFAULT_AUTO_CONFIG.action_horizon;
}
