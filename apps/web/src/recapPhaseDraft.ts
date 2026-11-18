import { newDatasetBindings } from './datasetBindingHelpers';
import { isStageAdvantageDataset, isStageAdvantageStage } from './recap';
import { DEFAULT_AUTO_CONFIG } from './trainingConfiguration';
import type { TrainingDraftSnapshot, TrainingPhaseSnapshot } from './trainingDraft';
import { DEFAULT_TRAINING_NORMALIZATION_METHOD } from './trainingNormalization';
import type { Asset, TrainingStage } from './types';

const phaseStages = {
  value: 'value',
  value_annotation: 'value_annotation',
  stage_progress: 'stage_progress',
  stage_annotation: 'stage_annotation',
  recap_acp: 'offline_rl',
  stage_advantage_acp: 'offline_rl',
} as const satisfies Record<string, TrainingStage>;

export type RecapPhase = keyof typeof phaseStages;
export type RecapMode = 'recap' | 'stage_advantage';
export const isRecapPhase = (phase: unknown): phase is RecapPhase =>
  typeof phase === 'string' && Object.hasOwn(phaseStages, phase);
export const recapPhaseStage = (phase: RecapPhase): TrainingStage => phaseStages[phase];
export const recapPhaseMode = (phase: RecapPhase): RecapMode =>
  phase === 'stage_advantage_acp' || isStageAdvantageStage(phase) ? 'stage_advantage' : 'recap';

export function recapPhaseFor(stage: TrainingStage, mode: RecapMode): RecapPhase {
  return stage === 'offline_rl'
    ? mode === 'stage_advantage'
      ? 'stage_advantage_acp'
      : 'recap_acp'
    : stage;
}

export function activeRecapPhase(snapshot: TrainingDraftSnapshot, assets: readonly Asset[]) {
  const stage = snapshot.request.training_stage;
  if (snapshot.recapPhase) return snapshot.recapPhase;
  const stageAdvantage = snapshot.request.dataset_bindings?.some((binding) =>
    assets.some((asset) => asset.id === binding.dataset_asset_id && isStageAdvantageDataset(asset)),
  );
  return stage ? recapPhaseFor(stage, stageAdvantage ? 'stage_advantage' : 'recap') : 'value';
}

export function newRecapPhase(phase: RecapPhase, initialNode = ''): TrainingDraftSnapshot {
  return {
    recapPhase: phase,
    step: 0,
    advancedRecipe: false,
    editedParameters: [],
    editedStageParameters: [],
    entrypointEdited: false,
    request: {
      name: '',
      node_id: initialNode,
      profile_id: '',
      template_id: 'recap',
      training_stage: recapPhaseStage(phase),
      inputs: {},
      dataset_bindings: newDatasetBindings([]),
      auto_config: { ...DEFAULT_AUTO_CONFIG },
      normalization_method: DEFAULT_TRAINING_NORMALIZATION_METHOD,
      parameters: {},
      gpu_count: 1,
    },
  };
}

const activeSnapshot = ({
  recapPhase: _phase,
  recapPhases: _other,
  ...snapshot
}: TrainingDraftSnapshot): TrainingPhaseSnapshot => snapshot;

export function recapDraftForPhase(
  current: TrainingDraftSnapshot | undefined,
  phase: RecapPhase,
  assets: readonly Asset[],
) {
  if (current?.request.template_id !== 'recap') return undefined;
  return activeRecapPhase(current, assets) === phase
    ? activeSnapshot(current)
    : current.recapPhases?.[phase];
}

export function replaceRecapPhase(
  current: TrainingDraftSnapshot | undefined,
  replacement: TrainingDraftSnapshot,
  assets: readonly Asset[],
): TrainingDraftSnapshot {
  if (replacement.request.template_id !== 'recap') return replacement;
  const phase = activeRecapPhase(replacement, assets);
  const others = { ...current?.recapPhases };
  if (current?.request.template_id === 'recap') {
    const active = activeRecapPhase(current, assets);
    if (active !== phase && current.request.training_stage)
      others[active] = activeSnapshot(current);
  }
  delete others[phase];
  return {
    ...activeSnapshot(replacement),
    recapPhase: phase,
    ...(Object.keys(others).length ? { recapPhases: others } : {}),
  };
}

export function switchRecapPhase(
  current: TrainingDraftSnapshot,
  phase: RecapPhase,
  assets: readonly Asset[],
  initialNode?: string,
): TrainingDraftSnapshot {
  const stored = recapDraftForPhase(current, phase, assets);
  return replaceRecapPhase(
    current,
    stored ? { ...stored, recapPhase: phase } : newRecapPhase(phase, initialNode),
    assets,
  );
}
