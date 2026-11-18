import { stageAnnotationSource } from './recap';
import { recapAnnotationInputs, recapValueCheckpoints } from './recapAnnotation';
import { retryTrainingSnapshot } from './runRetry';
import type { StageAdvantageHandoff } from './stageAdvantageHandoff';
import type { TrainingDraftSnapshot } from './trainingDraft';
import type { Asset, Run } from './types';

export interface ValueAnnotationHandoff {
  checkpoint: Asset;
  stage: 'value_annotation';
  sourceIndex: number;
}

export type RecapCheckpointHandoff = StageAdvantageHandoff | ValueAnnotationHandoff;

export function recapValueAnnotationSnapshot(
  run: Run,
  handoff: ValueAnnotationHandoff,
  assets: Asset[],
): TrainingDraftSnapshot {
  const checkpoint = assets.find((asset) => asset.id === handoff.checkpoint.id);
  if (
    run.state !== 'SUCCEEDED' ||
    checkpoint?.node_id !== run.node_id ||
    stageAnnotationSource(checkpoint, 'value_annotation')?.run_id !== run.id ||
    !recapValueCheckpoints(run).some((output) => output.id === checkpoint.id)
  )
    throw new Error('请选择已完成的 Value 作业所产出的本机检查点。');
  const source = recapAnnotationInputs(run, assets).find(
    ({ index }) => index === handoff.sourceIndex,
  );
  const snapshot = retryTrainingSnapshot(run);
  const { request } = snapshot;
  const binding = request.dataset_bindings?.[handoff.sourceIndex];
  if (!source || binding?.dataset_asset_id !== source.asset.id)
    throw new Error('原数据组当前不可用，请在优势值标注中手动选择数据集。');
  const norm = [checkpoint.norm_asset_id, binding.norm_asset_id]
    .flatMap((id) => assets.filter((asset) => asset.id === id))
    .find(
      (asset) =>
        asset.node_id === run.node_id &&
        asset.kind === 'NORM' &&
        ['READY', 'REGISTERED'].includes(asset.state ?? asset.status ?? '') &&
        asset.metadata?.availability !== 'removed',
    );
  return {
    ...snapshot,
    recapPhase: 'value_annotation',
    editedStageParameters: [],
    request: {
      ...request,
      name: `${request.name.slice(0, 148).replace(/[\uD800-\uDBFF]$/u, '')} · 优势值标注`,
      training_stage: 'value_annotation',
      stage_config: undefined,
      inputs: { model: checkpoint.id },
      dataset_bindings: [{ ...binding, norm_asset_id: norm?.id ?? '', weight: 1 }],
      parameters: Object.hasOwn(request.parameters, 'batch_size')
        ? { batch_size: request.parameters.batch_size }
        : {},
      gpu_count: 1,
      requested_gpu_uuids: undefined,
      training_gpu_policy: undefined,
    },
  };
}
