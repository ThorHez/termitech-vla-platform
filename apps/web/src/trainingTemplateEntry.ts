import { composedChoice, trainingProfileSeed } from './ComposedDatasetProcessing';
import { trainingCompositionDraft } from './composedProcessing';
import { newNormDataset, normProcessorChoice } from './normStats';
import { trainingParameterDefaults } from './trainingParameters';
import {
  applyResourceRecommendations,
  normResourceRecommendationTracking,
  recommendNormResources,
} from './trainingResourceRecommendations';
import { DEFAULT_AUTO_CONFIG } from './trainingConfiguration';
import { DEFAULT_TRAINING_NORMALIZATION_METHOD } from './trainingNormalization';
import type { TrainingDraftSnapshot } from './trainingDraft';
import type {
  Asset,
  DataProcessorChoice,
  DatasetBinding,
  Node,
  ParameterSpec,
  Profile,
  RecipeLiteral,
  Template,
} from './types';

/** An explicit recipe-card selection changes the mode of a restored draft once. */
export function trainingTemplateEntry(
  snapshot: TrainingDraftSnapshot | undefined,
  template: Template,
  resources: { assets: readonly Asset[]; profiles: readonly Profile[]; nodes: readonly Node[] },
): TrainingDraftSnapshot | undefined {
  if (!snapshot || snapshot.request.template_id === template.id) return snapshot;
  const previous = snapshot.request;
  const norm = template.id === 'openpi-norm';
  const local = template.id === 'openpi-sft' || template.id === 'recap';
  const asset = (id: string) => resources.assets.find((item) => item.id === id);
  const profiles = resources.profiles.filter(
    (profile) =>
      profile.node_id === previous.node_id &&
      profile.adapter === (template.runtime_adapter || template.id) &&
      profile.active !== false &&
      (!profile.execution_mode || profile.execution_mode === 'native'),
  );
  const profile = profiles.find((item) => item.id === previous.profile_id) || profiles[0];
  const parameterSpecs: Record<string, ParameterSpec> = Array.isArray(template.parameters)
    ? Object.fromEntries(
        template.parameters.filter((item) => item.name).map((item) => [item.name!, item]),
      )
    : Object.fromEntries(
        Object.entries(template.parameters || {}).map(([key, value]) => [
          key,
          typeof value === 'number' ? { default: value } : value,
        ]),
      );
  let parameters = {
    ...trainingParameterDefaults(parameterSpecs),
    ...Object.fromEntries(
      Object.entries(previous.parameters).filter(([key]) => Object.hasOwn(parameterSpecs, key)),
    ),
  };
  // SFT resource edits are not an explicit choice for this CPU statistics job.
  const editedParameters = norm
    ? normResourceRecommendationTracking(
        snapshot.editedParameters.filter((key) => key === 'batch_size'),
      )
    : new Set(snapshot.editedParameters.filter((key) => Object.hasOwn(parameterSpecs, key)));
  if (norm)
    parameters = applyResourceRecommendations(
      parameters,
      editedParameters,
      recommendNormResources(resources.nodes.find((item) => item.id === previous.node_id)?.metrics),
    );
  const selectedDatasets: {
    dataset_asset_id: string;
    processor?: DataProcessorChoice;
    norm_asset_id?: string;
    weight?: number;
  }[] = previous.dataset_bindings?.length
    ? previous.dataset_bindings
    : previous.norm_datasets?.length
      ? previous.norm_datasets
      : Object.entries(previous.inputs)
          .filter(([key]) => key === 'dataset' || /^dataset_\d+$/u.test(key))
          .map(([, id]) => ({ dataset_asset_id: id }));
  const rows = selectedDatasets.length ? selectedDatasets : [{ dataset_asset_id: '' }];
  const trainingBindings = (): DatasetBinding[] =>
    rows.map((row) => {
      const dataset = asset(row.dataset_asset_id);
      const seeded = trainingCompositionDraft(trainingProfileSeed(dataset));
      const processor = row.processor ? structuredClone(row.processor) : composedChoice(seeded);
      const composition = processor.parameters.composition;
      // Statistics deliberately omits vision; restore dataset camera defaults for training.
      if (
        previous.template_id === 'openpi-norm' &&
        processor.name === 'ComposableDataConfig' &&
        composition &&
        typeof composition === 'object' &&
        !Array.isArray(composition)
      )
        processor.parameters.composition = {
          ...composition,
          cameras: seeded.cameras as unknown as RecipeLiteral,
        };
      return {
        dataset_asset_id: row.dataset_asset_id,
        norm_asset_id: row.norm_asset_id ?? dataset?.norm_asset_id ?? '',
        weight: row.weight ?? 1,
        processor,
      };
    });
  return {
    request: {
      name: previous.name,
      node_id: previous.node_id,
      profile_id: profile?.id || '',
      template_id: template.id,
      inputs: local ? { ...previous.inputs } : {},
      parameters,
      gpu_count:
        norm || template.id === 'smoke' ? 0 : Math.max(template.min_gpus ?? 1, previous.gpu_count),
      ...(local && previous.requested_gpu_uuids !== undefined
        ? { requested_gpu_uuids: [...previous.requested_gpu_uuids] }
        : {}),
      ...(norm || local
        ? {
            auto_config: {
              ...(previous.auto_config || DEFAULT_AUTO_CONFIG),
              ...(norm ? { finetune_mode: 'full' as const } : {}),
            },
          }
        : {}),
      ...(norm
        ? {
            norm_datasets: rows.map((row) =>
              row.processor
                ? {
                    dataset_asset_id: row.dataset_asset_id,
                    processor: normProcessorChoice(structuredClone(row.processor)),
                  }
                : {
                    ...newNormDataset(true, asset(row.dataset_asset_id)),
                    dataset_asset_id: row.dataset_asset_id,
                  },
            ),
          }
        : local
          ? {
              dataset_bindings: trainingBindings(),
              normalization_method:
                previous.normalization_method || DEFAULT_TRAINING_NORMALIZATION_METHOD,
            }
          : { recipe: template.id === 'smoke' ? 'smoke' : undefined }),
    },
    step: 0,
    advancedRecipe: false,
    editedParameters: [...editedParameters],
    editedStageParameters: [],
    entrypointEdited: false,
  };
}
