import {
  decodeTrainingDraft,
  encodeTrainingDraft,
  type TrainingDraftSnapshot,
} from './trainingDraft';
import type { Asset, JsonObject, Node, Profile } from './types';

export const TRAINING_TEMPLATE_PROCESSING_MARKER = 'training_template.reuse_processing';

export interface TrainingTemplateSummary {
  id: string;
  name: string;
  description: string;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface TrainingTemplateRecord extends TrainingTemplateSummary {
  content: JsonObject;
}

export interface TrainingTemplateResources {
  nodes: readonly Node[];
  profiles: readonly Profile[];
  assets: readonly Asset[];
}

export interface AppliedTrainingTemplate {
  snapshot: TrainingDraftSnapshot;
  issues: string[];
}

const snapshotKeys = new Set([
  'request',
  'step',
  'advancedRecipe',
  'editedParameters',
  'editedStageParameters',
  'entrypointEdited',
  'recipeDatasets',
  'recipeDefaults',
  'stageRecipeDefaults',
  'stageRecipeInfo',
  'recipeEditorDraft',
]);
// These are transient wizard state, never template content or reusable approval.
const transientKeys = new Set([
  'previewId',
  'confirmedConfigSha256',
  'preview',
  'createdPreview',
  'idempotencyKey',
  'run',
  'runId',
  'attemptId',
  'codeSnapshot',
]);

function invalidTemplate(): never {
  throw new Error('训练模板内容无效、不兼容或包含不允许保存的字段，请重新保存模板。');
}

function cloneSnapshot(snapshot: TrainingDraftSnapshot): TrainingDraftSnapshot {
  if (!snapshot || typeof snapshot !== 'object' || Array.isArray(snapshot)) invalidTemplate();
  const prototype = Object.getPrototypeOf(snapshot);
  if (
    (prototype !== Object.prototype && prototype !== null) ||
    Object.keys(snapshot).some((key) => !snapshotKeys.has(key) && !transientKeys.has(key)) ||
    Object.values(Object.getOwnPropertyDescriptors(snapshot)).some(
      (property) => !('value' in property),
    )
  ) {
    invalidTemplate();
  }
  const cloned = decodeTrainingDraft(encodeTrainingDraft(snapshot));
  if (!cloned) invalidTemplate();
  return cloned;
}

function clearDatasetReferences(snapshot: TrainingDraftSnapshot): TrainingDraftSnapshot {
  const request = snapshot.request;
  request.name = '';
  request.inputs = request.inputs.model ? { model: request.inputs.model } : {};
  if (request.dataset_bindings !== undefined) {
    request.dataset_bindings = request.dataset_bindings.map((binding) => ({
      ...binding,
      dataset_asset_id: '',
      norm_asset_id: '',
    }));
  }
  snapshot.step = 0;
  if (!snapshot.editedParameters.includes(TRAINING_TEMPLATE_PROCESSING_MARKER)) {
    snapshot.editedParameters.push(TRAINING_TEMPLATE_PROCESSING_MARKER);
  }
  delete snapshot.recipeEditorDraft;
  return snapshot;
}

/** Save settings and processor structure; every future run chooses its own data. */
export function prepareTrainingTemplate(snapshot: TrainingDraftSnapshot): JsonObject {
  return encodeTrainingDraft(clearDatasetReferences(cloneSnapshot(snapshot)));
}

/** Decode anew for each application. Never select replacements or reuse preflight. */
export function applyTrainingTemplate(
  record: TrainingTemplateRecord,
  resources: TrainingTemplateResources,
): AppliedTrainingTemplate {
  const decoded = decodeTrainingDraft(record?.content);
  if (!decoded) invalidTemplate();
  const snapshot = clearDatasetReferences(decoded);
  const request = snapshot.request;
  const issues: string[] = [];
  const selectedNode = resources.nodes.find((node) => node.id === request.node_id);
  if (
    request.node_id &&
    (!selectedNode || ['REMOVED', 'DISCONNECTED'].includes(selectedNode.status))
  ) {
    issues.push('模板中的机器已不可用，请重新选择机器、运行环境和基础模型。');
    request.node_id = '';
  } else if (selectedNode && selectedNode.status !== 'ONLINE') {
    issues.push(
      `模板中的机器 ${selectedNode.name} 当前未在线；已保留原选择，上线或重新选择后才能预检。`,
    );
  }

  const selectedProfile = resources.profiles.find(
    (profile) => profile.id === request.profile_id && profile.node_id === request.node_id,
  );
  const adapter = ['openpi-sft', 'recap'].includes(request.template_id)
    ? 'openpi-sft'
    : request.template_id === 'smoke'
      ? 'smoke'
      : undefined;
  const profileUnavailable =
    !!request.profile_id &&
    (!selectedProfile ||
      (!!selectedProfile.execution_mode && selectedProfile.execution_mode !== 'native') ||
      (!!adapter && selectedProfile.adapter !== adapter));
  if (!request.node_id || profileUnavailable) {
    if (request.node_id && profileUnavailable) {
      issues.push('模板中的运行环境已不可用或不兼容，请在当前机器重新选择环境。');
    }
    request.profile_id = '';
  }

  const modelId = request.inputs.model;
  const model = resources.assets.find((asset) => asset.id === modelId);
  const modelState = model?.state ?? model?.status;
  if (
    modelId &&
    (!request.node_id ||
      !model ||
      model.kind !== 'MODEL' ||
      model.node_id !== request.node_id ||
      (!!modelState && modelState !== 'READY'))
  ) {
    issues.push('模板中的基础模型已不可用或不属于当前机器，请重新选择；不会自动替换模型。');
    request.inputs = {};
  }

  const savedRecipe = request.recipe_definition;
  const recipeScopeChanged =
    !request.node_id ||
    !request.profile_id ||
    (!!savedRecipe &&
      (savedRecipe.node_id !== request.node_id || savedRecipe.profile_id !== request.profile_id));
  if (recipeScopeChanged && (request.recipe || request.recipe_revision_id || savedRecipe)) {
    issues.push('模板中的配方所属机器或环境已不匹配，请重新选择配方。');
    delete request.recipe;
    delete request.recipe_revision_id;
    delete request.recipe_definition;
  }
  return { snapshot, issues };
}
