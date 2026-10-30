import type { JsonObject, RecipeDatasetSlot, RunRequest } from './types';
import { validRequestedGpuList } from './trainingGpuSelection';

export const MAX_TRAINING_DRAFT_BYTES = 512 * 1024;
export const MAX_TRAINING_DRAFT_DEPTH = 48;
export const MAX_TRAINING_DRAFT_NODES = 50000;

export interface RecipeEditorDraft {
  scope: string;
  sourceSha256?: string;
  fields: Record<string, string>;
  saveMode?: 'new' | 'revision';
  saveName?: string;
  configName?: string;
  description?: string;
}

export interface TrainingDraftSnapshot {
  request: RunRequest;
  step: 0 | 1;
  advancedRecipe: boolean;
  editedParameters: string[];
  editedStageParameters: string[];
  entrypointEdited: boolean;
  recipeDatasets?: RecipeDatasetSlot[];
  recipeDefaults?: Record<string, number>;
  stageRecipeDefaults?: Record<string, number>;
  stageRecipeInfo?: {
    values: Record<string, number>;
    mixedDropout: boolean;
    valueHead: { path: string; value: string }[];
  };
  recipeEditorDraft?: RecipeEditorDraft;
}

type Path = (string | number)[];
type SpecialValue =
  | 'nan'
  | 'positive_infinity'
  | 'negative_infinity'
  | 'negative_zero'
  | 'undefined';
interface SpecialEntry {
  path: Path;
  value: SpecialValue;
}

const requestKeys = new Set([
  'name',
  'node_id',
  'profile_id',
  'template_id',
  'inputs',
  'parameters',
  'gpu_count',
  'requested_gpu_uuids',
  'auto_config',
  'training_stage',
  'stage_config',
  'dataset_bindings',
  'entrypoint',
  'recipe',
  'recipe_revision_id',
  'recipe_definition',
]);
const envelopeKeys = new Set([
  'schema_version',
  'request',
  'step',
  'advanced_recipe',
  'edited_parameters',
  'edited_stage_parameters',
  'entrypoint_edited',
  'recipe_datasets',
  'recipe_defaults',
  'stage_recipe_defaults',
  'stage_recipe_info',
  'recipe_editor_draft',
  'special_values',
]);
// Match the server's normalized credential/approval names before anything can
// reach the per-account local outbox. Compatibility folding is conservative:
// alternate spellings must not make credentials persistable in the browser.
const forbiddenDraftKeys = new Set([
  'previewid',
  'confirmedconfigsha256',
  'codesnapshot',
  'idempotencykey',
  'executiontoken',
  'csrftoken',
  'password',
  'admintoken',
  'enrollmenttoken',
  'accesstoken',
  'refreshtoken',
  'sessiontoken',
  'authorization',
  'cookie',
  'passwordhash',
]);
const unsafeKey = (key: string) =>
  ['__proto__', 'prototype', 'constructor'].includes(key) ||
  forbiddenDraftKeys.has(
    key
      .normalize('NFKD')
      .toLowerCase()
      .replace(/ß/g, 'ss')
      .replace(/ſ/g, 's')
      .replace(/[^a-z0-9]/g, ''),
  );
const specialValues = new Set<SpecialValue>([
  'nan',
  'positive_infinity',
  'negative_infinity',
  'negative_zero',
  'undefined',
]);
const plainObject = (value: unknown): value is Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return (
    (prototype === Object.prototype || prototype === null) &&
    Object.values(Object.getOwnPropertyDescriptors(value)).every((item) => 'value' in item)
  );
};
function fail(): never {
  throw new Error('训练草稿格式无效、包含不支持的字段或超出保存上限。');
}
const own = (value: object, key: PropertyKey) => Object.prototype.hasOwnProperty.call(value, key);
const text = (value: unknown): value is string =>
  typeof value === 'string' && value.length <= 65536;
const stringList = (value: unknown) =>
  Array.isArray(value) &&
  value.length <= 512 &&
  value.every((item) => typeof item === 'string' && item.length <= 256);
const recordOf = (value: unknown, check: (item: unknown) => boolean) =>
  plainObject(value) && Object.values(value).every(check);
const numbers = (value: unknown) => recordOf(value, (item) => typeof item === 'number');

function cloneData(value: unknown, specials?: SpecialEntry[]): unknown {
  let nodes = 0;
  const parents = new Set<object>();
  const walk = (entry: unknown, path: Path): unknown => {
    if (++nodes > MAX_TRAINING_DRAFT_NODES || path.length > MAX_TRAINING_DRAFT_DEPTH) fail();
    if (
      entry === undefined ||
      (typeof entry === 'number' && (!Number.isFinite(entry) || Object.is(entry, -0)))
    ) {
      if (!specials) fail();
      const kind: SpecialValue =
        entry === undefined
          ? 'undefined'
          : Number.isNaN(entry)
            ? 'nan'
            : entry === Infinity
              ? 'positive_infinity'
              : entry === -Infinity
                ? 'negative_infinity'
                : 'negative_zero';
      specials.push({ path, value: kind });
      return null;
    }
    if (entry === null || typeof entry === 'boolean' || typeof entry === 'number') return entry;
    if (typeof entry === 'string') {
      if (entry.length > MAX_TRAINING_DRAFT_BYTES) fail();
      return entry;
    }
    if (!Array.isArray(entry) && !plainObject(entry)) fail();
    if (parents.has(entry)) fail();
    parents.add(entry);
    let result: unknown;
    if (Array.isArray(entry)) {
      if (entry.length > MAX_TRAINING_DRAFT_NODES) fail();
      result = Array.from({ length: entry.length }, (_, index) =>
        walk(entry[index], [...path, index]),
      );
    } else {
      const object: Record<string, unknown> = {};
      for (const [key, child] of Object.entries(entry)) {
        if (unsafeKey(key) || key.length > 4096) fail();
        object[key] = walk(child, [...path, key]);
      }
      result = object;
    }
    parents.delete(entry);
    return result;
  };
  return walk(value, []);
}

function boundedWire(value: unknown): JsonObject {
  const result = cloneData(value);
  if (!plainObject(result)) fail();
  const serialized = JSON.stringify(result);
  if (
    serialized.length > MAX_TRAINING_DRAFT_BYTES ||
    new TextEncoder().encode(serialized).byteLength > MAX_TRAINING_DRAFT_BYTES
  )
    fail();
  return result;
}

function validRequest(value: unknown): value is RunRequest {
  if (!plainObject(value) || Object.keys(value).some((key) => !requestKeys.has(key))) return false;
  if (
    !['name', 'node_id', 'profile_id', 'template_id'].every((key) => text(value[key])) ||
    !recordOf(value.inputs, text) ||
    !numbers(value.parameters) ||
    typeof value.gpu_count !== 'number'
  )
    return false;
  if (value.requested_gpu_uuids !== undefined && !validRequestedGpuList(value.requested_gpu_uuids))
    return false;
  for (const key of ['entrypoint', 'recipe', 'recipe_revision_id'])
    if (value[key] !== undefined && !text(value[key])) return false;
  if (
    value.training_stage !== undefined &&
    !['sft', 'value', 'offline_rl'].includes(String(value.training_stage))
  )
    return false;
  if (value.stage_config !== undefined && !numbers(value.stage_config)) return false;
  if (value.recipe_definition !== undefined && !plainObject(value.recipe_definition)) return false;
  if (value.auto_config !== undefined) {
    const config = value.auto_config;
    if (
      !plainObject(config) ||
      Object.keys(config).some(
        (key) => !['model_type', 'finetune_mode', 'action_horizon', 'action_dim'].includes(key),
      ) ||
      !text(config.model_type) ||
      !text(config.finetune_mode) ||
      typeof config.action_horizon !== 'number' ||
      typeof config.action_dim !== 'number'
    )
      return false;
  }
  if (value.dataset_bindings !== undefined) {
    if (!Array.isArray(value.dataset_bindings) || value.dataset_bindings.length > 64) return false;
    for (const binding of value.dataset_bindings) {
      if (
        !plainObject(binding) ||
        Object.keys(binding).some(
          (key) =>
            !['dataset_asset_id', 'norm_asset_id', 'weight', 'recipe_index', 'processor'].includes(
              key,
            ),
        ) ||
        !text(binding.dataset_asset_id) ||
        !text(binding.norm_asset_id) ||
        typeof binding.weight !== 'number'
      )
        return false;
      if (binding.recipe_index !== undefined && typeof binding.recipe_index !== 'number')
        return false;
      if (
        binding.processor !== undefined &&
        (!plainObject(binding.processor) ||
          Object.keys(binding.processor).some((key) => !['name', 'parameters'].includes(key)) ||
          !text(binding.processor.name) ||
          !plainObject(binding.processor.parameters))
      )
        return false;
    }
  }
  return true;
}

function validContext(value: Record<string, unknown>) {
  if (
    !stringList(value.edited_parameters) ||
    !stringList(value.edited_stage_parameters) ||
    typeof value.entrypoint_edited !== 'boolean' ||
    typeof value.advanced_recipe !== 'boolean'
  )
    return false;
  for (const key of ['recipe_defaults', 'stage_recipe_defaults'])
    if (value[key] !== undefined && !numbers(value[key])) return false;
  if (
    value.recipe_datasets !== undefined &&
    (!Array.isArray(value.recipe_datasets) ||
      value.recipe_datasets.length > 64 ||
      value.recipe_datasets.some(
        (slot) =>
          !plainObject(slot) ||
          Object.keys(slot).some(
            (key) => !['recipe_index', 'weight', 'processing', 'multiple'].includes(key),
          ) ||
          typeof slot.recipe_index !== 'number' ||
          typeof slot.weight !== 'number' ||
          !text(slot.processing) ||
          (slot.multiple !== undefined && typeof slot.multiple !== 'boolean'),
      ))
  )
    return false;
  if (value.stage_recipe_info !== undefined) {
    const info = value.stage_recipe_info;
    if (
      !plainObject(info) ||
      Object.keys(info).some((key) => !['values', 'mixedDropout', 'valueHead'].includes(key)) ||
      !numbers(info.values) ||
      typeof info.mixedDropout !== 'boolean' ||
      !Array.isArray(info.valueHead) ||
      info.valueHead.length > 64 ||
      info.valueHead.some(
        (row) =>
          !plainObject(row) ||
          Object.keys(row).some((key) => !['path', 'value'].includes(key)) ||
          !text(row.path) ||
          !text(row.value),
      )
    )
      return false;
  }
  if (value.recipe_editor_draft !== undefined) {
    const editor = value.recipe_editor_draft;
    if (
      !plainObject(editor) ||
      Object.keys(editor).some(
        (key) =>
          ![
            'scope',
            'sourceSha256',
            'fields',
            'saveMode',
            'saveName',
            'configName',
            'description',
          ].includes(key),
      ) ||
      !text(editor.scope) ||
      !recordOf(editor.fields, text)
    )
      return false;
    if (
      editor.sourceSha256 !== undefined &&
      (typeof editor.sourceSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(editor.sourceSha256))
    )
      return false;
    if (editor.saveMode !== undefined && !['new', 'revision'].includes(String(editor.saveMode)))
      return false;
    for (const key of ['saveName', 'configName', 'description'])
      if (editor[key] !== undefined && !text(editor[key])) return false;
  }
  return true;
}

/** A draft contains editable inputs, never a reusable preflight or launch approval. */
export function encodeTrainingDraft(snapshot: TrainingDraftSnapshot): JsonObject {
  if (!validRequest(snapshot.request)) fail();
  const request = Object.fromEntries(
    Object.entries(snapshot.request).filter(([key]) => requestKeys.has(key)),
  );
  const raw: JsonObject = {
    schema_version: 1,
    request,
    step: snapshot.step === 0 ? 0 : 1,
    advanced_recipe: snapshot.advancedRecipe,
    edited_parameters: snapshot.editedParameters,
    edited_stage_parameters: snapshot.editedStageParameters,
    entrypoint_edited: snapshot.entrypointEdited,
  };
  for (const [key, value] of Object.entries({
    recipe_datasets: snapshot.recipeDatasets,
    recipe_defaults: snapshot.recipeDefaults,
    stage_recipe_defaults: snapshot.stageRecipeDefaults,
    stage_recipe_info: snapshot.stageRecipeInfo,
    recipe_editor_draft: snapshot.recipeEditorDraft,
  }))
    if (value !== undefined) raw[key] = value;
  if (!validRequest(request) || !validContext(raw)) fail();
  const specials: SpecialEntry[] = [];
  const wire = cloneData(raw, specials) as JsonObject;
  wire.special_values = specials;
  return boundedWire(wire);
}

export function decodeTrainingDraft(content: unknown): TrainingDraftSnapshot | undefined {
  try {
    const wire = boundedWire(content);
    if (
      wire.schema_version !== 1 ||
      Object.keys(wire).some((key) => !envelopeKeys.has(key)) ||
      ![0, 1, 2].includes(Number(wire.step)) ||
      typeof wire.step !== 'number' ||
      !Array.isArray(wire.special_values)
    )
      return undefined;
    const paths = new Set<string>();
    for (const entry of wire.special_values) {
      if (
        !plainObject(entry) ||
        Object.keys(entry).some((key) => !['path', 'value'].includes(key)) ||
        !Array.isArray(entry.path) ||
        entry.path.length < 2 ||
        entry.path.length > MAX_TRAINING_DRAFT_DEPTH ||
        entry.path[0] === 'special_values' ||
        !specialValues.has(entry.value as SpecialValue)
      )
        return undefined;
      const path = entry.path as Path;
      if (
        path.some(
          (key) =>
            (typeof key !== 'string' && (!Number.isSafeInteger(key) || key < 0)) ||
            (typeof key === 'string' && unsafeKey(key)),
        )
      )
        return undefined;
      const identity = JSON.stringify(path);
      if (paths.has(identity)) return undefined;
      paths.add(identity);
      let parent: unknown = wire;
      for (const key of path.slice(0, -1)) {
        if ((!plainObject(parent) && !Array.isArray(parent)) || !own(parent, key)) return undefined;
        parent = (parent as Record<string | number, unknown>)[key];
      }
      const key = path.at(-1)!;
      if (
        (!plainObject(parent) && !Array.isArray(parent)) ||
        !own(parent, key) ||
        (parent as Record<string | number, unknown>)[key] !== null
      )
        return undefined;
      (parent as Record<string | number, unknown>)[key] =
        entry.value === 'undefined'
          ? undefined
          : entry.value === 'nan'
            ? NaN
            : entry.value === 'positive_infinity'
              ? Infinity
              : entry.value === 'negative_infinity'
                ? -Infinity
                : -0;
    }
    if (!validRequest(wire.request) || !validContext(wire)) return undefined;
    return {
      request: wire.request,
      step: wire.step === 0 ? 0 : 1,
      advancedRecipe: wire.advanced_recipe as boolean,
      editedParameters: wire.edited_parameters as string[],
      editedStageParameters: wire.edited_stage_parameters as string[],
      entrypointEdited: wire.entrypoint_edited as boolean,
      ...(wire.recipe_datasets !== undefined
        ? { recipeDatasets: wire.recipe_datasets as RecipeDatasetSlot[] }
        : {}),
      ...(wire.recipe_defaults !== undefined
        ? { recipeDefaults: wire.recipe_defaults as Record<string, number> }
        : {}),
      ...(wire.stage_recipe_defaults !== undefined
        ? { stageRecipeDefaults: wire.stage_recipe_defaults as Record<string, number> }
        : {}),
      ...(wire.stage_recipe_info !== undefined
        ? { stageRecipeInfo: wire.stage_recipe_info as TrainingDraftSnapshot['stageRecipeInfo'] }
        : {}),
      ...(wire.recipe_editor_draft !== undefined
        ? { recipeEditorDraft: wire.recipe_editor_draft as unknown as RecipeEditorDraft }
        : {}),
    };
  } catch {
    return undefined;
  }
}

export const stringifyTrainingDraft = (snapshot: TrainingDraftSnapshot) =>
  JSON.stringify(encodeTrainingDraft(snapshot));
export function parseTrainingDraft(value: string): TrainingDraftSnapshot | undefined {
  if (typeof value !== 'string' || value.length > MAX_TRAINING_DRAFT_BYTES) return undefined;
  try {
    return decodeTrainingDraft(JSON.parse(value));
  } catch {
    return undefined;
  }
}
