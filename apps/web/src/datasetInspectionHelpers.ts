import type { DatasetProfile } from './DatasetProfileForm';
import { sourceActionHorizon, type DatasetFieldAnnotation } from './datasetFieldAnnotations';
import type { ComposedProcessingSpec } from './composedProcessing';

export interface DatasetInspectionDTO {
  schema_version: 1;
  path: string;
  source: { info_sha256: string; tasks_sha256: string | null };
  features: { key: string; dtype: string; shape: number[]; names: string[] }[];
  suggestions: Partial<Omit<DatasetProfile, 'schema_version' | 'processing'>>;
  processing: {
    action_key?: string;
    prompt_key?: string;
    state_history?: number;
    state_representation?: 'eef' | 'joint';
    state_fields?: Omit<ComposedProcessingSpec['state_fields'][number], 'normalize'>[];
    action_layout?: Omit<ComposedProcessingSpec['action_layout'][number], 'normalize'>[];
    cameras?: { key: string; kind: 'rgb' | 'depth'; layout: 'chw' | 'hwc' }[];
  };
  evidence: { field: string; kind: 'metadata' | 'inferred'; reason: string; keys: string[] }[];
  warnings: string[];
}
export interface DatasetInspectionQuery {
  id: string;
  node_id: string;
  path: string;
  status: 'PENDING' | 'READY' | 'FAILED';
  inspection: DatasetInspectionDTO | null;
  error: { code: string; message: string } | null;
}
export interface InspectionDraftState {
  initial: DatasetProfile;
  registered: boolean;
  touched: Set<string>;
  automatic: Map<string, unknown>;
  originals?: Map<string, unknown>;
}
const copy = <T>(value: T): T => structuredClone(value);
const equal = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const related = (a: string, b: string) =>
  a === b ||
  a.startsWith(b + '.') ||
  b.startsWith(a + '.') ||
  a.startsWith(b + '[') ||
  b.startsWith(a + '[');
export const inspectionFieldTouched = (state: InspectionDraftState, path: string) =>
  [...state.touched].some((entry) => related(entry, path));

function fieldAt(value: unknown, path: string): unknown {
  if (!path) return value;
  return path
    .replace(/\[(\d+)\]/g, '.$1')
    .split('.')
    .reduce<unknown>(
      (current, key) =>
        current && typeof current === 'object'
          ? (current as Record<string, unknown>)[key]
          : undefined,
      value,
    );
}
export function clearAutomaticInspectionDefaults(
  current: DatasetProfile,
  state: InspectionDraftState,
) {
  const profile = copy(current);
  clearAutomaticAnnotations(profile, state);
  for (const [path, automatic] of [...state.automatic].sort(
    ([left], [right]) => left.length - right.length,
  )) {
    if (
      path === 'field_annotations' ||
      inspectionFieldTouched(state, path) ||
      !equal(fieldAt(profile, path), automatic)
    )
      continue;
    const keys = path.replace(/\[(\d+)\]/g, '.$1').split('.');
    const key = keys.pop()!,
      parent = fieldAt(profile, keys.join('.')) as Record<string, unknown> | undefined;
    if (!parent) continue;
    const original = state.originals?.get(path);
    if (original === undefined) delete parent[key];
    else parent[key] = copy(original);
  }
  state.automatic.clear();
  state.originals?.clear();
  return profile;
}

type AutomaticAnnotation = Partial<DatasetFieldAnnotation> & { key: string };

function sourceAnnotationHorizons(
  rows: DatasetFieldAnnotation[],
  inspection: DatasetInspectionDTO,
): DatasetFieldAnnotation[] {
  return rows.map((row) => {
    const { horizon, ...source } = row;
    if (row.role !== 'action') return source;
    const shape =
      inspection.features.find((feature) => feature.key === row.key)?.shape ?? row.shape;
    const actual = sourceActionHorizon(shape);
    // A model's prediction horizon is not evidence of a stored sequence. Do
    // not accept a conflicting declaration or invent an axis for higher ranks.
    return actual !== undefined && (horizon === undefined || horizon === actual)
      ? { ...source, horizon: actual }
      : source;
  });
}

function clearAutomaticAnnotations(profile: DatasetProfile, state: InspectionDraftState) {
  const automatic = state.automatic.get('field_annotations') as AutomaticAnnotation[] | undefined;
  if (!automatic || !profile.field_annotations || state.touched.has('field_annotations')) return;
  const originals = (state.originals?.get('field_annotations') || []) as DatasetFieldAnnotation[];
  const rows: DatasetFieldAnnotation[] = [];
  profile.field_annotations.forEach((current, index) => {
    const prior = originals.find((row) => row.key === current.key);
    const generated = automatic.find((row) => row.key === current.key);
    if (!generated) {
      rows.push(current);
      return;
    }
    const path = `field_annotations[${index}]`;
    if (!prior) {
      if (
        inspectionFieldTouched(state, path) ||
        Object.entries(generated).some(
          ([key, value]) => !equal(current[key as keyof DatasetFieldAnnotation], value),
        )
      )
        rows.push(current);
      return;
    }
    const row = copy(current) as unknown as Record<string, unknown>;
    for (const [key, value] of Object.entries(generated)) {
      if (
        key === 'key' ||
        inspectionFieldTouched(state, `${path}.${key}`) ||
        !equal(row[key], value)
      )
        continue;
      const original = prior[key as keyof DatasetFieldAnnotation];
      if (original === undefined) delete row[key];
      else row[key] = copy(original);
    }
    rows.push(row as unknown as DatasetFieldAnnotation);
  });
  if (rows.length || originals.length) profile.field_annotations = rows;
  else delete profile.field_annotations;
}

/** Older Agents supply source facts through the inspection DTO's bounded fields. */
function inspectionAnnotations(inspection: DatasetInspectionDTO): DatasetFieldAnnotation[] {
  if (inspection.suggestions.field_annotations)
    return sourceAnnotationHorizons(inspection.suggestions.field_annotations, inspection);
  const states = new Set(inspection.processing.state_fields?.map((row) => row.key));
  const actionKey = inspection.processing.action_key;
  const tokens = (key: string) => '_' + key.toLowerCase().replace(/[^a-z0-9]+/g, '_') + '_';
  const component = (key: string): DatasetFieldAnnotation['component'] => {
    const name = tokens(key);
    if (/_(gripper|grip|finger)_/.test(name)) return 'gripper';
    if (/_(joint\d*|joints)_/.test(name)) return 'joint';
    if (/_(rot|rotation|quaternion|quat|rotvec|rot6d|axis_angle)_/.test(name)) return 'rotation';
    if (/_(pos|position|translation)_/.test(name)) return 'position';
    return 'other';
  };
  return sourceAnnotationHorizons(
    inspection.features.flatMap((feature) => {
      if (
        !/^(?:u?int(?:8|16|32|64)?|float(?:16|32|64)?|double|bool)$/.test(feature.dtype) ||
        !feature.shape.length
      )
        return [];
      const name = tokens(feature.key);
      const role =
        feature.key === actionKey || /_(actions?|targets?|commands?|cmd)_/.test(name)
          ? 'action'
          : states.has(feature.key) || ['state', 'observation.state'].includes(feature.key)
            ? 'state'
            : undefined;
      if (!role) return [];
      const kind = component(feature.key);
      const declared =
        role === 'action'
          ? inspection.suggestions.action_representation
          : inspection.processing.state_representation;
      const representation =
        kind === 'gripper'
          ? 'gripper'
          : /_joint\d*_/.test(name)
            ? 'joint'
            : /_eef_/.test(name)
              ? 'eef'
              : declared || 'unknown';
      const arms = [
        /_((left)|(robot0)|(arm0))_/.test(name),
        /_((right)|(robot1)|(arm1))_/.test(name),
      ];
      // No numeric heuristic, default action mode, normalization or ordering is generated.
      return [
        {
          key: feature.key,
          role,
          representation,
          value_mode: 'unknown',
          component: kind,
          arm: arms[0] !== arms[1] ? (arms[0] ? 'left' : 'right') : 'unknown',
          shape: copy(feature.shape),
          dtype: feature.dtype,
          names: copy(feature.names),
        } as DatasetFieldAnnotation,
      ];
    }),
    inspection,
  );
}

function mergeAnnotations(
  profile: DatasetProfile,
  inspection: DatasetInspectionDTO,
  state: InspectionDraftState,
  applied: string[],
  kept: string[],
) {
  const candidates = inspectionAnnotations(inspection);
  if (!candidates.length) return;
  if (state.touched.has('field_annotations')) {
    kept.push('field_annotations');
    return;
  }
  const before = copy(profile.field_annotations || []);
  const automatic = (state.automatic.get('field_annotations') || []) as AutomaticAnnotation[];
  const generated: AutomaticAnnotation[] = copy(automatic);
  const rows = copy(before);
  for (const candidate of candidates) {
    const index = rows.findIndex((row) => row.key === candidate.key);
    if (index < 0) {
      rows.push(copy(candidate));
      generated.push(copy(candidate));
      continue;
    }
    const row = rows[index] as unknown as Record<string, unknown>;
    const prior = automatic.find((entry) => entry.key === candidate.key);
    const next = generated.find((entry) => entry.key === candidate.key) || { key: candidate.key };
    for (const [key, value] of Object.entries(candidate)) {
      if (key === 'key') continue;
      if (key === 'horizon' && row.role !== 'action') continue;
      const path = `field_annotations[${index}].${key}`;
      const isAutomatic =
        prior &&
        Object.hasOwn(prior, key) &&
        equal(row[key], prior[key as keyof AutomaticAnnotation]);
      if (!inspectionFieldTouched(state, path) && (row[key] === undefined || isAutomatic)) {
        row[key] = copy(value);
        (next as Record<string, unknown>)[key] = copy(value);
      } else kept.push(path);
    }
    const horizonPath = `field_annotations[${index}].horizon`;
    if (
      row.role !== 'action' ||
      (!Object.hasOwn(candidate, 'horizon') &&
        prior?.horizon !== undefined &&
        equal(row.horizon, prior.horizon) &&
        !inspectionFieldTouched(state, horizonPath))
    ) {
      delete row.horizon;
      delete next.horizon;
    }
    if (!generated.includes(next)) generated.push(next);
  }
  if (!equal(before, rows)) {
    state.originals ||= new Map();
    if (!state.originals.has('field_annotations'))
      state.originals.set('field_annotations', copy(profile.field_annotations));
    profile.field_annotations = rows;
    applied.push('field_annotations');
  }
  state.automatic.set('field_annotations', generated);
}

export function recordInspectionEdits(
  state: InspectionDraftState,
  before: DatasetProfile,
  after: DatasetProfile,
) {
  const walk = (a: unknown, b: unknown, path: string) => {
    if (Object.is(a, b)) return;
    if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
      a.forEach((value, index) => walk(value, b[index], `${path}[${index}]`));
      return;
    }
    if (
      a &&
      b &&
      typeof a === 'object' &&
      typeof b === 'object' &&
      !Array.isArray(a) &&
      !Array.isArray(b)
    ) {
      const left = a as Record<string, unknown>,
        right = b as Record<string, unknown>;
      for (const key of new Set([...Object.keys(left), ...Object.keys(right)]))
        walk(left[key], right[key], path ? `${path}.${key}` : key);
      return;
    }
    state.touched.add(path);
  };
  walk(before, after, '');
}

export function canonicalDatasetInspectionPath(path: string): string {
  const value = path.trim();
  if (
    !value.startsWith('/') ||
    value.length > 4096 ||
    /[\\\u0000-\u001f\u007f-\u009f]/.test(value) ||
    value.split('/').includes('..')
  )
    return '';
  return (
    '/' +
    value
      .split('/')
      .filter((part) => part && part !== '.')
      .join('/')
  );
}
export function validDatasetInspectionPath(path: string): boolean {
  return !!canonicalDatasetInspectionPath(path);
}
export function inspectionScope(nodeId: string | undefined, path: string) {
  return `${nodeId || ''}\0${canonicalDatasetInspectionPath(path)}`;
}
export function inspectionMatches(
  result: DatasetInspectionQuery,
  nodeId: string,
  path: string,
  id?: string,
) {
  return (
    result.node_id === nodeId &&
    result.path === canonicalDatasetInspectionPath(path) &&
    (!id || result.id === id) &&
    (result.status !== 'READY' ||
      (result.inspection?.schema_version === 1 &&
        result.inspection.path === canonicalDatasetInspectionPath(path)))
  );
}

export function mergeDatasetInspection(
  current: DatasetProfile,
  inspection: DatasetInspectionDTO,
  state: InspectionDraftState,
) {
  const profile = copy(current),
    applied: string[] = [],
    kept: string[] = [];
  const eligible = (path: string, value: unknown, empty: boolean) =>
    !inspectionFieldTouched(state, path) &&
    ((state.automatic.has(path) && equal(value, state.automatic.get(path))) || empty);
  const put = (path: string, value: unknown, assign: () => void) => {
    state.originals ||= new Map();
    if (!state.originals.has(path)) state.originals.set(path, copy(fieldAt(profile, path)));
    assign();
    state.automatic.set(path, copy(value));
    applied.push(path);
  };
  const simple = [
    'format',
    'robot_type',
    'arm_mode',
    'action_representation',
    'task_description',
    'action_dim',
    'state_dim',
    'fps',
    'action_mode',
    'rotation_representation',
    'feature_keys',
    'cameras',
  ] as const;
  for (const key of simple) {
    const value = inspection.suggestions[key];
    if (value === undefined) continue;
    if (
      key === 'state_dim' &&
      (!Number.isInteger(value) ||
        (value as number) <= 0 ||
        !inspection.features.some(
          (feature) =>
            ['state', 'observation.state'].includes(feature.key) &&
            feature.shape.length > 0 &&
            feature.shape[feature.shape.length - 1] === value,
        ))
    ) {
      // Older Agents summed segmented state fields; only a real unified vector proves this fact.
      kept.push(key);
      continue;
    }
    const old = profile[key];
    let empty =
      old === undefined ||
      (typeof old === 'string' && old.trim() === '') ||
      (Array.isArray(old) && old.length === 0);
    if (key === 'cameras')
      empty = profile.cameras.every((camera) => !camera.key.trim() && !camera.name.trim());
    if (
      !state.registered &&
      [
        'format',
        'arm_mode',
        'action_representation',
        'action_mode',
        'rotation_representation',
      ].includes(key)
    )
      empty = equal(old, state.initial[key]);
    if (
      state.registered &&
      [
        'format',
        'arm_mode',
        'action_representation',
        'action_mode',
        'rotation_representation',
      ].includes(key)
    )
      empty = false;
    // Physical action declarations require explicit metadata, never a dimension/name heuristic.
    if (
      ['action_mode', 'rotation_representation'].includes(key) &&
      !inspection.evidence.some(
        (evidence) => evidence.field === key && evidence.kind === 'metadata',
      )
    ) {
      kept.push(key);
      continue;
    }
    if (eligible(key, old, empty))
      put(key, value, () => {
        (profile as unknown as Record<string, unknown>)[key] = copy(value);
      });
    else kept.push(key);
  }
  // Inspection annotates source fields; training owns all processing choices.
  mergeAnnotations(profile, inspection, state, applied, kept);
  return { profile, applied: [...new Set(applied)], kept: [...new Set(kept)] };
}
