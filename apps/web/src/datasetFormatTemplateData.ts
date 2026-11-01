import {
  normalizedDatasetProfile,
  readDatasetProfile,
  type DatasetProfile,
} from './DatasetProfileForm';

export interface DatasetFormatTemplateSummary {
  id: string;
  name: string;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface DatasetFormatTemplate extends DatasetFormatTemplateSummary {
  profile: DatasetProfile;
}

/** Only source metadata belongs to a format template; legacy processing is training-owned. */
export function datasetFormatProfile(value: unknown): DatasetProfile | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const { processing: _processing, ...raw } = value as Record<string, unknown>;
  const profile = readDatasetProfile(raw);
  return profile ? structuredClone(normalizedDatasetProfile(profile)) : undefined;
}

export function datasetTemplateNameError(value: string): string | undefined {
  const name = value.trim();
  if (!name) return '请填写模板名称。';
  if (Array.from(name).length > 100 || /[\u0000-\u001f\u007f-\u009f]/.test(name))
    return '模板名称最多 100 个字符，不能包含控制字符。';
  return undefined;
}
