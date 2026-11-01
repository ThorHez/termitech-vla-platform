import { api } from './api';
import type { Run } from './types';

export function canDeleteRunRecord(run: Pick<Run, 'state'>) {
  return ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(run.state);
}

export function deleteRunRecord(id: string) {
  return api<void>(`/runs/${encodeURIComponent(id)}`, { method: 'DELETE' });
}
