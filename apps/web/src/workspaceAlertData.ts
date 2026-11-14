import type { Alert } from './types';

export const isDiskCapacityAlert = (alert: Alert) => alert.code === 'DISK_CAPACITY_HIGH';
const timestamp = (alert: Alert) => Date.parse(alert.timestamp || alert.created_at || '');

function canonicalMountpoint(value: string | undefined) {
  if (
    typeof value !== 'string' ||
    !value.startsWith('/') ||
    /[\u0000-\u001f\u007f-\u009f]/u.test(value)
  )
    return;
  const parts = value.split('/');
  if (parts.includes('..')) return;
  return `/${parts.filter((part) => part && part !== '.').join('/')}`;
}

export function deduplicateDiskAlerts(alerts: Alert[]) {
  const positions = new Map<string, number>();
  const result: Alert[] = [];
  for (const alert of alerts) {
    const mountpoint = isDiskCapacityAlert(alert) && canonicalMountpoint(alert.details?.mountpoint);
    if (!alert.node_id || !mountpoint) {
      result.push(alert);
      continue;
    }
    const key = JSON.stringify([alert.node_id, mountpoint]);
    const previous = positions.get(key);
    if (previous === undefined) {
      positions.set(key, result.length);
      result.push(alert);
      continue;
    }
    const next = timestamp(alert);
    const prior = timestamp(result[previous]);
    if (Number.isFinite(next) && (!Number.isFinite(prior) || next > prior))
      result[previous] = alert;
  }
  return result;
}
