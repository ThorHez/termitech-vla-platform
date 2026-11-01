import type { GpuReservation, Node } from './types';

export interface GpuReservationRecord {
  nodeId: string;
  nodeName: string;
  userId: string;
  username: string;
  displayName: string;
  gpuUuids: string[];
  gpuLabels: string[];
  createdAt: string;
}

const compareText = (left: string, right: string) => (left < right ? -1 : left > right ? 1 : 0);

function newestFirst(left: string, right: string) {
  const leftTime = Date.parse(left);
  const rightTime = Date.parse(right);
  if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime)
    return rightTime - leftTime;
  return compareText(right, left);
}

/** Present each saved batch without merging different users or reservation times. */
export function gpuReservationRecords(
  nodes: readonly Node[],
  reservations?: readonly GpuReservation[],
): GpuReservationRecord[] {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const rows =
    reservations ??
    nodes.flatMap((node) =>
      (node.gpus || []).flatMap((gpu) => (gpu.user_reservation ? [gpu.user_reservation] : [])),
    );
  const devices = new Map<string, GpuReservation>();
  for (const row of rows) {
    const key = JSON.stringify([row.node_id, row.gpu_uuid]);
    const previous = devices.get(key);
    // The server owns one row per device. If snapshots overlap, retain its
    // latest record rather than counting the same GPU twice.
    if (!previous || newestFirst(row.created_at, previous.created_at) < 0) devices.set(key, row);
  }
  const groups = new Map<string, GpuReservationRecord>();
  for (const row of devices.values()) {
    const key = JSON.stringify([row.node_id, row.user_id, row.created_at]);
    let group = groups.get(key);
    if (!group) {
      group = {
        nodeId: row.node_id,
        nodeName: nodeById.get(row.node_id)?.name || row.node_id,
        userId: row.user_id,
        username: row.username,
        displayName: row.display_name || row.username,
        gpuUuids: [],
        gpuLabels: [],
        createdAt: row.created_at,
      };
      groups.set(key, group);
    }
    group.gpuUuids.push(row.gpu_uuid);
  }
  for (const group of groups.values()) {
    const indices = new Map(
      nodeById.get(group.nodeId)?.gpus?.map((gpu, index) => [gpu.uuid, index]) || [],
    );
    group.gpuUuids.sort((left, right) => {
      const leftIndex = indices.get(left) ?? Number.POSITIVE_INFINITY;
      const rightIndex = indices.get(right) ?? Number.POSITIVE_INFINITY;
      return leftIndex - rightIndex || compareText(left, right);
    });
    group.gpuLabels = group.gpuUuids.map((uuid) => {
      const index = indices.get(uuid);
      return index === undefined ? `未上报 GPU（${uuid}）` : `GPU ${index}`;
    });
  }
  return [...groups.values()].sort(
    (left, right) =>
      compareText(left.nodeId, right.nodeId) ||
      newestFirst(left.createdAt, right.createdAt) ||
      compareText(left.userId, right.userId),
  );
}
