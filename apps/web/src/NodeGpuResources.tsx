import { useEffect, useId, useState } from 'react';
import type { ReactNode } from 'react';
import { LockKeyhole } from 'lucide-react';
import { date, useApi } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Loading, Panel } from './components';
import type { GpuReservation, Node } from './types';
import './gpu-reservations.css';

type GpuTab = 'status' | 'reservations';

export function GpuReservationDeviceTable({
  node,
  reservations,
}: {
  node: Node;
  reservations?: readonly GpuReservation[];
}) {
  const { user } = useAuth();
  const rows =
    reservations ??
    node.gpus?.flatMap((gpu) => (gpu.user_reservation ? [gpu.user_reservation] : [])) ??
    [];
  const byUuid = new Map(
    rows.filter((row) => row.node_id === node.id).map((row) => [row.gpu_uuid, row]),
  );
  const inventory = node.gpus || [];
  const devices = [
    ...inventory.map((gpu, index) => ({
      uuid: gpu.uuid,
      label: `GPU ${index}`,
      name: gpu.name,
      missing: false,
    })),
    ...[...byUuid.keys()]
      .filter((uuid) => !inventory.some((gpu) => gpu.uuid === uuid))
      .map((uuid) => ({ uuid, label: '未上报 GPU', name: '设备暂未上报', missing: true })),
  ];
  if (!devices.length)
    return <p className="gpu-reservation-empty">当前机器尚未上报 GPU，也没有预留信息。</p>;
  return (
    <div className="table-scroll">
      <table className="gpu-reservation-device-table" aria-label="所有 GPU 的预留信息">
        <thead>
          <tr>
            <th>GPU</th>
            <th>预留用户</th>
            <th>预留时间</th>
            <th>到期时间</th>
            <th>预留状态</th>
          </tr>
        </thead>
        <tbody>
          {devices.map((device) => {
            const reservation = byUuid.get(device.uuid);
            return (
              <tr key={device.uuid}>
                <td>
                  <strong>{device.label}</strong>
                  <span className="subtext">{device.name}</span>
                  <code className="gpu-reservation-device-uuid" title={device.uuid}>
                    {device.uuid}
                  </code>
                </td>
                <td>
                  {reservation ? (
                    <>
                      <strong>{reservation.display_name || reservation.username}</strong>
                      <span className="subtext">
                        {reservation.username}
                        {reservation.user_id === user?.id && ' · 我'}
                      </span>
                    </>
                  ) : (
                    <span className="gpu-reservation-public">公共资源</span>
                  )}
                </td>
                <td>
                  {reservation ? (
                    <time dateTime={reservation.created_at}>{date(reservation.created_at)}</time>
                  ) : (
                    '—'
                  )}
                </td>
                <td>
                  {reservation?.expires_at ? (
                    <time dateTime={reservation.expires_at}>{date(reservation.expires_at)}</time>
                  ) : (
                    '—'
                  )}
                </td>
                <td>
                  <span
                    className={reservation ? 'gpu-reservation-active' : 'gpu-reservation-public'}
                  >
                    {reservation && <LockKeyhole size={12} />}
                    {reservation ? '已预留' : '未预留'}
                  </span>
                  {device.missing && <span className="subtext">预留仍然有效</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function GpuResourceTabs({
  node,
  tab,
  onTabChange,
  onManage,
  reservations,
  loading,
  error,
  retry,
  children,
}: {
  node: Node;
  tab: GpuTab;
  onTabChange: (tab: GpuTab) => void;
  onManage: () => void;
  reservations?: readonly GpuReservation[];
  loading?: boolean;
  error?: Error;
  retry?: () => void;
  children?: ReactNode;
}) {
  const { can } = useAuth();
  const id = useId();
  return (
    <Panel
      title="GPU 资源"
      subtitle={`共 ${node.gpus?.length || 0} 张 GPU`}
      className="node-gpu-resources"
      action={
        can('training.write') && (
          <button className="button outline small" onClick={onManage}>
            <LockKeyhole size={14} />
            管理预留
          </button>
        )
      }
    >
      <div className="tabs gpu-resource-tabs" role="tablist" aria-label="GPU 资源详情">
        {(
          [
            ['status', 'GPU 状态'],
            ['reservations', '预留信息'],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={tab === value}
            aria-controls={`${id}-panel`}
            id={`${id}-${value}`}
            className={tab === value ? 'selected' : ''}
            onClick={() => onTabChange(value)}
          >
            {label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-${tab}`}>
        {tab === 'status' ? (
          children
        ) : (
          <>
            <ErrorNotice error={error} retry={retry} title="预留信息读取失败" />
            {loading && reservations === undefined ? (
              <Loading label="正在读取预留信息…" />
            ) : (
              <>
                <p className="gpu-reservation-tab-hint">
                  显示全部 GPU；已预留的卡仅供对应用户的任务使用，直到主动释放。
                </p>
                <GpuReservationDeviceTable node={node} reservations={reservations} />
              </>
            )}
          </>
        )}
      </div>
    </Panel>
  );
}

export function NodeGpuResources({
  node,
  onManage,
  children,
  refreshToken,
}: {
  node: Node;
  onManage: () => void;
  children: ReactNode;
  refreshToken?: number;
}) {
  const [tab, setTab] = useState<GpuTab>('status');
  const records = useApi<GpuReservation[]>(
    tab === 'reservations' ? `/nodes/${encodeURIComponent(node.id)}/gpu-reservations` : null,
    5000,
  );
  useEffect(() => {
    if (refreshToken) records.refresh();
  }, [refreshToken, records.refresh]);
  return (
    <GpuResourceTabs
      node={node}
      tab={tab}
      onTabChange={setTab}
      onManage={onManage}
      reservations={records.data}
      loading={records.loading}
      error={records.error}
      retry={records.refresh}
    >
      {children}
    </GpuResourceTabs>
  );
}
