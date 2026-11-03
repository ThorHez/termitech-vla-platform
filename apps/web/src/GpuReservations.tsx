import { useEffect, useRef, useState } from 'react';
import { CheckCircle2, LoaderCircle, LockKeyhole } from 'lucide-react';
import { api, date, useApi } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Loading, Modal } from './components';
import { useData } from './data';
import type { GpuReservation, Node } from './types';
import { gpuReservationRecords } from './gpuReservationRecords';
import './gpu-reservations.css';

export interface GpuReservationPolicy {
  max_per_user: number;
  ttl_seconds: number;
  reserved_count: number;
  remaining_count: number;
}

export function reservationSelectionLimit(
  policy: GpuReservationPolicy | undefined,
  currentNodeCount: number,
  selectedCount: number,
) {
  if (!policy) return undefined;
  const total = Math.max(0, policy.reserved_count - currentNodeCount) + selectedCount;
  return total > policy.max_per_user
    ? `每个账号跨机器最多预留 ${policy.max_per_user} 张 GPU；当前选择合计 ${total} 张，请减少选择。`
    : undefined;
}

export function RunGpuReservationNotice({
  node,
  gpuUuids,
}: {
  node?: Node;
  gpuUuids?: readonly string[];
}) {
  if (!node || !gpuUuids?.length) return null;
  const rows =
    node.gpus?.flatMap((gpu) =>
      gpuUuids.includes(gpu.uuid) && gpu.user_reservation ? [gpu.user_reservation] : [],
    ) || [];
  if (!rows.length) return null;
  return (
    <span className="run-gpu-reservation-notice">
      {gpuReservationRecords([node], rows).map((record) => (
        <small key={`${record.userId}-${record.createdAt}`}>
          <LockKeyhole size={12} />
          用户预留：{record.displayName || record.username} · {record.gpuLabels.join('、')}
        </small>
      ))}
      <a href={`#/nodes/${node.id}`}>查看预留记录</a>
    </span>
  );
}

export function GpuReservationSaveReceipt({
  node,
  rows,
}: {
  node: Node;
  rows: readonly GpuReservation[];
}) {
  const records = gpuReservationRecords([node], rows);
  return (
    <div className="gpu-reservation-save-receipt" role="status">
      <CheckCircle2 size={18} />
      <div>
        <strong>
          {rows.length ? `预留已保存，当前为你预留 ${rows.length} 张 GPU` : '已释放你的全部预留'}
        </strong>
        {records.map((record) => (
          <p key={record.createdAt}>
            {record.gpuLabels.join('、')} · {record.displayName || record.username} ·{' '}
            {date(record.createdAt)}
            {record.expiresAt && <> · 到期 {date(record.expiresAt)}</>}
          </p>
        ))}
        {rows.length > 0 && (
          <p>预留到期自动释放，也可由你或管理员提前释放；运行中的任务继续执行。</p>
        )}
      </div>
    </div>
  );
}

export function GpuReservations({
  node,
  close,
  changed,
}: {
  node: Node;
  close: () => void;
  changed?: () => void;
}) {
  const { user, can } = useAuth();
  const { nodes, overview } = useData();
  const path = `/nodes/${encodeURIComponent(node.id)}/gpu-reservations`;
  const reservations = useApi<GpuReservation[]>(path, 5000);
  const policy = useApi<GpuReservationPolicy>('/gpu-reservation-policy', 5000);
  const [selected, setSelected] = useState<string[]>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const [saved, setSaved] = useState<GpuReservation[]>();
  const pending = useRef(false);
  const writable = can('training.write') && Boolean(user);
  const liveNode = nodes.data?.find((item) => item.id === node.id) || node;
  const rows = reservations.data || [];
  const mine = rows.filter((row) => row.user_id === user?.id).map((row) => row.gpu_uuid);
  const selectionLimit = reservationSelectionLimit(policy.data, mine.length, selected?.length || 0);
  useEffect(() => {
    if (reservations.data && selected === undefined) {
      setSelected(
        reservations.data.filter((row) => row.user_id === user?.id).map((row) => row.gpu_uuid),
      );
    }
  }, [reservations.data, selected, user?.id]);
  const inventory = liveNode.gpus || [];
  // Retain missing devices in the editor so their reservations can be released.
  const devices = [
    ...inventory.map((gpu, index) => ({
      uuid: gpu.uuid,
      name: `GPU ${index} · ${gpu.name}`,
      present: true,
    })),
    ...rows
      .filter((row) => !inventory.some((gpu) => gpu.uuid === row.gpu_uuid))
      .map((row) => ({ uuid: row.gpu_uuid, name: '未上报的 GPU', present: false })),
  ];
  const unchanged =
    selected !== undefined &&
    selected.length === mine.length &&
    selected.every((uuid) => mine.includes(uuid));
  const safeClose = () => {
    if (!pending.current) close();
  };
  const mutate = async (gpuUuid?: string) => {
    if (pending.current || (!gpuUuid && (selected === undefined || selectionLimit)) || !writable)
      return;
    if (gpuUuid && !can('machines.manage')) return;
    pending.current = true;
    setBusy(true);
    setError(undefined);
    try {
      if (gpuUuid) {
        await api(`${path}/${encodeURIComponent(gpuUuid)}`, { method: 'DELETE' });
      } else {
        const result = await api<GpuReservation[]>(path, {
          method: 'PUT',
          body: JSON.stringify({ gpu_uuids: selected }),
        });
        const ownRows = result.filter((row) => row.user_id === user?.id);
        setSaved(ownRows);
        setSelected(ownRows.map((row) => row.gpu_uuid));
      }
      reservations.refresh();
      policy.refresh();
      nodes.refresh();
      overview.refresh();
      changed?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      pending.current = false;
      setBusy(false);
    }
  };
  return (
    <Modal
      title="GPU 预留"
      subtitle={node.name}
      className="gpu-reservations-modal"
      close={safeClose}
      footer={
        <>
          <button className="button outline" disabled={busy} onClick={safeClose}>
            关闭
          </button>
          {writable && (
            <button
              className="button primary"
              disabled={
                busy ||
                !reservations.data ||
                selected === undefined ||
                Boolean(selectionLimit) ||
                (unchanged && !selected.length)
              }
              onClick={() => void mutate()}
            >
              {busy && <LoaderCircle size={15} className="spin" />}
              {busy ? '保存中…' : unchanged && selected?.length ? '续期我的预留' : '保存我的预留'}
            </button>
          )}
        </>
      }
    >
      <p className="gpu-reservation-explanation">
        预留后，仅你的训练和推理任务可以使用这些 GPU。预留到期自动释放，运行中的任务会继续占用其
        GPU。
      </p>
      {policy.data && (
        <p className="field-hint">
          每个账号跨机器最多预留 {policy.data.max_per_user} 张，当前共 {policy.data.reserved_count}{' '}
          张； 每次保存或续期后有效 {policy.data.ttl_seconds / 3600} 小时。
        </p>
      )}
      {selectionLimit && (
        <p className="field-hint required" role="alert">
          {selectionLimit}
        </p>
      )}
      {saved && <GpuReservationSaveReceipt node={liveNode} rows={saved} />}
      <ErrorNotice
        error={error || reservations.error || policy.error}
        title={error ? '预留保存失败' : '预留信息读取失败'}
        retry={reservations.refresh}
      />
      {!reservations.data ? (
        !reservations.error && <Loading label="正在读取 GPU 预留…" />
      ) : (
        <>
          {writable && (
            <div className="gpu-reservation-toolbar">
              <span>
                预留用户：<strong>{user?.display_name || user?.username}</strong> · 已选{' '}
                {selected?.length || 0} 张
              </span>
              <button
                className="text-link"
                disabled={busy || !selected?.length}
                onClick={() => {
                  setSelected([]);
                  setSaved(undefined);
                }}
              >
                取消全部勾选
              </button>
            </div>
          )}
          <div className="gpu-reservation-options" aria-label="按用户预留 GPU">
            {!devices.length && <p className="field-hint">当前机器尚未上报 GPU。</p>}
            {devices.map((device) => {
              const row = rows.find((item) => item.gpu_uuid === device.uuid);
              const own = row?.user_id === user?.id;
              const foreign = Boolean(row && !own);
              const checked = selected?.includes(device.uuid) || false;
              const disabled =
                !writable ||
                busy ||
                (!checked &&
                  (foreign ||
                    !device.present ||
                    liveNode.status !== 'ONLINE' ||
                    Boolean(
                      reservationSelectionLimit(
                        policy.data,
                        mine.length,
                        (selected?.length || 0) + 1,
                      ),
                    )));
              return (
                <div className={`gpu-reservation-option ${own ? 'mine' : ''}`} key={device.uuid}>
                  <label>
                    {writable && (
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={disabled}
                        aria-label={`预留 ${device.name}`}
                        onChange={(event) => {
                          setSelected((previous) =>
                            event.target.checked
                              ? [...(previous || []), device.uuid]
                              : previous?.filter((uuid) => uuid !== device.uuid),
                          );
                          setError(undefined);
                          setSaved(undefined);
                        }}
                      />
                    )}
                    <span>
                      <strong>{device.name}</strong>
                      <code>{device.uuid}</code>
                      <span>
                        {row
                          ? `${own ? '我的预留' : '已预留'} · ${row.display_name || row.username}（${row.username}）`
                          : '公共资源'}
                        {!device.present && ' · 设备未上报'}
                      </span>
                      {row && (
                        <span>
                          预留于 {date(row.created_at)} ·{' '}
                          {row.expires_at ? `到期 ${date(row.expires_at)}` : '正在读取到期时间'}
                        </span>
                      )}
                    </span>
                  </label>
                  {foreign && can('machines.manage') && (
                    <button
                      className="button outline small"
                      disabled={busy}
                      aria-label={`释放 ${row?.username} 的 ${device.name}`}
                      onClick={() => void mutate(device.uuid)}
                    >
                      释放预留
                    </button>
                  )}
                </div>
              );
            })}
          </div>
          <p className="field-hint">
            {writable
              ? '勾选需要预留的卡，取消勾选可释放自己的预留。其他用户已预留或已占用的卡无法申请。'
              : '当前账号可查看预留；申请预留需要训练操作权限。'}
          </p>
        </>
      )}
    </Modal>
  );
}
