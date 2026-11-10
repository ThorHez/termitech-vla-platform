import { useEffect, useRef, useState } from 'react';
import { createOperationClient, navigate, useApi } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Field, Modal } from './components';
import { useData } from './data';
import { GeneratedConfigReview } from './GeneratedConfigReview';
import { PreflightChecks } from './PreflightChecks';
import { resumeCheckpoints, resumeTrainingRequest, retryTrainingName } from './runRetry';
import { submitTrainingRun } from './runSubmission';
import { configurationConfirmed } from './trainingConfiguration';
import type { Preview, Run } from './types';

export function ResumeTrainingWizard({ sourceRun, close }: { sourceRun?: Run; close: () => void }) {
  const { user, can } = useAuth();
  const { nodes, runs, refresh } = useData();
  const candidates = (sourceRun ? [sourceRun] : (runs.data ?? [])).flatMap((run) =>
    resumeCheckpoints(run, user?.id ?? '', user?.role === 'admin').map((checkpoint) => ({
      run,
      checkpoint,
      step: Number(checkpoint.metadata?.step),
    })),
  );
  const [checkpointId, setCheckpointId] = useState(() =>
    sourceRun ? (candidates[0]?.checkpoint.id ?? '') : '',
  );
  const [name, setName] = useState<string>();
  const [steps, setSteps] = useState<string>();
  const [pending, setPending] = useState<Preview>();
  const [confirmed, setConfirmed] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const active = useRef(true);
  const submitting = useRef(false);
  const idempotencyKey = useRef(crypto.randomUUID());
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const result = useApi<Preview>(pending ? `/run-previews/${pending.id}` : null, 1800);
  const preview = result.data?.id === pending?.id ? result.data : pending;
  const config = preview?.code_snapshot?.automatic_preflight?.plan?.generated_config;
  const selected = candidates.find(({ checkpoint }) => checkpoint.id === checkpointId);
  const minSteps = selected ? selected.step + 2 : 1;
  const previousTotal = selected?.run.parameters?.steps ?? 0;
  const totalSteps =
    steps === undefined && previousTotal >= minSteps ? previousTotal : Number(steps);
  const runName = name ?? (selected ? `${retryTrainingName(selected.run.name)} · 续训` : '');
  const valid =
    !!selected &&
    !!runName.trim() &&
    Number.isSafeInteger(totalSteps) &&
    totalSteps >= minSteps &&
    totalSteps <= 1000000;
  const checking = preview?.status === 'PENDING';

  function resetPreview() {
    setPending(undefined);
    setConfirmed(undefined);
    setError(undefined);
  }

  async function action(submit: boolean) {
    if (!can('training.write') || submitting.current || !selected || !valid) return;
    submitting.current = true;
    setBusy(true);
    setError(undefined);
    try {
      if (submit) {
        if (preview?.status !== 'READY' || !configurationConfirmed(config, confirmed)) return;
        await submitTrainingRun({
          preflightId: preview.id,
          confirmedConfigSha256: confirmed,
          idempotencyKey: idempotencyKey.current,
          finishDraft: async () => {},
          isActive: () => active.current,
          onSubmitted: (run) => {
            refresh();
            close();
            navigate(`/runs/${run.id}`);
          },
        });
      } else {
        resetPreview();
        const client = createOperationClient();
        const request = resumeTrainingRequest(
          selected.run,
          selected.checkpoint,
          totalSteps,
          runName.trim(),
        );
        const next = await client.post<Preview>('/run-previews', request);
        client.requireCurrent();
        if (active.current) {
          idempotencyKey.current = crypto.randomUUID();
          setPending(next);
        }
      }
    } catch (cause) {
      if (active.current) setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      submitting.current = false;
      if (active.current) setBusy(false);
    }
  }

  return (
    <Modal
      title="从检查点续训"
      subtitle="恢复 SFT 的模型、优化器、EMA 和全局步数，后续产出写入新任务。"
      close={() => {
        if (!busy) close();
      }}
      footer={
        <>
          <button
            type="button"
            className="button outline"
            disabled={busy || checking || !valid}
            onClick={() => void action(false)}
          >
            {preview ? '重新预检' : '预检续训'}
          </button>
          {preview?.status === 'READY' && (
            <button
              type="button"
              className="button primary"
              disabled={busy || !valid || !configurationConfirmed(config, confirmed)}
              onClick={() => void action(true)}
            >
              启动续训
            </button>
          )}
        </>
      }
    >
      <p className="field-hint">
        支持已结束的标准原生单数据集
        SFT。续训沿用源任务的数据、归一化、模型结构、优化器和学习率计划。
      </p>
      <ErrorNotice error={error || result.error || (!sourceRun ? runs.error : undefined)} />
      <Field label="续训检查点" required>
        <select
          value={checkpointId}
          disabled={busy}
          onChange={(event) => {
            setCheckpointId(event.target.value);
            setSteps(undefined);
            setName(undefined);
            resetPreview();
          }}
        >
          <option value="">
            {runs.loading && !sourceRun ? '正在读取训练记录…' : '选择已登记的检查点'}
          </option>
          {candidates.map(({ run, checkpoint, step }) => (
            <option key={checkpoint.id} value={checkpoint.id}>
              {run.name} · 检查点 {step}
            </option>
          ))}
        </select>
      </Field>
      {!candidates.length && (sourceRun || !runs.loading) && <p>当前没有可续训的 SFT 检查点。</p>}
      {selected && (
        <>
          <p className="field-hint">
            {nodes.data?.find((node) => node.id === selected.run.node_id)?.name ??
              selected.run.node_id}{' '}
            · 已完成 {selected.step + 1} 步 · {selected.run.gpu_count} 张 GPU
          </p>
          <Field label="训练名称" required>
            <input
              value={runName}
              maxLength={200}
              disabled={busy}
              onChange={(event) => {
                setName(event.target.value);
                resetPreview();
              }}
            />
          </Field>
          <Field
            label="目标总步数"
            required
            hint="填写续训结束时的累计步数，包含检查点已经完成的步数。"
          >
            <input
              type="number"
              step={1}
              min={minSteps}
              max={1000000}
              value={totalSteps || ''}
              disabled={busy}
              onChange={(event) => {
                setSteps(event.target.value);
                resetPreview();
              }}
            />
          </Field>
        </>
      )}
      {checking && <p role="status">正在核对检查点与原训练配置…</p>}
      {!!preview?.checks?.length && <PreflightChecks checks={preview.checks} />}
      {preview?.error && (
        <ErrorNotice error={new Error(preview.error.message || preview.error.code)} />
      )}
      {preview?.status === 'READY' && (
        <GeneratedConfigReview
          config={config}
          confirmedSha256={confirmed}
          onConfirm={setConfirmed}
        />
      )}
    </Modal>
  );
}
