import { useEffect, useState } from 'react';
import { ArrowRight, Database, LoaderCircle } from 'lucide-react';
import { post } from './api';
import { useAuth } from './auth';
import { CopyButton, ErrorNotice, Field, Modal, Panel } from './components';
import { useData } from './data';
import { RegisterAsset } from './pages';
import {
  recapAnnotationCommand,
  recapAnnotationInputs,
  recapValueCheckpoints,
} from './recapAnnotation';
import type { Asset, Run } from './types';
import './recap-handoff.css';

function stageOf(run: Run) {
  return (
    run.training_stage ??
    run.code_expectation?.training_stage ??
    run.attempts?.at(-1)?.code_snapshot?.training_stage
  );
}

function available(asset: Asset) {
  return (
    ['READY', 'REGISTERED'].includes(asset.state ?? asset.status ?? '') &&
    asset.metadata?.availability !== 'removed'
  );
}

function valueRunId(asset: Asset) {
  const source = asset.metadata?.recap_annotation_source;
  if (!source || typeof source !== 'object' || Array.isArray(source)) return undefined;
  return 'run_id' in source && typeof source.run_id === 'string' ? source.run_id : undefined;
}

function acpSourceRows(run: Run, assets: Asset[]) {
  const ids = new Set(
    Object.entries(run.inputs || {})
      .filter(([role]) => /^dataset(?:_\d+)?$/u.test(role))
      .map(([, value]) => (typeof value === 'string' ? value : value.id)),
  );
  return (run.input_assets || assets).filter(
    (asset) => asset.kind === 'DATASET' && ids.has(asset.id),
  );
}

export function RecapRunHandoff({
  run,
  onAcp,
}: {
  run: Run;
  onAcp?: (run: Run, dataset: Asset) => void;
}) {
  const { assets, runs, templates } = useData();
  const { can } = useAuth();
  const [showAnnotation, setShowAnnotation] = useState(false);
  const stage = stageOf(run);
  const finishedKai0 =
    run.template_id === 'recap' && stage === 'stage_progress' && run.state === 'SUCCEEDED';
  const refreshAssets = assets.refresh;
  useEffect(() => {
    if (finishedKai0) refreshAssets();
  }, [finishedKai0, refreshAssets]);
  if (run.template_id !== 'recap') return null;
  const catalog = assets.data || [];
  if (stage === 'offline_rl') {
    const sources = acpSourceRows(run, catalog).flatMap((dataset) => {
      const origin =
        valueRunId(dataset) ||
        dataset.origin_run_id ||
        (dataset.origin_attempt_id &&
        dataset.metadata?.registration_source === 'recap_stage_progress'
          ? runs.data?.find((candidate) =>
              candidate.attempts?.some((attempt) => attempt.id === dataset.origin_attempt_id),
            )?.id
          : undefined);
      return origin ? [{ dataset, origin }] : [];
    });
    if (!sources.length) return null;
    return (
      <Panel title="ReCap 阶段来源" className="recap-handoff-panel">
        <div className="recap-handoff-rows">
          {sources.map(({ dataset, origin }) => (
            <div className="recap-handoff-row" key={dataset.id}>
              <span>
                {dataset.name} · {dataset.version}
              </span>
              <a className="text-link" href={`#/runs/${encodeURIComponent(origin)}`}>
                查看上游训练作业
              </a>
            </div>
          ))}
        </div>
      </Panel>
    );
  }
  if (run.state !== 'SUCCEEDED' || (stage !== 'value' && stage !== 'stage_progress')) return null;

  const canStartAcp =
    can('training.write') &&
    !!onAcp &&
    templates.data?.some(
      (template) =>
        template.id === 'recap' &&
        template.status === 'AVAILABLE' &&
        template.stages?.some(({ id }) => id === 'offline_rl'),
    );
  const successfulAttempts = new Set(
    run.attempts
      ?.filter((attempt) => (attempt.state ?? attempt.status) === 'SUCCEEDED')
      .map(({ id }) => id),
  );
  const annotated =
    stage === 'value'
      ? catalog.filter(
          (asset) =>
            valueRunId(asset) === run.id &&
            asset.node_id === run.node_id &&
            asset.kind === 'DATASET' &&
            available(asset),
        )
      : (run.outputs || []).filter(
          (asset) =>
            asset.kind === 'DATASET' &&
            asset.node_id === run.node_id &&
            available(asset) &&
            !!asset.origin_attempt_id &&
            successfulAttempts.has(asset.origin_attempt_id) &&
            asset.metadata?.registration_source === 'recap_stage_progress',
        );
  const hasCheckpoint = stage === 'value' && recapValueCheckpoints(run).length > 0;
  if (!hasCheckpoint && !annotated.length) return null;

  return (
    <Panel
      title={stage === 'value' ? '下一步：优势值标注' : '下一步：ACP 训练'}
      subtitle={
        stage === 'value'
          ? '使用本次 Value 检查点标注数据，登记并关联派生数据后即可配置 ACP。'
          : '本次 Kai0 作业生成的标注数据可直接带入 ACP。'
      }
      className="recap-handoff-panel"
      action={
        hasCheckpoint && (
          <button
            type="button"
            className="button outline small"
            onClick={() => setShowAnnotation(true)}
          >
            配置优势值标注
            <ArrowRight size={14} />
          </button>
        )
      }
    >
      {annotated.length > 0 && (
        <div className="recap-handoff-rows">
          {annotated.map((dataset) => (
            <div className="recap-handoff-row" key={dataset.id}>
              <Database size={16} aria-hidden="true" />
              <span>
                {dataset.name} · {dataset.version}
              </span>
              {canStartAcp && onAcp && (
                <button
                  type="button"
                  className="button primary small"
                  disabled={!catalog.some((asset) => asset.id === dataset.id && available(asset))}
                  onClick={() => onAcp(run, dataset)}
                >
                  用此数据配置 ACP
                  <ArrowRight size={14} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
      {showAnnotation && hasCheckpoint && (
        <RecapAnnotationDialog run={run} close={() => setShowAnnotation(false)} />
      )}
    </Panel>
  );
}

function RecapAnnotationDialog({ run, close }: { run: Run; close: () => void }) {
  const { assets } = useData();
  const { can, user } = useAuth();
  const catalog = assets.data || [];
  const sources = recapAnnotationInputs(run, [...(run.input_assets || []), ...catalog]);
  const checkpoints = recapValueCheckpoints(run);
  const [sourceIndex, setSourceIndex] = useState<number>();
  const [checkpointId, setCheckpointId] = useState(checkpoints[0]?.id || '');
  const [outputPath, setOutputPath] = useState('');
  const [datasetId, setDatasetId] = useState('');
  const [recentRegistration, setRecentRegistration] = useState<Asset>();
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const source =
    sourceIndex === undefined ? sources[0] : sources.find(({ index }) => index === sourceIndex);
  const checkpoint = checkpoints.find(({ id }) => id === checkpointId);
  const targetPath = outputPath.trim();
  const command =
    source && checkpoint && recapAnnotationCommand(run, checkpoint, source, targetPath);
  const canAttach = can('assets.register');
  const candidates = [...catalog];
  if (recentRegistration && !catalog.some(({ id }) => id === recentRegistration.id)) {
    candidates.push(recentRegistration);
  }
  const choices = candidates.filter(
    (asset) =>
      asset.kind === 'DATASET' &&
      asset.node_id === run.node_id &&
      available(asset) &&
      !asset.origin_attempt_id &&
      !valueRunId(asset) &&
      !sources.some(({ asset: input }) => input.id === asset.id || input.path === asset.path) &&
      (user?.role === 'admin' || asset.registered_by_user_id === user?.id),
  );
  const selected = choices.find(({ id }) => id === datasetId);
  const attach = async () => {
    if (busy || !canAttach || !source || !checkpoint || !selected) return;
    setBusy(true);
    setError(undefined);
    try {
      await post<Asset>(`/runs/${encodeURIComponent(run.id)}/recap-annotations`, {
        checkpoint_asset_id: checkpoint.id,
        source_dataset_asset_id: source.asset.id,
        source_dataset_index: source.index,
        dataset_asset_id: selected.id,
      });
      assets.refresh();
      close();
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Modal
        title="关联优势值标注"
        wide
        close={() => {
          if (!busy) close();
        }}
        footer={
          canAttach && (
            <button
              type="button"
              className="button primary"
              disabled={busy || !source || !checkpoint || !selected}
              onClick={() => void attach()}
            >
              {busy && <LoaderCircle className="spin" size={15} />}
              关联标注数据
            </button>
          )
        }
      >
        <div className="form-grid">
          <Field label="本次作业的 Value 检查点">
            <select
              value={checkpointId}
              disabled={busy}
              onChange={(event) => setCheckpointId(event.target.value)}
            >
              {checkpoints.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {item.version}
                </option>
              ))}
            </select>
          </Field>
          <Field label="待标注的原训练数据">
            <select
              value={source?.index ?? ''}
              disabled={busy}
              onChange={(event) => setSourceIndex(Number(event.target.value))}
            >
              {!sources.length && <option value="">原数据资产不可用</option>}
              {sources.map(({ asset, index }) => (
                <option key={index} value={index}>
                  {asset.name} · {asset.version}
                </option>
              ))}
            </select>
          </Field>
        </div>
        {source && checkpoint && (
          <>
            <div className="recap-handoff-paths">
              <span>
                检查点 <code>{checkpoint.path}</code>
              </span>
              <span>
                原数据 <code>{source.asset.path}</code>
              </span>
            </div>
            <Field label="训练机上的派生数据目录" hint="使用新目录保存标注结果。">
              <input
                value={outputPath}
                disabled={busy}
                onChange={(event) => setOutputPath(event.target.value)}
                placeholder="/path/to/annotated_dataset"
              />
            </Field>
            {command ? (
              <div className="recap-handoff-command">
                <textarea
                  aria-label="优势值标注命令"
                  value={command}
                  rows={7}
                  readOnly
                  spellCheck={false}
                />
                <CopyButton text={command} label="复制标注命令" />
              </div>
            ) : (
              <p className="field-hint">
                在训练机上用上方检查点标注所选数据。PyTorch 作业可在填写有效输出目录后复制命令。历史
                JAX 作业使用原配方对数据副本运行 <code>scripts/lerobot_value_infer.py</code>。
              </p>
            )}
          </>
        )}
        {canAttach && (
          <div className="recap-handoff-attach">
            <Field
              label="已完成标注的派生数据"
              hint="选择生成了 is_positive 的数据。ACP 预检会核对标签。"
            >
              <select
                disabled={busy}
                value={selected?.id || ''}
                onChange={(event) => setDatasetId(event.target.value)}
              >
                <option value="">选择同一训练机上的派生数据</option>
                {choices.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} · {item.version} · {item.path}
                  </option>
                ))}
              </select>
            </Field>
            <button
              type="button"
              className="button outline small"
              disabled={busy || !source}
              onClick={() => setRegister(true)}
            >
              登记派生数据
            </button>
          </div>
        )}
        <ErrorNotice error={error} title="关联标注数据失败" />
      </Modal>
      {register && (
        <RegisterAsset
          initialNode={run.node_id}
          datasetDefaults={{ path: targetPath }}
          close={() => setRegister(false)}
          onRegistered={(asset) => {
            setRecentRegistration(asset);
            setDatasetId(asset.id);
          }}
        />
      )}
    </>
  );
}
