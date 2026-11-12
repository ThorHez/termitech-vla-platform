import type { Asset, Attempt, Run } from './types';

export interface RecapAnnotationSource {
  asset: Asset;
  index: number;
}

function valueRun(run: Run, attempt = run.attempts?.at(-1)) {
  return (
    run.template_id === 'recap' &&
    (run.training_stage ??
      attempt?.code_snapshot?.training_stage ??
      run.code_expectation?.training_stage) === 'value'
  );
}

export function recapAnnotationInputs(run: Run, assets: Asset[]): RecapAnnotationSource[] {
  if (!valueRun(run)) return [];

  const bindings =
    run.dataset_bindings ??
    run.attempts?.at(-1)?.code_snapshot?.dataset_bindings ??
    run.code_expectation?.dataset_bindings;
  const indexedInputs = Object.entries(run.inputs ?? {}).flatMap(([role, input]) => {
    const match = /^dataset_(0|[1-9]\d*)$/u.exec(role);
    if (!match) return [];
    const index = Number(match[1]);
    return Number.isSafeInteger(index) ? [{ input, index }] : [];
  });
  const rows = bindings?.length
    ? bindings.map((binding, index) => ({ input: binding.dataset_asset_id, index }))
    : indexedInputs.length
      ? indexedInputs.sort((left, right) => left.index - right.index)
      : [{ input: run.inputs?.dataset, index: 0 }];
  return rows.flatMap(({ input, index }) => {
    const id = typeof input === 'string' ? input : input?.id;
    const asset = assets.find(
      (candidate) =>
        candidate.id === id &&
        candidate.node_id === run.node_id &&
        candidate.kind === 'DATASET' &&
        !['ARCHIVED', 'INVALID', 'MISSING', 'UNAVAILABLE'].includes(
          candidate.state ?? candidate.status ?? '',
        ),
    );
    return asset ? [{ asset, index }] : [];
  });
}

function successfulAttempt(run: Run, output: Asset): Attempt | undefined {
  if (!output.origin_attempt_id) return undefined;
  return run.attempts?.find(
    (attempt) =>
      attempt.id === output.origin_attempt_id && (attempt.state ?? attempt.status) === 'SUCCEEDED',
  );
}

function checkpointStep(output: Asset) {
  const step = output.metadata?.step;
  if (typeof step === 'number' && Number.isSafeInteger(step)) return step;
  return /^\d+$/u.test(output.version) && Number.isSafeInteger(Number(output.version))
    ? Number(output.version)
    : -1;
}

export function recapValueCheckpoints(run: Run): Asset[] {
  return (run.outputs ?? [])
    .filter((output) => {
      const attempt = successfulAttempt(run, output);
      const role = output.metadata?.model_role;
      return (
        !!attempt &&
        valueRun(run, attempt) &&
        output.node_id === run.node_id &&
        output.kind === 'MODEL' &&
        (output.state ?? output.status) === 'READY' &&
        output.metadata?.availability !== 'removed' &&
        (role == null || role === 'value_estimator')
      );
    })
    .sort(
      (left, right) =>
        checkpointStep(right) - checkpointStep(left) ||
        (right.created_at ?? '').localeCompare(left.created_at ?? ''),
    );
}

function absolutePath(value?: string) {
  if (!value?.startsWith('/') || /[\u0000-\u001f\u007f]/u.test(value)) return undefined;
  const parts = value.split('/');
  if (parts.includes('..')) return undefined;
  return `/${parts.filter((part) => part && part !== '.').join('/')}`;
}

function atOrBelow(path: string, root: string) {
  return path === root || path.startsWith(root === '/' ? root : `${root}/`);
}

function shellQuote(value: string) {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

export function recapAnnotationCommand(
  run: Run,
  checkpoint: Asset,
  source: RecapAnnotationSource,
  outputPath: string,
): string | undefined {
  const output = recapValueCheckpoints(run).find(({ id }) => id === checkpoint.id);
  const attempt = output && successfulAttempt(run, output);
  if (!output || !attempt) return undefined;

  const backend =
    attempt.code_snapshot?.automatic_preflight?.plan.recap_stage_contract?.backend ??
    run.code_expectation?.automatic_preflight?.plan.recap_stage_contract?.backend;
  const checkpointBackend = output.metadata?.backend;
  const format = output.metadata?.format;
  if (
    (backend != null && backend !== 'pytorch') ||
    (checkpointBackend != null && checkpointBackend !== 'pytorch') ||
    (format != null && format !== 'openpi-pytorch') ||
    (backend !== 'pytorch' && checkpointBackend !== 'pytorch' && format !== 'openpi-pytorch')
  )
    return undefined;

  if (!recapAnnotationInputs(run, [source.asset]).some(({ index }) => index === source.index))
    return undefined;
  const dataset = absolutePath(source.asset.path);
  const destination = absolutePath(outputPath);
  const attemptDir = absolutePath(attempt.output_dir);
  const checkpointDir = absolutePath(output.path);
  if (
    !dataset ||
    !destination ||
    !attemptDir ||
    !checkpointDir ||
    checkpointDir === attemptDir ||
    !atOrBelow(checkpointDir, attemptDir) ||
    atOrBelow(destination, dataset) ||
    atOrBelow(dataset, destination)
  )
    return undefined;

  const codeDir = absolutePath(
    attempt.code_snapshot?.code_dir ??
      attempt.code_snapshot?.runtime_profile?.code_dir ??
      run.code_expectation?.code_dir ??
      run.code_expectation?.runtime_profile?.code_dir,
  );
  const args = [
    'uv run python scripts/acp/lerobot_value_infer_pytorch.py',
    `  --config-file ${shellQuote(`${attemptDir.replace(/\/$/u, '')}/training_config.py`)}`,
    `  --checkpoint-dir ${shellQuote(checkpointDir)}`,
    `  --dataset-root ${shellQuote(dataset)}`,
    `  --output-dir ${shellQuote(destination)}`,
    `  --dataset-index ${source.index}`,
  ];
  return `${codeDir ? `cd ${shellQuote(codeDir)} && \\\n` : ''}${args.join(' \\\n')}`;
}
