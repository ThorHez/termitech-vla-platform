import { ApiError, createOperationClient } from './api';
import type { Run } from './types';

/** The accepted run and its final page transition belong to the initiating session. */
export async function submitTrainingRun({
  preflightId,
  confirmedConfigSha256,
  idempotencyKey,
  finishDraft,
  isActive,
  onSubmitted,
}: {
  preflightId: string;
  confirmedConfigSha256?: string;
  idempotencyKey: string;
  finishDraft: () => Promise<void>;
  isActive: () => boolean;
  onSubmitted: (run: Run) => void;
}) {
  const client = createOperationClient();
  const requireActive = () => {
    client.requireCurrent();
    if (!isActive()) throw new DOMException('训练窗口已关闭。', 'AbortError');
  };
  requireActive();
  const run = await client.post<Run>(
    '/runs',
    { preflight_id: preflightId, confirmed_config_sha256: confirmedConfigSha256 },
    { headers: { 'Idempotency-Key': idempotencyKey } },
  );
  try {
    await finishDraft();
  } catch (cause) {
    // A normal draft cleanup error must not hide an already accepted run.
    // Session replacement must still stop the old account's continuation.
    if (cause instanceof ApiError && cause.code === 'SESSION_CHANGED') throw cause;
    requireActive();
  }
  requireActive();
  onSubmitted(run);
  return run;
}
