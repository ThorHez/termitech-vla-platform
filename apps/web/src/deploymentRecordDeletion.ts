import { api } from './api';
import { transferTerminal } from './checkpointTransfer';
import type { CheckpointTransfer } from './checkpointTransfer';
import type { DeploymentRecordSummary, PolicyDeployment } from './policyDeployment';

export function canDeleteDeploymentRecord(
  record: Pick<DeploymentRecordSummary, 'state'>,
  deployment?: PolicyDeployment | null,
  transfers?: CheckpointTransfer[],
) {
  if (transfers?.some((transfer) => !transferTerminal(transfer.state))) return false;
  if (deployment)
    return (
      ['STOPPED', 'FAILED'].includes(deployment.state) &&
      !deployment.pending_command &&
      !deployment.pid &&
      !deployment.endpoint
    );
  return ['DRAFT', 'READY_TO_DEPLOY', 'STOPPED', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(
    record.state,
  );
}

export function deleteDeploymentRecord(id: string) {
  return api<void>(`/deployment-records/${encodeURIComponent(id)}`, { method: 'DELETE' });
}
