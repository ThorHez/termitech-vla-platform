import { normalizeTransferPath } from './checkpointTransfer';
import type { Run } from './types';

export interface PolicyWarning {
  code: string;
  message: string;
}

export const checkpointSourceWarning: PolicyWarning = {
  code: 'POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH',
  message:
    '权重目录不属于所选训练的已记录产物；将继续使用该训练配置，请确认模型结构和归一化参数一致。',
};

export function checkpointSourceWarnings(
  run: Run | undefined,
  checkpoint: string,
): PolicyWarning[] {
  const path = normalizeTransferPath(checkpoint);
  if (!run || !path) return [];
  const roots = [
    ...(run.attempts || []).map((attempt) => attempt.output_dir),
    ...(run.outputs || [])
      .filter(
        (asset) =>
          asset.kind === 'MODEL' && asset.node_id === run.node_id && asset.state !== 'ARCHIVED',
      )
      .map((asset) => asset.path),
  ].filter((root): root is string => typeof root === 'string' && !!normalizeTransferPath(root));
  const recorded = roots.some((root) => {
    const directory = normalizeTransferPath(root);
    return path === directory || path.startsWith(directory === '/' ? '/' : directory + '/');
  });
  return recorded ? [] : [{ ...checkpointSourceWarning }];
}
