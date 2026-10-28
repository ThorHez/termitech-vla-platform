import { AlertTriangle } from 'lucide-react';
import type { PolicyWarning } from './policyWarningInfo';
import './policy-warnings.css';

export function PolicyWarnings({ warnings }: { warnings?: PolicyWarning[] }) {
  return warnings?.map((warning, index) => (
    <div
      className="inline-note amber policy-warning"
      role="status"
      key={`${warning.code}-${index}`}
    >
      <AlertTriangle size={16} />
      <div>
        <strong>配置来源提醒</strong>
        <div>{warning.message}</div>
        <code>{warning.code}</code>
      </div>
    </div>
  ));
}
