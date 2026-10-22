import { PathAutocomplete } from './PathAutocomplete';
import type { SshPathConnection } from './PathAutocomplete';

export type { SshPathConnection } from './PathAutocomplete';

export function SSHPathAutocomplete({
  connection,
  directoryOnly = true,
  ...props
}: {
  value: string;
  onChange: (value: string) => void;
  connection?: SshPathConnection;
  directoryOnly?: boolean;
  disabled?: boolean;
  required?: boolean;
  label: string;
  placeholder?: string;
  hint?: string;
}) {
  return <PathAutocomplete {...props} ssh={{ connection, directoryOnly }} />;
}
