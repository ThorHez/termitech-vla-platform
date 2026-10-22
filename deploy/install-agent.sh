#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$bundle_dir/agent-bootstrap.sh"
if ! vla_select_runtime; then
  # The selector reports a bounded diagnostic and never activates an existing
  # training environment or installs anything during discovery.
  exit 1
fi
exec env -u PYTHONHOME -u PYTHONPATH "$VLA_BOOTSTRAP_PYTHON" -I -u "$bundle_dir/install_agent.py" \
  --runtime-kind "$VLA_RUNTIME_KIND" --conda-executable "$VLA_CONDA_EXE" "$@"
