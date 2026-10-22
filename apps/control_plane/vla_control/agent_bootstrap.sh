# Shared by the SSH prerequisite check and the downloaded Agent installer.
# Sourcing this file does not execute Conda or change the caller's environment.

_vla_absolute_path() {
  [[ "$1" == /* && "$1" != *[[:cntrl:]]* ]]
}

_vla_bounded() {
  local timer child status duration=5
  if [[ -n "${_vla_runtime_deadline:-}" ]]; then
    duration=$((_vla_runtime_deadline - SECONDS))
    ((duration > 0)) || return 1
    ((duration <= 5)) || duration=5
  fi
  if type -P timeout >/dev/null 2>&1; then
    "$(type -P timeout)" -s KILL "$duration" "$@"
    return $?
  fi
  # Minimal Linux images may lack coreutils timeout. The watchdog has no output
  # descriptors, so it cannot keep a command substitution's output pipe open.
  "$@" &
  child=$!
  (sleep "$duration"; kill -KILL "$child" 2>/dev/null) >/dev/null 2>&1 &
  timer=$!
  if wait "$child"; then status=0; else status=$?; fi
  kill "$timer" 2>/dev/null || true
  wait "$timer" 2>/dev/null || true
  return "$status"
}

_vla_python_at_least() {
  _VLA_PYTHON_VERSION=
  _vla_absolute_path "$1" && [[ -f "$1" && -x "$1" ]] || return 1
  local output status
  if output=$(_vla_bounded "$1" -I -c "import sys; print('.'.join(map(str, sys.version_info[:3]))); raise SystemExit(sys.version_info < ($2, $3))" 2>/dev/null); then
    status=0
  else
    status=$?
  fi
  [[ "$output" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
  _VLA_PYTHON_VERSION=$output
  return "$status"
}

vla_select_runtime() {
  VLA_RUNTIME_KIND=
  VLA_BOOTSTRAP_PYTHON=
  VLA_CONDA_EXE=
  VLA_RUNTIME_ERROR=
  local candidate base root name environment selected default_python= default_version= duplicate
  local sudo_key= sudo_home= passwd_entry getent_exe ignored
  local _vla_runtime_deadline=$((SECONDS + 16))
  local conda_seen=0
  local -a candidates=() checked=() homes=("${HOME:-/nonexistent}")

  # A manually invoked sudo installer has root's HOME. Also inspect the original
  # account without running its shell or interpreting shell configuration.
  if [[ "${SUDO_USER:-}" =~ ^[a-zA-Z_][a-zA-Z0-9_.-]*\$?$ ]]; then
    sudo_key=$SUDO_USER
  elif [[ "${SUDO_UID:-}" =~ ^[0-9]+$ ]]; then
    sudo_key=$SUDO_UID
  fi
  if [[ -n "$sudo_key" ]]; then
    getent_exe=$(type -P getent 2>/dev/null) || getent_exe=
    if _vla_absolute_path "$getent_exe"; then
      passwd_entry=$(_vla_bounded "$getent_exe" passwd "$sudo_key" 2>/dev/null) || passwd_entry=
      if [[ "$passwd_entry" != *$'\n'* ]]; then
        IFS=: read -r ignored ignored ignored ignored ignored sudo_home ignored <<< "$passwd_entry"
      fi
    fi
    if ! _vla_absolute_path "$sudo_home" && [[ -n "${SUDO_USER:-}" && "$sudo_key" == "$SUDO_USER" ]]; then
      sudo_home="/home/$SUDO_USER"
    fi
    if _vla_absolute_path "$sudo_home" && [[ -d "$sudo_home" ]]; then
      homes+=("$sudo_home")
    fi
  fi

  # Never source shell startup files: activation changes PATH and can execute
  # unrelated user commands. Discover existing installations without activation.
  if [[ -n "${CONDA_EXE:-}" ]]; then
    candidates+=("$CONDA_EXE")
    conda_seen=1
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("$CONDA_PREFIX/bin/conda")
    if [[ "$CONDA_PREFIX" == */envs/* ]]; then
      candidates+=("${CONDA_PREFIX%%/envs/*}/bin/conda")
    fi
    conda_seen=1
  fi
  candidate=$(type -P conda 2>/dev/null) || candidate=
  [[ -z "$candidate" ]] || candidates+=("$candidate")
  for root in "${homes[@]}" /opt /usr/local /data /data1 /data2 /data3; do
    for name in miniconda3 anaconda3 miniforge3 mambaforge conda; do
      candidates+=("$root/$name/bin/conda")
      for candidate in "$root"/*/"$name"/bin/conda; do
        [[ ! -e "$candidate" && ! -L "$candidate" ]] || candidates+=("$candidate")
      done
    done
  done
  for root in "${homes[@]}"; do
    if [[ -r "$root/.conda/environments.txt" ]]; then
      while IFS= read -r environment || [[ -n "$environment" ]]; do
        _vla_absolute_path "$environment" || continue
        candidates+=("$environment/bin/conda")
        if [[ "$environment" == */envs/* ]]; then
          candidates+=("${environment%%/envs/*}/bin/conda")
        fi
      done < "$root/.conda/environments.txt"
    fi
  done

  for candidate in "${candidates[@]}"; do
    [[ -e "$candidate" || -L "$candidate" ]] || continue
    conda_seen=1
    duplicate=0
    for selected in ${checked[@]+"${checked[@]}"}; do
      [[ "$selected" != "$candidate" ]] || duplicate=1
    done
    [[ "$duplicate" == 0 ]] || continue
    checked+=("$candidate")
    _vla_absolute_path "$candidate" && [[ -f "$candidate" && -x "$candidate" ]] || continue
    base=$(unset PYTHONPATH PYTHONHOME; export CONDA_OFFLINE=true CONDA_NO_PLUGINS=true; _vla_bounded "$candidate" info --base 2>/dev/null) || continue
    _vla_absolute_path "$base" || continue
    _vla_python_at_least "$base/bin/python" 3 9 || continue
    VLA_RUNTIME_KIND=conda
    VLA_BOOTSTRAP_PYTHON="$base/bin/python"
    VLA_CONDA_EXE="$candidate"
    return 0
  done
  if [[ "$conda_seen" == 1 ]]; then
    VLA_RUNTIME_ERROR='已发现 Conda，但无法使用其 base Python（需要 3.9+）；请修复 Conda 或确认 CONDA_EXE 指向可执行的 conda，平台不会回退到其他 Python。'
    printf '%s\n' "$VLA_RUNTIME_ERROR" >&2
    return 1
  fi

  # Respect the machine's default interpreter before considering versioned
  # system commands. The Agent itself still needs Python 3.11 or newer.
  for selected in python3 python python3.14 python3.13 python3.12 python3.11; do
    candidate=$(type -P "$selected" 2>/dev/null) || continue
    if ! _vla_python_at_least "$candidate" 3 11; then
      if [[ -z "$default_python" ]] && _vla_absolute_path "$candidate"; then
        default_python=$candidate
        default_version=${_VLA_PYTHON_VERSION:-版本无法读取}
      fi
      continue
    fi
    VLA_RUNTIME_KIND=venv
    VLA_BOOTSTRAP_PYTHON="$candidate"
    return 0
  done
  VLA_RUNTIME_ERROR='未发现 Conda，且 SSH 环境中没有可用的 Python 3.11+；请安装 Python 3.11+ 或提供可用的 Conda 安装。'
  if [[ -n "$default_python" ]]; then
    VLA_RUNTIME_ERROR="SSH 默认 Python：$default_python（$default_version）；$VLA_RUNTIME_ERROR"
  fi
  printf '%s\n' "$VLA_RUNTIME_ERROR" >&2
  return 1
}
