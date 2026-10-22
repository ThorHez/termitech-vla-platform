from pathlib import Path
import shlex
import shutil
import subprocess
import time

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "apps/control_plane/vla_control/agent_bootstrap.sh"


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
    path.chmod(0o755)
    return path


def fake_python(path, minor=13):
    return executable(path, f"""
[[ "$1" == -I && "$2" == -c ]] || exit 97
case "$3" in
  *' < (3, 9)'*) minimum=9 ;;
  *' < (3, 11)'*) minimum=11 ;;
  *) exit 98 ;;
esac
printf '%s\\n' '3.{minor}.7'
[[ {minor} -ge "$minimum" ]]
""")


def fake_conda(base, *, minor=13, path=None):
    fake_python(base / "bin/python", minor)
    return executable(path or base / "bin/conda", f"""
[[ -z "${{PYTHONPATH+x}}" && -z "${{PYTHONHOME+x}}" ]] || exit 97
[[ "$CONDA_OFFLINE" == true && "$CONDA_NO_PLUGINS" == true ]] || exit 96
[[ "$1" == info && "$2" == --base ]] || exit 98
printf '%s\\n' {shlex.quote(str(base))}
""")


@pytest.fixture
def machine(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    system = tmp_path / "system"
    system.mkdir()
    # Redirect fixed installation roots into the fixture so tests cannot discover
    # the developer machine's real Conda. HOME and PATH are also isolated.
    source = SOURCE.read_text().replace(
        " /opt /usr/local /data /data1 /data2 /data3",
        " " + shlex.quote(str(system)),
    ).replace(
        'sudo_home="/home/$SUDO_USER"',
        f'sudo_home="{system}/userhomes/$SUDO_USER"',
    )
    script = tmp_path / "bootstrap.sh"
    script.write_text(source)
    for name in ("sleep", "timeout"):
        installed = shutil.which(name)
        if installed:
            (binaries / name).symlink_to(installed)
    return {"home": home, "bin": binaries, "system": system, "script": script, "cwd": tmp_path}


def select(machine, **variables):
    environment = {"HOME": str(machine["home"]), "PATH": str(machine["bin"]), "LANG": "C"}
    environment.update({key: str(value) for key, value in variables.items()})
    return subprocess.run(
        ["/bin/bash", "-c", """
set -eu
source "$1"
if vla_select_runtime; then
  printf 'OK\\n%s\\n%s\\n%s\\n' "$VLA_RUNTIME_KIND" "$VLA_BOOTSTRAP_PYTHON" "$VLA_CONDA_EXE"
else
  printf 'ERROR\\n%s\\n' "$VLA_RUNTIME_ERROR"
  exit 1
fi
""", "bootstrap-test", str(machine["script"])],
        env=environment, cwd=machine["cwd"], capture_output=True, text=True, timeout=22,
    )


def assert_runtime(result, kind, python, conda=""):
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"OK\n{kind}\n{python}\n{conda}\n"
    assert result.stderr == ""


def test_source_has_no_side_effects(machine):
    result = subprocess.run(
        ["/bin/bash", "-c", 'set -eu; source "$1"', "source-test", str(machine["script"])],
        env={"HOME": str(machine["home"]), "PATH": "", "CONDA_EXE": "/missing/conda"},
        capture_output=True, text=True, timeout=2,
    )
    assert result.returncode == 0
    assert result.stdout == result.stderr == ""


@pytest.mark.parametrize("minor", [9, 10, 13])
def test_conda_outside_path_uses_base_python_even_when_system_python_is_old(machine, minor):
    fake_python(machine["bin"] / "python3", 10)
    base = machine["home"] / "miniconda3"
    conda = fake_conda(base, minor=minor)
    assert_runtime(select(machine), "conda", base / "bin/python", conda)


def test_registered_named_environment_finds_nonstandard_conda_base(machine):
    base = machine["cwd"] / "custom conda location"
    conda = fake_conda(base)
    registry = machine["home"] / ".conda/environments.txt"
    registry.parent.mkdir()
    registry.write_text(str(base / "envs/training") + "\n")
    assert_runtime(select(machine), "conda", base / "bin/python", conda)


def test_conda_environment_and_path_candidates_are_supported(machine):
    base = machine["cwd"] / "provided-conda"
    conda = fake_conda(base)
    assert_runtime(select(machine, CONDA_PREFIX=base / "envs/training"), "conda", base / "bin/python", conda)
    (machine["bin"] / "conda").symlink_to(conda)
    assert_runtime(select(machine), "conda", base / "bin/python", machine["bin"] / "conda")


def test_common_system_installation_under_user_directory_is_found(machine):
    base = machine["system"] / "operator/miniforge3"
    conda = fake_conda(base)
    assert_runtime(select(machine), "conda", base / "bin/python", conda)


@pytest.mark.parametrize("identity", [{"SUDO_USER": "operator"}, {"SUDO_UID": "1001"}])
def test_sudo_finds_original_account_home_through_getent(machine, identity):
    original_home = machine["cwd"] / "original account"
    base = original_home / "miniconda3"
    conda = fake_conda(base, minor=10)
    executable(machine["bin"] / "getent", f"""
[[ "$1" == passwd && ( "$2" == operator || "$2" == 1001 ) ]] || exit 1
printf '%s\\n' {shlex.quote(f'operator:x:1001:1001:Original user:{original_home}:/bin/bash')}
""")
    assert_runtime(select(machine, **identity), "conda", base / "bin/python", conda)


def test_sudo_account_conda_registry_is_considered(machine):
    original_home = machine["cwd"] / "original-home"
    registry = original_home / ".conda/environments.txt"
    registry.parent.mkdir(parents=True)
    base = machine["cwd"] / "custom-sudo-conda"
    conda = fake_conda(base)
    registry.write_text(str(base / "envs/train") + "\n")
    executable(machine["bin"] / "getent", f"printf '%s\\n' {shlex.quote(f'operator:x:1001:1001::{original_home}:/bin/bash')}")
    assert_runtime(select(machine, SUDO_USER="operator"), "conda", base / "bin/python", conda)


def test_actual_home_conda_has_priority_over_sudo_account_home(machine):
    root_base = machine["home"] / "miniconda3"
    root_conda = fake_conda(root_base)
    original_home = machine["cwd"] / "original-home"
    fake_conda(original_home / "anaconda3")
    executable(machine["bin"] / "getent", f"printf '%s\\n' {shlex.quote(f'operator:x:1001:1001::{original_home}:/bin/bash')}")
    assert_runtime(select(machine, SUDO_USER="operator"), "conda", root_base / "bin/python", root_conda)


def test_sudo_home_fallback_without_getent(machine):
    base = machine["system"] / "userhomes/operator/miniconda3"
    conda = fake_conda(base)
    assert_runtime(select(machine, SUDO_USER="operator"), "conda", base / "bin/python", conda)


def test_sudo_user_with_shell_syntax_is_ignored(machine):
    python = fake_python(machine["bin"] / "python3")
    assert_runtime(select(machine, SUDO_USER="$(touch VLA_PWNED)"), "venv", python)
    assert not (machine["cwd"] / "VLA_PWNED").exists()


def test_first_usable_conda_wins_over_an_invalid_candidate(machine):
    invalid = executable(machine["cwd"] / "broken-conda", "exit 4")
    base = machine["home"] / "anaconda3"
    conda = fake_conda(base)
    assert_runtime(select(machine, CONDA_EXE=invalid), "conda", base / "bin/python", conda)


@pytest.mark.parametrize("failure", ["nonexecutable", "broken", "missing-explicit", "old-base", "invalid-output"])
def test_unusable_conda_never_falls_back_to_good_system_python(machine, failure):
    fake_python(machine["bin"] / "python3", 13)
    base = machine["home"] / "miniconda3"
    conda = fake_conda(base, minor=8 if failure == "old-base" else 13)
    if failure == "nonexecutable":
        conda.chmod(0o644)
    elif failure == "broken":
        conda.write_text("#!/bin/bash\nprintf 'private raw diagnostic' >&2\nexit 4\n")
    elif failure == "invalid-output":
        conda.write_text("#!/bin/bash\nprintf '/base\\nextra line\\n'\n")
    elif failure == "missing-explicit":
        conda.unlink()
    result = select(machine, **({"CONDA_EXE": conda} if failure == "missing-explicit" else {}))
    assert result.returncode == 1
    assert result.stdout.startswith("ERROR\n已发现 Conda")
    assert len(result.stdout.splitlines()) == 2
    assert "不会回退" in result.stdout
    assert "private raw diagnostic" not in result.stderr


def test_spaces_and_shell_metacharacters_in_conda_paths_are_literal(machine):
    base = machine["cwd"] / "conda `touch VLA_PWNED`; $(touch VLA_PWNED) ' spaced"
    conda = fake_conda(base)
    result = select(machine, CONDA_EXE=conda, PYTHONPATH="/unrelated-training", PYTHONHOME="/unrelated-home")
    assert_runtime(result, "conda", base / "bin/python", conda)
    assert not (machine["cwd"] / "VLA_PWNED").exists()


def test_no_conda_prefers_current_python3_over_versioned_python(machine):
    python = fake_python(machine["bin"] / "python3", 13)
    fake_python(machine["bin"] / "python3.11", 11)
    assert_runtime(select(machine, PYTHONPATH="/training", PYTHONHOME="/training"), "venv", python)


@pytest.mark.parametrize("name", ["python", "python3.13"])
def test_old_default_can_use_another_explicit_python_command(machine, name):
    fake_python(machine["bin"] / "python3", 10)
    python = fake_python(machine["bin"] / name, 13)
    assert_runtime(select(machine), "venv", python)


def test_old_default_diagnostic_includes_exact_path_and_version(machine):
    python = fake_python(machine["bin"] / "python3", 10)
    result = select(machine)
    assert result.returncode == 1
    assert len(result.stdout.splitlines()) == 2
    assert str(python) in result.stdout
    assert "3.10.7" in result.stdout
    assert "Python 3.11+" in result.stdout


def test_no_python_or_conda_gives_actionable_error(machine):
    result = select(machine)
    assert result.returncode == 1
    assert len(result.stdout.splitlines()) == 2
    assert "未发现 Conda" in result.stdout
    assert "Python 3.11+" in result.stdout


def test_relative_path_candidate_is_not_executed(machine):
    marker = machine["cwd"] / "executed"
    executable(machine["bin"] / "python3", f"touch {shlex.quote(str(marker))}")
    result = select(machine, PATH="bin")
    assert result.returncode == 1
    assert not marker.exists()


def test_control_characters_in_explicit_conda_path_are_rejected(machine):
    marker = machine["cwd"] / "executed"
    conda = executable(machine["cwd"] / "conda\ncontrol", f"touch {shlex.quote(str(marker))}")
    result = select(machine, CONDA_EXE=conda)
    assert result.returncode == 1
    assert len(result.stdout.splitlines()) == 2
    assert not marker.exists()


def test_hung_conda_check_is_bounded_and_not_repeated(machine):
    base = machine["home"] / "miniconda3"
    conda = executable(base / "bin/conda", f"exec {shlex.quote(shutil.which('sleep'))} 20")
    fake_python(machine["bin"] / "python3", 13)
    started = time.monotonic()
    result = select(machine, CONDA_EXE=conda)
    elapsed = time.monotonic() - started
    assert result.returncode == 1
    assert elapsed < 8
    assert "已发现 Conda" in result.stdout
