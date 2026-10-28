"""Human authentication stays enabled in generated deployment configuration."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('deployment_users_fixture', ROOT / 'scripts/deploy_control_plane.py')
deployment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deployment)


def test_user_deployment_does_not_require_shared_admin_token_and_defaults_private(tmp_path):
    credentials = tmp_path / 'private.json'
    credentials.write_text(json.dumps({'enrollment_token': 'isolated-enrollment-test'}))
    credentials.chmod(0o600)
    output = tmp_path / 'rendered'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/deploy_control_plane.py'), 'render',
                             '--credentials', str(credentials), '--release-dir', '/srv/vla',
                             '--state-dir', '/srv/vla/state', '--python', '/srv/vla/.venv/bin/python',
                             '--output-dir', str(output)], capture_output=True, text=True, check=True)
    environment = (output / 'control.env').read_text()
    assert 'VLA_HOST="127.0.0.1"' in environment
    assert 'VLA_SESSION_COOKIE_SECURE="true"' in environment
    assert 'VLA_SESSION_TTL_SECONDS="28800"' in environment
    assert 'VLA_SESSION_IDLE_SECONDS="3600"' in environment
    assert 'VLA_SESSION_REAUTH_SECONDS="300"' in environment
    assert 'VLA_LEGACY_ADMIN_ENABLED="false"' in environment
    assert 'isolated-enrollment-test' not in result.stdout
    assert (output / 'control.env').stat().st_mode & 0o777 == 0o600
    assert 'systemctl' not in (output / 'termitech-vla-control.service').read_text()


def test_existing_shared_credential_is_retained_only_for_backup(tmp_path):
    credentials = tmp_path / 'old-private.json'
    values = {'admin_token': 'old-controller-secret', 'enrollment_token': 'separate-agent-secret'}
    credentials.write_text(json.dumps(values))
    assert deployment.credentials(credentials) == values
