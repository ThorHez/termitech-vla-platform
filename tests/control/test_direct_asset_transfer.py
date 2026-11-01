"""Direct rsync peers on local subprocesses, without SSH or real user key edits.

The fake SSH executable runs the real restricted sender underneath the target
rsync process.  Controller channels only carry the fixed JSON control protocol.
"""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import pytest

from vla_control import direct_asset_transfer as module
from vla_control.checkpoint_transfer_transport import TransferError


spec = importlib.util.spec_from_file_location('direct_peer_helpers',
    Path(__file__).with_name('test_checkpoint_transfer_transport.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


_FAKE_SSH = r'''#!/usr/bin/env python3
import json,os,pathlib,re,shlex,sys,time
args=sys.argv[1:]
attempt=1
if os.environ.get('DIRECT_FAKE_SSH_COUNT'):
    counter=pathlib.Path(os.environ['DIRECT_FAKE_SSH_COUNT'])
    attempt=int(counter.read_text())+1 if counter.exists() else 1
    counter.write_text(str(attempt))
failure=os.environ.get('DIRECT_FAKE_SSH_FAILURE') or os.environ.get('DIRECT_FAKE_SSH_FAIL_FIRST') and attempt==1
diagnostic=os.environ.get('DIRECT_FAKE_SSH_LOG')
if diagnostic is None and failure and os.environ.get('DIRECT_FAKE_SSH_FAIL_FIRST'):
    diagnostic='kex_exchange_identification: banner line 0: Exceeded MaxStartups\n'
if not failure:diagnostic='Authenticated to synthetic-source using "publickey".\n'
if '-E' in args and diagnostic is not None:pathlib.Path(args[args.index('-E')+1]).write_text(diagnostic)
if failure:
    if os.environ.get('DIRECT_FAKE_SSH_STDERR'):print(os.environ['DIRECT_FAKE_SSH_STDERR'],file=sys.stderr)
    sys.exit(255)
if os.environ.get('DIRECT_FAKE_SSH_DELAY'):time.sleep(float(os.environ['DIRECT_FAKE_SSH_DELAY']))
key=pathlib.Path(args[args.index('-i')+1]);public=key.with_name(key.name+'.pub').read_text().split()[1]
authorized=pathlib.Path(os.environ['DIRECT_SOURCE_HOME'],'.ssh','authorized_keys').read_text()
lines=[line for line in authorized.splitlines() if ' ssh-ed25519 '+public+' ' in line]
if len(lines)!=1:sys.exit(255)
line=lines[0]
if not line.startswith('restrict,expiry-time=') or ',command=' not in line:sys.exit(255)
match=re.search(r'command="((?:\\.|[^"\\])*)"',line)
if not match:sys.exit(255)
command=re.sub(r'\\(.)',r'\1',match.group(1))
os.environ['SSH_ORIGINAL_COMMAND']='rm -rf /; this must never execute'
if os.environ.get('DIRECT_FAKE_SSH_RECORD'):
    pathlib.Path(os.environ['DIRECT_FAKE_SSH_RECORD']).write_text(json.dumps({'args':args,'pid':os.getpid()}))
argv=shlex.split(command);os.execv(argv[0],argv)
'''


class _HostKey:
    def get_name(self):
        return 'ssh-ed25519'

    def get_base64(self):
        return base64.b64encode(b'isolated synthetic pinned host key').decode()


class DirectLocalChannel(helpers.LocalChannel):
    def __init__(self, remote):
        super().__init__()
        self.remote = remote
        self.received_bytes = 0
        self.control_output = bytearray()

    def exec_command(self, command):
        argv = shlex.split(command)
        assert argv[:3] == ['python3', '-u', '-c']
        script = argv[3]
        if script == module._SOURCE_SCRIPT:
            # This modifies pwd only inside an isolated local process.  The
            # production fixed program always uses the authenticated Unix user.
            script = ('import pwd,types\npwd.getpwuid=lambda _uid: types.SimpleNamespace(pw_dir='
                      + repr(str(self.remote.home)) + ')\n' + script)
        environment = dict(os.environ, **self.remote.environment)
        self.process = subprocess.Popen([shutil.which('python3'), '-u', '-c', script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, bufsize=0, start_new_session=True)
        os.set_blocking(self.process.stdin.fileno(), self.timeout != 0)

    def recv(self, count):
        value = super().recv(count)
        self.received_bytes += len(value)
        if len(self.control_output) < 128 * 1024:self.control_output.extend(value[:128 * 1024-len(self.control_output)])
        return value


class DirectLocalRemote(helpers.LocalRemote):
    """Reusable real-rsync stand-in for tests of the public asset transport."""
    def __init__(self, home, source_home, ssh_bin_dir, *, host='source.test', username='fixture'):
        super().__init__()
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.host, self.port, self.username = host, 22, username
        raw = base64.b64decode(_HostKey().get_base64())
        self.fingerprint = 'SHA256:' + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        self.expected_fingerprint = self.fingerprint
        self.environment = {'DIRECT_SOURCE_HOME': str(source_home),
                            'PATH': str(ssh_bin_dir) + os.pathsep + os.environ['PATH']}

    def get_remote_server_key(self):
        return _HostKey()

    def open_session(self, timeout):
        assert timeout == 15
        channel = DirectLocalChannel(self)
        self.channels.append(channel)
        return channel

    def close(self):
        for channel in self.channels:
            channel.close()


def direct_local_pair(tmp_path):
    """Return local source/target remotes; no sockets, sshd or ~/.ssh are used."""
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    binaries = root / 'fake-bin'
    binaries.mkdir(exist_ok=True)
    executable = binaries / 'ssh'
    executable.write_text(_FAKE_SSH)
    executable.chmod(0o700)
    source_home, target_home = root / 'source-home', root / 'target-home'
    source = DirectLocalRemote(source_home, source_home, binaries)
    target = DirectLocalRemote(target_home, source_home, binaries, host='target.test')
    return source, target


@pytest.fixture
def peers(tmp_path):
    if any(shutil.which(binary) is None for binary in ('rsync', 'ssh-keygen')):
        pytest.skip('Direct transfer tests need rsync and ssh-keygen')
    source, target = direct_local_pair(tmp_path / 'peers')
    try:
        yield source, target
    finally:
        source.close()
        target.close()


def source_info(path):
    files = [path] if path.is_file() else [item for item in path.rglob('*') if item.is_file()]
    return {'path': str(path), 'name': path.name, 'kind': 'file' if path.is_file() else 'directory',
            'manifest': {'bytes_total': sum(item.stat().st_size for item in files), 'files_total': len(files)}}


def test_real_rsync_data_goes_between_peers_and_controller_receives_only_progress(tmp_path, peers):
    source, target = peers
    dataset = tmp_path / "dataset'; touch BAD; #"
    (dataset / 'empty').mkdir(parents=True)
    (dataset / '文件.bin').write_bytes(bytes(range(256)) * 32768)
    (dataset / 'task.json').write_text('{"task":"fold cloth"}')
    staging = tmp_path / '.owned.partial'
    staging.mkdir()
    events = []
    result = module.direct_copy(source, target, source_info(dataset), str(staging), 'copy-1', progress=events.append)
    assert result == {'transport': 'direct-ssh', **source_info(dataset)['manifest']}
    assert (staging / '文件.bin').read_bytes() == (dataset / '文件.bin').read_bytes()
    assert (staging / 'task.json').read_bytes() == (dataset / 'task.json').read_bytes()
    assert (staging / 'empty').is_dir()
    assert not (tmp_path / 'BAD').exists()
    assert sum(channel.received_bytes for remote in peers for channel in remote.channels) < 32768
    assert events[0]['stage'] == 'TRANSFERRING'
    assert (source.home / '.ssh' / 'authorized_keys').read_bytes() == b''
    assert not list((source.home / '.ssh').glob('.termitech-direct-*-*'))


def test_single_norm_file_and_owned_partial_retry_are_supported(tmp_path, peers):
    source, target = peers
    norm = tmp_path / '归一化 stats.json'
    norm.write_text('{"norm_stats":{"state":{"mean":[1,2]}}}')
    staging = tmp_path / '.norm.partial'
    staging.mkdir()
    (staging / norm.name).write_text('partial data')
    for _ in range(2):
        module.direct_copy(source, target, source_info(norm), str(staging), 'norm-1')
    assert list(staging.iterdir()) == [staging / norm.name]
    assert (staging / norm.name).read_bytes() == norm.read_bytes()


def test_direct_network_failure_never_falls_back_and_revokes_temporary_authorization(tmp_path, peers):
    source, target = peers
    source_file = tmp_path / 'asset'
    source_file.write_bytes(b'asset content')
    staging = tmp_path / '.failed.partial'
    staging.mkdir()
    ssh = source.home / '.ssh'
    ssh.mkdir(mode=0o700)
    original = b'# user key without trailing newline'
    (ssh / 'authorized_keys').write_bytes(original)
    (ssh / 'authorized_keys').chmod(0o640)
    target.environment['DIRECT_FAKE_SSH_FAILURE'] = '1'
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(source_file), str(staging), 'network-failure')
    assert error.value.code == 'DIRECT_CONNECTION_UNAVAILABLE'
    assert not list(staging.iterdir())
    assert (ssh / 'authorized_keys').read_bytes() == original
    assert (ssh / 'authorized_keys').stat().st_mode & 0o777 == 0o640


def test_cancellation_stops_target_and_cleans_keys_without_deleting_partial_files(tmp_path, peers):
    source, target = peers
    dataset = tmp_path / 'source'
    dataset.mkdir()
    (dataset / 'payload').write_bytes(b'payload')
    staging = tmp_path / '.cancelled.partial'
    staging.mkdir()
    (staging / 'retained').write_bytes(b'owned partial')
    target.environment['DIRECT_FAKE_SSH_DELAY'] = '30'
    started = time.monotonic()
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(dataset), str(staging), 'cancel-1',
                           cancelled=lambda: time.monotonic() - started > .8)
    assert error.value.code == 'TRANSFER_CANCELLED'
    assert time.monotonic() - started < 8
    assert (staging / 'retained').read_bytes() == b'owned partial'
    assert (source.home / '.ssh' / 'authorized_keys').read_bytes() == b''
    assert all(channel.exit_status_ready() for remote in peers for channel in remote.channels)


def test_source_symlink_ancestor_and_target_staging_symlinks_fail_closed(tmp_path, peers):
    source, target = peers
    dataset = tmp_path / 'source'
    dataset.mkdir()
    (dataset / 'payload').write_bytes(b'secret')
    link = tmp_path / 'linked'
    link.symlink_to(dataset, target_is_directory=True)
    staging = tmp_path / 'stage'
    staging.mkdir()
    with pytest.raises(TransferError):
        module.direct_copy(source, target, source_info(link), str(staging), 'linked-source')
    target_link = tmp_path / 'linked-stage'
    target_link.symlink_to(staging, target_is_directory=True)
    with pytest.raises(TransferError):
        module.direct_copy(source, target, source_info(dataset), str(target_link), 'linked-target')
    assert not list(staging.iterdir())


def test_pinned_host_key_mismatch_is_rejected_before_any_remote_operation(tmp_path, peers):
    source, target = peers
    source.expected_fingerprint = 'SHA256:unexpected'
    file = tmp_path / 'file'
    file.write_bytes(b'x')
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(file), str(tmp_path), 'bad-pin')
    assert error.value.code == 'DIRECT_HOST_KEY_MISMATCH'
    assert not source.channels and not target.channels


def test_invalid_destination_and_shell_identity_do_not_start_remote_programs(tmp_path, peers):
    source, target = peers
    file = tmp_path / 'file'
    file.write_bytes(b'x')
    source.host = '-oProxyCommand=evil'
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(file), str(tmp_path), 'bad-host')
    assert error.value.code == 'DIRECT_SSH_IDENTITY_INVALID'
    assert not source.channels and not target.channels


def test_source_watchdog_revokes_authorization_while_control_session_remains_open(tmp_path, peers):
    source, _ = peers
    dataset = tmp_path / 'watchdog-source'
    dataset.mkdir()
    channel = module._open_channel(source, shlex.join(['python3', '-u', '-c', module._SOURCE_SCRIPT]))
    module._send(channel, {'token': 'a' * 32, 'public_key': 'ssh-ed25519 ' + 'A' * 48,
                          'path': str(dataset), 'kind': 'directory', 'auth_seconds': .4, 'timeout': 10})
    messages = module._Messages(channel)
    ready = []
    deadline = time.monotonic() + 4
    while not ready and time.monotonic() < deadline:
        ready.extend(messages.poll())
        time.sleep(.02)
    assert ready == [{'type': 'ready'}]
    auth = source.home / '.ssh' / 'authorized_keys'
    assert b'termitech-direct-' in auth.read_bytes()
    while auth.read_bytes() and time.monotonic() < deadline:
        time.sleep(.05)
    assert auth.read_bytes() == b''
    assert not channel.exit_status_ready()  # Expiry must not kill a healthy SSH session.
    module._finish(channel)


def test_target_janitor_removes_private_key_even_before_start_request(tmp_path, peers):
    source, target = peers
    staging = tmp_path / 'waiting-stage'
    staging.mkdir()
    token = 'b' * 32
    script = module._TARGET_SCRIPT.replace("min(600,p['timeout']+30)", '.4')
    assert script != module._TARGET_SCRIPT
    channel = module._open_channel(target, shlex.join(['python3', '-u', '-c', script]))
    module._send(channel, {'token': token, 'transfer_id': 'abandoned', 'source': module._source_identity(source),
                          'staging': str(staging), 'timeout': 10, 'initial_bytes': 0, 'bytes_total': 0, 'files_total': 0})
    messages = module._Messages(channel)
    ready = []
    deadline = time.monotonic() + 4
    while not ready and time.monotonic() < deadline:
        ready.extend(messages.poll())
        time.sleep(.02)
    assert ready[0]['type'] == 'ready'
    assert set(ready[0]) == {'type', 'public_key'}  # No private key crosses the channel.
    keydirs = list(Path('/tmp').glob('.termitech-direct-' + token + '-*'))
    assert len(keydirs) == 1 and (keydirs[0] / 'key').stat().st_mode & 0o777 == 0o600
    while keydirs[0].exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert not keydirs[0].exists()
    assert not channel.exit_status_ready()
    module._finish(channel)


def test_direct_stalled_connection_has_bounded_deadline_and_cleans_authorization(tmp_path, peers):
    source, target = peers
    dataset = tmp_path / 'timeout-source'
    dataset.mkdir()
    (dataset / 'payload').write_bytes(b'x')
    staging = tmp_path / '.timeout.partial'
    staging.mkdir()
    target.environment['DIRECT_FAKE_SSH_DELAY'] = '30'
    started = time.monotonic()
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(dataset), str(staging), 'timeout-1', timeout=1)
    assert error.value.code == 'DIRECT_TIMEOUT'
    assert time.monotonic() - started < 8
    assert (source.home / '.ssh' / 'authorized_keys').read_bytes() == b''


@pytest.mark.parametrize('value', [None, 0, 1, 'true', [], {}])
def test_direct_exclusion_profile_rejects_non_boolean_before_opening_any_channel(tmp_path, peers, value):
    source, target = peers
    file = tmp_path / 'parameters'
    file.write_bytes(b'weights')
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(file), str(tmp_path), 'strict-profile', exclude_train_state=value)
    assert error.value.code == 'DIRECT_PROFILE_INVALID'
    assert not source.channels and not target.channels


def diagnostic_classifier():
    namespace = {}
    exec(module._REMOTE_COMMON + module._DIAGNOSTIC_HELPERS, namespace)
    return namespace['classify_direct_failure']


@pytest.mark.parametrize(('diagnostic', 'expected', 'retry'), [
    ('kex_exchange_identification: banner line 0: Exceeded MaxStartups\n', 'DIRECT_SSH_BUSY', True),
    ('ssh: connect to host sensitive-source port 22: Connection refused\n', 'DIRECT_CONNECTION_UNAVAILABLE', True),
    ('ssh: connect to host sensitive-source port 22: Connection timed out\n', 'DIRECT_CONNECTION_UNAVAILABLE', True),
    ('ssh: connect to host sensitive-source port 22: No route to host\n', 'DIRECT_CONNECTION_UNAVAILABLE', True),
    ('kex_exchange_identification: read: Connection reset by peer\n', 'DIRECT_CONNECTION_UNAVAILABLE', True),
    ('ssh: Could not resolve hostname sensitive-source: Temporary failure in name resolution\n', 'DIRECT_CONNECTION_UNAVAILABLE', True),
    ('Host key verification failed.\n', 'DIRECT_HOST_KEY_REJECTED', False),
    ('REMOTE HOST IDENTIFICATION HAS CHANGED!\n', 'DIRECT_HOST_KEY_REJECTED', False),
    ('root@sensitive-source: Permission denied (publickey,password).\n', 'DIRECT_AUTHENTICATION_FAILED', False),
    ('debug1: No more authentication methods to try.\n', 'DIRECT_AUTHENTICATION_FAILED', False),
    ('Unable to negotiate: no matching host key type found.\n', 'DIRECT_SSH_NEGOTIATION_FAILED', False),
    ('Load key "/private-sensitive/key": invalid format\n', 'DIRECT_TEMPORARY_KEY_UNAVAILABLE', False),
    ('ssh: Could not resolve hostname sensitive-source: Name or service not known\n', 'DIRECT_DNS_FAILED', False),
    ('unrecognized diagnostic containing PRIVATE-SECRET\n', 'DIRECT_CONNECTION_UNAVAILABLE', False),
])
def test_connection_classification_uses_bounded_ssh_labels_without_returning_raw_material(diagnostic, expected, retry):
    result = diagnostic_classifier()(255, diagnostic.encode(), b'rsync: connection unexpectedly closed\n', False, False)
    assert result['code'] == expected and result['retryable'] is retry
    assert 'sensitive' not in result['message'] and 'PRIVATE-SECRET' not in result['message']


@pytest.mark.parametrize(('authenticated', 'activity', 'truncated'), [
    (True, False, False), (False, True, False), (False, False, True),
])
def test_preconnection_diagnostic_is_never_enough_to_retry_authenticated_active_or_truncated_sessions(authenticated, activity, truncated):
    result = diagnostic_classifier()(255, b'kex_exchange_identification: banner line 0: Exceeded MaxStartups\n',
                                      b'', authenticated, activity, truncated)
    assert result['retryable'] is False


@pytest.mark.parametrize(('exit_code', 'stderr', 'expected'), [
    (23, b'rsync: [receiver] open private-file failed: Permission denied (13)\n', 'DIRECT_RSYNC_PERMISSION_DENIED'),
    (11, b'rsync: write private-file failed: No space left on device (28)\n', 'DIRECT_TARGET_SPACE_INSUFFICIENT'),
    (23, b'rsync: private-file read error\n', 'DIRECT_RSYNC_FILE_ERROR'),
    (24, b'rsync: source file vanished\n', 'DIRECT_SOURCE_CHANGED'),
    (12, b'rsync: broken protocol\n', 'DIRECT_RSYNC_PROTOCOL_FAILED'),
    (30, b'rsync: I/O timeout\n', 'DIRECT_TRANSFER_STALLED'),
    (14, b'rsync: local IPC error\n', 'DIRECT_RSYNC_FAILED'),
    (12, b'TERMITECH_DIRECT_SOURCE_ERROR DIRECT_SOURCE_SCOPE_UNAVAILABLE\n', 'DIRECT_SOURCE_SCOPE_UNAVAILABLE'),
])
def test_file_scope_and_established_stream_errors_are_not_hidden_as_connection_retry(exit_code, stderr, expected):
    result = diagnostic_classifier()(exit_code, b'Authenticated to sensitive-source using "publickey".\n', stderr, True, True)
    assert result['code'] == expected and result['retryable'] is False
    assert 'private-file' not in result['message'] and 'sensitive-source' not in result['message']


def test_rsync_filename_cannot_forge_an_ssh_connect_diagnosis():
    result = diagnostic_classifier()(12, b'', b'ssh: connect to host secret port 22: Connection refused\n', False, False)
    assert result['reason'] == 'unknown' and result['retryable'] is False


def test_maxstartups_failure_retries_only_one_connection_then_real_direct_copy_succeeds(tmp_path, peers):
    source, target = peers
    dataset = tmp_path / 'source-busy-once'
    dataset.mkdir()
    (dataset / 'payload').write_bytes(bytes(range(256)) * 4096)
    staging = tmp_path / '.busy.partial'
    staging.mkdir()
    count = tmp_path / 'ssh-attempts'
    target.environment.update(DIRECT_FAKE_SSH_COUNT=str(count), DIRECT_FAKE_SSH_FAIL_FIRST='1')
    events = []
    result = module.direct_copy(source, target, source_info(dataset), str(staging), 'busy-once', progress=events.append)
    assert result['transport'] == 'direct-ssh' and count.read_text() == '2'
    assert (staging / 'payload').read_bytes() == (dataset / 'payload').read_bytes()
    assert sum('仅重试一次连接' in event.get('message', '') for event in events) == 1
    assert (source.home / '.ssh' / 'authorized_keys').read_bytes() == b''


def test_persistent_maxstartups_is_bounded_and_reports_fixed_phase_and_exit_code(tmp_path, peers):
    source, target = peers
    file = tmp_path / 'weights'
    file.write_bytes(b'weights')
    staging = tmp_path / '.persistent-busy.partial'
    staging.mkdir()
    count = tmp_path / 'ssh-attempts'
    target.environment.update(DIRECT_FAKE_SSH_COUNT=str(count), DIRECT_FAKE_SSH_FAILURE='1',
                              DIRECT_FAKE_SSH_LOG='kex_exchange_identification: banner line 0: Exceeded MaxStartups\nPRIVATE-SECRET\n')
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(file), str(staging), 'persistent-busy')
    assert error.value.code == 'DIRECT_SSH_BUSY' and count.read_text() == '2'
    assert 'phase=SSH_CONNECT' in error.value.message and 'rsync_exit=' in error.value.message
    assert 'attempts=2' in error.value.message
    assert 'PRIVATE-SECRET' not in error.value.message
    assert all(b'PRIVATE-SECRET' not in channel.control_output for remote in peers for channel in remote.channels)


@pytest.mark.parametrize(('diagnostic', 'expected'), [
    ('Host key verification failed.\n', 'DIRECT_HOST_KEY_REJECTED'),
    ('root@sensitive-source: Permission denied (publickey).\n', 'DIRECT_AUTHENTICATION_FAILED'),
    ('Authenticated to sensitive-source using "publickey".\nkex_exchange_identification: read: Connection reset by peer\n', 'DIRECT_CONNECTION_LOST'),
    ('', 'DIRECT_CONNECTION_UNAVAILABLE'),
])
def test_host_key_authentication_authenticated_disconnect_and_unknown_failures_never_auto_retry(tmp_path, peers, diagnostic, expected):
    source, target = peers
    file = tmp_path / 'weights'
    file.write_bytes(b'weights')
    staging = tmp_path / '.permanent.partial'
    staging.mkdir()
    count = tmp_path / 'ssh-attempts'
    target.environment.update(DIRECT_FAKE_SSH_COUNT=str(count), DIRECT_FAKE_SSH_FAILURE='1', DIRECT_FAKE_SSH_LOG=diagnostic)
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(file), str(staging), 'permanent')
    assert error.value.code == expected and count.read_text() == '1'
    assert 'attempts=1' in error.value.message and 'sensitive-source' not in error.value.message


def test_actual_rsync_exit_23_remains_a_file_permission_failure_and_is_not_retried(tmp_path, peers):
    source, target = peers
    file = tmp_path / 'weights'
    file.write_bytes(b'weights')
    staging = tmp_path / '.file-error.partial'
    staging.mkdir()
    binaries = Path(target.environment['PATH'].split(os.pathsep)[0])
    counter = tmp_path / 'rsync-attempts'
    executable = binaries / 'rsync'
    executable.write_text('#!/usr/bin/env python3\nimport pathlib,sys\np=pathlib.Path(' + repr(str(counter))
                          + ')\np.write_text(str(int(p.read_text())+1 if p.exists() else 1))\n'
                          'print("rsync: [receiver] open private-sensitive-file failed: Permission denied (13)",file=sys.stderr)\n'
                          'sys.exit(23)\n')
    executable.chmod(0o700)
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(file), str(staging), 'file-error')
    assert error.value.code == 'DIRECT_RSYNC_PERMISSION_DENIED' and counter.read_text() == '1'
    assert 'rsync_exit=23' in error.value.message and 'attempts=1' in error.value.message
    assert 'private-sensitive-file' not in error.value.message


def test_cancellation_during_connection_retry_backoff_does_not_start_another_ssh_session(tmp_path, peers):
    source, target = peers
    file = tmp_path / 'weights'
    file.write_bytes(b'weights')
    staging = tmp_path / '.retry-cancelled.partial'
    staging.mkdir()
    count = tmp_path / 'ssh-attempts'
    target.environment.update(DIRECT_FAKE_SSH_COUNT=str(count), DIRECT_FAKE_SSH_FAIL_FIRST='1')
    cancelled = False
    def progress(event):
        nonlocal cancelled
        if '仅重试一次连接' in event.get('message', ''):cancelled = True
    with pytest.raises(TransferError) as error:
        module.direct_copy(source, target, source_info(file), str(staging), 'retry-cancelled',
                           progress=progress, cancelled=lambda: cancelled)
    assert error.value.code == 'TRANSFER_CANCELLED' and count.read_text() == '1'
    assert (source.home / '.ssh' / 'authorized_keys').read_bytes() == b''


def test_ssh_diagnostic_reader_bounds_memory_rejects_symlinks_and_never_uses_truncation_as_retry_proof(tmp_path):
    namespace = {}
    exec(module._REMOTE_COMMON + module._DIAGNOSTIC_HELPERS, namespace)
    read = namespace['read_ssh_diagnostic']
    classify = namespace['classify_direct_failure']
    log = tmp_path / 'ssh.log'
    log.write_bytes(b'ssh: connect to host secret port 22: Connection refused\n' + b'x' * 40000)
    raw, truncated = read(str(log))
    assert truncated is True and len(raw) <= 32768 + 8193
    assert classify(255, raw, b'', False, False, truncated)['retryable'] is False
    link = tmp_path / 'linked.log'
    link.symlink_to(log)
    assert read(str(link)) == (b'', True)
    log.write_bytes(b'x' * (128 * 1024 + 1))
    with pytest.raises(ValueError, match='DIRECT_OUTPUT_LIMIT'):
        read(str(log))
