"""The transport moves real rsync protocol bytes between isolated local peers."""
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import time

import pytest

from vla_control.checkpoint_transfer_transport import TransferError, transfer_checkpoint


class LocalChannel:
    def __init__(self):
        self.process = None
        self.stdout_eof = False
        self.stderr_eof = False
        self.stderr = bytearray()

    def settimeout(self, _value):
        pass

    def exec_command(self, command):
        self.process = subprocess.Popen(['/bin/sh', '-c', command], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, start_new_session=True)

    def recv_ready(self):
        return not self.stdout_eof and bool(select.select([self.process.stdout], [], [], 0)[0])

    def recv_stderr_ready(self):
        return not self.stderr_eof and bool(select.select([self.process.stderr], [], [], 0)[0])

    def recv(self, count):
        value = os.read(self.process.stdout.fileno(), count)
        if not value:
            self.stdout_eof = True
        return value

    def recv_stderr(self, count):
        value = os.read(self.process.stderr.fileno(), count)
        self.stderr.extend(value)
        if not value:
            self.stderr_eof = True
        return value

    def sendall(self, value):
        while value:
            size = os.write(self.process.stdin.fileno(), value)
            value = value[size:]

    def shutdown_write(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()

    def exit_status_ready(self):
        return self.process.poll() is not None

    def recv_exit_status(self):
        return self.process.wait(timeout=2)

    def close(self):
        if self.process is None:
            return
        self.shutdown_write()
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()


class LocalRemote:
    client = None

    def __init__(self):
        self.client = self
        self.channels = []

    def get_transport(self):
        return self

    def is_active(self):
        return True

    def open_session(self, timeout):
        assert timeout == 15
        channel = LocalChannel()
        self.channels.append(channel)
        return channel


@pytest.fixture
def checkpoint(tmp_path):
    if not shutil.which('rsync'):
        pytest.skip('Real transport tests require rsync')
    source = tmp_path / "source checkpoint';echo BAD; #"
    target = tmp_path / 'target'
    (source / 'params' / 'd').mkdir(parents=True)
    (source / 'params' / 'd' / 'weights').write_bytes(bytes(range(256)) * 4096)
    (source / 'assets' / 'robot').mkdir(parents=True)
    (source / 'assets' / 'robot' / 'norm_stats.json').write_text('{"state":{}}')
    (source / '元信息.json').write_text('{"fixture":true}')
    target.mkdir()
    return tmp_path, source, target


def transfer(checkpoint, **changes):
    root, source, target = checkpoint
    arguments = dict(source_remote=LocalRemote(), target_remote=LocalRemote(), source_path=str(source),
        target_parent=str(target), transfer_id='transfer-123', source_roots=[str(root)], target_roots=[str(root)])
    arguments.update(changes)
    try:
        return transfer_checkpoint(**arguments)
    except BaseException:
        for remote in (arguments['source_remote'],arguments['target_remote']):
            for channel in remote.channels:
                if channel.stderr:
                    print(channel.stderr.decode(errors='replace'))
        raise


def test_real_rsync_ssh_stream_bridge_complete_checkpoint_and_sidecar(checkpoint):
    events = []
    result = transfer(checkpoint, progress=events.append, extra_files={'termitech-policy.json': '{"config":{},"digest":"fixture"}'})
    root, source, target = checkpoint
    published = Path(result['target_path'])
    assert published == target / 'transfer-123'
    assert (published / 'params' / 'd' / 'weights').read_bytes() == (source / 'params' / 'd' / 'weights').read_bytes()
    assert (published / 'assets' / 'robot' / 'norm_stats.json').is_file()
    assert json.loads((published / 'termitech-policy.json').read_text())['digest'] == 'fixture'
    manifest = result['manifest']
    assert manifest['files_total'] == 4
    assert manifest['bytes_total'] == sum(item['size'] for item in manifest['files'])
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert result['manifest_sha256'] == hashlib.sha256(canonical.encode()).hexdigest()
    assert events[-1]['stage'] == 'SUCCEEDED'
    assert events[-1]['percent'] == 100
    assert {event['bytes_total'] for event in events if 'bytes_total' in event} == {result['bytes_total']}
    assert {event['files_total'] for event in events if 'files_total' in event} == {result['files_total']}
    assert any(item['stage'] == 'TRANSFERRING' for item in events)
    assert not (target / '.transfer-123.partial').exists()
    assert not (root / 'BAD').exists()


def test_train_state_directory_is_never_read_hashed_or_sent_but_model_assets_and_tokenizer_remain(checkpoint, monkeypatch):
    from vla_control import checkpoint_transfer_transport as module
    _, source, target = checkpoint
    state = source / 'train_state'
    state.mkdir()
    with (state / 'large-optimizer').open('wb') as writer:
        writer.truncate(16 * 1024**3)
    os.mkfifo(state / 'unreadable-special-state')
    (source / 'tokenizer.model').write_bytes(b'tokenizer')
    # A sparse fake is large enough to detect unintended IO. Fail immediately
    # if inventory opens any excluded child, even when the test runs as root.
    checked = module._REMOTE_SCRIPT.replace("def manifest(path):", """
original_open=os.open
def checked_open(value,*args,**kwargs):
    base=p.get('path')
    if base and isinstance(value,(str,os.PathLike)) and str(value).startswith(str(base)+'/train_state/'):
        raise AssertionError('excluded training state must never be read')
    return original_open(value,*args,**kwargs)
os.open=checked_open
def manifest(path):""")
    monkeypatch.setattr(module, '_REMOTE_SCRIPT', checked)
    result = transfer(checkpoint)
    published = Path(result['target_path'])
    assert not (published / 'train_state').exists()
    assert (published / 'tokenizer.model').read_bytes() == b'tokenizer'
    assert (published / 'params' / 'd' / 'weights').is_file()
    assert (published / 'assets' / 'robot' / 'norm_stats.json').is_file()
    assert result['files_total'] == 4
    assert all(not item['path'].startswith('train_state/') for item in result['manifest']['files'])
    assert result['bytes_total'] == sum(path.stat().st_size for path in source.rglob('*')
        if path.is_file() and not path.is_relative_to(state))
    assert (state / 'large-optimizer').stat().st_size == 16 * 1024**3
    assert (state / 'unreadable-special-state').exists()


def test_train_state_regular_file_and_nested_directory_are_strictly_transferred(checkpoint):
    _, source, _ = checkpoint
    (source / 'train_state').write_bytes(b'ordinary top-level file')
    (source / 'params' / 'train_state').mkdir()
    (source / 'params' / 'train_state' / 'kept').write_bytes(b'nested model data')
    result = transfer(checkpoint)
    published = Path(result['target_path'])
    assert (published / 'train_state').read_bytes() == b'ordinary top-level file'
    assert (published / 'params' / 'train_state' / 'kept').read_bytes() == b'nested model data'
    assert {'train_state', 'params/train_state/kept'} <= {item['path'] for item in result['manifest']['files']}


def test_train_state_symlink_is_not_mistaken_for_an_excluded_directory(checkpoint):
    root, source, _ = checkpoint
    (source / 'train_state').symlink_to(root, target_is_directory=True)
    with pytest.raises(TransferError) as error:
        transfer(checkpoint)
    assert error.value.code == 'TRANSFER_UNSAFE_FILE'


def test_retry_keeps_same_named_train_state_file_in_strict_source_identity(checkpoint):
    source = checkpoint[1]
    (source / 'train_state').write_bytes(b'first-file')
    cancelled = False
    def progress(event):
        nonlocal cancelled
        if event['stage'] == 'TRANSFERRING':
            cancelled = True
    with pytest.raises(TransferError):
        transfer(checkpoint, progress=progress, cancelled=lambda: cancelled)
    (source / 'train_state').write_bytes(b'other-file')
    with pytest.raises(TransferError) as error:
        transfer(checkpoint)
    assert error.value.code == 'TRANSFER_SOURCE_CHANGED'


def test_retry_normalizes_only_legacy_train_state_receipt_and_preserves_partial_directory(checkpoint):
    _, source, target = checkpoint
    cancelled = False
    def progress(event):
        nonlocal cancelled
        if event['stage'] == 'TRANSFERRING':
            cancelled = True
    with pytest.raises(TransferError):
        transfer(checkpoint, progress=progress, cancelled=lambda: cancelled)
    partial = target / '.transfer-123.partial'
    (partial / 'train_state').mkdir()
    (partial / 'train_state' / 'old-optimizer').write_bytes(b'keep partial user data')
    receipt = target / '.transfer-123.transfer.json'
    previous = json.loads(receipt.read_text())
    extra = {'path': 'train_state/old-optimizer', 'size': 22, 'sha256': 'a' * 64}
    for key in ('manifest', 'source_manifest'):
        previous[key]['files'].append(extra)
        previous[key]['files'].sort(key=lambda item: item['path'])
        previous[key]['files_total'] += 1
        previous[key]['bytes_total'] += extra['size']
    receipt.write_text(json.dumps(previous))
    result = transfer(checkpoint)
    published = Path(result['target_path'])
    assert (published / 'train_state' / 'old-optimizer').read_bytes() == b'keep partial user data'
    assert all(not item['path'].startswith('train_state/') for item in result['manifest']['files'])
    assert transfer(checkpoint) == result


def test_retry_resumes_partial_and_does_not_replace_other_files(checkpoint):
    root, source, target = checkpoint
    cancelled = False
    def progress(event):
        nonlocal cancelled
        if event['stage'] == 'TRANSFERRING':
            cancelled = True
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, progress=progress, cancelled=lambda:cancelled)
    assert error.value.code == 'TRANSFER_CANCELLED'
    partial = target / '.transfer-123.partial'
    (partial / 'params' / 'd').mkdir(parents=True,exist_ok=True)
    content = (source / 'params' / 'd' / 'weights').read_bytes()
    (partial / 'params' / 'd' / 'weights').write_bytes(content[:100000])
    unrelated = target / 'user-model'
    unrelated.mkdir()
    (unrelated / 'keep').write_text('preserved')
    result = transfer(checkpoint)
    assert (Path(result['target_path']) / 'params' / 'd' / 'weights').read_bytes() == content
    assert (unrelated / 'keep').read_text() == 'preserved'
    recovered = transfer(checkpoint)
    assert recovered == result


def test_symlinks_special_files_and_extra_conflicts_are_rejected(checkpoint):
    root, source, target = checkpoint
    linked = source / 'escape'
    linked.symlink_to(root)
    with pytest.raises(TransferError) as error:
        transfer(checkpoint)
    assert error.value.code == 'TRANSFER_UNSAFE_FILE'
    linked.unlink()
    (source / 'termitech-policy.json').write_text('unrelated')
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, extra_files={'termitech-policy.json': 'new'})
    assert error.value.code == 'TRANSFER_EXTRA_CONFLICT'
    assert not list(target.iterdir())


def test_outside_roots_cancel_and_invalid_extra_do_not_publish(checkpoint):
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, source_roots=['/nonexistent'])
    assert error.value.code == 'TRANSFER_PATH_OUTSIDE_ROOTS'
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, cancelled=lambda: True)
    assert error.value.code == 'TRANSFER_CANCELLED'
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, extra_files={'../escape': 'bad'})
    assert error.value.code == 'TRANSFER_EXTRA_INVALID'
    assert not list(checkpoint[2].iterdir())


def test_partial_directory_with_unrelated_file_is_never_deleted(checkpoint):
    partial = checkpoint[2] / '.transfer-123.partial'
    partial.mkdir()
    (partial / 'keep').write_text('user content')
    with pytest.raises(TransferError) as error:
        transfer(checkpoint)
    assert error.value.code == 'TRANSFER_PARTIAL_CONFLICT'
    assert (partial / 'keep').read_text() == 'user content'


def test_source_change_during_copy_and_between_retry_is_never_adopted(checkpoint):
    source = checkpoint[1]
    changed = False
    def progress(event):
        nonlocal changed
        if event['stage'] == 'VERIFYING' and not changed:
            (source / 'params' / 'd' / 'weights').write_bytes(b'changed source')
            changed = True
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, progress=progress)
    assert error.value.code == 'TRANSFER_SOURCE_CHANGED'
    with pytest.raises(TransferError) as error:
        transfer(checkpoint)
    assert error.value.code == 'TRANSFER_SOURCE_CHANGED'
    assert not (checkpoint[2] / 'transfer-123').exists()


def test_checksum_corruption_blocks_publish_and_retry_repairs_copy(checkpoint):
    target = checkpoint[2]
    def corrupt(event):
        if event['stage'] == 'VERIFYING':
            (target / '.transfer-123.partial' / 'params' / 'd' / 'weights').write_bytes(b'corrupted')
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, progress=corrupt)
    assert error.value.code == 'TRANSFER_CHECKSUM_MISMATCH'
    assert not (target / 'transfer-123').exists()
    assert Path(transfer(checkpoint)['target_path']).is_dir()


def test_target_space_check_prevents_rsync_and_preserves_source(checkpoint, monkeypatch):
    from vla_control import checkpoint_transfer_transport as module
    monkeypatch.setattr(module, '_REMOTE_SCRIPT', module._REMOTE_SCRIPT.replace('shutil.disk_usage(parent).free', '0'))
    with pytest.raises(TransferError) as error:
        transfer(checkpoint)
    assert error.value.code == 'TRANSFER_SPACE_INSUFFICIENT'
    assert (checkpoint[1] / 'params' / 'd' / 'weights').is_file()
    assert not (checkpoint[2] / 'transfer-123').exists()


def test_creates_nested_target_only_inside_managed_root(checkpoint):
    root, _, target = checkpoint
    destination = target / 'new' / 'deployed-models'
    result = transfer(checkpoint, target_parent=str(destination))
    assert Path(result['target_path']) == destination / 'transfer-123'
    escaped = target / 'escape'
    escaped.symlink_to(root.parent)
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, target_parent=str(escaped / 'outside'), transfer_id='other-transfer')
    assert error.value.code == 'TRANSFER_PATH_INVALID'


def test_source_control_character_name_is_rejected_before_rsync(checkpoint):
    (checkpoint[1] / 'bad\nname').write_text('unsafe')
    with pytest.raises(TransferError) as error:
        transfer(checkpoint)
    assert error.value.code == 'TRANSFER_UNSAFE_FILE'
    assert not list(checkpoint[2].iterdir())


def test_retry_reuses_identical_sidecar_but_never_overwrites_changed_configuration(checkpoint):
    extras = {'termitech-policy.json': '{"config":{},"digest":"fixture"}'}
    target = checkpoint[2]
    def corrupt(event):
        if event['stage'] == 'VERIFYING':
            (target / '.transfer-123.partial' / 'params' / 'd' / 'weights').write_bytes(b'corrupted')
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, progress=corrupt, extra_files=extras)
    assert error.value.code == 'TRANSFER_CHECKSUM_MISMATCH'
    assert (target / '.transfer-123.partial' / 'termitech-policy.json').is_file()
    with pytest.raises(TransferError) as error:
        transfer(checkpoint, extra_files={'termitech-policy.json': 'different configuration'})
    assert error.value.code == 'TRANSFER_SOURCE_CHANGED'
    result = transfer(checkpoint, extra_files=extras)
    assert transfer(checkpoint, extra_files=extras) == result
