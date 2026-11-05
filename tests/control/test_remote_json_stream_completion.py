"""SSH process exit can precede the final stdout/stderr channel packets."""

import json
from types import SimpleNamespace

import pytest

from vla_control import checkpoint_transfer_transport as transport


class StreamClock:
    def __init__(self):
        self.value = 1000.0
        self.events = []

    def monotonic(self):
        return self.value

    def later(self, seconds, callback):
        self.events.append((self.value + seconds, callback))
        self.events.sort(key=lambda event: event[0])

    def sleep(self, seconds):
        self.value += seconds
        while self.events and self.events[0][0] <= self.value:
            _, callback = self.events.pop(0)
            callback()


class StreamChannel:
    def __init__(self, stdout=b'', stderr=b'', *, status_ready=True, eof=False, status=0):
        self.stdout = bytearray(stdout)
        self.stderr = bytearray(stderr)
        self.status_ready = status_ready
        self.eof_received = eof
        self.closed = False
        self.status = status
        self.sent = bytearray()
        self.stdout_received = 0
        self.stderr_received = 0
        self.status_reads = 0
        self.close_calls = 0

    def sendall(self, value):
        self.sent.extend(value)

    def recv_ready(self):
        return bool(self.stdout)

    def recv(self, count):
        value = bytes(self.stdout[:count])
        del self.stdout[:len(value)]
        self.stdout_received += len(value)
        return value

    def recv_stderr_ready(self):
        return bool(self.stderr)

    def recv_stderr(self, count):
        value = bytes(self.stderr[:count])
        del self.stderr[:len(value)]
        self.stderr_received += len(value)
        return value

    def exit_status_ready(self):
        return self.status_ready

    def recv_exit_status(self):
        assert self.status_ready
        self.status_reads += 1
        return self.status

    def finish(self, *, stdout=b'', stderr=b'', eof=True, closed=False, status_ready=True):
        self.stdout.extend(stdout)
        self.stderr.extend(stderr)
        self.eof_received = eof
        self.closed = closed
        self.status_ready = status_ready

    def close(self):
        self.close_calls += 1
        self.closed = True


@pytest.fixture
def stream(monkeypatch):
    clock = StreamClock()
    monkeypatch.setattr(transport, 'time', SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))

    def run(channel, *, cancelled=lambda: False, **kwargs):
        monkeypatch.setattr(transport, '_open_channel', lambda _remote, _command: channel)
        return transport._remote_json(object(), {}, cancelled, **kwargs)

    return clock, run


def test_exit_status_before_large_manifest_tail_waits_for_stream_eof(stream):
    clock, run = stream
    expected = {'text': 'x' * 1048576}
    encoded = json.dumps({'result': expected}).encode()
    channel = StreamChannel(encoded[:196608])
    clock.later(.02, lambda: channel.finish(stdout=encoded[196608:]))

    assert run(channel) == expected
    assert channel.stdout_received == len(encoded)
    assert channel.status_reads == 1
    assert channel.close_calls == 1
    assert clock.value >= 1000.02


def test_complete_json_does_not_close_channel_before_late_stderr(stream):
    clock, run = stream
    channel = StreamChannel(b'{"result":{"ok":true}}')
    diagnostic = b'late remote diagnostic\n'
    clock.later(.02, lambda: channel.finish(stderr=diagnostic))

    assert run(channel) == {'ok': True}
    assert channel.stderr_received == len(diagnostic)
    assert not channel.stderr
    assert channel.close_calls == 1


def test_eof_before_exit_status_waits_for_status(stream):
    clock, run = stream
    channel = StreamChannel(b'{"result":42}', eof=True, status_ready=False)
    clock.later(.02, lambda: channel.finish())

    assert run(channel) == 42
    assert clock.value >= 1000.02
    assert channel.status_reads == 1


def test_closed_channel_still_drains_buffered_output(stream):
    _, run = stream
    expected = {'text': 'x' * 200000}
    encoded = json.dumps({'result': expected}).encode()
    channel = StreamChannel(encoded)
    channel.closed = True

    assert run(channel) == expected
    assert channel.stdout_received == len(encoded)
    assert not channel.stdout
    assert channel.close_calls == 1


def test_cancellation_remains_effective_after_exit_status(stream):
    clock, run = stream
    channel = StreamChannel(b'{"result":42}')

    with pytest.raises(transport.TransferError) as exc:
        run(channel, cancelled=lambda: clock.value >= 1000.02)

    assert exc.value.code == 'TRANSFER_CANCELLED'
    assert channel.status_reads == 0
    assert channel.close_calls == 1


def test_deadline_remains_effective_after_exit_status_without_eof(stream):
    _, run = stream
    channel = StreamChannel(b'{"result":42}')

    with pytest.raises(transport.TransferError) as exc:
        run(channel, timeout=.015)

    assert exc.value.code == 'TRANSFER_TIMEOUT'
    assert channel.status_reads == 0
    assert channel.close_calls == 1


@pytest.mark.parametrize('output', ['stdout', 'stderr'])
def test_result_limits_remain_effective_for_late_packets(stream, output):
    clock, run = stream
    channel = StreamChannel(b'{"result":42}')
    clock.later(.02, lambda: channel.finish(**{output: b'x' * 200}))

    with pytest.raises(transport.TransferError) as exc:
        run(channel, max_bytes=128)

    assert exc.value.code == 'TRANSFER_MANIFEST_LIMIT'
    assert channel.close_calls == 1


@pytest.mark.parametrize('value', [b'', b'not JSON', b'[]', b'{"result":42} extra', b'\xff'])
def test_completed_malformed_output_is_still_rejected(stream, value):
    _, run = stream
    channel = StreamChannel(value, eof=True)

    with pytest.raises(transport.TransferError) as exc:
        run(channel)

    assert exc.value.code == 'TRANSFER_REMOTE_FAILED'
    assert 'invalid result' in exc.value.message
    assert channel.close_calls == 1


def test_nonzero_exit_status_still_fails_after_stream_completion(stream):
    clock, run = stream
    channel = StreamChannel(b'{"result":42}', status=7)
    clock.later(.02, lambda: channel.finish())

    with pytest.raises(transport.TransferError) as exc:
        run(channel)

    assert exc.value.code == 'TRANSFER_REMOTE_FAILED'
    assert 'process failed' in exc.value.message
    assert channel.close_calls == 1


def test_remote_error_envelope_is_preserved(stream):
    _, run = stream
    channel = StreamChannel(b'{"error":{"code":"TRANSFER_SOURCE_CHANGED","message":"source changed"}}', eof=True)

    with pytest.raises(transport.TransferError) as exc:
        run(channel)

    assert exc.value.code == 'TRANSFER_SOURCE_CHANGED'
    assert exc.value.message == 'source changed'
    assert channel.close_calls == 1
