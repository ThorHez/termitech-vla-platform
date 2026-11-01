"""Exercise SSH flow control independently of a fast local rsync subprocess.

An SSH channel can accept only part of a write and then lose its send window.
The peer model below exposes those exact byte counts, temporary timeouts and
bidirectional dependencies without opening a network connection.
"""
from collections import deque
import socket

import pytest

from vla_control import checkpoint_transfer_transport as transport


class Clock:
    def __init__(self):
        self.value = 1000.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(seconds, .005)


class Peer:
    def __init__(self, output=b'', expected=b'', *, send_plan=(), writable=None, max_send=4096):
        self.output = bytearray(output)
        self.expected = expected
        self.received = bytearray()
        self.send_plan = deque(send_plan)
        self.writable = writable or (lambda: True)
        self.max_send = max_send
        self.closed = False
        self.shutdown_sizes = []
        self.bytes_read = 0
        self.read_observer = None
        self.timeouts = 0
        self.send_calls = 0
        self.timeout = None

    def settimeout(self, value):
        self.timeout = value

    def recv_ready(self):
        return bool(self.output)

    def recv(self, count):
        chunk = bytes(self.output[:count])
        del self.output[:len(chunk)]
        self.bytes_read += len(chunk)
        if self.read_observer:
            self.read_observer()
        return chunk

    def recv_stderr_ready(self):
        return False

    def send_ready(self):
        return self.writable()

    def send(self, data):
        self.send_calls += 1
        if not self.writable():
            self.timeouts += 1
            raise socket.timeout()
        result = self.send_plan.popleft() if self.send_plan else self.max_send
        if isinstance(result, BaseException):
            self.timeouts += isinstance(result, socket.timeout)
            raise result
        count = min(result, len(data))
        self.received.extend(data[:count])
        return count

    def sendall(self, data):
        # Match SSH sendall semantics: a timeout provides no acknowledged offset
        # to the caller even when earlier send() calls already accepted bytes.
        while data:
            count = self.send(data)
            if count == 0:
                raise EOFError()
            data = data[count:]

    def exit_status_ready(self):
        return not self.output and self.received == self.expected

    def recv_exit_status(self):
        assert self.exit_status_ready()
        return 0

    def shutdown_write(self):
        self.shutdown_sizes.append(len(self.received))

    def close(self):
        self.closed = True


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(transport.time, 'monotonic', value.monotonic)
    monkeypatch.setattr(transport.time, 'sleep', value.sleep)
    return value


def bridge(source, target, *, cancelled=lambda: False, idle_timeout=.15):
    transport._bridge(source, target, cancelled, lambda _event: None,
                      source.bytes_read + len(source.output), 1, idle_timeout=idle_timeout)


def test_partial_write_then_empty_timeout_preserves_each_protocol_byte(clock):
    data = bytes(range(256)) * 17
    reply = b'acknowledgement-and-rsync-control'
    source = Peer(data, reply, max_send=3)
    target = Peer(reply, data, send_plan=[7, socket.timeout(), 11], max_send=113)

    bridge(source, target)

    assert target.timeouts == 1
    assert bytes(target.received) == data
    assert bytes(source.received) == reply
    assert source.closed and target.closed
    assert all(size == len(data) for size in target.shutdown_sizes)


def test_reverse_direction_continues_while_forward_send_window_is_closed(clock):
    data = bytes(range(256)) * 1024
    reply = b'open-forward-window-after-this-control-message'
    source = Peer(data, reply, max_send=5)
    target = Peer(reply, data, writable=lambda: source.received == reply, max_send=997)
    pending = []
    source.read_observer = lambda: pending.append(source.bytes_read - len(target.received))

    bridge(source, target)

    assert bytes(source.received) == reply
    assert bytes(target.received) == data
    assert max(pending) <= 65536, 'A stalled destination must not buffer the dataset in the controller'
    assert source.closed and target.closed


def test_eof_is_forwarded_only_after_the_last_pending_bytes_are_delivered(clock):
    data = b'final-rsync-block' * 128
    source = Peer(data, b'')
    target = Peer(b'', data, writable=lambda: clock.value >= 1000.025, max_send=13)

    bridge(source, target)

    assert target.shutdown_sizes == [len(data)]
    assert bytes(target.received) == data


def test_cancellation_remains_responsive_during_backpressure(clock):
    source = Peer(b'pending-bytes')
    target = Peer(expected=b'pending-bytes', writable=lambda: False)

    with pytest.raises(transport.TransferError) as caught:
        bridge(source, target, cancelled=lambda: clock.value >= 1000.025)

    assert caught.value.code == 'TRANSFER_CANCELLED'
    assert str(caught.value)
    assert not target.received
    assert source.closed and target.closed


def test_backpressure_without_any_progress_has_explicit_idle_timeout(clock):
    source = Peer(b'pending-bytes')
    target = Peer(expected=b'pending-bytes', writable=lambda: False)

    with pytest.raises(transport.TransferError) as caught:
        bridge(source, target, idle_timeout=.03)

    assert caught.value.code == 'TRANSFER_IDLE_TIMEOUT'
    assert str(caught.value)
    assert source.closed and target.closed


@pytest.mark.parametrize('failure', [EOFError(), OSError(), 0])
def test_unrecoverable_write_failure_is_a_nonempty_transfer_error(clock, failure):
    source = Peer(b'pending-bytes')
    target = Peer(expected=b'pending-bytes', send_plan=[failure])

    with pytest.raises(transport.TransferError) as caught:
        bridge(source, target)

    assert caught.value.code not in {'TRANSFER_IDLE_TIMEOUT', 'TRANSFER_CANCELLED'}
    assert str(caught.value)
    assert source.closed and target.closed
