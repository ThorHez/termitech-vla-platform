from pathlib import Path
import sys

import pytest

from vla_agent import network_process_metrics as metrics
from vla_agent.main import Agent


def test_socket_capture_ignores_loopback_same_host_and_unverified_owners(monkeypatch):
    rows = '\n'.join([
        '0 0 192.0.2.1:22 198.51.100.1:20001 users:(("sshd",pid=20,fd=4),("sshd",pid=20,fd=9)) '
        'ino:83 sk:a2 bytes_acked:804 bytes_received:204 send 9000000bps',
        'ESTAB 0 0 [2001:db8::1]:443 [2001:db8::2]:20001 users:(("proxy",pid=21,fd=4)) '
        'ino:84 sk:a3 bytes_received:500',
        '0 0 192.0.2.1:10 192.0.2.2:20001 users:(("samehost",pid=22,fd=4)) ino:85 sk:a4 bytes_acked:500',
        '0 0 [::ffff:127.0.0.1]:10 [::ffff:127.0.0.2]:20001 users:(("loop",pid=23,fd=4)) '
        'ino:86 sk:a5 bytes_acked:500',
        '0 0 192.0.2.1:10 198.51.100.1:20002 users:(("foo",pid=24,fd=4),("foo",pid=25,fd=4)) '
        'ino:87 sk:a6 bytes_acked:500',
        '0 0 192.0.2.1:10 198.51.100.1:20003 ino:88 sk:a7 bytes_acked:500',
        '0 0 192.0.2.1:10 198.51.100.1:20004 users:(("stale",pid=26,fd=4)) '
        'ino:89 sk:a8 bytes_acked:500',
    ])
    monkeypatch.setattr(metrics, '_ss_snapshot', lambda: rows)
    monkeypatch.setattr(metrics, '_local_addresses', lambda: {metrics._address('192.0.2.2')})
    monkeypatch.setattr(metrics, '_owns_socket', lambda pid, fd, inode: (pid, fd, inode) in {(20, 9, 83), (21, 4, 84)})
    monkeypatch.setattr(metrics, '_process_identity', lambda pid: (pid + 123, {20: 'sshd', 21: 'proxy'}[pid]))
    captured, partial = metrics._capture_sockets()

    assert partial is True
    assert list(captured.values()) == [
        metrics.SocketSample(20, 143, 'sshd', 204, 804),
        metrics.SocketSample(21, 144, 'proxy', 500, 0),
    ]


def test_process_label_uses_kernel_comm_and_starttime(monkeypatch):
    def stat(path, **options):
        assert path == Path('/proc/20/stat')
        return '20 (renderer:worker) S ' + ' '.join(['0'] * 18) + ' 123456 0'

    monkeypatch.setattr(Path, 'read_text', stat)
    assert metrics._process_identity(20) == (123456, 'renderer_worker')


def test_process_rates_require_two_matching_samples_and_add_sockets(monkeypatch):
    sample = metrics.SocketSample
    previous = {1: sample(20, 500, 'ssh', 40, 80), 2: sample(20, 500, 'ssh', 0, 90),
                3: sample(21, 600, 'old', 120, 900), 4: sample(20, 500, 'ssh', 50, 50),
                5: sample(24, 700, 'closed', 0, 10000)}
    current = {1: sample(20, 500, 'ssh', 140, 280), 2: sample(20, 500, 'ssh', 30, 160),
               3: sample(21, 601, 'new', 820, 9900), 4: sample(20, 500, 'ssh', 20, 70),
               6: sample(23, 700, 'fresh', 0, 5410)}
    snapshots = iter([(previous, True), (current, False)])
    monkeypatch.setattr(metrics.sys, 'platform', 'linux')
    monkeypatch.setattr(metrics, '_capture_sockets', lambda: next(snapshots))
    monkeypatch.setattr(metrics, 'monotonic', iter([10, 20]).__next__)
    monitor = metrics.NetworkProcessMonitor()

    assert monitor.sample() == {'status': 'warming_up', 'processes': [], 'partial': True}
    assert monitor.sample() == {
        'status': 'ready', 'partial': True,
        'processes': [{'pid': 20, 'name': 'ssh', 'rx_bytes_per_second': 13, 'tx_bytes_per_second': 27}],
    }


def test_report_is_ranked_bounded_and_restarts_after_capture_failure(monkeypatch):
    sample = metrics.SocketSample
    before = {pid: sample(pid, pid, 'worker', 0, 1) for pid in range(1, 14)}
    after = {pid: sample(pid, pid, 'worker', pid * 100, 1) for pid in before}
    snapshots = iter([(before, False), (after, False), None, (after, False)])

    def capture():
        if (result := next(snapshots)) is None:
            raise OSError('raw collector detail')
        return result

    monkeypatch.setattr(metrics.sys, 'platform', 'linux')
    monkeypatch.setattr(metrics, '_capture_sockets', capture)
    monkeypatch.setattr(metrics, 'monotonic', iter([10, 20, 40]).__next__)
    monitor = metrics.NetworkProcessMonitor()

    assert monitor.sample()['status'] == 'warming_up'
    assert [row['pid'] for row in monitor.sample()['processes']] == list(range(13, 3, -1))
    assert monitor.sample() == {'status': 'unavailable', 'processes': []}
    assert monitor.sample()['status'] == 'warming_up'


@pytest.mark.parametrize('code,error', [
    ('print("x" * 10000)', OSError),
    ('import time; time.sleep(3)', TimeoutError),
])
def test_ss_capture_is_bounded(tmp_path, monkeypatch, code, error):
    fake = tmp_path / 'ss'
    fake.write_text(f'#!{sys.executable}\n{code}\n')
    fake.chmod(0o700)
    monkeypatch.setattr(metrics.shutil, 'which', lambda *args, **kwargs: str(fake))
    monkeypatch.setattr(metrics, 'MAX_OUTPUT_BYTES', 4096)
    # Give Python enough startup time for the output limit assertion.
    monkeypatch.setattr(metrics, 'COMMAND_TIMEOUT_SECONDS', 0.5)
    with pytest.raises(error):
        metrics._ss_snapshot()


def test_agent_heartbeat_includes_network_process_report(agent_fixture, monkeypatch):
    seen = []

    class Client:
        def request(self, method, path, token, body=None, headers=None):
            if path == '/agent/v1/heartbeat':
                seen.append(body['metrics']['network_processes'])
            return {}

    report = {'status': 'ready', 'processes': [
        {'pid': 100, 'name': 'sshd', 'rx_bytes_per_second': 400, 'tx_bytes_per_second': 500},
    ]}
    monkeypatch.setattr(metrics.NetworkProcessMonitor, 'sample', lambda _: report)
    monkeypatch.setattr('vla_agent.main.gpu_inventory', lambda: ([], {}))
    agent = Agent(agent_fixture['config'], client=Client())
    agent.credentials = {'node_id': 'fixture-node', 'token': 'fixture-token'}
    try:
        agent.heartbeat()
    finally:
        agent.close()
    assert seen == [report]
