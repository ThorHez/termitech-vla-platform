from pathlib import Path
import json
import re
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
    monkeypatch.setattr(metrics, '_today', lambda: '2026-11-15')
    monkeypatch.setattr(metrics, '_process_identity', lambda pid, **kwargs: (500, 'ssh'))
    monitor = metrics.NetworkProcessMonitor()

    assert monitor.sample() == {'status': 'warming_up', 'processes': [], 'partial': True,
                                'today': {'date': '2026-11-15', 'processes': [], 'partial': True}}
    result = monitor.sample()
    today = result.pop('today')
    assert result == {
        'status': 'ready', 'partial': True,
        'processes': [{'pid': 20, 'name': 'ssh', 'rx_bytes_per_second': 13, 'tx_bytes_per_second': 27}],
    }
    assert today['date'] == '2026-11-15' and today['partial']
    row, = today['processes']
    assert re.fullmatch(r'[0-9a-f]{32}', row.pop('instance_id'))
    assert row == {'pid': 20, 'name': 'ssh', 'rx_bytes': 130, 'tx_bytes': 270, 'active': True}


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
    monkeypatch.setattr(metrics, '_today', lambda: '2026-11-15')
    monitor = metrics.NetworkProcessMonitor()

    assert monitor.sample()['status'] == 'warming_up'
    assert [row['pid'] for row in monitor.sample()['processes']] == list(range(13, 3, -1))
    unavailable = monitor.sample()
    assert unavailable['status'] == 'unavailable' and unavailable['processes'] == []
    assert [row['pid'] for row in unavailable['today']['processes']] == list(range(13, 3, -1))
    assert monitor.sample()['status'] == 'warming_up'


def test_today_aggregates_processes_outside_the_realtime_top_ten(monkeypatch):
    sample = metrics.SocketSample
    before = {pid: sample(pid, pid, 'worker', 500, 0) for pid in range(1, 14)}
    middle = {pid: sample(pid, pid, 'worker', 500 + pid, 0) for pid in before}
    after = {**middle, 1: sample(1, 1, 'worker', 521, 0)}
    snapshots = iter([(before, False), (middle, False), (after, False)])
    monkeypatch.setattr(metrics.sys, 'platform', 'linux')
    monkeypatch.setattr(metrics, '_capture_sockets', lambda: next(snapshots))
    monkeypatch.setattr(metrics, 'monotonic', iter([10, 20, 30]).__next__)
    monkeypatch.setattr(metrics, '_today', lambda: '2026-11-15')
    monkeypatch.setattr(metrics, '_process_identity', lambda pid, **kwargs: (pid, 'worker'))
    monitor = metrics.NetworkProcessMonitor()
    monitor.sample()
    first = monitor.sample()
    assert all(row['pid'] != 1 for row in first['processes'])
    assert all(row['pid'] != 1 for row in first['today']['processes'])
    total = monitor.sample()['today']['processes'][0]
    assert total['pid'] == 1 and total['rx_bytes'] == 21


def test_today_survives_agent_restart_and_keeps_exited_process_instances(tmp_path, monkeypatch):
    sample = metrics.SocketSample
    socket_key = (23, 55, '192.0.2.1:22', '198.51.100.1:3333')
    snapshots = iter([
        ({socket_key: sample(20, 500, 'sshd', 50, 200)}, False),
        ({socket_key: sample(20, 500, 'sshd', 70, 500)}, False),
        ({socket_key: sample(20, 600, 'python', 900, 4500)}, False),
        ({socket_key: sample(20, 600, 'python', 1000, 4510)}, False),
    ])
    monkeypatch.setattr(metrics.sys, 'platform', 'linux')
    monkeypatch.setattr(metrics, '_capture_sockets', lambda: next(snapshots))
    monkeypatch.setattr(metrics, 'monotonic', iter([10, 20, 40, 50]).__next__)
    monkeypatch.setattr(metrics, '_today', lambda: '2026-11-15')
    started = 500
    monkeypatch.setattr(metrics, '_boot_id', lambda: 'boot-three')
    monkeypatch.setattr(metrics, '_process_identity', lambda pid, **kwargs: (started, 'sshd'))
    first = metrics.NetworkProcessMonitor(tmp_path)
    first.sample()
    record, = first.sample()['today']['processes']
    assert (record['active'], record['rx_bytes'], record['tx_bytes']) == (True, 20, 300)
    text = (tmp_path / 'network-process-today.json').read_text()
    assert '198.51.100.1' not in text and '"active"' not in text

    started = 600
    restarted = metrics.NetworkProcessMonitor(tmp_path)
    restored = restarted.sample()
    assert restored['status'] == 'warming_up' and restored['processes'] == []
    assert restored['today']['processes'] == [{**record, 'active': False}]
    rows = restarted.sample()['today']['processes']
    assert len(rows) == 2
    assert rows[0] == {**record, 'active': False}
    assert rows[1]['instance_id'] != record['instance_id']
    assert (rows[1]['pid'], rows[1]['active'], rows[1]['rx_bytes'], rows[1]['tx_bytes']) == (20, True, 100, 10)


def test_today_resets_at_china_midnight_and_skips_the_crossing_delta(tmp_path, monkeypatch):
    sample = metrics.SocketSample
    snapshots = iter([({1: sample(20, 500, 'sshd', rx, 0)}, False) for rx in (500, 700, 950, 1000)])
    dates = iter(['2026-11-15'] * 4 + ['2026-11-16'] * 4)
    monkeypatch.setattr(metrics.sys, 'platform', 'linux')
    monkeypatch.setattr(metrics, '_capture_sockets', lambda: next(snapshots))
    monkeypatch.setattr(metrics, 'monotonic', iter([10, 20, 30, 40]).__next__)
    monkeypatch.setattr(metrics, '_today', lambda: next(dates))
    monkeypatch.setattr(metrics, '_process_identity', lambda pid, **kwargs: (500, 'sshd'))
    monitor = metrics.NetworkProcessMonitor(tmp_path)
    monitor.sample()
    assert monitor.sample()['today']['processes'][0]['rx_bytes'] == 200
    crossing = monitor.sample()
    assert crossing['processes'][0]['rx_bytes_per_second'] == 25
    assert crossing['today'] == {'date': '2026-11-16', 'processes': [], 'partial': True}
    assert monitor.sample()['today']['processes'][0]['rx_bytes'] == 50
    saved = json.loads((tmp_path / 'network-process-today.json').read_text())
    assert saved['date'] == '2026-11-16' and saved['processes'][0]['rx_bytes'] == 50


def test_capture_spanning_midnight_is_not_used_as_today_baseline(monkeypatch):
    sample = metrics.SocketSample
    snapshots = iter([({1: sample(20, 500, 'sshd', rx, 0)}, False) for rx in (500, 700, 950, 1000)])
    dates = iter(['2026-11-15'] * 3 + ['2026-11-16'] * 5)
    monkeypatch.setattr(metrics.sys, 'platform', 'linux')
    monkeypatch.setattr(metrics, '_capture_sockets', lambda: next(snapshots))
    monkeypatch.setattr(metrics, 'monotonic', iter([10, 20, 30, 40]).__next__)
    monkeypatch.setattr(metrics, '_today', lambda: next(dates))
    monkeypatch.setattr(metrics, '_process_identity', lambda pid, **kwargs: (500, 'sshd'))
    monitor = metrics.NetworkProcessMonitor()
    monitor.sample()
    assert monitor.sample()['today']['processes'] == []
    assert monitor.sample()['today']['processes'] == []
    assert monitor.sample()['today']['processes'][0]['rx_bytes'] == 50


def test_today_keeps_previous_boot_totals_separate(tmp_path, monkeypatch):
    boot_ids = iter(['boot-one', 'boot-two'])
    monkeypatch.setattr(metrics, '_boot_id', lambda: next(boot_ids))
    monkeypatch.setattr(metrics, '_process_identity', lambda pid, **kwargs: (500, 'sshd'))
    before_reboot = metrics._DailyProcessUsage(tmp_path)
    before_reboot.begin_day('2026-11-15')
    before_reboot.add({(20, 500): ('sshd', 100, 200)})
    previous, = before_reboot.report()['processes']

    after_reboot = metrics._DailyProcessUsage(tmp_path)
    after_reboot.begin_day('2026-11-15')
    after_reboot.add({(20, 500): ('sshd', 10, 20)})
    retained, current = after_reboot.report()['processes']
    assert retained == {**previous, 'active': False}
    assert current['instance_id'] != previous['instance_id'] and current['active']


def test_today_survives_storage_failure_and_limits_persisted_identities(tmp_path, monkeypatch):
    sample = metrics.SocketSample
    before = {pid: sample(pid, pid, 'worker', 0, 0) for pid in range(1, 5)}
    after = {pid: sample(pid, pid, 'worker', pid * 100, 0) for pid in before}
    snapshots = iter([(before, False), (after, False), (after, False)])
    monkeypatch.setattr(metrics, 'MAX_DAY_PROCESSES', 2)
    monkeypatch.setattr(metrics.sys, 'platform', 'linux')
    monkeypatch.setattr(metrics, '_capture_sockets', lambda: next(snapshots))
    monkeypatch.setattr(metrics, 'monotonic', iter([10, 20, 30]).__next__)
    monkeypatch.setattr(metrics, '_today', lambda: '2026-11-15')

    def denied(pid, **kwargs):
        raise PermissionError

    monkeypatch.setattr(metrics, '_process_identity', denied)
    state_dir = tmp_path / 'temporarily-missing'
    monitor = metrics.NetworkProcessMonitor(state_dir)
    monitor.sample()
    result = monitor.sample()
    assert result['status'] == 'ready'
    assert [(row['pid'], row['active']) for row in result['today']['processes']] == [(4, None), (3, None)]
    state_dir.mkdir()
    monitor.sample()
    persisted = json.loads((state_dir / 'network-process-today.json').read_text())
    assert [row['pid'] for row in persisted['processes']] == [4, 3]


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
    ], 'today': {'date': '2026-11-15', 'partial': True, 'processes': [
        {'pid': 100, 'name': 'sshd', 'instance_id': 'a' * 32,
         'rx_bytes': 800, 'tx_bytes': 1000, 'active': True},
    ]}}
    monkeypatch.setattr(metrics.NetworkProcessMonitor, 'sample', lambda _: report)
    monkeypatch.setattr('vla_agent.main.gpu_inventory', lambda: ([], {}))
    agent = Agent(agent_fixture['config'], client=Client())
    agent.credentials = {'node_id': 'fixture-node', 'token': 'fixture-token'}
    try:
        agent.heartbeat()
    finally:
        agent.close()
    assert seen == [report]
