from collections import namedtuple
from threading import Event
from types import SimpleNamespace

from vla_agent import disk_capacity
from vla_agent.main import Agent
from vla_agent.metrics import system_metrics


def test_linux_mounts_report_root_and_native_data_skip_remote_and_pseudo(monkeypatch):
    lines = [
        '1 0 0:1 / / rw - overlay overlay rw',
        '2 1 8:2 / /data1 ro - xfs /dev/sda2 ro',
        '3 1 8:2 /part /data1/bind rw - xfs /dev/sda2 rw',
        '4 1 8:3 / /data2\\040vol rw - ext4 /dev/sda3 rw',
        '6 1 0:12 / /tmp rw - tmpfs tmpfs rw',
        '7 1 0:13 / /nfs rw - nfs host:/share rw',
        '8 1 0:14 / /var/lib/docker/overlay rw - overlay overlay rw',
    ]
    mounts = list(disk_capacity._linux_mounts(lines))
    monkeypatch.setattr(disk_capacity, '_mounts', lambda: mounts)
    monkeypatch.setattr(disk_capacity, '_read_mount', lambda mount: {
        'mountpoint': mount.path, 'filesystem': mount.filesystem, 'total_bytes': 100,
        'used_bytes': 80, 'free_bytes': 10, 'error': None,
    })
    result = disk_capacity._local_disks()
    assert result['error'] is None
    assert [(row['mountpoint'], row['filesystem']) for row in result['mounts']] == [
        ('/', 'overlay'), ('/data1', 'xfs'), ('/data2 vol', 'ext4'),
    ]


def test_unreadable_bind_falls_back_and_failed_mount_has_no_capacity(monkeypatch):
    mounts = [disk_capacity.Mount('/data', 'ext4', '8:1'), disk_capacity.Mount('/data/bind', 'ext4', '8:1'),
              disk_capacity.Mount('/unreadable', 'xfs', '8:2')]
    monkeypatch.setattr(disk_capacity, '_mounts', lambda: mounts)
    monkeypatch.setattr(disk_capacity, 'sys', SimpleNamespace(platform='darwin'))

    def usage(path):
        if path != '/data/bind':
            raise PermissionError('unavailable')
        return SimpleNamespace(total=100, used=81, free=9)

    monkeypatch.setattr(disk_capacity.psutil, 'disk_usage', usage)
    result = disk_capacity._local_disks()
    assert result == {'error': None, 'mounts': [
        {'mountpoint': '/data/bind', 'filesystem': 'ext4', 'total_bytes': 100, 'used_bytes': 81,
         'free_bytes': 9, 'error': None},
        {'mountpoint': '/unreadable', 'filesystem': 'xfs', 'total_bytes': None, 'used_bytes': None,
         'free_bytes': None, 'error': 'PermissionError'},
    ]}


def test_sampling_caches_every_minute_skips_legacy_wait_and_marks_timeout(monkeypatch):
    entered, release = Event(), Event()
    now = [500.0]
    monkeypatch.setattr(disk_capacity, 'monotonic', lambda: now[0])
    clock = iter(('first', 'second', 'third', 'fourth'))
    monkeypatch.setattr(disk_capacity, 'utc_now', lambda: next(clock))
    paths = []

    def slow_read(path):
        paths.append(path)
        if path == '/output':
            entered.set()
            assert release.wait(5)
        return {'total_bytes': 123, 'used_bytes': 94, 'free_bytes': 20}

    monkeypatch.setattr(disk_capacity, '_local_disks', lambda: {'mounts': [], 'error': None})
    monkeypatch.setattr(disk_capacity, '_disk_bytes', slow_read)
    monitor = disk_capacity.DiskCapacityMonitor()
    monitor.heartbeat_metrics('/output')
    assert entered.wait(1)
    try:
        monitor.local.worker.join(1)
        assert monitor.heartbeat_metrics('/output') == {
            'disk_capacity': {'sampled_at': 'first', 'mounts': [], 'error': None},
        }
        now[0] += disk_capacity.SAMPLE_TIMEOUT + 1
        assert monitor.heartbeat_metrics('/output')['collection_errors'] == {'disk': 'CollectionTimeout'}
        now[0] += disk_capacity.SAMPLE_INTERVAL
        metrics = monitor.heartbeat_metrics('/output')
        assert metrics['collection_errors'] == {'disk': 'CollectionTimeout'}
        assert paths == ['/output']
        monitor.local.worker.join(1)
    finally:
        release.set()
        monitor.output.worker.join(1)
    monitor.heartbeat_metrics('/new-output')
    monitor.output.worker.join(1)
    metrics = monitor.heartbeat_metrics('/new-output')
    assert paths == ['/output', '/new-output']
    assert metrics['disk_total_bytes'] == 123 and metrics['disk_free_bytes'] == 20
    assert 'collection_errors' not in metrics
    assert metrics['disk_capacity']['sampled_at'] == 'third'


def test_heartbeat_and_direct_call_use_expected_storage_metrics(agent_fixture, monkeypatch):
    captured = []
    legacy = namedtuple('Usage', 'total free')(160, 42)
    monkeypatch.setattr(disk_capacity.psutil, 'disk_usage', lambda _: legacy)
    assert system_metrics(str(agent_fixture['root']))['disk_free_bytes'] == 42

    class Client:
        def request(self, method, path, token, body=None):
            captured.append(body)
            return {}

    def blocked_usage(_):
        raise AssertionError('synchronous disk probe inside heartbeat')

    monkeypatch.setattr(disk_capacity.psutil, 'disk_usage', blocked_usage)
    sample = {'sampled_at': '2026-11-14T08:00:00+00:00', 'mounts': [], 'error': 'NoLocalMounts'}
    monkeypatch.setattr(disk_capacity.DiskCapacityMonitor, 'heartbeat_metrics', lambda self, path: {
        'disk_capacity': sample, 'disk_total_bytes': 160, 'disk_free_bytes': 42,
    })
    monkeypatch.setattr('vla_agent.main.gpu_inventory', lambda: ([], {}))
    agent = Agent(agent_fixture['config'], client=Client())
    agent.credentials = {'node_id': 'fixture-node', 'token': 'fixture-token'}
    try:
        agent.heartbeat()
        metrics = captured[-1]['metrics']
        assert metrics['disk_capacity'] == sample and metrics['disk_free_bytes'] == 42
    finally:
        agent.close()
