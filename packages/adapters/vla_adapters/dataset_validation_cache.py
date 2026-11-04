"""Private, content/configuration-bound receipts for successful data checks.

Receipts are never validation authority supplied by the client. Registration can
refresh missing evidence with get_or_validate; lookup_cached only reads existing
evidence and never falls back to a validator. Evidence is returned unchanged so
cache hits cannot change execution-plan digests.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import time


SCHEMA_VERSION = 1
KINDS = {"timestamps", "samples", "source"}
MAX_ENTRIES = 30000
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_HASH_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
SCAN_SECONDS = 15
LOCK_SECONDS = 300
LIBRARIES = ("numpy", "pyarrow", "datasets", "torch", "torchvision", "av", "torchcodec", "pillow",
             "lerobot", "openpi", "openpi-client", "termitech-vla-platform")


class _UnsafeCache(Exception):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _version(details):
    return [details.st_mode, details.st_dev, details.st_ino, details.st_size,
            details.st_mtime_ns, details.st_ctime_ns]


def _absolute(value):
    if not isinstance(value, (str, Path)):
        raise _UnsafeCache()
    path = Path(value)
    try:
        canonical = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise _UnsafeCache() from exc
    if not path.is_absolute() or ".." in path.parts or path != canonical:
        raise _UnsafeCache()
    return path


def _managed(value, spec):
    path = _absolute(value)
    roots = spec.get("allowed_roots")
    if (not isinstance(roots, (tuple, list)) or not roots
            or not any(path.is_relative_to(_absolute(root)) for root in roots)):
        raise _UnsafeCache()
    return path


def _open_path(path, *, directory=False, create=False):
    """Walk every component through dirfds, refusing links including parents."""
    path = _absolute(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parts[1:]):
            is_dir = directory or index < len(path.parts) - 2
            if create and is_dir:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
            if is_dir:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _hash_file(descriptor, budget):
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
        raise _UnsafeCache()
    budget[0] += before.st_size
    if budget[0] > MAX_HASH_BYTES:
        raise _UnsafeCache()
    value = hashlib.sha256()
    read_bytes = 0
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk:
            break
        read_bytes += len(chunk)
        if read_bytes > MAX_FILE_BYTES:
            raise _UnsafeCache()
        value.update(chunk)
    if _version(before) != _version(os.fstat(descriptor)):
        raise _UnsafeCache()
    return value.hexdigest()


def _tree(path, budget, *, python_only=False):
    """Read stable regular-file identities without following directory links."""
    result, deadline = [], time.monotonic() + SCAN_SECONDS

    def walk(descriptor, relative, depth):
        before = os.fstat(descriptor)
        if depth > 24:
            raise _UnsafeCache()
        names = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                budget[1] += 1
                if budget[1] > MAX_ENTRIES or time.monotonic() > deadline:
                    raise _UnsafeCache()
                names.append(entry.name)
        names.sort()
        for name in names:
            if time.monotonic() > deadline or len(result) > MAX_ENTRIES:
                raise _UnsafeCache()
            if python_only and name in {"__pycache__", ".git"}:
                continue
            details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(details.st_mode):
                raise _UnsafeCache()
            location = relative + "/" + name if relative else name
            if stat.S_ISDIR(details.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    if _version(details) != _version(os.fstat(child)):
                        raise _UnsafeCache()
                    if not python_only:
                        result.append([location, _version(details)])
                    walk(child, location, depth + 1)
                finally:
                    os.close(child)
            elif stat.S_ISREG(details.st_mode):
                if python_only and not name.endswith(".py"):
                    continue
                row = [location] if python_only else [location, _version(details)]
                if python_only or path.name == "meta":
                    child = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=descriptor)
                    try:
                        if _version(details) != _version(os.fstat(child)):
                            raise _UnsafeCache()
                        row.append(_hash_file(child, budget))
                    finally:
                        os.close(child)
                result.append(row)
            else:
                raise _UnsafeCache()
        after = os.fstat(descriptor)
        if _version(before) != _version(after):
            raise _UnsafeCache()

    descriptor = _open_path(path, directory=True)
    try:
        if not python_only:
            result.append([".", _version(os.fstat(descriptor))])
        walk(descriptor, "", 0)
    finally:
        os.close(descriptor)
    return result


def _profile(data, spec):
    explicit = spec.get("registered_profile")
    profiles = [asset.get("dataset_profile") for asset in spec.get("inputs", {}).values()
                if isinstance(asset, dict) and asset.get("path") == data["path"] and asset.get("dataset_profile") is not None]
    if explicit is not None:
        profiles.append(explicit)
    unique = {_json(profile) for profile in profiles}
    if len(unique) > 1:
        raise _UnsafeCache()
    return profiles[0] if profiles else None


def _environment():
    executable = Path(sys.executable).resolve(strict=True)
    versions = {}
    for name in LIBRARIES:
        try:
            distribution = importlib.metadata.distribution(name)
            versions[name] = distribution.version
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"executable": str(executable), "executable_version": _version(executable.stat()),
            "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "python": sys.version, "platform": sys.platform, "libraries": versions,
            "video_backend": os.environ.get("LEROBOT_VIDEO_BACKEND")}


def cache_key(kind, data, spec, *, context=None):
    """Return a fresh receipt key, or None when safe reuse cannot be established.

    Timestamp/source keys deliberately exclude run, recipe, asset IDs and model
    parameters. Samples pass their full processing binding/model/stage context.
    The cache location is trusted Agent configuration, not client metadata.
    """
    if kind not in KINDS or not spec.get("validation_cache_dir"):
        return None
    try:
        if callable(context):
            context = context()
        root = _managed(data["path"], spec)
        cache = _managed(spec["validation_cache_dir"], spec)
        if cache.is_relative_to(root):
            raise _UnsafeCache()
        code = _managed(spec["code_dir"], spec)
        budget = [0, 0]
        descriptor = _open_path(root, directory=True)
        try:
            root_version = _version(os.fstat(descriptor))
        finally:
            os.close(descriptor)
        trees = {}
        for name in ("meta", "data", "videos", "tasks"):
            target = root / name
            if not target.exists() and not target.is_symlink():
                if name in {"meta", "data"}:
                    raise _UnsafeCache()
                trees[name] = None
            elif target.is_dir() and not target.is_symlink():
                trees[name] = _tree(target, budget)
            else:
                if name in {"meta", "data"}:
                    raise _UnsafeCache()
                descriptor = _open_path(target)
                try:
                    trees[name] = [_version(os.fstat(descriptor)), _hash_file(descriptor, budget)]
                finally:
                    os.close(descriptor)
        sources = {relative: _tree(code / relative, budget, python_only=True) for relative in (
            "src/openpi", "third_party/lerobot/lerobot", "packages/openpi-client")}
        sources["adapters"] = _tree(Path(__file__).resolve().parent, budget, python_only=True)
        norm = None
        if kind == "samples":
            norm_path = (context or {}).get("norm_path") or data.get("norm_path")
            if not norm_path:
                raise _UnsafeCache()
            path = _managed(norm_path, spec)
            descriptor = _open_path(path)
            try:
                norm = [str(path), _version(os.fstat(descriptor)), _hash_file(descriptor, budget)]
            finally:
                os.close(descriptor)
        return _digest({"schema_version": SCHEMA_VERSION, "kind": kind, "path": str(root),
                        "root_version": root_version, "trees": trees, "norm": norm,
                        "data": {key: data.get(key) for key in (
                            "metadata_sha256", "data_manifest_digest", "frames", "episodes", "effective_fps")},
                        "registered_profile": _profile(data, spec), "context": context,
                        "code": {"path": str(code), "sources": sources}, "environment": _environment()})
    except (OSError, ValueError, TypeError, KeyError, RecursionError, _UnsafeCache):
        return None


@contextmanager
def _cache_directory(spec):
    descriptor = _open_path(_managed(spec["validation_cache_dir"], spec), directory=True, create=True)
    try:
        details = os.fstat(descriptor)
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) != 0o700:
            raise _UnsafeCache()
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _lock(directory, name):
    flags = os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
    except FileExistsError:
        descriptor = os.open(name, flags, dir_fd=directory)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) != 0o600 or details.st_nlink != 1:
            raise _UnsafeCache()
        deadline = time.monotonic() + LOCK_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise _UnsafeCache()
                time.sleep(0.05)
        yield
    finally:
        os.close(descriptor)


def _read(directory, filename, kind, key):
    descriptor = os.open(filename, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid()
                or stat.S_IMODE(details.st_mode) != 0o600 or details.st_nlink != 1 or details.st_size > MAX_RECEIPT_BYTES):
            raise _UnsafeCache()
        contents = stream.read(MAX_RECEIPT_BYTES + 1)
        if _version(details) != _version(os.fstat(stream.fileno())):
            raise _UnsafeCache()
    value = json.loads(contents)
    evidence = value["evidence"]
    if (value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != kind or value.get("key") != key
            or not _successful(evidence) or value.get("evidence_sha256") != _digest(evidence)):
        raise _UnsafeCache()
    return evidence


def _successful(evidence):
    return (isinstance(evidence, dict) and type(evidence.get("schema_version")) is int
            and evidence["schema_version"] == 1 and evidence.get("status") == "verified")


def _write(directory, filename, kind, key, evidence):
    contents = _json({"schema_version": SCHEMA_VERSION, "kind": kind, "key": key,
                      "evidence": evidence, "evidence_sha256": _digest(evidence)})
    if len(contents) > MAX_RECEIPT_BYTES:
        raise _UnsafeCache()
    temporary = ".receipt-" + secrets.token_hex(16)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filename, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


def lookup_cached(kind, data, spec, *, context=None):
    """Read stable successful evidence without creating or refreshing anything.

    The receipt must already exist in the Agent's private cache. Both source
    fingerprints are freshly computed; missing, unsafe, corrupt or changed
    evidence is a miss. In particular, this path creates no directory or lock
    and never invokes a dataset validator, loader or decoder.
    """
    key = cache_key(kind, data, spec, context=context)
    if key is None:
        return None
    try:
        directory = _open_path(_managed(spec["validation_cache_dir"], spec), directory=True)
        try:
            details = os.fstat(directory)
            if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) != 0o700:
                raise _UnsafeCache()
            evidence = _read(directory, kind + "-" + key + ".json", kind, key)
            return evidence if cache_key(kind, data, spec, context=context) == key else None
        finally:
            os.close(directory)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, _UnsafeCache):
        return None


def get_or_validate(kind, data, spec, validate, *, context=None):
    """Reuse stable successful evidence; otherwise execute the full validator."""
    key = cache_key(kind, data, spec, context=context)
    if key is None:
        return validate()
    executed = False
    validation_failed = False
    result = None

    def run():
        nonlocal executed, result, validation_failed
        executed = True
        try:
            result = validate()
        except BaseException:
            validation_failed = True
            raise
        if cache_key(kind, data, spec, context=context) != key:
            raise ValueError("DATASET_VALIDATION_SOURCE_CHANGED: 数据、处理配置或校验代码在检查期间发生变化，请重新校验")
        return result

    try:
        with _cache_directory(spec) as directory, _lock(directory, kind + "-" + key + ".lock"):
            if cache_key(kind, data, spec, context=context) != key:
                return run()
            try:
                evidence = _read(directory, kind + "-" + key + ".json", kind, key)
            except (OSError, ValueError, TypeError, KeyError, RecursionError, _UnsafeCache):
                evidence = None
            if evidence is not None and cache_key(kind, data, spec, context=context) == key:
                return evidence
            result = run()
            if _successful(result):
                try:
                    _write(directory, kind + "-" + key + ".json", kind, key, result)
                except (OSError, ValueError, TypeError, _UnsafeCache):
                    pass  # A completed real check remains valid if persistence fails.
            return result
    except (OSError, _UnsafeCache):
        if validation_failed:
            raise
        return result if executed else run()
