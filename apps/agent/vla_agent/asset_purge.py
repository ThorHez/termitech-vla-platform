"""Inspect and delete a single authorized asset without following links."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys

from .profile_rebinding import profile_configuration_lock
from .runner_runtime import installed_runtime, training_start_lock
from .safety import AgentError


CAPABILITIES = ("ASSET_PURGE_INSPECT", "ASSET_PURGE")
MAX_ENTRIES = 250_000
MAX_DEPTH = 64
MAX_PROTECTED_PATHS = 10_000
MAX_PROTECTED_PATHS_BYTES = 1024 * 1024
ITEM = "asset"
QUARANTINE_PREFIX = ".vla-asset-purge-"
SYSTEM_PATHS = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/proc",
                "/run", "/sbin", "/sys", "/usr", "/var/lib", "/var/log", "/var/spool", "/var/run",
                "/var/cache", "/root", "/System", "/Library", "/Applications", "/opt/homebrew")
PROJECT_PATHS = (".git", ".venv", "src", "scripts", "packages", "third_party", "vendor", "node_modules")
LOCAL_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "f2fs", "overlay", "tmpfs"}
PROTECTED_COMPONENTS = {".git", ".venv", ".ssh", ".gnupg"}


def _fail(code, message):
    raise AgentError("ASSET_PURGE_" + code, message)


def _overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def _protected_path_values(payload):
    values = payload.get("protected_paths", [])
    if not isinstance(values, list) or len(values) > MAX_PROTECTED_PATHS:
        _fail("INVALID", "受保护的资产路径清单无效或超出上限")
    size = 0
    for value in values:
        if (not isinstance(value, str) or not value.startswith("/") or len(value) > 4096
                or ".." in value.split("/") or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)):
            _fail("INVALID", "受保护的资产路径必须是有效的本机绝对路径")
        try:
            size += len(value.encode("utf-8"))
        except UnicodeError:
            _fail("INVALID", "受保护的资产路径包含无效字符")
        if size > MAX_PROTECTED_PATHS_BYTES:
            _fail("INVALID", "受保护的资产路径清单超出大小上限")
    return values


def _requested_path(config, raw):
    if (not isinstance(raw, str) or not raw.startswith("/") or raw.startswith("//") or len(raw) > 4096
            or ".." in raw.split("/") or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in raw)):
        _fail("INVALID_PATH", "资产路径必须是无上级目录或控制字符的本机绝对路径")
    path = Path(raw)
    roots = [Path(value) for value in (*config["roots"], config["output_root"])]
    if (not any(path != root and path.is_relative_to(root) for root in roots)
            or any(root.is_relative_to(path) for root in roots)):
        _fail("PROTECTED_PATH", "只可清理受管根目录下的注册资产")
    if any(part.startswith(QUARANTINE_PREFIX) or part in PROTECTED_COMPONENTS for part in path.parts):
        _fail("PROTECTED_PATH", "不能清理已暂存或受保护的目录")

    config_path = Path(config["config_path"])
    protected = [item for value in SYSTEM_PATHS for item in (Path(value), Path(value).resolve())]
    protected += [Path(config["state_dir"]), config_path, Path(sys.prefix).resolve()]
    if config_path.parent.name == "config" and config_path.name == "agent.json":
        protected.append(config_path.parent.parent)
    runtime = installed_runtime()
    if runtime:
        protected.append(Path(runtime["runtime_path"]).parent.parent)
    if config.get("cache_root"):
        protected.append(Path(config["cache_root"]))
    guard = config.get("resource_guard")
    if guard is not None:
        if not isinstance(guard, dict) or not guard.get("database"):
            _fail("PROTECTED_PATH", "无法确认资源保护数据库位置")
        for key in ("database", "lock_directory"):
            value = guard.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
                _fail("PROTECTED_PATH", "无法确认资源保护目录或数据库位置")
            paths = (Path(value), Path(value).resolve())
            protected.extend(paths)
            if key == "database":
                protected.extend(item.with_name(item.name + suffix)
                                 for item in paths for suffix in ("-wal", "-shm", "-journal"))
    output_roots = [config["output_root"], *config.get("_output_root_history", [])]
    if any(Path(value).is_relative_to(path) for value in output_roots):
        _fail("PROTECTED_PATH", "不能清理训练输出根目录或其上级目录")
    if any(path == home or path in home.parents for home in (Path.home(), Path.home().resolve())):
        _fail("PROTECTED_PATH", "不能清理用户主目录或其上级目录")
    for profile in config["profiles"]:
        project = Path(profile["code_dir"]).resolve()
        python = Path(profile["python"])
        # Preserve the interpreter's whole environment, including symlink spelling.
        for executable in (python, python.resolve()):
            protected.append(executable.parent.parent if executable.parent.name == "bin" else executable)
        if project.is_relative_to(path):
            _fail("PROTECTED_PATH", "不能清理训练代码目录或其上级目录")
        protected.extend(project / item for item in PROJECT_PATHS)
        if (path.is_relative_to(project) and any(
                part.startswith(".") for part in path.relative_to(project).parts)):
            _fail("PROTECTED_PATH", "不能清理训练代码的隐藏子目录")
        if path.parent == project and path.suffix in {".py", ".toml", ".lock", ".sh", ".yaml", ".yml"}:
            _fail("PROTECTED_PATH", "不能清理训练项目代码或配置")
    if any(_overlaps(path, item) for item in protected):
        _fail("PROTECTED_PATH", "资产路径与系统、Agent、缓存或运行环境目录重叠")
    return path


def _protect_training_source(config, path):
    repository = next((folder for folder in path.parents if os.path.lexists(folder / ".git")), None)
    if repository is None:
        return
    for profile in config["profiles"]:
        project = Path(profile["code_dir"]).resolve()
        project_repo = next((folder for folder in (project, *project.parents)
                             if os.path.lexists(folder / ".git")), None)
        if project_repo == repository:
            break
    else:
        _fail("PROTECTED_PATH", "不能清理未登记的 Git 仓库中的文件")
    try:
        # Use a literal pathspec and disable optional Git monitors.
        tracked = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", str(repository),
            "--literal-pathspecs", "ls-files", "-z", "--", path.relative_to(repository).as_posix()],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentError("ASSET_PURGE_PROTECTED_PATH", "无法确认资产未包含受版本管理的训练文件") from exc
    if tracked:
        _fail("PROTECTED_PATH", "不能清理受版本管理的训练文件")


@contextmanager
def _directory_at(parent, name):
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _parent(path):
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            except FileNotFoundError:
                yield None
                return
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _entry(parent, name):
    try:
        return os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _inode(details):
    return [details.st_dev, details.st_ino, stat.S_IFMT(details.st_mode)]


def _avoid_protected_paths(path, values, *, alternate=None):
    """Check aliases by resolved spelling and inode ancestry without walking asset trees."""
    if not values:
        return
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        _fail("PATH_IN_USE", "无法可靠核对待清理路径及其引用")
    stat_cache = {}

    def inode_at(candidate):
        if candidate not in stat_cache:
            try:
                stat_cache[candidate] = tuple(_inode(candidate.stat()))
            except (FileNotFoundError, NotADirectoryError):
                stat_cache[candidate] = None
        return stat_cache[candidate]

    target = alternate or resolved
    target_inode = inode_at(target)
    target_parents = {inode for item in (target, *target.parents) if (inode := inode_at(item)) is not None}
    for value in dict.fromkeys(values):
        try:
            protected = Path("/" + value.lstrip("/")).resolve(strict=False)
            if (_overlaps(resolved, protected) or _overlaps(target, protected)
                    or inode_at(protected) in target_parents
                    or target_inode is not None and any(inode_at(parent) == target_inode for parent in protected.parents)):
                _fail("PATH_IN_USE", "此路径与其他已登记资产或受保护目录指向同一位置")
        except (OSError, RuntimeError):
            _fail("PATH_IN_USE", "无法可靠核对其他已登记资产或受保护目录的路径")


def _version(details):
    return [*_inode(details), details.st_uid, details.st_gid, details.st_mode, details.st_nlink,
            details.st_size, details.st_mtime_ns, details.st_ctime_ns]


def _mount(descriptor):
    if sys.platform.startswith("linux"):
        for line in Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
            if line.startswith("mnt_id:"):
                return line.partition(":")[2].strip()
        _fail("UNSAFE_PATH", "无法确认资产所在的文件系统挂载点")
    return os.fstat(descriptor).st_dev


def _local_mount(descriptor):
    mount = _mount(descriptor)
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            if line.split(" ", 1)[0] == mount:
                filesystem = line.partition(" - ")[2].split(" ", 1)[0]
                if filesystem in LOCAL_FILESYSTEMS:
                    return mount
                break
        _fail("UNSAFE_MOUNT", "只可清理已知本机文件系统上的资产，共享存储或未知挂载需要人工处理")
    return mount


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def _regular_file(parent, name, expected, mount):
    if sys.platform.startswith("linux"):
        # O_PATH can inspect bind mounts without opening file contents.
        descriptor = os.open(name, os.O_PATH | os.O_NOFOLLOW, dir_fd=parent)
        try:
            if _mount(descriptor) != mount or _version(os.fstat(descriptor)) != _version(expected):
                _fail("UNSAFE_PATH", "资产文件已变化或位于其他挂载点")
        finally:
            os.close(descriptor)


def _manifest(parent, name, top, mount):
    """Hash lstat identities, without reading asset contents or following links."""
    entries = 0
    digest = hashlib.sha256()

    def visit(directory, child, details, relative, depth):
        nonlocal entries
        entries += 1
        if depth > MAX_DEPTH or entries > MAX_ENTRIES:
            _fail("LIMIT", "资产包含过多文件或嵌套层级，无法安全核对")
        if depth and (child in PROTECTED_COMPONENTS or child.startswith(QUARANTINE_PREFIX)):
            _fail("UNSAFE_PATH", "资产包含受保护的代码、账户或暂存目录")
        if details.st_dev != top.st_dev or not (stat.S_ISREG(details.st_mode) or stat.S_ISDIR(details.st_mode)):
            _fail("UNSAFE_PATH", "资产包含符号链接、挂载点或非普通文件")
        if stat.S_ISREG(details.st_mode) and (
                details.st_nlink > 1 or details.st_mode & (stat.S_ISUID | stat.S_ISGID)):
            _fail("UNSAFE_PATH", "资产包含硬链接或具有特殊权限的文件，无法安全确认")
        version = _version(details)
        # rename updates the root ctime. The initial identity separately pins
        # that time, while the manifest can also be confirmed after rename.
        digest.update(json.dumps([relative, version if depth else version[:-1]],
                                 ensure_ascii=True, separators=(",", ":")).encode())
        if stat.S_ISDIR(details.st_mode):
            with _directory_at(directory, child) as descriptor:
                if _mount(descriptor) != mount or _version(os.fstat(descriptor)) != version:
                    _fail("UNSAFE_PATH", "资产目录已变化或经过其他挂载点")
                with os.scandir(descriptor) as listing:
                    names = []
                    for entry in listing:
                        if entries + len(names) >= MAX_ENTRIES:
                            _fail("LIMIT", "资产包含过多文件，无法安全核对")
                        names.append(entry.name)
                for item in sorted(names):
                    item_details = _entry(descriptor, item)
                    if item_details is None:
                        _fail("IDENTITY_CHANGED", "核对期间资产文件已变化")
                    visit(descriptor, item, item_details, relative + "/" + item, depth + 1)
                if _version(os.fstat(descriptor)) != version:
                    _fail("IDENTITY_CHANGED", "核对期间资产目录已变化")
        else:
            _regular_file(directory, child, details, mount)
        if (after := _entry(directory, child)) is None or _version(after) != version:
            _fail("IDENTITY_CHANGED", "核对期间资产文件已变化")

    visit(parent, name, top, "", 0)
    return digest.hexdigest()


def _active_attempts(state, path):
    for attempt in state.active_attempts():
        payload = attempt["payload"]
        resume = payload.get("resume_source") or {}
        sources = [attempt.get("output_dir"), resume.get("path"), resume.get("output_dir")]
        inputs = payload.get("inputs", {})
        if not isinstance(inputs, dict) or any(
                not isinstance(asset, dict) or not isinstance(asset.get("path"), str) for asset in inputs.values()):
            _fail("IN_USE", "无法核对本机训练任务的资产引用")
        sources.extend(asset["path"] for asset in inputs.values())
        if any(source and _overlaps(path, Path(source).resolve()) for source in sources):
            _fail("IN_USE", "此资产仍被本机训练任务引用")


def _quarantine_matches(details, record):
    if (details is None or _inode(details) != record["quarantine_inode"]
            or details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) != 0o700):
        _fail("INDETERMINATE", "无法确认资产暂存目录身份，已停止清理")


def _inspect(state, result, path, parent, record):
    if record is not None:
        _fail("ALREADY_ATTEMPTED", "此资产已有物理清理记录，请查看原记录")
    _active_attempts(state, path)
    details = _entry(parent, path.name) if parent is not None else None
    if details is None:
        return {**result, "status": "MISSING", "identity": None}
    manifest = _manifest(parent, path.name, details, _local_mount(parent))
    identity = _hash([_inode(os.fstat(parent)), _version(details), manifest])
    return {**result, "status": "READY", "identity": identity}


def _start(state, result, path, parent, identity):
    details = _entry(parent, path.name) if parent is not None else None
    record = {"path": result["path"], "identity": identity}
    if details is None:
        record["phase"] = "MISSING"
    else:
        manifest = _manifest(parent, path.name, details, _local_mount(parent))
        parent_inode = _inode(os.fstat(parent))
        if identity != _hash([parent_inode, _version(details), manifest]):
            _fail("IDENTITY_CHANGED", "资产路径或文件内容已变更，请重新核对后确认")
        name = QUARANTINE_PREFIX + secrets.token_hex(16)
        os.mkdir(name, mode=0o700, dir_fd=parent)
        with _directory_at(parent, name) as quarantine:
            quarantine_inode = _inode(os.fstat(quarantine))
            os.fsync(quarantine)
        os.fsync(parent)
        record.update(phase="PREPARED", parent_inode=parent_inode, name=name, quarantine_inode=quarantine_inode,
                      inode=_inode(details), manifest=manifest)
    state.save_asset_purge(result["asset_id"], record)
    return record


def _remove(parent, name, expected, mount, *, device, budget, depth=0):
    details = _entry(parent, name)
    if details is None or _version(details) != _version(expected) or details.st_dev != device:
        _fail("INDETERMINATE", "清理期间暂存文件已变化，已停止后续清理")
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_DEPTH:
        _fail("INDETERMINATE", "暂存文件过多或嵌套层级过深，已停止清理")
    if stat.S_ISDIR(details.st_mode):
        with _directory_at(parent, name) as descriptor:
            if _version(os.fstat(descriptor)) != _version(details) or _mount(descriptor) != mount:
                _fail("INDETERMINATE", "清理期间暂存目录已变化或包含挂载点")
            children = []
            with os.scandir(descriptor) as listing:
                for entry in listing:
                    if len(children) >= budget[0]:
                        _fail("INDETERMINATE", "暂存文件过多，已停止清理")
                    children.append((entry.name, entry.stat(follow_symlinks=False)))
            for child, item in children:
                _remove(descriptor, child, item, mount, device=device, budget=budget, depth=depth + 1)
            if (current := _entry(parent, name)) is None or _inode(current) != _inode(details):
                _fail("INDETERMINATE", "清理期间暂存目录已变化，已停止后续清理")
            os.rmdir(name, dir_fd=parent)
    elif stat.S_ISREG(details.st_mode):
        if details.st_nlink > 1 or details.st_mode & (stat.S_ISUID | stat.S_ISGID):
            _fail("INDETERMINATE", "暂存文件已成为受保护的硬链接或特殊权限文件")
        _regular_file(parent, name, details, mount)
        os.unlink(name, dir_fd=parent)
    else:
        _fail("INDETERMINATE", "暂存区包含符号链接或非普通文件，已停止清理")


def _purge(state, result, path, parent, identity, record, protected_paths):
    if record is None:
        _active_attempts(state, path)
        record = _start(state, result, path, parent, identity)
    elif (record["path"], record["identity"]) != (result["path"], identity):
        _fail("IDENTITY_CHANGED", "清理请求与已记录的资产身份不符")
    if record["phase"] in {"MISSING", "DELETED"}:
        return {**result, "status": record["phase"]}
    if record["phase"] not in {"PREPARED", "DELETING"}:
        _fail("INDETERMINATE", "无法确认暂存清理状态，已停止清理")
    if parent is None or _inode(os.fstat(parent)) != record["parent_inode"]:
        _fail("INDETERMINATE", "无法确认资产原父目录，已停止清理")
    _active_attempts(state, path)
    _quarantine_matches(_entry(parent, record["name"]), record)
    with _directory_at(parent, record["name"]) as quarantine:
        _quarantine_matches(os.fstat(quarantine), record)
        mount = _local_mount(parent)
        if _mount(quarantine) != mount:
            _fail("INDETERMINATE", "资产暂存区不在原文件系统，已停止清理")
        if record["phase"] == "DELETING":
            if _entry(quarantine, ITEM) is not None:
                _fail("INDETERMINATE", "上次清理中断，请检查原暂存目录，当前路径上的文件未处理")
        else:
            staged = _entry(quarantine, ITEM)
            if staged is None:
                source = _entry(parent, path.name)
                if source is None:
                    _fail("INDETERMINATE", "未找到原资产或暂存副本，已停止清理")
                manifest = _manifest(parent, path.name, source, mount)
                if identity != _hash([record["parent_inode"], _version(source), manifest]):
                    _fail("IDENTITY_CHANGED", "资产已变化，已停止清理")
                _avoid_protected_paths(path, protected_paths)
                if (current := _entry(parent, path.name)) is None or _version(current) != _version(source):
                    _fail("IDENTITY_CHANGED", "资产已变化，已停止清理")
                with _parent(path) as current_parent:
                    if current_parent is None or _inode(os.fstat(current_parent)) != record["parent_inode"]:
                        _fail("IDENTITY_CHANGED", "资产父目录已变化，已停止清理")
                os.rename(path.name, ITEM, src_dir_fd=parent, dst_dir_fd=quarantine)
                os.fsync(parent)
                os.fsync(quarantine)
                staged = _entry(quarantine, ITEM)
            if (staged is None or _inode(staged) != record["inode"]
                    or _manifest(quarantine, ITEM, staged, mount) != record["manifest"]):
                _fail("INDETERMINATE", "暂存副本与原资产不符，已停止清理")
            _avoid_protected_paths(path, protected_paths, alternate=path.parent / record["name"] / ITEM)
            record["phase"] = "DELETING"
            state.save_asset_purge(result["asset_id"], record)
            _remove(quarantine, ITEM, staged, mount, device=staged.st_dev, budget=[MAX_ENTRIES])
        os.fsync(quarantine)
        record["phase"] = "DELETED"
        state.save_asset_purge(result["asset_id"], record)
    try:
        if (remaining := _entry(parent, record["name"])) is not None and _inode(remaining) == record["quarantine_inode"]:
            os.rmdir(record["name"], dir_fd=parent)
            os.fsync(parent)
    except OSError:
        pass  # The durable success applies only to the inspected asset.
    return {**result, "status": "DELETED"}


def execute_asset_purge_command(config, state, command_type, payload):
    inspect = command_type == CAPABILITIES[0]
    asset_id = None
    record = None
    try:
        keys = {"asset_id", "path"} if inspect else {"asset_id", "path", "identity"}
        if (command_type not in CAPABILITIES or not isinstance(payload, dict)
                or not keys <= set(payload) <= keys | {"protected_paths"}):
            _fail("INVALID", "物理清理请求格式无效")
        protected_paths = _protected_path_values(payload)
        asset_id = payload["asset_id"]
        if not isinstance(asset_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", asset_id):
            _fail("INVALID", "资产登记标识无效")
        identity = payload.get("identity")
        if identity is not None and (not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity)):
            _fail("INVALID", "资产文件身份无效，请重新核对")
        # Coordinate with Agent maintenance and START without stopping runners.
        with training_start_lock(config["state_dir"]), profile_configuration_lock(config, busy_code="ASSET_PURGE_BUSY"):
            record = state.asset_purge(asset_id)
            result = {"asset_id": asset_id, "path": payload["path"]}
            if (not inspect and record is not None
                    and (record["path"], record["identity"]) == (payload["path"], identity)
                    and record["phase"] in {"DELETED", "MISSING"}):
                return {**result, "status": record["phase"]}
            path = _requested_path(config, payload["path"])
            _protect_training_source(config, path)
            with _parent(path) as parent:
                _avoid_protected_paths(path, protected_paths)
                if inspect:
                    return _inspect(state, result, path, parent, record)
                return _purge(state, result, path, parent, identity, record, protected_paths)
    except (AgentError, OSError, RuntimeError) as exc:
        error = exc.as_dict() if isinstance(exc, AgentError) else {
            "code": "ASSET_PURGE_UNSAFE_PATH" if isinstance(exc, OSError) else "ASSET_PURGE_FAILED",
            "message": "无法安全核对或清理资产，请检查路径状态和访问权限",
        }
        if isinstance(asset_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", asset_id):
            record = state.asset_purge(asset_id) or record
        return {"error": error, **({"staged": True} if record is not None else {})}
