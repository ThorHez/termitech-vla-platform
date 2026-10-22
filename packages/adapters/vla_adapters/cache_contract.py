"""Check the registered training environment's real local cache contract.

Builder construction exercises Hugging Face's own locks without preparing Arrow
files, reading training examples, or constructing a model/dataloader.
"""

import hashlib
import os
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType

if __package__:
    from .entrypoints import read_entrypoint
else:
    from entrypoints import read_entrypoint


CACHE_ENV_KEYS = (
    "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "JAX_COMPILATION_CACHE_DIR",
    "OPENPI_DATA_HOME", "TMPDIR", "TEMP", "TMP",
)
MAX_CACHE_ENTRIES = 20000


def _managed_path(value, spec, *, label, strict=False):
    if not isinstance(value, (str, Path)) or not Path(value).expanduser().is_absolute():
        raise ValueError(f"TRAINING_CACHE_UNAVAILABLE: {label} 必须是本机绝对路径：{value}")
    try:
        path = Path(value).expanduser().resolve(strict=strict)
        roots = spec.get("allowed_roots")
        if not isinstance(roots, list) or not roots or not any(
            path.is_relative_to(Path(root).resolve(strict=True)) for root in roots
        ):
            raise ValueError(f"TRAINING_CACHE_UNAVAILABLE: {label} 超出 Agent 受管目录：{path}")
        return path
    except OSError as exc:
        raise ValueError(f"TRAINING_CACHE_UNAVAILABLE: 无法访问 {label}：{value}: {exc}") from exc


def _runtime_paths(spec):
    code = _managed_path(spec["code_dir"], spec, label="训练项目", strict=True)
    candidate = code / "src/openpi/shared/runtime_paths.py"
    if not candidate.exists() and not candidate.is_symlink():
        return None
    path = _managed_path(candidate, spec, label="缓存初始化源码", strict=True)
    if not path.is_relative_to(code):
        raise ValueError("TRAINING_CACHE_UNAVAILABLE: 缓存初始化源码必须位于登记的训练项目内")
    return path, read_entrypoint(path)


def configure_cache(spec):
    """Apply the project's cache helper before importing OpenPI/datasets.

    Only the known helper module is executed. The selected training entrypoint is
    left untouched until all final checks have passed.
    """
    try:
        source = _runtime_paths(spec)
        if source is None:
            return
        path, contents = source
        name = "vla_registered_openpi_runtime_paths"
        module = ModuleType(name)
        module.__file__ = str(path)
        previous = sys.modules.get(name)
        try:
            sys.modules[name] = module
            exec(compile(contents, str(path), "exec"), module.__dict__)
            configure = getattr(module, "configure_training_cache", None)
            if not callable(configure):
                raise ValueError("缓存初始化源码缺少 configure_training_cache()")
            configure()
        finally:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    except Exception as exc:
        if str(exc).startswith("TRAINING_CACHE_"):
            raise
        code = "TRAINING_CACHE_UNWRITABLE" if isinstance(exc, PermissionError) else "TRAINING_CACHE_UNAVAILABLE"
        raise ValueError(f"{code}: 无法初始化训练缓存：{exc}") from exc


def _existing_locks(root):
    """Check write access without truncating or acquiring existing lock files."""
    pending = [root]
    count = 0
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                count += 1
                if count > MAX_CACHE_ENTRIES:
                    raise ValueError(f"TRAINING_CACHE_UNAVAILABLE: 缓存文件数量超过预检上限 {MAX_CACHE_ENTRIES}：{root}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.name.endswith(".lock"):
                    # O_NONBLOCK avoids blocking on special files, O_NOFOLLOW
                    # rejects links, and omitting O_TRUNC preserves old locks.
                    descriptor = os.open(entry.path, os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
                    try:
                        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                            raise ValueError(f"TRAINING_CACHE_UNAVAILABLE: 缓存锁必须是普通文件：{entry.path}")
                    finally:
                        os.close(descriptor)


def _new_lock(root, lock_type):
    path = None
    try:
        descriptor, filename = tempfile.mkstemp(prefix=".vla-cache-probe-", suffix=".lock", dir=root)
        path = Path(filename)
        os.close(descriptor)
        with lock_type(str(path), timeout=1):
            pass
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def inspect_training_cache(datasets, spec):
    """Return stable cache paths after checking actual per-dataset builder locks."""
    cache = None
    try:
        import datasets as hf_datasets
        from filelock import FileLock

        cache = _managed_path(hf_datasets.config.HF_DATASETS_CACHE, spec, label="HF_DATASETS_CACHE")
        paths = {key: str(_managed_path(os.environ[key], spec, label=key))
                 for key in CACHE_ENV_KEYS if os.environ.get(key)}
        if paths.get("HF_DATASETS_CACHE", str(cache)) != str(cache):
            raise ValueError("TRAINING_CACHE_UNAVAILABLE: datasets 实际缓存路径与 HF_DATASETS_CACHE 环境变量不一致；请在导入训练库之前初始化缓存")
        paths["HF_DATASETS_CACHE"] = str(cache)
        cache.mkdir(parents=True, exist_ok=True)
        if not cache.is_dir():
            raise ValueError(f"TRAINING_CACHE_UNAVAILABLE: 数据集缓存必须是目录：{cache}")
        _existing_locks(cache)
        _new_lock(cache, FileLock)
        builders = []
        for dataset in datasets:
            path = _managed_path(dataset["path"], spec, label="数据集", strict=True)
            data = _managed_path(path / "data", spec, label="parquet 数据目录", strict=True)
            if not data.is_relative_to(path):
                raise ValueError(f"TRAINING_CACHE_UNAVAILABLE: parquet 数据目录不能逃逸数据集目录：{data}")
            # LeRobotDataset.load_hf_dataset uses this same builder request.
            # Do not call download_and_prepare, as_dataset, or load_dataset.
            builder = hf_datasets.load_dataset_builder("parquet", data_dir=str(data))
            directory = _managed_path(builder.cache_dir, spec, label="parquet builder 缓存目录")
            if not directory.is_relative_to(cache):
                raise ValueError(f"TRAINING_CACHE_UNAVAILABLE: parquet builder 使用了另一缓存目录：{directory}")
            builders.append({"path": str(path), "cache_dir": str(directory)})
        result = {"datasets_cache": str(cache), "paths": paths, "builders": builders}
        source = _runtime_paths(spec)
        if source is not None:
            result["runtime_paths_sha256"] = hashlib.sha256(source[1]).hexdigest()
        return result
    except Exception as exc:
        if str(exc).startswith("TRAINING_CACHE_"):
            raise
        if isinstance(exc, (PermissionError, OSError)):
            code = "TRAINING_CACHE_UNWRITABLE"
        else:
            code = "TRAINING_CACHE_UNAVAILABLE"
        raise ValueError(f"{code}: 无法检查训练数据集缓存 {cache or 'HF_DATASETS_CACHE'}：{exc}") from exc
