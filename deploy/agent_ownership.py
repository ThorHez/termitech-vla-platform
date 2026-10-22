"""Reversible ownership changes for an idle Agent's own directories."""

from __future__ import annotations

import os
from pathlib import Path
import stat


class OwnershipMigration:
    """Never follow links or take ownership of an unrelated user's files.

    The installer supplies a previous account only after checking the existing
    unit and Agent configuration. Output directories use recursive=False so
    historical training artifacts retain their ownership.
    """

    def __init__(self, account, previous_account):
        self.account = account
        self.previous_account = previous_account
        self.allowed_uids = {0, account.pw_uid}
        if previous_account is not None:
            self.allowed_uids.add(previous_account.pw_uid)
        self._original = {}
        self._recursive_roots = {}
        self._missing_roots = {}

    def _validate_path(self, path: Path) -> None:
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Agent 账户迁移需要规范绝对路径：{path}")
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError(f"Agent 账户迁移不能跟随符号链接：{path}")

    def _entries(self, path: Path, recursive: bool):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return []
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Agent 账户迁移不能跟随符号链接：{path}")
        entries = [(path, metadata)]
        if recursive and stat.S_ISDIR(metadata.st_mode):
            for child in sorted(path.iterdir()):
                entries.extend(self._entries(child, True))
        return entries

    def migrate(self, value: Path, *, recursive: bool = False) -> None:
        path = Path(value)
        self._validate_path(path)
        entries = self._entries(path, recursive)
        if not entries and self.previous_account is not None:
            self._missing_roots.setdefault(path, recursive)
        # Validate the entire tree before changing its first entry.
        for entry, metadata in entries:
            if metadata.st_uid not in self.allowed_uids:
                raise ValueError(f"目录含有其他账户的数据，不迁移其权限：{entry}")
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
                raise ValueError(f"Agent 账户迁移不能更改共享硬链接文件：{entry}")
        if recursive and entries:
            root_metadata = entries[0][1]
            self._recursive_roots.setdefault(path, (root_metadata.st_uid, root_metadata.st_gid))
        for entry, metadata in entries:
            if entry not in self._original:
                self._original[entry] = (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode))
            if (metadata.st_uid, metadata.st_gid) != (self.account.pw_uid, self.account.pw_gid):
                os.chown(entry, self.account.pw_uid, self.account.pw_gid, follow_symlinks=False)

    def rollback(self) -> None:
        # Upgrades can recreate a missing home/tmp/state directory. The previous
        # service account must be able to use those mode-0700 paths on recovery.
        for root, recursive in self._missing_roots.items():
            self._validate_path(root)
            for entry, metadata in self._entries(root, recursive):
                if metadata.st_uid not in self.allowed_uids or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1):
                    raise ValueError(f"Agent 账户迁移恢复时发现共享或其他账户的数据：{entry}")
                os.chown(entry, self.previous_account.pw_uid, self.previous_account.pw_gid, follow_symlinks=False)
        # A briefly started replacement Agent can create new SQLite WAL/SHM or
        # lock files. Restore those to the previous directory owner as well.
        for root, (uid, gid) in self._recursive_roots.items():
            self._validate_path(root)
            for entry, metadata in self._entries(root, True):
                if entry in self._original:
                    continue
                if metadata.st_uid not in self.allowed_uids or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1):
                    raise ValueError(f"Agent 账户迁移恢复时发现共享或其他账户的数据：{entry}")
                os.chown(entry, uid, gid, follow_symlinks=False)
        for entry, (uid, gid, mode) in reversed(tuple(self._original.items())):
            self._validate_path(entry)
            try:
                metadata = entry.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Agent 账户迁移恢复时发现符号链接：{entry}")
            os.chown(entry, uid, gid, follow_symlinks=False)
            os.chmod(entry, mode, follow_symlinks=False)
        self._original.clear()
        self._recursive_roots.clear()
        self._missing_roots.clear()
