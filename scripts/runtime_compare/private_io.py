"""Descriptor-pinned private files and directories for runtime comparison."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any


def _parts(path: Path) -> tuple[str, ...]:
    original = Path(path)
    if ".." in original.parts:
        raise ValueError("parent traversal is not allowed")
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise ValueError("path must be absolute and contain no parent traversal")
    return tuple(part for part in absolute.parts if part not in {"/", ""})


def _open_directory_chain(path: Path) -> int:
    if os.name != "posix":
        raise OSError("descriptor-pinned private output requires Unix")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in _parts(path):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _basename(name: str) -> str:
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\x00" in name:
        raise ValueError("expected a single safe path component")
    return name


def _open_regular(path: Path) -> int:
    raw_path = Path(path)
    _parts(raw_path)
    path = Path(os.path.abspath(os.fspath(raw_path)))
    parent = _open_directory_chain(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=parent)
    finally:
        os.close(parent)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ValueError("input must be a regular file")
    return fd


def open_regular_file(path: Path) -> int:
    """Open a regular file through pinned, no-follow directory descriptors."""
    return _open_regular(Path(path))


def read_bounded(path: Path, cap: int) -> bytes:
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("invalid read cap")
    fd = _open_regular(Path(path))
    try:
        if os.fstat(fd).st_size > cap:
            raise ValueError("file exceeds byte limit")
        chunks = bytearray()
        while len(chunks) <= cap:
            piece = os.read(fd, min(65536, cap + 1 - len(chunks)))
            if not piece:
                break
            chunks.extend(piece)
        if len(chunks) > cap:
            raise ValueError("file exceeds byte limit")
        return bytes(chunks)
    finally:
        os.close(fd)


def validate_regular_file(path: Path, executable: bool = False) -> Path:
    fd = _open_regular(Path(path))
    try:
        info = os.fstat(fd)
        if executable and not (info.st_mode & 0o111):
            raise ValueError("executable file has no execute bits")
    finally:
        os.close(fd)
    return Path(os.path.abspath(os.fspath(path)))


def read_bounded_json(path: Path, cap: int) -> Any:
    try:
        return json.loads(read_bounded(path, cap))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid bounded JSON") from exc


class PrivateDirectory:
    """A private directory held open by fd; all children are addressed by basename."""

    def __init__(self, fd: int, path: Path):
        self._fd = fd
        self.path = path
        self._closed = False

    @classmethod
    def create(cls, path: Path, repo_root: Path) -> "PrivateDirectory":
        if os.name != "posix":
            raise OSError("private output permission enforcement requires Unix")
        _parts(Path(path))
        target = Path(os.path.abspath(os.fspath(path)))
        repo = Path(os.path.realpath(repo_root))
        if target == repo or repo in target.parents:
            raise ValueError("output directory must be outside the source repository")
        name = _basename(target.name)
        parent_fd = _open_directory_chain(target.parent)
        try:
            parent_info = os.fstat(parent_fd)
            if parent_info.st_uid != os.geteuid() or stat.S_IMODE(parent_info.st_mode) != 0o700:
                raise PermissionError("output parent must be owned and mode 0700")
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        os.fchmod(fd, 0o700)
        return cls(fd, target)

    def _ensure_open(self):
        if self._closed:
            raise ValueError("private directory is closed")

    def mkdir(self, name: str) -> "PrivateDirectory":
        self._ensure_open()
        name = _basename(name)
        os.mkdir(name, mode=0o700, dir_fd=self._fd)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=self._fd)
        os.fchmod(fd, 0o700)
        return PrivateDirectory(fd, self.path / name)

    def write(self, name: str, data: bytes, cap: int) -> None:
        self._ensure_open()
        name = _basename(name)
        if len(data) > cap:
            raise ValueError("serialized output exceeds byte limit")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=self._fd)
        try:
            os.fchmod(fd, 0o600)
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

    def read(self, name: str, cap: int) -> bytes:
        self._ensure_open()
        name = _basename(name)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=self._fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("artifact must be a regular file")
            if os.fstat(fd).st_size > cap:
                raise ValueError("artifact exceeds byte limit")
            data = bytearray()
            while len(data) <= cap:
                piece = os.read(fd, min(65536, cap + 1 - len(data)))
                if not piece:
                    break
                data.extend(piece)
            if len(data) > cap:
                raise ValueError("artifact exceeds byte limit")
            return bytes(data)
        finally:
            os.close(fd)

    def size(self, cap: int | None = None) -> int:
        """Count regular files below this pinned directory without following symlinks."""
        self._ensure_open()
        def walk(fd: int) -> int:
            total = 0
            for name in os.listdir(fd):
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
                elif stat.S_ISDIR(info.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                    dir_fd=fd)
                    try:
                        total += walk(child)
                    finally:
                        os.close(child)
                else:
                    raise ValueError("non-regular object found in private artifact directory")
                if cap is not None and total > cap:
                    return total
            return total
        return walk(self._fd)

    def close(self) -> None:
        if not self._closed:
            os.close(self._fd)
            self._closed = True

    def limit_retained_artifacts(self, budget: int) -> dict:
        """Trim only this session's private files after all children are reaped.

        Preserve file prefixes where space permits, never follow links, and
        return numeric loss accounting for the failure report. This is not a
        concurrent writer quota; callers must first stop every owned writer.
        """
        self._ensure_open()
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError("invalid retention budget")
        entries = 0

        def visit(fd, action, depth=0):
            nonlocal entries
            if depth > 16:
                raise ValueError("artifact directory nesting exceeds retention budget")
            for name in sorted(os.listdir(fd)):
                entries += 1
                if entries > 4096:
                    raise ValueError("artifact count exceeds retention budget")
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                    dir_fd=fd)
                    try:
                        child_info = os.fstat(child)
                        if child_info.st_uid != os.geteuid() or stat.S_IMODE(child_info.st_mode) != 0o700:
                            raise PermissionError("retention requires an owned private directory")
                        visit(child, action, depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    file_fd = os.open(name, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                                      dir_fd=fd)
                    try:
                        pinned = os.fstat(file_fd)
                        if (not stat.S_ISREG(pinned.st_mode) or pinned.st_uid != os.geteuid()
                                or pinned.st_nlink != 1 or stat.S_IMODE(pinned.st_mode) != 0o600):
                            raise PermissionError("retention requires an unlinked owned mode0600 regular artifact")
                        action(file_fd, pinned.st_size)
                    finally:
                        os.close(file_fd)
                else:
                    raise ValueError("retention refuses non-regular artifacts")

        # Check the entire bounded tree first, so an unsafe later entry cannot
        # cause partial truncation of earlier files in this pass.
        visit(self._fd, lambda _fd, _size: None)
        entries = 0
        remaining = budget
        result = {"truncated_files": 0, "discarded_bytes": 0}

        def trim(fd, size):
            nonlocal remaining
            retained = min(size, remaining)
            remaining -= retained
            if retained < size:
                os.ftruncate(fd, retained)
                result["truncated_files"] += 1
                result["discarded_bytes"] += size - retained

        visit(self._fd, trim)
        return result

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_exc):
        self.close()
