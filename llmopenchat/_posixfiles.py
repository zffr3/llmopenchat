"""Linux repository I/O using pinned, no-follow directory descriptors.

Every component is opened relative to its verified parent. Content is never
opened by an absolute pathname, and directories and files are revalidated after
approval. Replacements are published atomically from private temporary files,
so replacements never modify an existing inode through an outside hardlink.

POSIX advisory locks cannot stop an unrelated process running as the same user
from renaming directories or editing files. Descriptor pinning and identity
checks detect changes at validation boundaries; Linux deployments must isolate
the service account from untrusted concurrent writers for that stronger threat
model. Windows retains its mandatory share-mode protections.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
import sys
from pathlib import Path

from ._winfiles import FileAccessError


def supported() -> bool:
    return (
        sys.platform.startswith("linux")
        and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
        and os.scandir in os.supports_fd
    )


def _identity(info) -> tuple[int, int, int]:
    return info.st_dev, 0, info.st_ino


def _version(info) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _failure(error: OSError) -> FileAccessError:
    return FileAccessError(f"Не удалось безопасно открыть путь: {error.strerror or error}.")


class LockedHandle:
    def __init__(self, path: Path, *, dir_fd: int, name: str, directory=False,
                 write=False, create=False, validate=None, device: int | None = None):
        self.path = path
        self.handle: int | None = None
        self._parent = dir_fd
        self._name = name
        self._directory = directory
        self._validate_parent = validate
        self._temporary: str | None = None
        self._create = create
        self._device = device
        try:
            if create:
                self._temporary, self.handle = self._new_temporary()
            else:
                flags = os.O_RDONLY if directory or not write else os.O_RDWR
                flags |= os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
                if directory:
                    flags |= os.O_DIRECTORY
                self.handle = os.open(name, flags, dir_fd=dir_fd)
            self._info = os.fstat(self.handle)
            self._check(self._info)
        except OSError as error:
            self.close()
            raise _failure(error) from error
        except BaseException:
            self.close()
            raise

    def _check(self, info):
        if self._directory:
            if not stat.S_ISDIR(info.st_mode):
                raise FileAccessError("Ожидалась обычная папка без символических ссылок.")
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise FileAccessError("Разрешены только обычные файлы без символических и жёстких ссылок.")
        if self._device is not None and info.st_dev != self._device:
            raise FileAccessError("Переход на другую файловую систему внутри репозитория запрещён.")

    def _new_temporary(self) -> tuple[str, int]:
        for _ in range(16):
            name = ".llmopenchat-" + secrets.token_hex(16) + ".tmp"
            try:
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=self._parent)
                return name, fd
            except FileExistsError:
                continue
        raise FileAccessError("Не удалось создать временный файл внутри репозитория.")

    def validate(self):
        if self._validate_parent is not None:
            self._validate_parent()
        current = os.fstat(self.handle)
        self._check(current)
        name = self._temporary if self._create else self._name
        linked = os.stat(name, dir_fd=self._parent, follow_symlinks=False)
        if _identity(linked) != _identity(current) or stat.S_IFMT(linked.st_mode) != stat.S_IFMT(current.st_mode):
            raise FileAccessError("Путь был заменён во время подтверждения; повторите запрос.")
        if not self._directory and not self._create and _version(current) != _version(self._info):
            raise FileAccessError("Файл изменился во время подтверждения; повторите запрос.")

    @property
    def identity(self) -> tuple[int, int, int]:
        return _identity(os.fstat(self.handle))

    @property
    def size(self) -> int:
        return os.fstat(self.handle).st_size

    def read(self, limit: int) -> bytes:
        self.validate()
        if self.size > limit:
            raise FileAccessError(f"Файл превышает лимит {limit} байт.")
        os.lseek(self.handle, 0, os.SEEK_SET)
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(self.handle, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        self.validate()
        if len(content) > limit:
            raise FileAccessError(f"Файл превышает лимит {limit} байт.")
        return content

    def write(self, content: bytes):
        self.validate()
        temporary, fd = (self._temporary, self.handle) if self._create else self._new_temporary()
        try:
            view = memoryview(content)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise FileAccessError("Запись не завершена; проверьте файл.")
                view = view[written:]
            if not self._create:
                os.fchmod(fd, stat.S_IMODE(self._info.st_mode) & 0o777)
            os.fsync(fd)
            self.validate()
            staged = os.stat(temporary, dir_fd=self._parent, follow_symlinks=False)
            if _identity(staged) != _identity(os.fstat(fd)) or staged.st_nlink != 1:
                raise FileAccessError("Временный файл был заменён; запись запрещена.")
            if self._create:
                # link() publishes the completed inode with CREATE_NEW semantics.
                os.link(temporary, self._name, src_dir_fd=self._parent,
                        dst_dir_fd=self._parent, follow_symlinks=False)
                os.unlink(temporary, dir_fd=self._parent)
                self._temporary = None
            else:
                # Do not truncate a pinned inode: an outside alias to the
                # original file must retain its original bytes.
                os.replace(temporary, self._name, src_dir_fd=self._parent, dst_dir_fd=self._parent)
            os.fsync(self._parent)
        finally:
            if not self._create:
                os.close(fd)
            if temporary:
                try:
                    os.unlink(temporary, dir_fd=self._parent)
                except FileNotFoundError:
                    pass

    def close(self):
        if self.handle is not None:
            os.close(self.handle)
            self.handle = None
        if self._temporary is not None:
            try:
                os.unlink(self._temporary, dir_fd=self._parent)
            except FileNotFoundError:
                pass
            self._temporary = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LockedRepository:
    def __init__(self, root: Path, identity: tuple[int, int, int] | None = None):
        if not supported():
            raise FileAccessError("Строгий доступ к репозиторию поддерживается в Windows и Linux.")
        self.root = root
        self.handles: list[LockedHandle] = []
        self._directories: dict[tuple[str, ...], LockedHandle] = {}
        self._anchor: int | None = None
        try:
            self._anchor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            parent = self._anchor
            path = Path("/")
            for part in root.parts[1:]:
                path = path / part
                handle = LockedHandle(path, dir_fd=parent, name=part, directory=True)
                self.handles.append(handle)
                parent = handle.handle
            # Opening '.' from a pinned descriptor also supports choosing '/'.
            handle = LockedHandle(root, dir_fd=parent, name=".", directory=True)
            self.handles.append(handle)
            self._directories[()] = handle
            self.identity = handle.identity
            self._device = self.identity[0]
            if identity is not None and identity != self.identity:
                raise FileAccessError("Выбранная папка была заменена. Выберите репозиторий заново в меню.")
        except OSError as error:
            self.close()
            raise _failure(error) from error
        except BaseException:
            self.close()
            raise

    def validate(self):
        for handle in self.handles:
            handle.validate()

    def directory(self, parts: tuple[str, ...]) -> Path:
        self.validate()
        for index, part in enumerate(parts):
            prefix = parts[:index + 1]
            if prefix not in self._directories:
                parent = self._directories[parts[:index]].handle
                handle = LockedHandle(self.root.joinpath(*prefix), dir_fd=parent,
                                      name=part, directory=True, device=self._device)
                self.handles.append(handle)
                self._directories[prefix] = handle
        return self.root.joinpath(*parts)

    def plan_directory(self, parts: tuple[str, ...]) -> int:
        for index in range(len(parts)):
            try:
                self.directory(parts[:index + 1])
            except FileAccessError as error:
                if isinstance(error.__cause__, OSError) and error.__cause__.errno == errno.ENOENT:
                    return index
                raise
        return len(parts)

    def create_directories(self, parts: tuple[str, ...], existing_count: int) -> list[str]:
        self.validate()
        created = []
        for index in range(existing_count, len(parts)):
            parent = self._directories[parts[:index]].handle
            made = False
            try:
                os.mkdir(parts[index], mode=0o700, dir_fd=parent)
                made = True
            except FileExistsError:
                pass
            self.directory(parts[:index + 1])
            if made:
                created.append("/".join(parts[:index + 1]))
        return created

    def open_file(self, parts: tuple[str, ...], *, write=False, create=False) -> LockedHandle:
        self.directory(parts[:-1])
        return LockedHandle(self.root.joinpath(*parts), dir_fd=self._directories[parts[:-1]].handle,
                            name=parts[-1], write=write, create=create, validate=self.validate,
                            device=self._device)

    def open_directory(self, parts: tuple[str, ...]) -> LockedHandle:
        self.directory(parts[:-1])
        return LockedHandle(self.root.joinpath(*parts), dir_fd=self._directories[parts[:-1]].handle,
                            name=parts[-1], directory=True, validate=self.validate, device=self._device)

    def exists(self, parts: tuple[str, ...]) -> bool:
        self.directory(parts[:-1])
        try:
            os.stat(parts[-1], dir_fd=self._directories[parts[:-1]].handle, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False

    def scandir(self, parts: tuple[str, ...]):
        self.directory(parts)
        return os.scandir(self._directories[parts].handle)

    def close(self):
        for handle in reversed(self.handles):
            handle.close()
        self.handles = []
        if self._anchor is not None:
            os.close(self._anchor)
            self._anchor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
