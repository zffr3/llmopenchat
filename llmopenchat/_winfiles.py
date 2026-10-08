"""Windows file handles used by the repository harness.

Directory handles deny write and delete sharing, so ancestors cannot become
reparse points or be renamed
while a tool waits for approval or runs. Files are opened exclusively and checked
through their handles before any content I/O. No fallback to ordinary open().
"""

from __future__ import annotations

import ctypes
import ntpath
import os
from pathlib import Path


class FileAccessError(RuntimeError):
    def __init__(self, message: str, winerror: int | None = None):
        super().__init__(message)
        self.winerror = winerror


if os.name == "nt":
    from ctypes import wintypes

    class _Info(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation", wintypes.FILETIME),
            ("access", wintypes.FILETIME),
            ("write", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    _kernel.CreateFileW.restype = wintypes.HANDLE
    _kernel.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
    _kernel.CreateDirectoryW.restype = wintypes.BOOL
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL
    _kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Info)]
    _kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    _kernel.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    _kernel.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _kernel.GetFileType.argtypes = [wintypes.HANDLE]
    _kernel.GetFileType.restype = wintypes.DWORD
    _kernel.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    _kernel.GetDriveTypeW.restype = wintypes.UINT
    _kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _kernel.ReadFile.restype = wintypes.BOOL
    _kernel.WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _kernel.WriteFile.restype = wintypes.BOOL
    _kernel.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
    _kernel.SetFilePointerEx.restype = wintypes.BOOL
    _kernel.SetEndOfFile.argtypes = [wintypes.HANDLE]
    _kernel.SetEndOfFile.restype = wintypes.BOOL
    _kernel.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    _kernel.FlushFileBuffers.restype = wintypes.BOOL


def supported() -> bool:
    return os.name == "nt"


def _failure(operation: str) -> FileAccessError:
    code = ctypes.get_last_error()
    return FileAccessError(f"{operation}: Windows error {code}. Файл может быть занят.", code)


class LockedHandle:
    def __init__(self, path: Path, *, directory: bool = False, write: bool = False, create: bool = False):
        if not supported():
            raise FileAccessError("Строгий доступ к репозиторию поддерживается только в Windows.")
        self.path = path
        self.handle = None
        # Extended paths avoid Win32 stripping trailing dots/spaces; the caller
        # additionally rejects them and all device/ADS/short-name forms.
        extended = "\\\\?\\" + str(path)
        # FILE_LIST_DIRECTORY is required for Windows share-mode enforcement.
        # FILE_READ_ATTRIBUTES alone would not stop a second GENERIC_WRITE
        # handle from changing a directory into a junction during approval.
        access = 0x81 if directory else (0x40000000 if write else 0x80000000)
        share = 0x1 if directory else 0
        flags = 0x00200000 | (0x02000000 if directory else 0)  # OPEN_REPARSE_POINT, BACKUP_SEMANTICS
        handle = _kernel.CreateFileW(extended, access, share, None, 1 if create else 3, flags, None)
        if handle == ctypes.c_void_p(-1).value or handle is None:
            raise _failure("Не удалось безопасно открыть путь")
        self.handle = handle
        try:
            info = self.info()
            if info.attributes & 0x400:
                raise FileAccessError("Символические ссылки, junction и другие reparse points запрещены.")
            if bool(info.attributes & 0x10) != directory:
                raise FileAccessError("Ожидалась папка." if directory else "Ожидался обычный файл.")
            if _kernel.GetFileType(handle) != 1:
                raise FileAccessError("Разрешены только обычные файлы на локальном диске.")
            if not directory and info.links != 1:
                raise FileAccessError("Файлы с жёсткими ссылками запрещены.")
            buffer = ctypes.create_unicode_buffer(32768)
            length = _kernel.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
            if not length or length >= len(buffer):
                raise _failure("Не удалось проверить фактический путь")
            actual = buffer.value.removeprefix("\\\\?\\")
            if ntpath.normcase(ntpath.normpath(actual)) != ntpath.normcase(ntpath.normpath(str(path))):
                raise FileAccessError("Фактический путь отличается от разрешённого; доступ запрещён.")
        except BaseException:
            self.close()
            raise

    def info(self):
        info = _Info()
        if not _kernel.GetFileInformationByHandle(self.handle, ctypes.byref(info)):
            raise _failure("Не удалось проверить файл")
        return info

    @property
    def identity(self) -> tuple[int, int, int]:
        info = self.info()
        return (info.volume, info.index_high, info.index_low)

    @property
    def size(self) -> int:
        info = self.info()
        return (info.size_high << 32) | info.size_low

    def read(self, limit: int) -> bytes:
        # Size checking uses handle metadata; the exclusive handle excludes an
        # uncooperative writer while content is being read.
        if self.size > limit:
            raise FileAccessError(f"Файл превышает лимит {limit} байт.")
        buffer = ctypes.create_string_buffer(limit + 1)
        count = wintypes.DWORD()
        if not _kernel.ReadFile(self.handle, buffer, limit + 1, ctypes.byref(count), None):
            raise _failure("Не удалось прочитать файл")
        if count.value > limit:
            raise FileAccessError(f"Файл превышает лимит {limit} байт.")
        return buffer.raw[:count.value]

    def write(self, content: bytes):
        # No truncation happens in CreateFile: writes begin only after approval.
        if not _kernel.SetFilePointerEx(self.handle, 0, None, 0):
            raise _failure("Не удалось перейти к началу файла")
        buffer = ctypes.create_string_buffer(content)
        count = wintypes.DWORD()
        if not _kernel.WriteFile(self.handle, buffer, len(content), ctypes.byref(count), None) or count.value != len(content):
            raise _failure("Запись не завершена; проверьте файл")
        if not _kernel.SetEndOfFile(self.handle) or not _kernel.FlushFileBuffers(self.handle):
            raise _failure("Не удалось завершить запись; проверьте файл")

    def close(self):
        if self.handle is not None:
            _kernel.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LockedRepository:
    """Keep ancestors locked against both renames and reparse-point changes."""

    def __init__(self, root: Path, identity: tuple[int, int, int] | None = None):
        if not supported():
            raise FileAccessError("Строгий доступ к репозиторию поддерживается только в Windows.")
        if _kernel.GetDriveTypeW(root.anchor) != 3:
            raise FileAccessError("Выберите папку на локальном фиксированном диске Windows.")
        self.root = root
        self.handles: list[LockedHandle] = []
        try:
            for path in [*reversed(root.parents), root]:
                self.handles.append(LockedHandle(path, directory=True))
            self.identity = self.handles[-1].identity
            if identity is not None and identity != self.identity:
                raise FileAccessError("Выбранная папка была заменена. Выберите репозиторий заново в меню.")
        except BaseException:
            self.close()
            raise

    def directory(self, parts: tuple[str, ...]) -> Path:
        path = self.root
        for part in parts:
            path = path / part
            self.handles.append(LockedHandle(path, directory=True))
        return path

    def plan_directory(self, parts: tuple[str, ...]) -> int:
        """Pin the existing prefix without creating or enumerating anything.

        Return its component count. Only FILE_NOT_FOUND/PATH_NOT_FOUND denotes
        a missing suffix; reparse points, files, sharing conflicts and access
        failures are never interpreted as permissions to create a directory.
        """
        path = self.root
        for index, part in enumerate(parts):
            path = path / part
            try:
                handle = LockedHandle(path, directory=True)
            except FileAccessError as error:
                if error.winerror in (2, 3):
                    return index
                raise
            self.handles.append(handle)
        return len(parts)

    def create_directories(self, parts: tuple[str, ...], existing_count: int) -> list[str]:
        """Create a consented missing suffix under pinned parents, one at a time.

        A competing ordinary directory can be accepted after no-follow handle
        validation. A competing junction, file or write-open directory fails
        before any descent. Each successful child is pinned before proceeding.
        """
        path = self.root.joinpath(*parts[:existing_count])
        created: list[str] = []
        for index in range(existing_count, len(parts)):
            path = path / parts[index]
            made = bool(_kernel.CreateDirectoryW("\\\\?\\" + str(path), None))
            if not made and ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
                raise _failure("Не удалось создать каталог")
            self.handles.append(LockedHandle(path, directory=True))
            if made:
                created.append("/".join(parts[:index + 1]))
        return created

    def close(self):
        for handle in reversed(self.handles):
            handle.close()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
