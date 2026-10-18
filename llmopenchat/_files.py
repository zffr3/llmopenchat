"""Select the native repository boundary without weakening Windows locking."""

from __future__ import annotations

import os
if os.name == "nt":
    from ._winfiles import FileAccessError, LockedHandle, supported
    from ._winfiles import LockedRepository as _WindowsRepository

    class LockedRepository(_WindowsRepository):
        def open_file(self, parts: tuple[str, ...], *, write=False, create=False):
            return LockedHandle(self.root.joinpath(*parts), write=write, create=create)

        def open_directory(self, parts: tuple[str, ...]):
            return LockedHandle(self.root.joinpath(*parts), directory=True)

        def exists(self, parts: tuple[str, ...]) -> bool:
            try:
                os.lstat(self.root.joinpath(*parts))
                return True
            except FileNotFoundError:
                return False

        def scandir(self, parts: tuple[str, ...]):
            return os.scandir(self.root.joinpath(*parts))

        def validate(self):
            # Windows share modes already prevent ancestor and target swaps.
            pass
else:
    from ._posixfiles import FileAccessError, LockedHandle, LockedRepository, supported
