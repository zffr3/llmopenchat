"""Let service termination unwind owned servers, workers and open files."""

from __future__ import annotations

from contextlib import contextmanager
import os
import signal
import threading


@contextmanager
def termination_signals():
    # signal.signal is only available in the main interpreter thread. Calls to
    # main() by embedding applications must retain their caller's handlers.
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        # A second stop request must not interrupt cleanup in a finally block.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
