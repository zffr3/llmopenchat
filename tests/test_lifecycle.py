"""Service signals must unwind cleanup without changing embedding callers."""

from __future__ import annotations

import signal
import threading
import unittest
from unittest.mock import patch

from llmopenchat import lifecycle


class TerminationTests(unittest.TestCase):
    def test_sigterm_exits_cleanly_and_restores_previous_handler(self):
        original = object()
        cleaned = []
        with (patch.object(lifecycle.os, "name", "posix"),
              patch.object(signal, "getsignal", return_value=original),
              patch.object(signal, "signal") as install):
            with self.assertRaises(SystemExit) as raised:
                with lifecycle.termination_signals():
                    handler = install.call_args.args[1]
                    try:
                        handler(signal.SIGTERM, None)
                    finally:
                        cleaned.append(True)
            self.assertEqual(raised.exception.code, 0)
            self.assertEqual(cleaned, [True])
            self.assertEqual(install.call_args.args, (signal.SIGTERM, original))
            self.assertIn(((signal.SIGTERM, signal.SIG_IGN),), install.call_args_list)

    def test_embedded_worker_keeps_callers_signal_handlers(self):
        with (patch.object(lifecycle.os, "name", "posix"),
              patch.object(threading, "current_thread", return_value=object()),
              patch.object(signal, "signal") as install):
            with lifecycle.termination_signals():
                pass
            install.assert_not_called()

    def test_windows_signal_handlers_stay_unchanged(self):
        with (patch.object(lifecycle.os, "name", "nt"),
              patch.object(signal, "signal") as install):
            with lifecycle.termination_signals():
                pass
            install.assert_not_called()
