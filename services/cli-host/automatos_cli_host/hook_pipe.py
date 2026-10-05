"""The host's hook channel on Windows: a named pipe both ends authenticate (#818).

Python's standard library has no Unix socket on Windows, so the hooks reach the
host over a named pipe, through ``multiprocessing.connection`` (stdlib):

- The pipe's name is random for each host start, and its first instance is
  created exclusively, so nothing can claim the name before the host does.
- Before a payload moves, each end proves it holds the host's key with an HMAC
  challenge. The key is 32 random bytes, handed to each session in
  ``AUTOMATOS_HOST_KEY``.
- The shim also checks that the process serving the pipe is the host
  (``GetNamedPipeServerProcessId`` against ``AUTOMATOS_HOST_PID``, in
  ``hook_shim``). A command the CLI runs inherits the key with the rest of the
  session's environment, and could serve its own instance of the pipe; the
  process check, F234's on Unix, keeps it from answering the policy gate.
- A pipe's default security lets only the same user, LocalSystem and the
  administrators open it for writing.

Each connection carries one payload and one answer, as on the Unix socket.
``family="AF_UNIX"`` runs the same server over a Unix socket, which is how the
tests exercise it on Linux and macOS.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from typing import Any, Dict, Optional

from .hook_server import HookRegistry

log = logging.getLogger("automatos.cli_host.hooks")

PIPE_PREFIX = "\\\\.\\pipe\\"
KEY_BYTES = 32
PAYLOAD_WAIT_SECONDS = 10.0
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024      # a Write tool call carries the whole file
_RETRY_AFTER_ERROR_SECONDS = 0.5
_WAKE_TIMEOUT_SECONDS = 2.0


class PipeHookServer(HookRegistry):
    def __init__(self, address: Optional[str] = None, family: str = "AF_PIPE", key: Optional[bytes] = None):
        super().__init__()
        self.address = address or f"{PIPE_PREFIX}automatos-cli-host-{secrets.token_hex(16)}"
        self.family = family
        self.key = key or secrets.token_bytes(KEY_BYTES)
        self._listener: Any = None
        self._stopping = threading.Event()

    def session_env(self) -> Dict[str, str]:
        return {"AUTOMATOS_HOST_SOCK": self.address, "AUTOMATOS_HOST_KEY": self.key.hex()}

    def start(self) -> None:
        from multiprocessing.connection import Listener

        self._stopping.clear()
        self._listener = Listener(self.address, family=self.family, backlog=64, authkey=self.key)
        threading.Thread(target=self._serve, args=(self._listener,), name="automatos-hook-pipe", daemon=True).start()

    def ensure_listening(self) -> bool:
        """A pipe has no file that a previous host could remove, so nothing to heal."""
        return False

    def stop(self) -> None:
        self._stopping.set()
        listener, self._listener = self._listener, None
        if listener is None:
            return
        # accept() has no timeout: a throwaway caller wakes it. On its own thread,
        # because a caller would wait forever if nothing were accepting any more.
        waker = threading.Thread(target=self._wake, daemon=True, name="automatos-hook-pipe-wake")
        waker.start()
        waker.join(_WAKE_TIMEOUT_SECONDS)
        listener.close()

    def _wake(self) -> None:
        from multiprocessing.connection import Client

        try:
            Client(self.address, family=self.family, authkey=self.key).close()
        except (OSError, EOFError):
            pass

    def _serve(self, listener: Any) -> None:
        from multiprocessing import AuthenticationError

        while not self._stopping.is_set():
            try:
                conn = listener.accept()
            except AuthenticationError:
                log.warning("refused a hook caller that does not hold the host's key")
                continue
            except (OSError, EOFError):
                if self._stopping.is_set():
                    break
                log.warning("hook pipe accept failed; retrying", exc_info=True)
                time.sleep(_RETRY_AFTER_ERROR_SECONDS)
                continue
            if self._stopping.is_set():
                conn.close()
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: Any) -> None:
        answer: dict = {}
        try:
            if conn.poll(PAYLOAD_WAIT_SECONDS):
                answer = self.answer_for(conn.recv_bytes(MAX_PAYLOAD_BYTES))
            else:
                log.warning("a hook caller sent no payload within %ss", PAYLOAD_WAIT_SECONDS)
        except Exception:  # noqa: BLE001 — never let a hook call crash the host
            log.exception("hook handling failed")
            answer = {}
        try:
            conn.send_bytes(json.dumps(answer).encode("utf-8"))
        except OSError:
            pass
        finally:
            conn.close()


__all__ = ["PIPE_PREFIX", "PipeHookServer"]
