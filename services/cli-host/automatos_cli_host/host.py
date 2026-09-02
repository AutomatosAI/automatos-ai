"""The host loop: pair once, then heartbeat · claim · run · report, forever.

* Refuses to serve anything but a LOCAL edition backend with session mode on.
* Pairs with the one-time code from Settings → Session mode; the token it gets
  back is the only secret it keeps (``~/.automatos/cli-host/host.json``, 0600).
* Each claimed ticket runs in its own thread as a supervised interactive
  session (``session.py``); hook events are shipped in batches, and the
  backend's answer carries control (``cancel``).
* On start, any process left in the table by a previous host run is killed:
  the pseudo-terminal that kept it alive died with that host, and the backend
  requeues the ticket when its lease lapses. Nothing is re-attached blind.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import state
from .api import BackendClient, BackendError
from .config import HostConfig, parse_args
from .hook_server import HookServer
from .session import Session, host_capabilities

log = logging.getLogger("automatos.cli_host")


class HostRefused(RuntimeError):
    """The host will not run against this backend / configuration."""


def check_backend(api: BackendClient) -> Dict[str, Any]:
    """The backend must be the local edition with session mode enabled."""
    health = api.health()
    edition = health.get("edition")
    if edition is None:
        raise HostRefused("the backend does not report its edition — update Automatos to a build with PRD-234 S1b")
    if edition != "local":
        raise HostRefused(f"refusing to serve a '{edition}' edition backend: session mode is local-only (PRD-234 §Terms)")
    if not health.get("cli_runtime_enabled"):
        raise HostRefused("session mode is off on this backend — set CLI_RUNTIME_ENABLED=true in .env and restart the stack")
    return health


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Host:
    def __init__(self, cfg: HostConfig):
        self.cfg = cfg
        self.api = BackendClient(cfg.url)
        self.identity: Optional[Dict[str, Any]] = None
        self.hooks = HookServer(cfg.socket_path)
        self.sessions: Dict[str, Session] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.pending_results: Dict[str, Dict[str, Any]] = {}
        self.allow_roots: List[str] = []
        self.stop = threading.Event()
        self._last_heartbeat = 0.0
        self._last_flush = 0.0
        self._claimed_once = False
        self._capabilities: Optional[Dict[str, Any]] = None
        self._announced_parked: set = set()

    # ── setup ───────────────────────────────────────────────────────────────
    def prepare(self) -> None:
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        check_backend(self.api)
        saved = state.load_allowlist(self.cfg.allowlist_path)
        for d in self.cfg.allow_dirs:
            resolved = str(Path(d).expanduser().resolve())
            if resolved not in saved:
                saved.append(resolved)
        state.save_allowlist(self.cfg.allowlist_path, saved)
        self.allow_roots = saved
        if not self.allow_roots:
            raise HostRefused("no directories registered — start with `--allow <dir>` (make cli-host registers ./workspaces)")

        self.identity = state.load_host_identity(self.cfg.token_path)
        if self.identity and self.identity.get("url") not in (None, self.cfg.url):
            log.warning("host was paired with %s, now talking to %s", self.identity.get("url"), self.cfg.url)
        if self.identity is None:
            if not self.cfg.pair_code:
                raise HostRefused("this host is not paired — get a code from Settings → Session mode and run: make cli-host PAIR=<code>")
            paired = self.api.pair(self.cfg.pair_code, self.cfg.name, self.capabilities())
            state.save_host_identity(
                self.cfg.token_path, host_id=paired["host_id"], token=paired["token"],
                workspace_id=paired.get("workspace_id", ""), url=self.cfg.url,
            )
            self.identity = state.load_host_identity(self.cfg.token_path)
            log.info("paired as host %s", paired["host_id"])
        self.api.token = self.identity["token"]
        self._reap_previous_run()
        self.hooks.start()

    def capabilities(self) -> Dict[str, Any]:
        if self._capabilities is None:
            self._capabilities = host_capabilities(self.cfg)
        return self._capabilities

    def _reap_previous_run(self) -> None:
        table = state.load_process_table(self.cfg.process_table_path)
        for task_id, entry in table.items():
            pid = entry.get("pid")
            if isinstance(pid, int) and _pid_alive(pid):
                log.warning("killing orphan session process %s for task %s from a previous host run", pid, task_id)
                try:
                    os.killpg(entry.get("pgid") or pid, signal.SIGTERM)
                except OSError:
                    pass
        state.save_process_table(self.cfg.process_table_path, {})

    # ── loop ────────────────────────────────────────────────────────────────
    def run_forever(self) -> int:
        host_id = self.identity["host_id"]
        log.info("CLI host %s serving %s — directories: %s", host_id, self.cfg.url, ", ".join(self.allow_roots))
        try:
            while not self.stop.is_set():
                now = time.time()
                if now - self._last_heartbeat >= self.cfg.heartbeat_seconds:
                    self._heartbeat(host_id)
                    self._last_heartbeat = now
                self._claim_and_start(host_id)
                if now - self._last_flush >= self.cfg.event_flush_seconds:
                    self._flush_events(host_id)
                    self._last_flush = now
                self._reap_finished(host_id)
                self._retry_results(host_id)
                if self.cfg.once and self._claimed_once and not self.sessions and not self.pending_results:
                    return 0
                time.sleep(self.cfg.poll_seconds if not self.sessions else min(1.0, self.cfg.poll_seconds))
        finally:
            for s in self.sessions.values():
                s.request_cancel()
            for t in self.threads.values():
                t.join(timeout=15)
            self._flush_events(host_id)
            self._reap_finished(host_id)
            self.hooks.stop()
        return 0

    def _heartbeat(self, host_id: str) -> None:
        running = [{"task_id": int(tid), "session_id": s.session_id, "attempt": s.attempt}
                   for tid, s in self.sessions.items()]
        try:
            out = self.api.heartbeat(host_id, self.capabilities(), running)
        except BackendError as exc:
            log.warning("heartbeat failed: %s", exc)
            return
        for stale in out.get("stale") or []:
            s = self.sessions.get(str(stale))
            if s is not None:
                log.info("task %s is no longer ours — stopping its session", stale)
                s.request_cancel()

    def _free_slots(self) -> int:
        if self.cfg.max_sessions <= 0:
            return self.cfg.claim_batch
        return max(0, self.cfg.max_sessions - len(self.sessions))

    def _claim_and_start(self, host_id: str) -> None:
        free = self._free_slots()
        if free <= 0 or (self.cfg.once and self._claimed_once):
            return
        try:
            claimed = self.api.claim(host_id, min(free, self.cfg.claim_batch))
        except BackendError as exc:
            log.warning("claim failed: %s", exc)
            return
        self._claimed_once = True
        for held in claimed["parked"]:
            key = str(held.get("task_id"))
            if key not in self._announced_parked:
                self._announced_parked.add(key)
                log.info("task %s (%s) is waiting for the operator: %s — approve it in the Command Centre and it comes back",
                         key, held.get("title"), held.get("reason"))
        for ticket in claimed["tasks"]:
            self._start(ticket)

    def _start(self, ticket: Dict[str, Any]) -> None:
        task_id = str(ticket.get("task_id"))
        session = Session(ticket, self.cfg, self.allow_roots, self.cfg.socket_path, default_root=self.allow_roots[0])
        self.sessions[task_id] = session
        self.hooks.register(task_id, session.handle_hook)

        def _runner() -> None:
            outcome = session.run()
            self.pending_results[task_id] = outcome.as_result_payload(session.attempt)

        t = threading.Thread(target=_runner, name=f"session-{task_id}", daemon=True)
        self.threads[task_id] = t
        t.start()
        self._record_process(task_id, session)
        log.info("task %s claimed (agent %s, provider %s)", task_id, ticket.get("agent_name"), ticket.get("provider"))

    def _record_process(self, task_id: str, session: Session) -> None:
        # The pid is known only after spawn; record a placeholder now and the pid on flush.
        table = state.load_process_table(self.cfg.process_table_path)
        table[task_id] = {"pid": session.proc.pid if session.proc else None, "pgid": session.pgid,
                          "session_id": session.session_id, "attempt": session.attempt,
                          "started_at": session.started_at}
        state.save_process_table(self.cfg.process_table_path, table)

    def _flush_events(self, host_id: str) -> None:
        for task_id, session in list(self.sessions.items()):
            batch: List[Dict[str, Any]] = []
            while not session.events.empty() and len(batch) < 200:
                batch.append(session.events.get_nowait())
            if session.proc is not None:
                table = state.load_process_table(self.cfg.process_table_path)
                if table.get(task_id, {}).get("pid") != session.proc.pid:
                    table[task_id] = {**table.get(task_id, {}), "pid": session.proc.pid, "pgid": session.pgid}
                    state.save_process_table(self.cfg.process_table_path, table)
            try:
                out = self.api.events(host_id, int(task_id), batch)
            except BackendError as exc:
                log.warning("events for task %s failed (%s) — will retry", task_id, exc)
                for ev in batch:
                    session.events.put(ev)
                continue
            if "cancel" in (out.get("control") or []):
                session.request_cancel()

    def _reap_finished(self, host_id: str) -> None:
        for task_id, thread in list(self.threads.items()):
            if thread.is_alive():
                continue
            self.threads.pop(task_id, None)
            session = self.sessions.pop(task_id, None)
            self.hooks.unregister(task_id)
            if session is not None:
                self._flush_session_events(host_id, task_id, session)
            table = state.load_process_table(self.cfg.process_table_path)
            table.pop(task_id, None)
            state.save_process_table(self.cfg.process_table_path, table)
        self._retry_results(host_id)

    def _flush_session_events(self, host_id: str, task_id: str, session: Session) -> None:
        batch: List[Dict[str, Any]] = []
        while not session.events.empty():
            batch.append(session.events.get_nowait())
        if batch:
            try:
                self.api.events(host_id, int(task_id), batch)
            except BackendError as exc:
                log.warning("final events for task %s failed: %s", task_id, exc)

    def _retry_results(self, host_id: str) -> None:
        for task_id, payload in list(self.pending_results.items()):
            try:
                out = self.api.result(host_id, int(task_id), payload)
            except BackendError as exc:
                if exc.status in (401, 403, 404):
                    log.error("result for task %s rejected (%s) — dropping", task_id, exc)
                    self.pending_results.pop(task_id, None)
                else:
                    log.warning("result for task %s not accepted yet (%s) — retrying", task_id, exc)
                continue
            self.pending_results.pop(task_id, None)
            log.info("task %s → %s (applied=%s)", task_id, out.get("status"), out.get("applied"))


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    host = Host(cfg)
    try:
        host.prepare()
    except HostRefused as exc:
        log.error("%s", exc)
        return 2
    except BackendError as exc:
        log.error("backend error: %s", exc)
        return 3

    def _sigterm(_signum, _frame):
        host.stop.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    return host.run_forever()


if __name__ == "__main__":
    sys.exit(main())
