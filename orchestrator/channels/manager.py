"""
Channel Manager (PRD-55 US-019)
================================
Manages the lifecycle of all channel adapters.

``reconcile()`` makes the running adapters match a desired set (connection
id -> ``AdapterSpec``): it starts the missing ones, stops the unwanted ones
and restarts one whose spec changed. ``channels.polling`` feeds it the
active polling-mode rows, on the one worker that owns channel polling, at
boot and on a tick.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, Mapping, Optional, Tuple

from .base import BaseChannelAdapter

logger = logging.getLogger(__name__)

#: A connection whose adapter failed to start is tried again after this long, at
#: once when its spec changes, or when someone acts on it (``retry``). A revoked
#: token is then not retried, and logged, on every tick.
START_RETRY_SECONDS = 120


@dataclass(frozen=True)
class AdapterSpec:
    """What a connection's adapter is built from: ``_create_adapter``'s inputs.

    The config-change check is equality of these three fields against the spec
    the running adapter was started with: the rows are read for the desired set
    anyway, so it costs a dict comparison and no extra query.
    """

    workspace_id: str
    platform: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class ReconcileReport:
    """What one ``reconcile()`` pass changed, as connection ids."""

    started: Tuple[str, ...] = ()
    restarted: Tuple[str, ...] = ()
    stopped: Tuple[str, ...] = ()
    failed: Tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.started or self.restarted or self.stopped or self.failed)


class ChannelManager:
    """Manages all active channel adapters."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._adapters: Dict[str, BaseChannelAdapter] = {}  # connection_id -> adapter
        self._specs: Dict[str, AdapterSpec] = {}  # connection_id -> the running adapter's spec
        self._failures: Dict[str, Tuple[AdapterSpec, float]] = {}  # last failed start: spec, when
        self._owns_polling = False
        self._clock = clock

    # ------------------------------------------------------------------
    # Polling ownership
    # ------------------------------------------------------------------

    @property
    def owns_polling(self) -> bool:
        """True on the worker that runs this process's polling adapters
        (``channels.polling`` decides at boot); False on every other worker."""
        return self._owns_polling

    def take_polling_ownership(self) -> None:
        self._owns_polling = True

    def release_polling_ownership(self) -> None:
        """At shutdown: a request still in flight must not start adapters again."""
        self._owns_polling = False

    def running_ids(self) -> Tuple[str, ...]:
        """The connection ids this worker runs an adapter for."""
        return tuple(self._adapters)

    # ------------------------------------------------------------------
    # Bulk lifecycle
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        desired: Mapping[str, AdapterSpec],
        *,
        retry: Collection[str] = (),
    ) -> ReconcileReport:
        """Make the running adapters match ``desired`` (connection id -> spec).

        Stops the adapters ``desired`` lacks, then starts each wanted one that
        is not running, or runs with another spec (a restart, through
        ``start_adapter``). A failed start waits ``START_RETRY_SECONDS`` unless
        its spec changed or ``retry`` names it. Callers serialise passes
        (``channels.polling`` holds a lock around each).
        """
        stopped = await self._stop_unwanted(desired)
        started, restarted, failed = await self._start_wanted(desired, retry)
        report = ReconcileReport(started=started, restarted=restarted, stopped=stopped, failed=failed)
        if report.changed:
            logger.info(
                "[ChannelManager] reconcile: started=%d restarted=%d stopped=%d failed=%d",
                len(started), len(restarted), len(stopped), len(failed),
            )
        return report

    async def _stop_unwanted(self, desired: Mapping[str, AdapterSpec]) -> Tuple[str, ...]:
        unwanted = tuple(conn_id for conn_id in self._adapters if conn_id not in desired)
        for conn_id in unwanted:
            try:
                await self.stop_adapter(conn_id)
            except Exception:
                # stop_adapter dropped it before stopping: it is no longer ours to run.
                logger.exception("[ChannelManager] Error stopping %s", conn_id)
        self._failures = {conn_id: failure for conn_id, failure in self._failures.items() if conn_id in desired}
        return unwanted

    async def _start_wanted(
        self,
        desired: Mapping[str, AdapterSpec],
        retry: Collection[str],
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        started, restarted, failed = [], [], []
        for conn_id, spec in desired.items():
            was_running = conn_id in self._adapters
            if was_running and self._specs.get(conn_id) == spec:
                continue
            if conn_id not in retry and self._waiting_to_retry(conn_id, spec):
                continue
            if await self._try_start(conn_id, spec):
                self._failures.pop(conn_id, None)
                (restarted if was_running else started).append(conn_id)
            else:
                self._failures[conn_id] = (spec, self._clock())
                failed.append(conn_id)
        return tuple(started), tuple(restarted), tuple(failed)

    async def _try_start(self, conn_id: str, spec: AdapterSpec) -> bool:
        try:
            return await self.start_adapter(conn_id, spec.workspace_id, spec.platform, dict(spec.config))
        except Exception:
            # start_adapter catches the new adapter's failure; this is the old one's stop on a restart.
            logger.exception("[ChannelManager] Could not start %s", conn_id)
            return False

    def _waiting_to_retry(self, conn_id: str, spec: AdapterSpec) -> bool:
        failure = self._failures.get(conn_id)
        if failure is None or failure[0] != spec:
            return False
        return self._clock() - failure[1] < START_RETRY_SECONDS

    async def stop_all(self) -> Tuple[str, ...]:
        """Stop all running adapters. Returns the connection ids it stopped."""
        stopped = tuple(self._adapters)
        for conn_id, adapter in list(self._adapters.items()):
            try:
                await adapter.stop()
                logger.info("[ChannelManager] Stopped adapter %s", conn_id)
            except Exception as e:
                logger.error("[ChannelManager] Error stopping %s: %s", conn_id, e)
        self._adapters.clear()
        self._specs.clear()
        self._failures.clear()
        return stopped

    # ------------------------------------------------------------------
    # Single-adapter lifecycle
    # ------------------------------------------------------------------

    async def start_adapter(
        self,
        connection_id: str,
        workspace_id: str,
        platform: str,
        config: dict,
    ) -> bool:
        """Start (or restart) a specific adapter. Returns True on success.

        A restart stops and drops the old adapter first, so a failed restart
        leaves nothing that reads as running.
        """
        if connection_id in self._adapters:
            await self.stop_adapter(connection_id)

        adapter = self._create_adapter(connection_id, workspace_id, platform, config)
        if adapter is None:
            return False

        try:
            await adapter.start()
            self._adapters[connection_id] = adapter
            self._specs[connection_id] = AdapterSpec(workspace_id, platform, dict(config))
            logger.info(
                "[ChannelManager] Started %s adapter for connection %s",
                platform,
                connection_id,
            )
            return True
        except Exception as e:
            logger.error("[ChannelManager] Failed to start %s: %s", platform, e)
            return False

    async def stop_adapter(self, connection_id: str):
        """Stop a specific adapter by connection ID."""
        adapter = self._adapters.pop(connection_id, None)
        self._specs.pop(connection_id, None)
        if adapter:
            await adapter.stop()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def _create_adapter(
        self,
        connection_id: str,
        workspace_id: str,
        platform: str,
        config: dict,
    ) -> Optional[BaseChannelAdapter]:
        """Create a platform-specific adapter instance.

        Uses lazy imports so missing optional dependencies (e.g.
        python-telegram-bot) only surface when the adapter is actually
        requested.
        """
        _ADAPTER_MAP = {
            "telegram":    (".telegram_adapter",     "TelegramAdapter"),
            "slack":       (".slack_adapter",        "SlackAdapter"),
            "discord":     (".discord_adapter",      "DiscordAdapter"),
            "whatsapp":    (".whatsapp_adapter",     "WhatsAppAdapter"),
        }

        entry = _ADAPTER_MAP.get(platform)
        if not entry:
            logger.warning("[ChannelManager] Unknown platform: %s", platform)
            return None

        module_path, class_name = entry
        try:
            import importlib
            mod = importlib.import_module(module_path, package="channels")
            adapter_cls = getattr(mod, class_name)
            return adapter_cls(connection_id, workspace_id, config)
        except ImportError as e:
            logger.warning(
                "[ChannelManager] %s adapter dependencies not installed: %s",
                platform,
                e,
            )
            return None
        except Exception as e:
            logger.error(
                "[ChannelManager] Failed to create %s adapter: %s",
                platform,
                e,
            )
            return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_running(self, connection_id: str) -> bool:
        """True iff a polling adapter for ``connection_id`` is loaded.

        Consults this worker's ``_adapters`` dict (TelegramAdapter,
        SlackAdapter, …), the running set the polling worker publishes
        (``channels.polling_status``: on any other worker the dict is
        empty, so without it this answered False for every channel), and
        the per-driver state in ``channels.drivers.*`` — any is
        authoritative. Used by ``GET /api/channels`` and Auto's list tool
        to report a running channel as active.
        """
        legacy = self._adapters.get(connection_id)
        if legacy is not None and getattr(legacy, "is_running", False):
            return True
        if self._running_on_polling_worker(connection_id):
            return True
        # Ask every registered driver — cheap dict lookups, no I/O.
        # Each driver is isolated: one failing check must not hide a
        # channel that another driver is still polling (issue #1075).
        from channels.drivers import get_driver, list_platforms
        try:
            platforms = list_platforms()
        except Exception:
            logger.warning(
                "[ChannelManager] Failed to list channel drivers",
                exc_info=True,
            )
            return False
        for platform in platforms:
            try:
                driver = get_driver(platform)()
                if driver.is_polling_running(connection_id=connection_id):
                    return True
            except Exception:
                logger.warning(
                    "[ChannelManager] Driver %s failed is_polling_running check",
                    platform,
                    exc_info=True,
                )
        return False

    @staticmethod
    def _running_on_polling_worker(connection_id: str) -> bool:
        """The shared running set, isolated like each driver below (issue #1075):
        a Redis error must not hide a channel a driver still reports."""
        try:
            from channels import polling_status

            return polling_status.is_running(connection_id)
        except Exception:
            logger.warning(
                "[ChannelManager] Failed to read the polling worker's running set",
                exc_info=True,
            )
            return False

    def get_status(self) -> dict:
        """Return a status snapshot of all managed adapters."""
        return {
            "total_adapters": len(self._adapters),
            "adapters": {
                conn_id: {
                    "platform": type(adapter).__name__,
                    "running": adapter.is_running,
                }
                for conn_id, adapter in self._adapters.items()
            },
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_channel_manager: Optional[ChannelManager] = None


def get_channel_manager() -> ChannelManager:
    """Return the global ChannelManager singleton."""
    global _channel_manager
    if _channel_manager is None:
        _channel_manager = ChannelManager()
    return _channel_manager
