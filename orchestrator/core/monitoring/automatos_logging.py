"""
Automatos Log Relay Client — Python Logging Handler

Drop-in logging handler that ships logs to the Automatos log-relay service,
which forwards them to Loki for querying via Grafana.

Works on any platform — Railway, ECS, bare metal, local dev.

Usage:
    from automatos_logging import setup_logging
    setup_logging(service="automatos-backend")

    # Then use standard Python logging everywhere:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Request processed", extra={"request_id": "abc123"})
"""

import json
import logging
import os
import queue
import threading
import time
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


LOG_RELAY_URL = os.environ.get(
    "LOG_RELAY_URL",
    "http://log-relay.railway.internal:8080/push",
)
LOG_RELAY_ENABLED = os.environ.get("LOG_RELAY_ENABLED", "true").lower() == "true"
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown")
ENVIRONMENT = os.environ.get("ENVIRONMENT", os.environ.get("RAILWAY_ENVIRONMENT", "development"))

# Batch settings
BATCH_SIZE = int(os.environ.get("LOG_RELAY_BATCH_SIZE", "50"))
FLUSH_INTERVAL = float(os.environ.get("LOG_RELAY_FLUSH_INTERVAL", "2.0"))


class LogRelayHandler(logging.Handler):
    """Async logging handler that batches and ships logs to log-relay.

    Uses a background thread with a queue — never blocks the caller.
    Gracefully degrades if log-relay is unreachable (logs to stderr).
    """

    def __init__(
        self,
        url: str = LOG_RELAY_URL,
        service: str = SERVICE_NAME,
        environment: str = ENVIRONMENT,
        batch_size: int = BATCH_SIZE,
        flush_interval: float = FLUSH_INTERVAL,
    ):
        super().__init__()
        self.url = url
        self.service = service
        self.environment = environment
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="log-relay-flusher",
            daemon=True,
        )
        self._thread.start()
        self._consecutive_failures = 0

    def emit(self, record: logging.LogRecord):
        try:
            entry = self._format_entry(record)
            self._queue.put_nowait(entry)
        except queue.Full:
            pass  # Drop silently — better than blocking the app

    def _format_entry(self, record: logging.LogRecord) -> dict:
        extra = {}
        # Capture extra fields passed via logger.info("msg", extra={...})
        for key in ("request_id", "module", "user_id", "workspace_id",
                     "agent_id", "task_id", "trace_id", "duration_ms"):
            val = getattr(record, key, None)
            if val is not None:
                extra[key] = str(val)

        # Also capture any custom extra fields
        standard_attrs = logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()
        for key, val in record.__dict__.items():
            if key not in standard_attrs and key not in extra and not key.startswith("_"):
                extra[key] = str(val)

        return {
            "service": self.service,
            "level": record.levelname.lower(),
            "message": self.format(record) if self.formatter else record.getMessage(),
            "extra": {
                "logger": record.name,
                "module": record.module,
                "funcName": record.funcName,
                "lineno": record.lineno,
                **extra,
            },
        }

    def _flush_loop(self):
        while not self._shutdown.is_set():
            batch = self._drain_queue()
            if batch:
                self._send_batch(batch)
            self._shutdown.wait(timeout=self.flush_interval)

        # Final flush on shutdown
        batch = self._drain_queue()
        if batch:
            self._send_batch(batch)

    def _drain_queue(self) -> list:
        batch = []
        while len(batch) < self.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send_batch(self, batch: list):
        payload = json.dumps(batch).encode("utf-8")
        req = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=5) as resp:
                if resp.status < 400:
                    self._consecutive_failures = 0
                    return
        except (URLError, OSError, TimeoutError):
            pass

        self._consecutive_failures += 1
        if self._consecutive_failures <= 3:
            import sys
            print(
                f"[log-relay] Failed to send {len(batch)} entries to {self.url} "
                f"(attempt {self._consecutive_failures})",
                file=sys.stderr,
            )

    def close(self):
        self._shutdown.set()
        self._thread.join(timeout=5)
        super().close()


def setup_logging(
    service: str,
    level: int = logging.INFO,
    relay_url: Optional[str] = None,
    environment: Optional[str] = None,
    enable_relay: Optional[bool] = None,
) -> logging.Logger:
    """Configure Python logging with both console and log-relay output.

    Args:
        service: Service name (e.g. "automatos-backend", "agent-opt-worker")
        level: Logging level (default INFO)
        relay_url: Override LOG_RELAY_URL env var
        environment: Override ENVIRONMENT env var
        enable_relay: Override LOG_RELAY_ENABLED env var

    Returns:
        Root logger (already configured)
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler (always present)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        root.addHandler(console)

    # Log-relay handler (ships to Loki via log-relay)
    should_enable = enable_relay if enable_relay is not None else LOG_RELAY_ENABLED
    if should_enable:
        relay = LogRelayHandler(
            url=relay_url or LOG_RELAY_URL,
            service=service,
            environment=environment or ENVIRONMENT,
        )
        relay.setLevel(level)
        root.addHandler(relay)

    return root
