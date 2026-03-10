"""
Automatos Log Relay Client — Phase 2: Structured Observability

Drop-in logging handler that ships structured logs to the Automatos log-relay
service, which forwards them to Loki for querying via Grafana.

Features:
- Auto-captures ContextVars (request_id, workspace_id, user_id, agent_id, etc.)
- Error fingerprinting — groups identical errors by exception type + location
- Stack trace hashing — captures full call path for deduplication
- Structured context — every log carries method, path, module, function, line
- Correlation IDs — trace requests across service boundaries
- Async batching — never blocks the caller

Works on any platform — Railway, ECS, bare metal, local dev.

Usage:
    from automatos_logging import setup_logging
    setup_logging(service="automatos-backend")

    import logging
    logger = logging.getLogger(__name__)

    # Basic usage — ContextVars auto-captured from middleware:
    logger.info("Request processed")

    # With explicit context:
    logger.info("Agent heartbeat", extra={"agent_id": "cto", "duration_ms": 1523})

    # Errors auto-fingerprinted:
    try:
        do_something()
    except Exception:
        logger.exception("Failed to process")  # fingerprint + stack hash auto-added
"""

import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from contextvars import ContextVar
from typing import Optional


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

LOG_RELAY_URL = os.environ.get(
    "LOG_RELAY_URL",
    "http://log-relay.railway.internal:8080/push",
)
LOG_RELAY_ENABLED = os.environ.get("LOG_RELAY_ENABLED", "true").lower() == "true"
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown")
ENVIRONMENT = os.environ.get(
    "ENVIRONMENT", os.environ.get("RAILWAY_ENVIRONMENT", "development")
)

# Batch settings
BATCH_SIZE = int(os.environ.get("LOG_RELAY_BATCH_SIZE", "50"))
FLUSH_INTERVAL = float(os.environ.get("LOG_RELAY_FLUSH_INTERVAL", "2.0"))


# ─────────────────────────────────────────────
# Context Variables (set by middleware, read by handler)
# ─────────────────────────────────────────────

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
workspace_id_var: ContextVar[str] = ContextVar("workspace_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
agent_id_var: ContextVar[str] = ContextVar("agent_id", default="")
workflow_id_var: ContextVar[str] = ContextVar("workflow_id", default="")
run_id_var: ContextVar[str] = ContextVar("run_id", default="")
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="")
http_method_var: ContextVar[str] = ContextVar("http_method", default="")
http_path_var: ContextVar[str] = ContextVar("http_path", default="")


# ─────────────────────────────────────────────
# Error Fingerprinting
# ─────────────────────────────────────────────

def compute_error_fingerprint(exc_type: str, filename: str, func_name: str) -> str:
    """Stable fingerprint from exception type + location.

    Same exception in the same function = same fingerprint,
    regardless of error message content or timestamp.
    """
    raw = f"{exc_type}:{filename}:{func_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def compute_stack_hash(tb_text: str) -> str:
    """Hash of normalized traceback frames.

    Captures the full call path — two different entry points
    hitting the same underlying error get different stack hashes.
    """
    # Normalize: strip line numbers and variable content, keep structure
    lines = []
    for line in tb_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("File "):
            # Keep file + function, strip line number variations
            parts = stripped.split(",")
            if len(parts) >= 3:
                lines.append(f"{parts[0].strip()},{parts[2].strip()}")
            else:
                lines.append(stripped)
        elif stripped.startswith("in ") or stripped.startswith("raise "):
            lines.append(stripped)
    normalized = "\n".join(lines)
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def extract_error_info(record: logging.LogRecord) -> Optional[dict]:
    """Extract structured error info from a log record with exception."""
    if not record.exc_info or record.exc_info[0] is None:
        return None

    exc_type, exc_value, exc_tb = record.exc_info
    type_name = exc_type.__name__ if exc_type else "Unknown"
    message = str(exc_value) if exc_value else ""

    # Get traceback text
    tb_text = ""
    if exc_tb:
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    # Extract location from the innermost frame
    filename = record.pathname or ""
    func_name = record.funcName or ""
    if exc_tb:
        # Walk to the deepest frame for the actual error location
        tb_frame = exc_tb
        while tb_frame.tb_next:
            tb_frame = tb_frame.tb_next
        filename = tb_frame.tb_frame.f_code.co_filename
        func_name = tb_frame.tb_frame.f_code.co_name

    fingerprint = compute_error_fingerprint(type_name, filename, func_name)
    stack_hash = compute_stack_hash(tb_text) if tb_text else ""

    return {
        "type": type_name,
        "message": message[:500],  # Truncate long error messages
        "fingerprint": fingerprint,
        "stack_hash": stack_hash,
        "traceback": tb_text[:5000] if tb_text else "",  # Truncate massive traces
    }


# ─────────────────────────────────────────────
# Log Relay Handler
# ─────────────────────────────────────────────

class LogRelayHandler(logging.Handler):
    """Async logging handler that batches and ships structured logs to log-relay.

    Auto-captures ContextVars set by middleware + error fingerprints.
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
        # Build context from ContextVars (auto-captured from middleware)
        context = {}
        for var_name, ctx_var in _CONTEXT_VARS.items():
            val = ctx_var.get("")
            if val:
                context[var_name] = val

        # Override with explicit extra fields (extra={} takes precedence)
        for key in _EXTRA_FIELDS:
            val = getattr(record, key, None)
            if val is not None:
                context[key] = str(val)

        # Capture any non-standard extra fields
        standard_attrs = _STANDARD_RECORD_ATTRS
        for key, val in record.__dict__.items():
            if (
                key not in standard_attrs
                and key not in context
                and not key.startswith("_")
            ):
                context[key] = str(val)

        # Module info — always included
        context["module"] = record.module
        context["function"] = record.funcName
        context["lineno"] = record.lineno
        context["logger"] = record.name

        # Error info — auto-fingerprinted
        error = extract_error_info(record)

        # Build the structured entry
        entry = {
            "service": self.service,
            "level": record.levelname.lower(),
            "message": self.format(record) if self.formatter else record.getMessage(),
            "timestamp": record.created,
            "context": context,
        }

        if error:
            entry["error"] = error

        # Metrics context (if present)
        metrics = {}
        for metric_key in ("duration_ms", "tokens_in", "tokens_out", "cost"):
            val = getattr(record, metric_key, None)
            if val is not None:
                metrics[metric_key] = val
        if context.get("duration_ms"):
            metrics["duration_ms"] = int(context.pop("duration_ms"))
        if metrics:
            entry["metrics"] = metrics

        return entry

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
        from urllib.request import Request, urlopen
        from urllib.error import URLError

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
            print(
                f"[log-relay] Failed to send {len(batch)} entries to {self.url} "
                f"(attempt {self._consecutive_failures})",
                file=sys.stderr,
            )

    def close(self):
        self._shutdown.set()
        self._thread.join(timeout=5)
        super().close()


# ─────────────────────────────────────────────
# ContextVar → extra field mapping
# ─────────────────────────────────────────────

_CONTEXT_VARS = {
    "request_id": request_id_var,
    "correlation_id": correlation_id_var,
    "workspace_id": workspace_id_var,
    "user_id": user_id_var,
    "agent_id": agent_id_var,
    "workflow_id": workflow_id_var,
    "run_id": run_id_var,
    "tenant_id": tenant_id_var,
    "method": http_method_var,
    "path": http_path_var,
}

_EXTRA_FIELDS = frozenset({
    "request_id", "correlation_id", "workspace_id", "user_id",
    "agent_id", "workflow_id", "run_id", "tenant_id",
    "method", "path", "task_id", "trace_id", "duration_ms",
    "tokens_in", "tokens_out", "cost", "model",
})

# Cache standard LogRecord attributes to avoid re-creating every time
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
)


# ─────────────────────────────────────────────
# Setup Function
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Middleware Helper
# ─────────────────────────────────────────────

def set_request_context(
    request_id: str = "",
    workspace_id: str = "",
    user_id: str = "",
    method: str = "",
    path: str = "",
    correlation_id: str = "",
    agent_id: str = "",
):
    """Set ContextVars for the current async context.

    Call this from FastAPI middleware to auto-enrich all logs
    within the request lifecycle. No need to pass extra={} at call sites.

    Usage in middleware:
        from automatos_logging import set_request_context
        set_request_context(
            request_id=request.state.request_id,
            workspace_id=str(ctx.workspace_id),
            user_id=ctx.user.id,
            method=request.method,
            path=request.url.path,
        )
    """
    if request_id:
        request_id_var.set(request_id)
    if workspace_id:
        workspace_id_var.set(workspace_id)
    if user_id:
        user_id_var.set(user_id)
    if method:
        http_method_var.set(method)
    if path:
        http_path_var.set(path)
    if correlation_id:
        correlation_id_var.set(correlation_id)
    if agent_id:
        agent_id_var.set(agent_id)
