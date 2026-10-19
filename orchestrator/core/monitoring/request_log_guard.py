"""The request-log guard (#1051, #1178): no request URL, and no SDK request body, in the logs.

httpx and its relatives log every request line at INFO with the full URL, and a URL
can carry a secret (Telegram's bot<token> reached Loki on 23 Sep). The model and
integration SDKs log a request's options, its body (the prompt) included, at DEBUG.
``RequestLogGuard`` drops both at the handlers. It imports nothing but ``logging``, so a
script that sets up its own logging (``logging.basicConfig``) can call
``guard_request_logs()`` without loading the app's config.
"""
import logging


# HTTP libraries whose INFO lines carry a full request URL, and a URL can carry a secret
# (Telegram's bot<token> reached Loki on 23 Sep): only their warnings and errors pass.
REQUEST_LIBRARIES = ("httpx", "httpcore", "httpx2", "httpcore2", "urllib3")
# Model and integration SDKs whose DEBUG lines carry a request's options, its body
# (the prompt) included: their retry lines (INFO, a relative path) pass, DEBUG never.
SDK_LIBRARIES = ("anthropic", "openai", "composio_client")


def _from(name: str, roots: tuple) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


class RequestLogGuard(logging.Filter):
    """A handler filter: a request library's line below WARNING, or an SDK's DEBUG line,
    never reaches the console or Loki. It sits on the handlers, not the loggers, because
    an SDK sets its own and httpx's logger level when it's imported (ANTHROPIC_LOG,
    COMPOSIO_LOG, OPENAI_LOG), whatever setup_logging set before."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _from(record.name, REQUEST_LIBRARIES):
            return record.levelno >= logging.WARNING
        if _from(record.name, SDK_LIBRARIES):
            return record.levelno >= logging.INFO
        return True


def guard_request_logs() -> None:
    """Put the guard on every handler the root logger has: ours, one an SDK's own
    basicConfig added first, or a script's ``logging.basicConfig``. A script that
    configures logging itself calls this right after. It covers the handlers there
    now: code that adds a root handler later (a log exporter, say) calls it again."""
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, RequestLogGuard) for f in handler.filters):
            handler.addFilter(RequestLogGuard())
