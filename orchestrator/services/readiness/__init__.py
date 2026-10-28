"""PRD-268 §6.11.3 — readiness: what Auto depends on, and what couldn't be checked.

One probe per server item (``probes``: core, model_route, web_search, push), each bounded by
``config.READINESS_PROBE_TIMEOUT_SECONDS``; the runner and the held check (``service``);
``Readiness@1`` (``contract``); the copy in English and Portuguese (Brazil) (``strings``).
Served by ``api/readiness.py``. The client adds microphone, speech and the phone's own push.
"""
