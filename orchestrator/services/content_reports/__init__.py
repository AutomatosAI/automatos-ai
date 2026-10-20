"""Content reports: a member reports an Auto message or a Deliverable (PRD-266 §4.8, D15).

A report is one ``audit_logs`` row (action ``content_reported``; no table of its own),
an in-app notice to the workspace's owner and, on SaaS, a short summary to the
Automatos moderation inbox. The owner reads reports in the existing audit view.

This module holds the report contract only, so the router, the target resolver and
the service share one set of names without importing each other:

* ``targets`` finds the reported content in the caller's workspace;
* ``service`` files the report: rate limit, target, duplicate check, audit row, notice;
* ``moderation`` posts a new report's summary to the Automatos inbox (SaaS only).
"""

from __future__ import annotations

# What can be reported (K8). The app's "Report this" offers these two.
REPORT_TARGET_TYPES: tuple[str, ...] = ("message", "deliverable")

# Why it is reported (K8): a kit choice the app's picker follows.
REPORT_REASONS: tuple[str, ...] = ("harmful", "offensive", "inaccurate", "privacy", "other")

# The longest excerpt of the reported content a report keeps or sends anywhere.
EXCERPT_MAX_CHARS = 200

# The longest note a reporter may add.
NOTE_MAX_CHARS = 500

# A target id is a UUID today; the bound keeps a malformed one short.
TARGET_ID_MAX_CHARS = 64

# The audit row's action, and the in-app event the owner's notice is raised as.
AUDIT_ACTION = "content_reported"

__all__ = [
    "AUDIT_ACTION",
    "EXCERPT_MAX_CHARS",
    "NOTE_MAX_CHARS",
    "REPORT_REASONS",
    "REPORT_TARGET_TYPES",
    "TARGET_ID_MAX_CHARS",
]
