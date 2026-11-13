"""Heartbeats (PRD-55): code split out of :mod:`services.heartbeat_service`.

- :mod:`services.heartbeats.triage_shadow`: the decision engine's shadow read of a
  heartbeat, beside the route the platform takes (PRD-248 S5).
- :mod:`services.heartbeats.auto_review`: ``report_to: auto``'s review ticket for Auto (PRD-72).
- :mod:`services.heartbeats.notification`: the dispatcher's pieces (PRD-128): the message, the
  source and its ``report_to``, the workspace fan-out.
- :mod:`services.heartbeats.session_ticks`: a Claude Code agent's heartbeat reports when its
  session ends, not when its ticket is queued.
"""
