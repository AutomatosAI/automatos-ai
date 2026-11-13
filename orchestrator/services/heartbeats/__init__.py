"""Heartbeats (PRD-55): code split out of :mod:`services.heartbeat_service`.

- :mod:`services.heartbeats.triage_shadow`: the decision engine's shadow read of a
  heartbeat, beside the route the platform takes (PRD-248 S5).
- :mod:`services.heartbeats.auto_review`: ``report_to: auto``'s review ticket for Auto, and a
  Claude Code agent's heartbeat filing it when its session ends, not when it is queued.
"""
