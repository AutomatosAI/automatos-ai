"""Proactive Auto (PRD-268): the services that read the database for a check-in.

The rules are pure functions in :mod:`modules.proactive`; these services read what they
decide on.

- :mod:`services.proactive.signals`: what changed in a workspace over a check-in's
  window, the signals Auto's workspace check-in starts from (§6.2.1).
"""
