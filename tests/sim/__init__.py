"""Automatos simulation harness (PRD-247, P0 — the instrument).

Drives the local platform the way a customer would — creates a throwaway
workspace, seeds agents from a pack, files tasks and talks to Auto, answers the
questions the agents raise, then scores the night on four rows: usability,
cost, quality, usefulness. Report-only: it never changes the product.

Standard library only, so the scheduled job runs on the operator's system
``python3`` with no virtualenv. Run from the repo root::

    python3 -m tests.sim.night run --pack smoke

Outputs live under ``~/.automatos-sim/`` (never in the repo).
"""

__version__ = "0.1.0"
