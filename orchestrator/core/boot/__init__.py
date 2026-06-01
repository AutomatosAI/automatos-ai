"""Boot-time tasks (PRD-142 Wave 1 · WS-C).

Houses startup seeds and the orphaned-run reaper so that boot work is
importable, unit-testable, and observable rather than buried in ``main.py`` as
fire-and-forget closures.
"""
