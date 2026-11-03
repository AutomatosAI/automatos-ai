"""PRD-270 — Sync with Automatos Cloud: the hosted catalogue a local install reads, and the local side's pull.

The hosted edition (SaaS) serves the catalogue (``catalog.py``) and its files; a local install pulls it,
checks every item's signature and lists what verifies. ``urls.py`` holds the catalogue's paths, which both
sides read the same way.
"""
