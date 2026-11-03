"""PRD-270 §5.7 — the public Marketplace's reads: what a visitor with no account sees of each tab.

The hosted edition answers ``GET /api/public/marketplace/{type}`` (``api/public_marketplace.py``) from
these readers. Each type runs its own query over the global catalogue rows and maps each row, field by
field, through its allowlist in ``fields.py``; ``reads.py`` names the reader of each type. Themes are not
here: the catalogue (``services/marketplace_cloud/catalog.py``) is their public read.
"""
