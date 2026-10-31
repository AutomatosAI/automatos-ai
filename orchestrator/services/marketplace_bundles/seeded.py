"""PRD-270 W2 (K9) — the Marketplace rows every local edition seeds itself, which Automatos Cloud never exports.

* An agent is seeded when it has a slug: every seeder keys its marketplace agents by slug
  (seed_shopify_agents.py, seed_socials_package.py) and the Agent model calls slug the seeding key,
  while a community agent never has one (``api/marketplace.py`` ``submit_item`` sets none). A
  Playbook's step on a seeded agent names it by that slug, which a local install resolves to its own.
* A Playbook is seeded when its template id is one of ``SOCIALS_PLAYBOOKS``' (seed_socials_package.py),
  the only marketplace Playbooks a seeder writes.
* A package is seeded when its slug is one of ``PACKAGES``' (seed_packages.py); a package has no submit
  route, so every other package row is one an admin made on Automatos Cloud.

The seed modules are read lazily, on the first export, never when this module is imported: the bundle
format stays importable on its own.
"""
from __future__ import annotations

from functools import lru_cache
from typing import FrozenSet, Optional


def is_seeded_agent(slug: Optional[str]) -> bool:
    """Whether a marketplace agent with this slug is one a local edition seeds: it has a slug at all."""
    return slug is not None


@lru_cache(maxsize=1)
def seeded_template_ids() -> FrozenSet[str]:
    """The template ids of the marketplace Playbooks every local edition seeds (SOCIALS_PLAYBOOKS')."""
    from core.seeds.seed_socials_package import SOCIALS_PLAYBOOKS

    return frozenset(spec["template_id"] for spec in SOCIALS_PLAYBOOKS)


@lru_cache(maxsize=1)
def seeded_package_slugs() -> FrozenSet[str]:
    """The slugs of the packages every local edition seeds (seed_packages.PACKAGES')."""
    from core.seeds.seed_packages import PACKAGES

    return frozenset(spec["slug"] for spec in PACKAGES)


__all__ = ["is_seeded_agent", "seeded_package_slugs", "seeded_template_ids"]
