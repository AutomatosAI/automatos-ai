#!/usr/bin/env python3
"""Sync Auto's platform-management skill seed FROM the automatos-skills repo.

THE RULE (Gerard, 2026-08-29): skills are authored in the automatos-skills
repo ONLY — never edited live in the platform. This script is the one
sanctioned way content reaches orchestrator/core/seeds/platform-management-skill.md,
which is a GENERATED artifact:

    automatos-skills/team/auto/SKILL.md   (author + PR here)
        │  scripts/sync-auto-skill.py     (this script, run at version bumps)
        ▼
    orchestrator/core/seeds/platform-management-skill.md   (generated copy)
        │  seed_auto_agent.py             (new workspaces: seeds the row)
        │  _refresh_builtin_if_stale      (existing workspaces: hash-compare
        ▼                                  at load time, refresh inline)
    every workspace's Auto, next skill load after deploy

Both platform readers strip the YAML frontmatter and hash the markdown body,
so the file is written frontmatter-intact with a generated-file banner as the
first body line (the banner travels into the prompt — two short lines, and it
tells anyone reading the prompt where the truth lives).

Usage:
    python3 scripts/sync-auto-skill.py [path-to-skills-repo-SKILL.md]

Default source: ../automatos-skills/team/auto/SKILL.md (sibling checkout).
Self-checks after writing: the dispatch-contract fragment and all nine
doctrine anchors must be present, or the sync exits non-zero.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT.parent.parent.parent / "automatos-skills" / "team" / "auto" / "SKILL.md"
# When run from a normal checkout (not a worktree), the sibling lives one level up.
if not DEFAULT_SOURCE.exists():
    DEFAULT_SOURCE = REPO_ROOT.parent / "automatos-skills" / "team" / "auto" / "SKILL.md"
TARGET = REPO_ROOT / "orchestrator" / "core" / "seeds" / "platform-management-skill.md"

# The banner is INSIDE the body (after frontmatter) so both platform readers,
# which strip frontmatter before hashing, stay consistent with each other.
BANNER = (
    "<!-- GENERATED FILE — DO NOT EDIT IN THIS REPO. Source of truth:\n"
    "     automatos-skills/team/auto/SKILL.md (v{version}). Re-sync: python3 scripts/sync-auto-skill.py -->\n"
)

# Sanity anchors — the platform pins these (test_prd226_contract.py /
# test_prd226_doctrine.py). A sync that drops one must fail HERE, not in CI.
DOCTRINE_ANCHORS = [
    "Awareness",
    "Three lanes, chosen deliberately",
    "Delegate, don't implement",
    "Reuse before creating",
    "Dispatch as a contract",
    "Board as ledger",
    "Asks are decisions, not reports",
    "Recurring work becomes a Playbook",
    "Narrate",
]
CONTRACT_OPENER = "A dispatch contract has four parts, written so the owner needs nothing else to do the work:"


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source.exists():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        return 1

    raw = source.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        print("ERROR: source has no YAML frontmatter — refusing (wrong file?)", file=sys.stderr)
        return 1

    parts = raw.split("---", 2)
    if len(parts) < 3:
        print("ERROR: malformed frontmatter", file=sys.stderr)
        return 1
    frontmatter, body = parts[1], parts[2].lstrip("\n")

    m = re.search(r'^version:\s*"?([\d.]+)"?', frontmatter, re.M)
    version = m.group(1) if m else "unknown"

    new_body = BANNER.format(version=version) + "\n" + body
    out = f"---{frontmatter}---\n\n{new_body}"

    # Self-checks BEFORE writing.
    missing = [a for a in DOCTRINE_ANCHORS if a not in new_body]
    problems = []
    if missing:
        problems.append(f"doctrine anchors missing: {missing}")
    if CONTRACT_OPENER not in new_body:
        problems.append("dispatch-contract fragment opener missing")
    if problems:
        print("ERROR: refusing to write a seed that would fail the platform pins:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    old_hash = (
        hashlib.sha256(TARGET.read_text(encoding="utf-8").encode()).hexdigest()[:12]
        if TARGET.exists()
        else "(none)"
    )
    TARGET.write_text(out, encoding="utf-8")
    new_hash = hashlib.sha256(out.encode()).hexdigest()[:12]
    print(f"synced v{version}: {source}")
    print(f"  → {TARGET}")
    print(f"  file hash {old_hash} → {new_hash}  ({len(out)} bytes)")
    print("  self-checks: 9 doctrine anchors ✓  contract fragment ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
