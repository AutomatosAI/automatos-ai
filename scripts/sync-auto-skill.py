#!/usr/bin/env python3
"""Sync Auto's platform skill seeds FROM the automatos-skills repo.

THE RULE (Gerard, 2026-08-29): skills are authored in the automatos-skills
repo ONLY — never edited live in the platform. This script is the one
sanctioned way content reaches the GENERATED seed artifacts under
orchestrator/core/seeds/.

PRD-231 (the context diet) splits Auto's one fat skill into two:

    automatos-skills/team/auto/SKILL.md       (the CHARTER — always-on)
    automatos-skills/team/auto-ops/SKILL.md    (the OPS cookbook — on-demand)
        │  scripts/sync-auto-skill.py          (this script, run at version bumps)
        ▼
    orchestrator/core/seeds/platform-management-skill.md   (generated copy)
    orchestrator/core/seeds/platform-operations-skill.md   (generated copy)
        │  seed_auto_agent.py                  (new workspaces: seeds the rows)
        │  _refresh_builtin_if_stale           (existing workspaces: hash-compare
        ▼                                       at load time, refresh inline)
    every workspace's Auto, next skill load after deploy

Both platform readers strip the YAML frontmatter and hash the markdown body,
so each file is written frontmatter-intact with a generated-file banner as the
FIRST BODY LINE (the banner travels into the prompt — it tells anyone reading
the prompt where the truth lives). The banner also records a sha256[:12] of the
SOURCE body (banner excluded) so the drift guard (PRD-231 US-005) can detect a
hand-edit to a seed WITHOUT the sibling repo present.

Per-file self-checks run BEFORE any write and REFUSE to write on failure — a
sync that would drop a platform pin fails HERE, not in CI.

Usage:
    python3 scripts/sync-auto-skill.py            # regenerate both seeds
    python3 scripts/sync-auto-skill.py --check     # no-write; exit non-zero on drift
    python3 scripts/sync-auto-skill.py --skills-repo /path/to/automatos-skills

The sibling automatos-skills checkout is auto-discovered by walking up from this
repo; --skills-repo overrides it.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = REPO_ROOT / "orchestrator" / "core" / "seeds"

# The banner is an HTML comment placed INSIDE the body (after frontmatter) so
# both platform readers — which strip frontmatter before hashing — stay
# byte-consistent with each other. Its markers are fixed so the drift guard can
# strip it back off to recover the source body it hashed.
_BANNER_OPEN = "<!-- GENERATED FILE"
_BANNER_CLOSE = "-->"
_SHA_LABEL = "source-body-sha256[:12]="
_SHA_RE = re.compile(re.escape(_SHA_LABEL) + r"([0-9a-f]{12})")


# ── Per-file self-check anchors ──────────────────────────────────────────────
# The platform pins these (test_prd226_contract.py / test_prd226_doctrine.py for
# the charter). A sync that drops one must fail HERE.
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
CONTRACT_OPENER = (
    "A dispatch contract has four parts, written so the owner needs nothing "
    "else to do the work:"
)
# The ops cookbook's own top-level header — it must live ONLY in the ops seed,
# never leak into the charter (that leak is exactly what PRD-231 undoes).
OPS_REFERENCE_HEADER = "# Platform Operations Reference"


def _check_charter(frontmatter: str, body: str) -> list[str]:
    """The charter seed keeps the doctrine + dispatch contract and must NOT carry
    the ops cookbook body."""
    problems: list[str] = []
    missing = [a for a in DOCTRINE_ANCHORS if a not in body]
    if missing:
        problems.append(f"doctrine anchors missing: {missing}")
    if CONTRACT_OPENER not in body:
        problems.append("dispatch-contract fragment opener missing")
    if OPS_REFERENCE_HEADER in body:
        problems.append(
            f"contains '{OPS_REFERENCE_HEADER}' — ops cookbook leaked into the charter"
        )
    return problems


def _check_ops(frontmatter: str, body: str) -> list[str]:
    """The ops seed is the tool-by-tool cookbook: it opens at section 0, runs to
    section 19, and its frontmatter names it platform-operations."""
    problems: list[str] = []
    if "## 0." not in body:
        problems.append("'## 0.' section head missing")
    if "## 19." not in body:
        problems.append("'## 19.' section head missing")
    if "name: platform-operations" not in frontmatter:
        problems.append("frontmatter name is not 'platform-operations'")
    return problems


@dataclass(frozen=True)
class Source:
    """One authored SKILL.md → one generated seed."""

    key: str  # human label for logs
    src_rel: str  # display path recorded in the banner (machine-independent)
    src_subpath: tuple  # path under the skills-repo root
    target: Path  # generated seed file
    check: Callable[[str, str], list[str]]  # (frontmatter, body) -> problems


SOURCES = (
    Source(
        key="platform-management (charter)",
        src_rel="automatos-skills/team/auto/SKILL.md",
        src_subpath=("team", "auto", "SKILL.md"),
        target=SEEDS_DIR / "platform-management-skill.md",
        check=_check_charter,
    ),
    Source(
        key="platform-operations (ops cookbook)",
        src_rel="automatos-skills/team/auto-ops/SKILL.md",
        src_subpath=("team", "auto-ops", "SKILL.md"),
        target=SEEDS_DIR / "platform-operations-skill.md",
        check=_check_ops,
    ),
)


# ── Pure helpers (imported by the US-005 drift-guard tests) ───────────────────

def find_skills_repo() -> Optional[Path]:
    """Locate the sibling automatos-skills checkout by walking up from this repo.

    Works from a normal checkout (``<ws>/automatos-ai``) and from a worktree
    (``<ws>/automatos-ai/.claude/worktrees/<name>``); returns None if not found.
    """
    for base in [REPO_ROOT, *REPO_ROOT.parents]:
        cand = base / "automatos-skills"
        if (cand / "team" / "auto" / "SKILL.md").exists():
            return cand
    return None


def split_frontmatter(raw: str) -> tuple[str, str]:
    """Return (frontmatter, body-after-frontmatter). Raises ValueError if absent.

    For a SEED the returned body still carries the leading banner; for a SOURCE
    it is the bare markdown. ``lstrip('\\n')`` mirrors the write side so a
    round-trip is byte-stable.
    """
    if not raw.startswith("---"):
        raise ValueError("no YAML frontmatter")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise ValueError("malformed frontmatter")
    return parts[1], parts[2].lstrip("\n")


def strip_banner(body_after_frontmatter: str) -> str:
    """Recover the source markdown body from a seed's frontmatter-stripped text
    by removing the leading generated-file banner comment (a no-op on a source
    body, which carries no banner)."""
    b = body_after_frontmatter.lstrip()
    if b.startswith(_BANNER_OPEN):
        idx = b.find(_BANNER_CLOSE)
        if idx != -1:
            return b[idx + len(_BANNER_CLOSE):].lstrip("\n")
    return body_after_frontmatter


def body_sha12(source_body: str) -> str:
    """sha256[:12] of the source markdown body (whitespace-normalized). This is
    the value recorded in the banner and recomputed by the drift guard."""
    return hashlib.sha256(source_body.strip().encode("utf-8")).hexdigest()[:12]


def extract_recorded_sha(body_after_frontmatter: str) -> Optional[str]:
    """The sha the banner claims, or None if the banner carries none."""
    m = _SHA_RE.search(body_after_frontmatter)
    return m.group(1) if m else None


def seed_drift(seed_text: str) -> tuple[Optional[str], str]:
    """(sha recorded in the seed's banner, sha recomputed from its body).

    A mismatch means the seed body was hand-edited after generation — the
    'never edit live in the platform' violation. Needs no sibling repo.
    """
    _fm, body = split_frontmatter(seed_text)
    recorded = extract_recorded_sha(body)
    actual = body_sha12(strip_banner(body))
    return recorded, actual


def _extract_version(frontmatter: str) -> str:
    m = re.search(r'^version:\s*"?([\d.]+)"?', frontmatter, re.M)
    return m.group(1) if m else "unknown"


def _build_banner(src_rel: str, version: str, sha12: str) -> str:
    return (
        "<!-- GENERATED FILE — DO NOT EDIT IN THIS REPO. Source of truth:\n"
        f"     {src_rel} (v{version}). Re-sync: python3 scripts/sync-auto-skill.py\n"
        f"     {_SHA_LABEL}{sha12} -->\n"
    )


def render_seed(source_text: str, src_rel: str) -> str:
    """Turn an authored SKILL.md into the generated seed text (frontmatter-intact,
    banner as the first body line, sha recorded). Raises ValueError on bad
    frontmatter."""
    frontmatter, body = split_frontmatter(source_text)
    version = _extract_version(frontmatter)
    banner = _build_banner(src_rel, version, body_sha12(body))
    new_body = banner + "\n" + body
    return f"---{frontmatter}---\n\n{new_body}"


def render_for_source(src: Source, skills_repo: Path) -> tuple[str, list[str]]:
    """(rendered seed text, self-check problems) for one source. Raises
    FileNotFoundError / ValueError on unreadable sources — self-checks run on the
    rendered body (banner + body), the exact text the platform readers see."""
    source_path = skills_repo.joinpath(*src.src_subpath)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_text = source_path.read_text(encoding="utf-8")
    seed_text = render_seed(source_text, src.src_rel)
    frontmatter, seed_body = split_frontmatter(seed_text)
    problems = src.check(frontmatter, seed_body)
    return seed_text, problems


# ── CLI operations ───────────────────────────────────────────────────────────

def sync(skills_repo: Path) -> int:
    rc = 0
    for src in SOURCES:
        try:
            seed_text, problems = render_for_source(src, skills_repo)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR [{src.key}]: unreadable source: {exc}", file=sys.stderr)
            rc = 1
            continue
        if problems:
            print(
                f"ERROR [{src.key}]: refusing to write a seed that would fail the "
                f"platform pins:",
                file=sys.stderr,
            )
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            rc = 1
            continue
        old_hash = (
            hashlib.sha256(src.target.read_text(encoding="utf-8").encode()).hexdigest()[:12]
            if src.target.exists()
            else "(none)"
        )
        src.target.write_text(seed_text, encoding="utf-8")
        new_hash = hashlib.sha256(seed_text.encode()).hexdigest()[:12]
        print(f"synced [{src.key}] v{_extract_version(seed_text.split('---', 2)[1])}")
        print(f"  → {src.target}")
        print(f"  file hash {old_hash} → {new_hash}  ({len(seed_text)} bytes)")
        print("  self-checks: ✓")
    return rc


def check(skills_repo: Path) -> int:
    """No-write freshness check: re-render each seed in memory and compare to
    disk. Exits non-zero naming any seed that differs (or whose source fails its
    self-checks)."""
    rc = 0
    for src in SOURCES:
        try:
            seed_text, problems = render_for_source(src, skills_repo)
        except (FileNotFoundError, ValueError) as exc:
            print(f"FAIL [{src.key}]: unreadable source: {exc}", file=sys.stderr)
            rc = 1
            continue
        if problems:
            print(f"FAIL [{src.key}]: source fails self-checks: {problems}", file=sys.stderr)
            rc = 1
            continue
        if not src.target.exists():
            print(f"DRIFT [{src.key}]: seed missing: {src.target}", file=sys.stderr)
            rc = 1
            continue
        if src.target.read_text(encoding="utf-8") != seed_text:
            print(
                f"DRIFT [{src.key}]: {src.target.name} differs from its source — "
                f"re-run: python3 scripts/sync-auto-skill.py",
                file=sys.stderr,
            )
            rc = 1
        else:
            print(f"OK [{src.key}]: {src.target.name} matches source")
    return rc


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync Auto's platform skill seeds.")
    parser.add_argument("--check", action="store_true", help="no-write; exit non-zero on drift")
    parser.add_argument("--skills-repo", type=Path, default=None, help="path to automatos-skills")
    args = parser.parse_args(argv)

    skills_repo = args.skills_repo or find_skills_repo()
    if skills_repo is None:
        if args.check:
            print(
                "SKIP: automatos-skills sibling not found — --check needs it; the "
                "sha-layer drift guard (tests) still covers hand-edits."
            )
            return 0
        print(
            "ERROR: automatos-skills sibling not found — pass --skills-repo <path>.",
            file=sys.stderr,
        )
        return 1

    return check(skills_repo) if args.check else sync(skills_repo)


if __name__ == "__main__":
    sys.exit(main())
