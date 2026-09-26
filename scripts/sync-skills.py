#!/usr/bin/env python3
"""Sync the NAMED built-in skills FROM the automatos-skills repo.

THE RULE (Gerard, 2026-08-29): skills are authored in the automatos-skills repo
ONLY, never edited in the platform. This script is the one sanctioned way a
built-in skill's content reaches this repo, and every file it writes is a
GENERATED artifact:

    automatos-skills/<source>                  (author + PR here)
        │  scripts/sync-skills.py <name> ...   (this script: ONLY the skills named)
        ▼
    orchestrator/core/seeds/<seed>             (generated copy)
        │  seed_builtin_skills                 (boot, and Auto's seeding: creates a missing row)
        │  _refresh_builtin_if_stale           (the loader: hash-compare at load, refresh inline)
        ▼
    the global skill row every workspace reads

orchestrator/core/seeds/skills/manifest.json lists the built-in skills: name →
seed file → automatos-skills source. There is no sync-everything mode. The skills
repo can be ahead of what the platform is ready for: its team/auto/SKILL.md is
v2.4.0 (PRD-231's split, which moved the ops cookbook into team/auto-ops) while
this platform stays on v2.3.0 until PRD-231 lands, so a blanket sync would
silently strip Auto's ops cookbook. platform-management is synced only when it
is named, and then only if Auto's doctrine checks pass.

Both platform readers strip the YAML frontmatter and hash the markdown body, so
each seed is written frontmatter-intact with a generated-file banner as the
first body line (the banner travels into the prompt: two short lines, and it
tells anyone reading the prompt where the truth lives).

Usage:
    python3 scripts/sync-skills.py social-video-director social-ops
    python3 scripts/sync-skills.py platform-management
    python3 scripts/sync-skills.py --skills-repo ~/src/automatos-skills social-ops

Default --skills-repo: the sibling automatos-skills checkout, read from its
WORKING TREE (no push needed). A name the manifest does not list, a missing
source, a source without YAML frontmatter (or whose frontmatter has no
description, or names another skill), or a failed self-check refuses the whole
run: nothing is written.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
_ORCHESTRATOR = str(REPO_ROOT / "orchestrator")
if _ORCHESTRATOR not in sys.path:
    sys.path.insert(0, _ORCHESTRATOR)

from core.builtin_skills import MANIFEST_PATH, BuiltinSkill, ManifestError, load_manifest  # noqa: E402

SKILLS_REPO_NAME = "automatos-skills"

# The banner is INSIDE the body (after frontmatter) so both platform readers,
# which strip frontmatter before hashing, stay consistent with each other.
BANNER = (
    "<!-- GENERATED FILE — DO NOT EDIT IN THIS REPO. Source of truth:\n"
    "     automatos-skills/{source} (v{version}). Re-sync: python3 scripts/sync-skills.py {name} -->\n"
)

# Auto's sanity anchors: the platform pins these (test_prd226_contract.py /
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

_NAME = re.compile(r"""^name:\s*["']?(.*?)["']?\s*$""", re.M)
_DESCRIPTION = re.compile(r"^description:\s*\S", re.M)
_VERSION = re.compile(r'^version:\s*"?([\d.]+)"?', re.M)


def auto_doctrine_problems(body: str) -> List[str]:
    missing = [anchor for anchor in DOCTRINE_ANCHORS if anchor not in body]
    problems = [f"doctrine anchors missing: {missing}"] if missing else []
    if CONTRACT_OPENER not in body:
        problems.append("dispatch-contract fragment opener missing")
    return problems


# Extra checks for the skills that carry platform pins, by name.
SELF_CHECKS: Dict[str, Callable[[str], List[str]]] = {
    "platform-management": auto_doctrine_problems,
}
SELF_CHECK_REPORT = {
    "platform-management": f"{len(DOCTRINE_ANCHORS)} doctrine anchors ✓  contract fragment ✓",
}


class Planned(NamedTuple):
    entry: BuiltinSkill
    source: Path
    version: str
    text: str


def default_skills_repo() -> Path:
    """The sibling checkout: beside this repo, or beside the workspace when this
    is a worktree (<workspace>/.worktrees/<repo>/<branch>)."""
    beside = REPO_ROOT.parent / SKILLS_REPO_NAME
    if beside.is_dir():
        return beside
    return REPO_ROOT.parent.parent.parent / SKILLS_REPO_NAME


def plan(entry: BuiltinSkill, skills_repo: Path) -> Tuple[Optional[Planned], List[str]]:
    """The seed file ``entry`` gets, or why it gets none. Reads, never writes."""
    source = skills_repo / entry.source
    if not source.is_file():
        return None, [f"source not found: {source}"]
    raw = source.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        return None, ["source has no YAML frontmatter — refusing (wrong file?)"]
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return None, ["malformed frontmatter"]
    frontmatter, body = parts[1], parts[2].lstrip("\n")

    named = _NAME.search(frontmatter)
    if not named or named.group(1) != entry.name:
        found = named.group(1) if named else "no skill"
        return None, [f"frontmatter names {found!r}, not {entry.name!r} — wrong source?"]
    if not _DESCRIPTION.search(frontmatter):
        return None, ["frontmatter has no description (the skill's L1 trigger text)"]

    version_match = _VERSION.search(frontmatter)
    version = version_match.group(1) if version_match else "unknown"
    new_body = BANNER.format(source=entry.source, version=version, name=entry.name) + "\n" + body
    check = SELF_CHECKS.get(entry.name)
    problems = check(new_body) if check else []
    if problems:
        return None, problems
    return Planned(entry, source, version, f"---{frontmatter}---\n\n{new_body}"), []


def _short_hash(path: Path) -> str:
    if not path.exists():
        return "(none)"
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()[:12]


def write(planned: Planned) -> None:
    target = planned.entry.seed_path
    old_hash = _short_hash(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(planned.text, encoding="utf-8")
    new_hash = hashlib.sha256(planned.text.encode()).hexdigest()[:12]
    print(f"synced {planned.entry.name} v{planned.version}: {planned.source}")
    print(f"  → {target}")
    print(f"  file hash {old_hash} → {new_hash}  ({len(planned.text)} bytes)")
    report = SELF_CHECK_REPORT.get(planned.entry.name)
    if report:
        print(f"  self-checks: {report}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync the named built-in skills from the automatos-skills repo into this repo's seeds.",
    )
    parser.add_argument(
        "names", nargs="+", metavar="NAME",
        help="built-in skills to sync, as named in the manifest (there is no sync-everything mode)",
    )
    parser.add_argument(
        "--skills-repo", type=Path, default=None,
        help="the automatos-skills checkout (default: the sibling checkout)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_PATH,
        help="the built-in skills manifest (default: the platform's)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    skills_repo = args.skills_repo or default_skills_repo()
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    names = list(dict.fromkeys(args.names))
    unknown = [name for name in names if name not in manifest]
    if unknown:
        print(f"ERROR: not built-in skills (add them to {args.manifest} first): {unknown}", file=sys.stderr)
        return 1

    results = [(name, *plan(manifest[name], skills_repo)) for name in names]
    refused = [(name, problems) for name, _planned, problems in results if problems]
    if refused:
        print("ERROR: refusing — nothing written:", file=sys.stderr)
        for name, problems in refused:
            for problem in problems:
                print(f"  - {name}: {problem}", file=sys.stderr)
        return 1

    for _name, planned, _problems in results:
        write(planned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
