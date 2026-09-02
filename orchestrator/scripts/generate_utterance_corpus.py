"""
PRD-232 US-005 — synthetic utterance corpus: format contract + coverage linter.
================================================================================

This is a **regeneration aid, not a generator**. It authors NOTHING and calls
NO LLM or network service. The corpus itself
(``orchestrator/core/seeds/utterances/<category>.yaml``) is hand-authored seed
data, committed to the repo (decision §6.3). This script:

  1. loads the live ActionRegistry (lightweight — leaf-loaded so the heavy
     ``modules/tools/__init__`` transformers/torch chain is never imported),
  2. AST-parses AutoBrain's ``_PLATFORM_KEYWORDS`` phrase map out of
     ``consumers/chatbot/auto.py`` (no import — the map is a literal dict),
  3. parses every corpus YAML file,
  4. validates format + coverage and prints a per-action authoring checklist.

Coverage contract (enforced here and by
``tests/test_prd232_us005_utterance_corpus.py``):

  * super_admin_only actions are EXCLUDED (fail-closed — read from the registry
    flag, never a hardcoded list).
  * >= 15 utterances for >= 90% of non-su actions.
  * every ``ActionDefinition.examples`` string appears in its action's corpus
    (source: example).
  * every ``_PLATFORM_KEYWORDS`` phrase whose key maps to a registered non-su
    action (directly, or via the legacy recipe->playbook remap) appears in that
    action's corpus (source: phrase_map). Phrases keyed on su-only or
    unregistered actions are reported as skipped — the corpus deliberately
    cannot house them.

Usage (human-run aid — never against a DB, it touches none):

    cd orchestrator
    python -m scripts.generate_utterance_corpus            # full checklist
    python -m scripts.generate_utterance_corpus --summary  # one-line verdict
    python -m scripts.generate_utterance_corpus --json     # machine report
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── paths ────────────────────────────────────────────────────────────────────
_ORCH_ROOT = Path(__file__).resolve().parent.parent            # orchestrator/
_DISCOVERY_DIR = _ORCH_ROOT / "modules" / "tools" / "discovery"
_AUTO_PY = _ORCH_ROOT / "consumers" / "chatbot" / "auto.py"
CORPUS_DIR = _ORCH_ROOT / "core" / "seeds" / "utterances"

CORPUS_SCHEMA_VERSION = 1
MIN_UTTERANCES = 15          # per-action floor (spec: 15-25)
SOFT_MAX_UTTERANCES = 30     # over this is a warning, not a failure
COVERAGE_FLOOR_PCT = 90.0    # >= this share of non-su actions must hit the floor

VALID_SOURCES = frozenset({"authored", "example", "phrase_map"})

# Legacy AutoBrain phrase-map keys (pre-canonical "recipe" naming, PRD unregistered)
# whose vocabulary migrates onto the canonical Playbook actions. Keeps the phrase
# map's real coverage alive in the corpus without resurrecting dead action names.
PHRASE_MAP_LEGACY_REMAP: Dict[str, str] = {
    "platform_list_recipes": "platform_list_playbooks",
    "platform_execute_recipe": "platform_execute_playbook",
    "platform_get_recipe_execution": "platform_get_playbook_execution",
    "platform_delete_recipe": "platform_delete_playbook",
}


# ── registry (lightweight leaf-load — no transformers/torch) ─────────────────
_LEAF_PKG = "_prd232_corpus_discovery"


def _leaf_load_registry_module():
    """Load the discovery ActionRegistry + platform_actions under a synthetic
    package so ``modules/tools/__init__`` (execution -> agents -> llm ->
    sentence_transformers) is never executed. Every ``actions_*.py`` imports
    only ``.action_registry``, so the whole registration graph is stdlib-light."""
    if _LEAF_PKG not in sys.modules:
        pkg = types.ModuleType(_LEAF_PKG)
        pkg.__path__ = [str(_DISCOVERY_DIR)]
        sys.modules[_LEAF_PKG] = pkg

    def _leaf(mod_name: str):
        full = f"{_LEAF_PKG}.{mod_name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, _DISCOVERY_DIR / f"{mod_name}.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = _LEAF_PKG
        sys.modules[full] = module
        spec.loader.exec_module(module)   # relative imports resolve via _LEAF_PKG.__path__
        return module

    action_registry = _leaf("action_registry")
    platform_actions = _leaf("platform_actions")   # pulls every actions_*.py via the package path
    return action_registry, platform_actions


@dataclass(frozen=True)
class ActionMeta:
    name: str
    category: str
    description: str
    super_admin_only: bool
    admin_only: bool
    promoted: bool
    examples: tuple
    param_enums: Dict[str, tuple]   # param name -> enum values
    tags: tuple


def load_registry_actions() -> List[ActionMeta]:
    """Every registered ActionDefinition as a lightweight, hashable snapshot.

    Includes su-only actions — callers filter with ``super_admin_only`` so the
    exclusion stays fail-closed and reads the live flag."""
    action_registry, platform_actions = _leaf_load_registry_module()
    reg = action_registry.ActionRegistry()
    platform_actions.register_all_actions(reg)
    reg._initialized = True  # we registered manually; block get_all()'s re-init
    out: List[ActionMeta] = []
    for a in reg._actions.values():
        props = (a.parameters or {}).get("properties", {}) or {}
        enums = {
            pname: tuple(pspec["enum"])
            for pname, pspec in props.items()
            if isinstance(pspec, dict) and isinstance(pspec.get("enum"), list)
        }
        out.append(ActionMeta(
            name=a.name,
            category=a.category,
            description=a.description,
            super_admin_only=bool(a.super_admin_only),
            admin_only=bool(a.admin_only),
            promoted=bool(a.promoted),
            examples=tuple(a.examples or ()),
            param_enums=enums,
            tags=tuple(a.tags or ()),
        ))
    return out


def load_phrase_map() -> Dict[str, List[str]]:
    """AST-extract ``_PLATFORM_KEYWORDS`` from auto.py (no import — literal dict)."""
    tree = ast.parse(_AUTO_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_PLATFORM_KEYWORDS":
                    return {k: list(v) for k, v in ast.literal_eval(node.value).items()}
    raise RuntimeError("_PLATFORM_KEYWORDS not found in auto.py")


# ── corpus loading ───────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


@dataclass
class Utterance:
    text: str
    source: str


@dataclass
class CorpusFile:
    path: Path
    category: str
    version: int
    actions: Dict[str, List[Utterance]] = field(default_factory=dict)


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> List[CorpusFile]:
    """Parse every ``<category>.yaml`` file. Raises on structural violations
    (they are contract breaches, not soft coverage gaps)."""
    import yaml  # local: keep module import cheap for callers that only lint metadata

    files: List[CorpusFile] = []
    for path in sorted(corpus_dir.glob("*.yaml")):
        if path.name.startswith("_"):
            continue  # reserved for non-corpus helpers (e.g. manifests)
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name}: top level must be a mapping")
        category = raw.get("category")
        version = raw.get("version")
        actions_block = raw.get("actions")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{path.name}: missing/invalid 'category'")
        if version != CORPUS_SCHEMA_VERSION:
            raise ValueError(f"{path.name}: 'version' must be {CORPUS_SCHEMA_VERSION}, got {version!r}")
        if category != path.stem:
            raise ValueError(f"{path.name}: 'category' {category!r} != filename stem {path.stem!r}")
        if not isinstance(actions_block, dict):
            raise ValueError(f"{path.name}: 'actions' must be a mapping")
        cf = CorpusFile(path=path, category=category, version=version)
        for action_name, utt_list in actions_block.items():
            if not isinstance(utt_list, list) or not utt_list:
                raise ValueError(f"{path.name}: action {action_name!r} must have a non-empty list")
            parsed: List[Utterance] = []
            seen_norm = set()
            for item in utt_list:
                if not isinstance(item, dict) or "text" not in item or "source" not in item:
                    raise ValueError(f"{path.name}:{action_name}: each utterance needs 'text' and 'source'")
                text = item["text"]
                source = item["source"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{path.name}:{action_name}: empty utterance text")
                if source not in VALID_SOURCES:
                    raise ValueError(
                        f"{path.name}:{action_name}: source {source!r} not in {sorted(VALID_SOURCES)}"
                    )
                norm = _normalize(text)
                if norm in seen_norm:
                    raise ValueError(f"{path.name}:{action_name}: duplicate utterance {text!r}")
                seen_norm.add(norm)
                parsed.append(Utterance(text=text, source=source))
            cf.actions[action_name] = parsed
        files.append(cf)
    return files


# ── validation ───────────────────────────────────────────────────────────────
@dataclass
class ActionReport:
    name: str
    category: str
    count: int
    needed: int                       # how many more to reach MIN_UTTERANCES
    missing_examples: List[str] = field(default_factory=list)
    missing_phrases: List[str] = field(default_factory=list)
    mistagged: List[str] = field(default_factory=list)


@dataclass
class CorpusReport:
    ok: bool
    errors: List[str]
    warnings: List[str]
    n_nonsu_actions: int
    n_at_floor: int
    coverage_pct: float
    per_action: Dict[str, ActionReport]
    su_present: List[str]
    unknown_actions: List[str]
    skipped_phrase_keys: Dict[str, str]


def validate(
    corpus_dir: Path = CORPUS_DIR,
    actions: Optional[List[ActionMeta]] = None,
    phrase_map: Optional[Dict[str, List[str]]] = None,
) -> CorpusReport:
    actions = actions if actions is not None else load_registry_actions()
    phrase_map = phrase_map if phrase_map is not None else load_phrase_map()
    corpus_files = load_corpus(corpus_dir)

    by_name = {a.name: a for a in actions}
    nonsu = [a for a in actions if not a.super_admin_only]
    nonsu_names = {a.name for a in nonsu}
    su_names = {a.name for a in actions if a.super_admin_only}

    # Flatten corpus: action_name -> list[Utterance]  (files partition by category
    # but we index by action so cross-file mistakes surface).
    corpus: Dict[str, List[Utterance]] = {}
    corpus_category: Dict[str, str] = {}
    for cf in corpus_files:
        for action_name, utts in cf.actions.items():
            corpus.setdefault(action_name, []).extend(utts)
            corpus_category[action_name] = cf.category

    errors: List[str] = []
    warnings: List[str] = []

    # su-only actions must never appear (fail-closed).
    su_present = sorted(n for n in corpus if n in su_names)
    for n in su_present:
        errors.append(f"super_admin_only action {n!r} present in corpus (must be excluded)")

    # unknown / stale action names.
    unknown_actions = sorted(n for n in corpus if n not in by_name)
    for n in unknown_actions:
        errors.append(f"corpus action {n!r} is not a registered action")

    # category placement sanity (warning only).
    for n, cat in corpus_category.items():
        if n in by_name and by_name[n].category != cat:
            warnings.append(
                f"{n!r} lives in {cat}.yaml but registry category is {by_name[n].category!r}"
            )

    # Build required phrase sets per non-su action (direct keys + legacy remap).
    required_phrases: Dict[str, List[str]] = {}
    skipped_phrase_keys: Dict[str, str] = {}
    for key, phrases in phrase_map.items():
        target = key if key in nonsu_names else PHRASE_MAP_LEGACY_REMAP.get(key)
        if target and target in nonsu_names:
            required_phrases.setdefault(target, []).extend(phrases)
        elif key in su_names:
            skipped_phrase_keys[key] = "su-only action (excluded from corpus)"
        else:
            skipped_phrase_keys[key] = "unregistered action name (no corpus home)"

    per_action: Dict[str, ActionReport] = {}
    n_at_floor = 0
    for a in nonsu:
        utts = corpus.get(a.name, [])
        norms = {_normalize(u.text) for u in utts}
        count = len(utts)
        if count >= MIN_UTTERANCES:
            n_at_floor += 1
        if count > SOFT_MAX_UTTERANCES:
            warnings.append(f"{a.name}: {count} utterances (> soft max {SOFT_MAX_UTTERANCES})")

        missing_examples = [ex for ex in a.examples if _normalize(ex) not in norms]
        missing_phrases = [
            ph for ph in dict.fromkeys(required_phrases.get(a.name, []))
            if _normalize(ph) not in norms
        ]

        # Provenance sanity: a text tagged example/phrase_map must actually be one.
        example_norms = {_normalize(ex) for ex in a.examples}
        phrase_norms = {_normalize(ph) for ph in required_phrases.get(a.name, [])}
        mistagged: List[str] = []
        for u in utts:
            if u.source == "example" and _normalize(u.text) not in example_norms:
                mistagged.append(f"{u.text!r} tagged 'example' but not in registry examples")
            if u.source == "phrase_map" and _normalize(u.text) not in phrase_norms:
                mistagged.append(f"{u.text!r} tagged 'phrase_map' but not a mapped phrase")

        per_action[a.name] = ActionReport(
            name=a.name,
            category=a.category,
            count=count,
            needed=max(0, MIN_UTTERANCES - count),
            missing_examples=missing_examples,
            missing_phrases=missing_phrases,
            mistagged=mistagged,
        )
        for ex in missing_examples:
            errors.append(f"{a.name}: example not in corpus: {ex!r}")
        for ph in missing_phrases:
            errors.append(f"{a.name}: phrase-map phrase not in corpus: {ph!r}")
        for m in mistagged:
            errors.append(f"{a.name}: {m}")

    coverage_pct = (100.0 * n_at_floor / len(nonsu)) if nonsu else 0.0
    if coverage_pct < COVERAGE_FLOOR_PCT:
        errors.append(
            f"coverage {coverage_pct:.1f}% of non-su actions have >= {MIN_UTTERANCES} "
            f"utterances (floor {COVERAGE_FLOOR_PCT:.0f}%)"
        )

    ok = not errors
    return CorpusReport(
        ok=ok,
        errors=errors,
        warnings=warnings,
        n_nonsu_actions=len(nonsu),
        n_at_floor=n_at_floor,
        coverage_pct=coverage_pct,
        per_action=per_action,
        su_present=su_present,
        unknown_actions=unknown_actions,
        skipped_phrase_keys=skipped_phrase_keys,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def _print_checklist(report: CorpusReport) -> None:
    print(f"PRD-232 utterance corpus — {CORPUS_DIR}")
    print(f"  non-su actions:        {report.n_nonsu_actions}")
    print(f"  at >= {MIN_UTTERANCES} utterances:   {report.n_at_floor} "
          f"({report.coverage_pct:.1f}%, floor {COVERAGE_FLOOR_PCT:.0f}%)")
    print(f"  skipped phrase keys:   {len(report.skipped_phrase_keys)} "
          f"(su-only / unregistered)")
    print()
    incomplete = [
        r for r in report.per_action.values()
        if r.needed or r.missing_examples or r.missing_phrases or r.mistagged
    ]
    if incomplete:
        print("── per-action authoring checklist (incomplete actions) ──")
        for r in sorted(incomplete, key=lambda x: (x.category, x.name)):
            print(f"  [{r.category}] {r.name}: {r.count} utterances")
            if r.needed:
                print(f"      + author {r.needed} more to reach {MIN_UTTERANCES}")
            for ex in r.missing_examples:
                print(f"      + missing example (source: example): {ex!r}")
            for ph in r.missing_phrases:
                print(f"      + missing phrase  (source: phrase_map): {ph!r}")
            for m in r.mistagged:
                print(f"      ! mistagged: {m}")
    else:
        print("All non-su actions complete: floor met, every example + phrase present.")

    if report.warnings:
        print("\n── warnings ──")
        for w in report.warnings:
            print(f"  ! {w}")

    print("\nVERDICT:", "PASS" if report.ok else "FAIL")
    if not report.ok:
        print(f"  {len(report.errors)} error(s). First few:")
        for e in report.errors[:8]:
            print(f"    - {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PRD-232 utterance corpus linter (no LLM, no network)")
    parser.add_argument("--summary", action="store_true", help="one-line verdict only")
    parser.add_argument("--json", action="store_true", help="emit the machine report as JSON")
    args = parser.parse_args(argv)

    report = validate()

    if args.json:
        import json
        print(json.dumps({
            "ok": report.ok,
            "n_nonsu_actions": report.n_nonsu_actions,
            "n_at_floor": report.n_at_floor,
            "coverage_pct": report.coverage_pct,
            "errors": report.errors,
            "warnings": report.warnings,
            "su_present": report.su_present,
            "unknown_actions": report.unknown_actions,
            "skipped_phrase_keys": report.skipped_phrase_keys,
        }, indent=2))
    elif args.summary:
        print(f"{'PASS' if report.ok else 'FAIL'}: "
              f"{report.n_at_floor}/{report.n_nonsu_actions} non-su actions at floor "
              f"({report.coverage_pct:.1f}%), {len(report.errors)} errors")
    else:
        _print_checklist(report)

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
