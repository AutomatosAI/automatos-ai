"""PRD-226 — The Manager's Doctrine (US-003).

Pure, LLM-free coverage of the 4-part dispatch contract:

- ONE shared fragment (DISPATCH_CONTRACT_FRAGMENT), defined once and imported by
  BOTH the planner prompt builder and the PRD-224 ASSIGN directive — a copy-paste
  would be a hard failure (identity + single-definition-site tests);
- the planner parser extracts `definition_of_done` into the task spec and it
  rides the EXISTING input_context JSONB — no schema change, rebuild-don't-mutate;
- verification scores against the DoD when present and is byte-for-byte identical
  to before when it is absent (the sacred back-compat guarantee).

No live model is called — _build_judge_prompt and the planner parse are pure.
"""

import inspect
import os
from pathlib import Path

from modules.coordination.deterministic_checks import DeterministicResult
from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT
from modules.coordination.verification import VerificationService, _build_judge_prompt

_ROOT = Path(__file__).resolve().parent.parent
_SKIP_DIRS = {"tests", "__pycache__", ".git", "node_modules", "alembic", ".venv", "venv"}
_SKILL_MD = _ROOT / "core" / "seeds" / "platform-management-skill.md"


# ---------------------------------------------------------------------------
# P226-RVW-2 — the always-on skill seed must not carry a hand-copy of the
# contract that drifts from the code fragment behind CI's back. The single-
# source guards above walk .py files only; the skill's Markdown home was
# invisible to them. It now embeds DISPATCH_CONTRACT_FRAGMENT verbatim, so this
# fails the moment the fragment is edited without mirroring the skill.
# ---------------------------------------------------------------------------

def test_skill_seed_embeds_the_fragment_verbatim():
    """The platform-management skill (always-on, injected into every Auto turn,
    refreshed to existing installs by skill_loader) carries the 4-part contract
    as the SAME text as the code fragment — not an independently-maintained copy.
    Verbatim lock-step: drift fails CI."""
    skill_md = _SKILL_MD.read_text(encoding="utf-8")
    assert DISPATCH_CONTRACT_FRAGMENT in skill_md, (
        "platform-management-skill.md §17.5 no longer embeds DISPATCH_CONTRACT_FRAGMENT "
        "verbatim — the skill's dispatch-contract text has drifted from the single "
        "source in modules/coordination/dispatch_contract.py"
    )


# ---------------------------------------------------------------------------
# AC1 — one fragment definition, imported by ≥2 consumers (never copy-pasted)
# ---------------------------------------------------------------------------

def _iter_py_files():
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def test_fragment_has_exactly_one_definition_site():
    """grep proof: the fragment is ASSIGNED in exactly one file (no copies)."""
    defs = [
        p.relative_to(_ROOT).as_posix()
        for p in _iter_py_files()
        if "DISPATCH_CONTRACT_FRAGMENT = " in p.read_text(encoding="utf-8")
    ]
    assert defs == ["modules/coordination/dispatch_contract.py"], defs


def test_fragment_imported_by_planner_and_assign():
    importers = [
        p.relative_to(_ROOT).as_posix()
        for p in _iter_py_files()
        if "import DISPATCH_CONTRACT_FRAGMENT" in p.read_text(encoding="utf-8")
    ]
    assert "modules/coordination/planner.py" in importers
    assert "consumers/chatbot/auto.py" in importers
    assert len(importers) >= 2


def test_both_consumers_bind_the_same_object():
    """Runtime proof of single-source: every name is the SAME object."""
    from consumers.chatbot.auto import DISPATCH_CONTRACT_FRAGMENT as from_assign
    from modules.coordination.planner import DISPATCH_CONTRACT_FRAGMENT as from_planner
    assert from_assign is DISPATCH_CONTRACT_FRAGMENT
    assert from_planner is DISPATCH_CONTRACT_FRAGMENT


def test_fragment_is_four_parts():
    for part in ("OBJECTIVE", "OUTPUT", "TOOLS", "BOUNDARIES"):
        assert part in DISPATCH_CONTRACT_FRAGMENT
    assert "definition of done" in DISPATCH_CONTRACT_FRAGMENT.lower()


def test_planner_prompt_embeds_the_fragment():
    from modules.coordination.planner import _OUTPUT_SCHEMA_INSTRUCTIONS
    assert DISPATCH_CONTRACT_FRAGMENT in _OUTPUT_SCHEMA_INSTRUCTIONS
    assert "definition_of_done" in _OUTPUT_SCHEMA_INSTRUCTIONS


def test_assign_directive_embeds_the_fragment_both_paths():
    from consumers.chatbot.auto import build_assign_directive
    resolved = build_assign_directive(target_agent_name="Jim", resolved=True, deferred=False)
    ask = build_assign_directive(target_agent_name=None, resolved=False, deferred=False)
    assert DISPATCH_CONTRACT_FRAGMENT in resolved
    assert DISPATCH_CONTRACT_FRAGMENT in ask
    # 224's behaviour is intact around the shared fragment:
    assert "file this as a board ticket" in resolved and "in_progress" in resolved
    assert "confirm the agent first" in ask and "Do NOT guess or auto-pick" in ask


# ---------------------------------------------------------------------------
# AC2 — planner parse stores definition_of_done in the existing JSONB spec
# ---------------------------------------------------------------------------

def _one_task_plan(**extra):
    task = {
        "temp_id": "task_1",
        "title": "Draft the report",
        "description": "OBJECTIVE: … OUTPUT: … TOOLS: … BOUNDARIES: …",
        "agent_role": "researcher",
        "sequence_number": 1,
        "task_type": "llm_generation",
    }
    task.update(extra)
    return {"tasks": [task]}


def test_parse_extracts_definition_of_done():
    from modules.coordination.planner import _parse_plan
    errors = []
    tasks, _ = _parse_plan(_one_task_plan(definition_of_done="A 3-section report, cited"), errors)
    assert errors == []
    assert tasks[0].definition_of_done == "A 3-section report, cited"


def test_parse_absent_dod_is_none():
    from modules.coordination.planner import _parse_plan
    errors = []
    tasks, _ = _parse_plan(_one_task_plan(), errors)
    assert errors == []
    assert tasks[0].definition_of_done is None


def test_parse_blank_dod_is_none():
    from modules.coordination.planner import _parse_plan
    errors = []
    tasks, _ = _parse_plan(_one_task_plan(definition_of_done="   "), errors)
    assert tasks[0].definition_of_done is None


def test_planned_task_is_frozen_rebuild_not_mutate():
    """rebuild-don't-mutate is structurally enforced: PlannedTask is frozen."""
    import dataclasses
    from modules.coordination.planner import _parse_plan
    tasks, _ = _parse_plan(_one_task_plan(definition_of_done="x"), [])
    try:
        object.__setattr__  # sanity
        raised = False
        try:
            tasks[0].definition_of_done = "mutated"
        except dataclasses.FrozenInstanceError:
            raised = True
        assert raised, "PlannedTask must be frozen (rebuild-don't-mutate)"
    finally:
        pass


def test_no_schema_change_dod_rides_input_context_jsonb():
    """DoD lives inside the EXISTING input_context JSONB — no new column."""
    from core.models.orchestration import OrchestrationTask
    cols = {c.name for c in OrchestrationTask.__table__.columns}
    assert "definition_of_done" not in cols, "DoD must NOT be a new column"
    assert "input_context" in cols, "DoD rides the existing input_context JSONB"


# ---------------------------------------------------------------------------
# AC3 — verification consumes DoD when present; byte-identical when absent
# ---------------------------------------------------------------------------

def _judge_kwargs():
    return dict(
        task_title="Draft the report",
        task_description="Write it up",
        output="Here is the finished report with content.",
        verification_criteria=None,
        deterministic_result=DeterministicResult(passed=True),
    )


def test_judge_prompt_absent_dod_is_byte_identical():
    """No DoD ⇒ the prompt is byte-for-byte what it was before this story."""
    base = _build_judge_prompt(**_judge_kwargs())
    explicit_none = _build_judge_prompt(**_judge_kwargs(), definition_of_done=None)
    blank = _build_judge_prompt(**_judge_kwargs(), definition_of_done="   ")
    assert base == explicit_none == blank
    assert "Definition of Done" not in base
    # the bytes between the criteria and the deterministic sections are untouched
    assert (
        "## Verification Criteria\nNone specified.\n\n## Deterministic Check Results"
        in base
    )


def test_judge_prompt_with_dod_scores_against_it():
    with_dod = _build_judge_prompt(
        **_judge_kwargs(), definition_of_done="Must include exactly 3 sections"
    )
    assert "## Definition of Done" in with_dod
    assert "Must include exactly 3 sections" in with_dod
    assert with_dod != _build_judge_prompt(**_judge_kwargs())


def test_verify_task_accepts_definition_of_done_kwarg():
    sig = inspect.signature(VerificationService.verify_task)
    assert "definition_of_done" in sig.parameters
    assert sig.parameters["definition_of_done"].default is None
