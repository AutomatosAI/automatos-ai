"""PRD-239 S1 — the agent's soul rides its Claude Code session: description,
persona and skill bodies rendered stable per agent and capped; the claim carries
it, and asks the same host to resume the conversation's previous session (S2).

Pure units: fake rows, fake session, no Postgres."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services.cli_session_prompt import (
    GAP_LINE_PREFIX,
    OMITTED_NOTE,
    SESSION_TOOLS_AVAILABLE,
    SESSION_UNAVAILABLE_TOOL_PREFIXES,
    SKILLS_HEADER,
    TOOLS_HEADER,
    TRUNCATED_NOTE,
    session_system_prompt,
    session_tool_gaps,
    skills_block,
    tools_block,
)

WS = uuid4()


def _skill(sid, name, desc, body, active=True):
    return SimpleNamespace(id=sid, name=name, description=desc, prompt_template=body,
                           is_active=active, content_hash=None, tools_schema=None)


def _agent(**over):
    base = dict(
        id=15, name="Bob", description="Automatos AI specialist.",
        use_custom_persona=True, custom_persona_prompt="Blunt and precise.",
        persona=SimpleNamespace(system_prompt="Row persona"),
        skills=[
            _skill(1, "automatos-platform", "Knows the platform.", "Body A"),
            _skill(2, "writing", "Writes briefs.", "Body B"),
        ],
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── the soul ─────────────────────────────────────────────────────────────────

def test_soul_has_description_persona_and_full_skill_bodies_in_that_order():
    text = session_system_prompt(_agent())
    assert text.index("## About you") < text.index("## Persona & Communication Style") < text.index(SKILLS_HEADER)
    assert "Automatos AI specialist." in text and "Blunt and precise." in text
    assert "### automatos-platform\nKnows the platform.\n\nBody A" in text
    assert "### writing\nWrites briefs.\n\nBody B" in text
    assert "Row persona" not in text  # the custom persona wins, as on the API path


def test_persona_row_is_used_when_no_custom_persona():
    text = session_system_prompt(_agent(use_custom_persona=False))
    assert "Row persona" in text and "Blunt and precise." not in text


def test_the_soul_is_stable_per_agent():
    assert session_system_prompt(_agent()) == session_system_prompt(_agent())


def test_skill_cap_truncates_the_first_and_omits_the_rest_by_name():
    agent = _agent(skills=[_skill(1, "big", "First.", "x" * 500), _skill(2, "second", "Second.", "y" * 500)])
    block = skills_block(agent, max_chars=200)
    assert block.startswith(SKILLS_HEADER)
    assert TRUNCATED_NOTE in block and "x" * 10 in block
    assert "### second\nSecond.\n" + OMITTED_NOTE in block and "y" * 10 not in block


def test_inactive_and_duplicate_skills_are_skipped():
    agent = _agent(skills=[_skill(1, "a", "d", "B"), _skill(1, "a", "d", "B"), _skill(2, "b", "d", "C", active=False)])
    assert skills_block(agent).count("### ") == 1


def test_an_agent_with_nothing_to_say_still_learns_the_ticket_sessions_tools():
    bare = SimpleNamespace(id=1, name="X", description=None, use_custom_persona=False,
                           custom_persona_prompt=None, persona=None, skills=[])
    # PRD-245 S0.6: a ticket session is policy-gated — the gate is worth knowing even without a soul…
    assert session_system_prompt(bare) == tools_block(bare)
    # …the Canvas terminal has no gate (PRD-239 S7 v2): soul only, and nothing when there is none.
    assert session_system_prompt(bare, ticket_session=False) == ""
    assert session_system_prompt(None) == ""


# ── PRD-245 S0.6: the prompt tells the truth about tools ─────────────────────

RESEARCH_BODY = (
    "Search with `composio_execute` (COMPOSIO_SEARCH), check `search_knowledge` for what we already "
    "know, then file the brief with `platform_submit_report`. Retry platform_submit_report once on failure. "
    "Never touch platform_ tools you were not given; scratchpads are for drafts."
)


def _researcher():
    return _agent(id=57, name="RESEARCHER", description="Finds things out.",
                  skills=[_skill(9, "web-research", "Researches the web.", RESEARCH_BODY),
                          _skill(10, "writing", "Writes briefs.", "Keep it short.")])


def test_the_tools_block_names_what_a_session_has_and_what_it_has_not():
    text = session_system_prompt(_researcher())
    block = text[text.index(TOOLS_HEADER):text.index(SKILLS_HEADER)]
    assert "- Files:" in block and "- Bash:" in block and "- Web:" in block
    assert "`git status`" in block and "`sort`" in block and "`pytest`" in block   # the host's allowlist, rendered
    assert "may be HELD until the operator" in block and "Questions tab" in block  # F167: not every host holds
    # Only the families the session does NOT have, and only in the unavailable
    # line — asserting over the whole block let `composio_execute` and
    # `search_knowledge` match the AVAILABLE line, so a regression that listed
    # them as unavailable again (the contradiction W1/W3 created) would pass.
    unavailable_line = next(l for l in block.splitlines() if "NOT available in a session" in l)
    offered = set(SESSION_TOOLS_AVAILABLE)
    for prefix in SESSION_UNAVAILABLE_TOOL_PREFIXES:
        label = f"`{prefix}*`" if prefix.endswith("_") else f"`{prefix}`"
        if prefix.endswith("_") or prefix not in offered:
            assert label in unavailable_line, label
        else:
            assert label not in unavailable_line, f"{label} is offered — it must not be called unavailable"
    assert "NOT available in a session" in block and "do not wait for them" in block
    assert text.index("## About you") < text.index(TOOLS_HEADER) < text.index(SKILLS_HEADER)


def test_the_researchers_skill_header_names_what_it_asks_for_that_a_session_works_differently_on():
    """With NO tools offered (the W0 state) all three are simply missing. What the
    prompt actually renders is the W1 state — ``SESSION_TOOLS_AVAILABLE`` — where
    two of the three have a session equivalent and only one truly does not."""
    agent = _researcher()
    assert session_tool_gaps(agent, ()) == [
        {"skill": "web-research", "tools": ["composio_execute", "search_knowledge", "platform_submit_report"]},
    ]
    # What the prompt renders follows the bridge's CURRENT tool list, so derive
    # the expectation from it rather than freezing one wave's snapshot: each name
    # with an equivalent is pointed at it, each name without one is named as
    # absent, and a skill with nothing to say gets no line at all.
    from services.cli_session_prompt import SESSION_TOOLS_AVAILABLE

    rendered = session_system_prompt(agent)
    assert "### web-research\nResearches the web.\n" in rendered
    assert "### writing\nWrites briefs.\n\nKeep it short." in rendered
    live = session_tool_gaps(agent, SESSION_TOOLS_AVAILABLE)
    assert [g["skill"] for g in live] == ["web-research"]          # the writing skill names none
    entry = live[0]
    for mentioned, replacement in (entry.get("instead") or {}).items():
        assert f"`{replacement}` instead of `{mentioned}`" in rendered
    if entry.get("tools"):
        assert GAP_LINE_PREFIX in rendered
        for missing in entry["tools"]:
            assert f"`{missing}`" in rendered
    else:
        assert GAP_LINE_PREFIX not in rendered                    # nothing it names is missing


def test_gaps_shrink_with_what_the_session_offers():
    """The bridge adds tools once per deploy; the gap lines follow by themselves.
    A name with a session EQUIVALENT moves from "cannot call" to "call this
    instead" — the work is available, under another name."""
    gaps = session_tool_gaps(_researcher(), ["search_knowledge", "submit_report"])
    assert gaps == [{"skill": "web-research", "tools": ["composio_execute"],
                     "instead": {"platform_submit_report": "submit_report"}}]
    assert session_tool_gaps(_researcher(), ["composio_execute", "search_knowledge", "platform_submit_report"]) == []


def test_a_skill_with_no_platform_names_has_no_gap():
    assert session_tool_gaps(_agent(), ()) == []                  # bodies "Body A" / "Body B"
    assert GAP_LINE_PREFIX not in session_system_prompt(_agent())
    inactive = _agent(skills=[_skill(1, "a", "d", "call platform_x", active=False)])
    assert session_tool_gaps(inactive, ()) == []


def test_two_renders_are_byte_identical_and_carry_no_ticket_id():
    import re

    agent = _researcher()
    first, second = session_system_prompt(agent), session_system_prompt(agent)
    assert first == second
    assert re.search(r"#\d", first) is None and "ticket #" not in first.lower()
    assert "57" not in first.replace("RESEARCHER", "")           # no agent id either


def test_the_agents_own_bash_allowlist_is_named():
    agent = _researcher()
    agent.configuration = {"runtime": "cli", "allowed_tools": ["pip --version", " poetry run "]}
    text = session_system_prompt(agent)
    assert "This agent's own allowlist adds `pip --version`, `poetry run`." in text
    assert "own allowlist" not in session_system_prompt(_researcher())


def test_an_omitted_skill_keeps_its_gap_line():
    agent = _agent(skills=[_skill(1, "big", "First.", "x" * 500),
                           _skill(2, "second", "Second.", "use platform_board_summary " + "y" * 500)])
    block = skills_block(agent, max_chars=200)
    # ``platform_board_summary`` has a session equivalent, so the line points at it
    assert ("### second\nSecond.\nIn a session, call `board_summary` instead of "
            "`platform_board_summary`.\n" + OMITTED_NOTE) in block


# ── the claim ────────────────────────────────────────────────────────────────

from services import cli_host_service as svc  # noqa: E402


class _Query:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result


class _DB:
    def __init__(self, agent):
        self.agent = agent
        self.commits = 0

    def query(self, model):
        return _Query(self.agent)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        pass


def _task(host_id, **over):
    base = dict(
        id=7, workspace_id=WS, assigned_agent_id=15, blocked_reason=None, attempts=1,
        title="Chat with Bob: hi", raw_prompt=None, description="hi", review_feedback=None,
        review_mode="auto", attachment_ids=[], lease_until="lease", status="in_progress",
        source_type="chat", source_id="chat:c1:m1",
        runtime_ref={"resume_session_id": "sess-1", "resume_host_id": str(host_id)},
    )
    base.update(over)
    return SimpleNamespace(**base)


def _claim_env(monkeypatch, task, soul="SOUL"):
    import services.cli_ticket_lane as lane

    monkeypatch.setattr(svc, "claim_tasks", lambda db, **kw: [task])
    monkeypatch.setattr(svc, "_blocked_pending_approval", lambda db, t: False)
    monkeypatch.setattr(svc, "explorer_root_for", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_session_system_prompt", lambda agent: soul)


def _agent_row():
    return SimpleNamespace(id=15, name="Bob", configuration={"runtime": "cli", "provider": "claude"})


def test_claim_carries_the_soul_and_resumes_on_the_same_host(monkeypatch):
    host = SimpleNamespace(id=uuid4(), workspace_id=WS)
    task = _task(host.id)
    _claim_env(monkeypatch, task)
    out = svc.claim_for_host(_DB(_agent_row()), host, 1)["tasks"][0]
    assert out["system_prompt"] == "SOUL"
    assert out["resume_session_id"] == "sess-1" and out["session_id"]
    assert task.runtime_ref["resume_session_id"] == "sess-1" and task.runtime_ref["host_id"] == str(host.id)


def test_claim_never_resumes_a_session_another_host_ran(monkeypatch):
    host = SimpleNamespace(id=uuid4(), workspace_id=WS)
    task = _task(uuid4())  # the previous session lived on a different machine
    _claim_env(monkeypatch, task)
    out = svc.claim_for_host(_DB(_agent_row()), host, 1)["tasks"][0]
    assert out["resume_session_id"] is None and "resume_session_id" not in task.runtime_ref


def test_a_prompt_rendering_failure_never_blocks_a_claim(monkeypatch):
    import services.cli_session_prompt as prompt_mod

    def boom(agent):
        raise RuntimeError("template exploded")

    monkeypatch.setattr(prompt_mod, "session_system_prompt", boom)
    assert svc._session_system_prompt(_agent_row()) == ""
    assert svc._session_system_prompt(None) == ""


def test_the_sessions_real_directory_always_wins(monkeypatch):
    """PRD-239: the SessionStart / result cwd (a --worktree for a repo) replaces the
    configured one, and the explorer root follows it."""
    monkeypatch.setattr(svc, "explorer_root_for", lambda task_id, cwd, ws, projects: f"root-for:{cwd}")
    ref = {"cwd": "/repo", "explorer_root": "projects/repo"}
    svc._record_session_cwd(ref, SimpleNamespace(id=7, workspace_id=WS), "/repo/.claude/worktrees/automatos-7")
    assert ref["cwd"] == "/repo/.claude/worktrees/automatos-7"
    assert ref["explorer_root"] == "root-for:/repo/.claude/worktrees/automatos-7"


def test_host_contract_version_moved_with_the_claim_shape():
    # 0.8.0 (2026-09-17, PRD-245 W1): the claim carries session_tools,
    # session_tools_path and a per-ticket session_token.
    # 0.7.0 (2026-09-11, CLI adapter design): capabilities carry every CLI under
    # ``clis`` with served/reason; ``providers`` = the served ids.
    assert svc.EXPECTED_CLI_HOST_VERSION == "0.8.0"
