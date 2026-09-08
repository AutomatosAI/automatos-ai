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

from services.cli_session_prompt import (  # noqa: E402
    OMITTED_NOTE,
    SKILLS_HEADER,
    TRUNCATED_NOTE,
    session_system_prompt,
    skills_block,
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


def test_an_agent_with_nothing_to_say_renders_nothing():
    bare = SimpleNamespace(id=1, name="X", description=None, use_custom_persona=False,
                           custom_persona_prompt=None, persona=None, skills=[])
    assert session_system_prompt(bare) == ""
    assert session_system_prompt(None) == ""


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


def _claim_env(monkeypatch, task, predecessor=None, soul="SOUL"):
    import services.cli_ticket_lane as lane

    monkeypatch.setattr(svc, "claim_tasks", lambda db, **kw: [task])
    monkeypatch.setattr(svc, "_blocked_pending_approval", lambda db, t: False)
    monkeypatch.setattr(svc, "explorer_root_for", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_session_system_prompt", lambda agent: soul)
    monkeypatch.setattr(lane, "running_predecessor_of", lambda db, t: predecessor)


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


def test_claim_holds_a_chat_ticket_while_its_predecessor_still_runs(monkeypatch):
    host = SimpleNamespace(id=uuid4(), workspace_id=WS)
    task = _task(host.id)
    _claim_env(monkeypatch, task, predecessor=SimpleNamespace(id=3))
    out = svc.claim_for_host(_DB(_agent_row()), host, 1)
    assert out["tasks"] == [] and out["parked"] == []
    assert task.status == "assigned" and task.lease_until is None
    assert task.runtime_ref["resume_session_id"] == "sess-1"  # the hint survives the release


def test_a_chat_ticket_filed_mid_turn_resumes_the_session_that_has_since_ended(monkeypatch):
    """The message arrived while the previous turn still ran (no session to name at
    filing); by claim time that turn has ended — continue it, on the same host."""
    import services.cli_ticket_lane as lane

    host = SimpleNamespace(id=uuid4(), workspace_id=WS)
    task = _task(host.id, runtime_ref=None)  # no hint from the filing
    _claim_env(monkeypatch, task)
    monkeypatch.setattr(lane, "previous_session_of", lambda db, ws, cid, aid: ("sess-prev", str(host.id)))
    out = svc.claim_for_host(_DB(_agent_row()), host, 1)["tasks"][0]
    assert out["resume_session_id"] == "sess-prev" and task.runtime_ref["resume_session_id"] == "sess-prev"
    # …but never a session another host ran, and never for a non-chat ticket.
    monkeypatch.setattr(lane, "previous_session_of", lambda db, ws, cid, aid: ("sess-prev", "other-host"))
    task2 = _task(host.id, runtime_ref=None)
    _claim_env(monkeypatch, task2)
    assert svc.claim_for_host(_DB(_agent_row()), host, 1)["tasks"][0]["resume_session_id"] is None
    heartbeat = _task(host.id, runtime_ref=None, source_type="heartbeat", source_id="agent:15")
    _claim_env(monkeypatch, heartbeat)
    monkeypatch.setattr(lane, "previous_session_of", lambda db, ws, cid, aid: ("sess-prev", str(host.id)))
    assert svc.claim_for_host(_DB(_agent_row()), host, 1)["tasks"][0]["resume_session_id"] is None


def test_a_prompt_rendering_failure_never_blocks_a_claim(monkeypatch):
    import services.cli_session_prompt as prompt_mod

    def boom(agent):
        raise RuntimeError("template exploded")

    monkeypatch.setattr(prompt_mod, "session_system_prompt", boom)
    assert svc._session_system_prompt(_agent_row()) == ""
    assert svc._session_system_prompt(None) == ""


def test_host_contract_version_moved_with_the_claim_shape():
    assert svc.EXPECTED_CLI_HOST_VERSION == "0.3.0"
