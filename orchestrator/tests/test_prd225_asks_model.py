"""PRD-225 US-001 — the asks model: approval_grants extended into questions.

Pure tests (no Postgres): the alembic single-head invariant, the new columns +
``kind`` default on the SQLAlchemy model, the ``to_dict`` surface, and the
migration's chain + up/down shape. THE wave migration must land on ONE head.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


# ---------------------------------------------------------------------------
# The single-head invariant — house rule after the 4-heads incident.
# ---------------------------------------------------------------------------

def test_alembic_single_head():
    """After the wave's only migration, ``alembic heads`` must be exactly one."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", str(_ORCH / "alembic"))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    assert len(heads) == 1, f"expected a single alembic head, got {heads}"


def test_prd225_revision_chains_onto_prior_head():
    """The new revision descends from the prior single head and defines both
    directions — a rival head or a missing downgrade is a hard failure."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", str(_ORCH / "alembic"))
    rev = ScriptDirectory.from_config(cfg).get_revision("prd225_s1_asks_on_grants")

    # Assert the INTENT, not a literal parent: this revision must be the single
    # head and must descend from whatever the prior head was. Hard-coding the
    # parent SHA-name breaks on every rebase onto a moved main (it did, when the
    # PRD-222 merges buried prd185_s1b_toollog_user_nullable mid-chain), which
    # would tempt a future rebase into "fixing" the test instead of the chain.
    assert rev.down_revision, "the revision must descend from the prior head, not sit at base"
    heads = tuple(ScriptDirectory.from_config(cfg).get_heads())
    assert heads == ("prd225_s1_asks_on_grants",), (
        f"must be the SINGLE head (rival heads are the trap this guards): {heads}"
    )
    assert callable(rev.module.upgrade) and callable(rev.module.downgrade)


def test_prd225_is_reachable_from_the_head():
    """Walking down from the head reaches the new revision — it is on the live
    chain, not an orphan file."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", str(_ORCH / "alembic"))
    sd = ScriptDirectory.from_config(cfg)
    (head,) = sd.get_heads()
    chain = {rev.revision for rev in sd.walk_revisions(base="base", head=head)}
    assert "prd225_s1_asks_on_grants" in chain


# ---------------------------------------------------------------------------
# The model surface — additive; the classic approval row is unchanged.
# ---------------------------------------------------------------------------

def test_asks_columns_present_on_model():
    from core.models.approval_grants import ApprovalGrant

    g = ApprovalGrant(
        workspace_id=uuid4(), subject_type="board_task", subject_id="42",
    )
    for col in (
        "kind", "question_md", "options", "answer_text", "answered_by",
        "answered_at", "asked_by_agent_id", "channel_refs",
    ):
        assert hasattr(g, col), f"ApprovalGrant missing PRD-225 column {col}"


def test_kind_defaults_to_approval_and_is_not_null():
    """Existing rows and flows are untouched: ``kind`` defaults to 'approval'
    at both the Python and server layer, and is NOT NULL."""
    from core.models.approval_grants import ApprovalGrant, KIND_APPROVAL

    kind_col = ApprovalGrant.__table__.c.kind
    assert kind_col.nullable is False
    assert kind_col.default is not None and kind_col.default.arg == KIND_APPROVAL
    assert kind_col.server_default is not None
    assert kind_col.server_default.arg == KIND_APPROVAL


def test_kind_constants():
    from core.models.approval_grants import KIND_APPROVAL, KIND_QUESTION

    assert KIND_APPROVAL == "approval"
    assert KIND_QUESTION == "question"
    assert KIND_APPROVAL != KIND_QUESTION


def test_to_dict_carries_ask_fields():
    """The list/detail surface exposes the ask fields; a bare row degrades
    cleanly (kind → 'approval', channel_refs → {})."""
    from core.models.approval_grants import ApprovalGrant, KIND_QUESTION

    g = ApprovalGrant(
        workspace_id=uuid4(), subject_type="board_task", subject_id="7",
        kind=KIND_QUESTION, question_md="Ship it? **A/B**",
        options=["A", "B"], asked_by_agent_id=9,
    )
    d = g.to_dict()
    assert d["kind"] == KIND_QUESTION
    assert d["question_md"] == "Ship it? **A/B**"
    assert d["options"] == ["A", "B"]
    assert d["asked_by_agent_id"] == 9
    assert d["answer_text"] is None
    assert d["answered_at"] is None
    assert d["channel_refs"] == {}

    # A classic approval row (no kind set) still reads as an approval.
    plain = ApprovalGrant(
        workspace_id=uuid4(), subject_type="board_task", subject_id="8",
    ).to_dict()
    assert plain["kind"] == "approval"
    assert plain["question_md"] is None
