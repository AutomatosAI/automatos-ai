"""PRD-226 — The Manager's Doctrine (US-002).

Pure, LLM-free coverage of the assessment-rubric extension. The three-lane
rubric (PRD-224 US-004) is extended IN PLACE with reuse-before-create signals
and a one-line lane-narration instruction — one rubric definition site, no
duplication. No live model is called (build_assessment_prompt is a pure string
builder).
"""

from pathlib import Path

from consumers.chatbot.auto import build_assessment_prompt

_AUTO_PY = Path(__file__).resolve().parent.parent / "consumers" / "chatbot" / "auto.py"


# ---------------------------------------------------------------------------
# AC1 — single rubric definition site (extended in place, not duplicated)
# ---------------------------------------------------------------------------

def test_rubric_has_exactly_one_definition_site():
    src = _AUTO_PY.read_text(encoding="utf-8")
    assert src.count("def build_assessment_prompt") == 1


# ---------------------------------------------------------------------------
# AC1 — reuse-before-create + named-routing + narrate-one-line in the prompt
# ---------------------------------------------------------------------------

def test_prompt_has_reuse_before_create_signals():
    p = build_assessment_prompt("build me a thing", 0, "\n- Jim (dev)\n")
    assert "Reuse before creating" in p
    # prefer an existing owner; creating requires nothing fitting + saying you checked
    assert "Prefer an existing roster agent" in p
    assert "nothing on the roster fits" in p
    assert "say you checked" in p


def test_prompt_honours_named_routing():
    p = build_assessment_prompt("have Jim do it", 0, "")
    assert "Honour named routing" in p
    assert "never silently substitute another" in p


def test_prompt_instructs_one_line_lane_narration():
    p = build_assessment_prompt("x", 0, "")
    assert "which lane and why in one line" in p
    assert "narrates every routing decision" in p


# ---------------------------------------------------------------------------
# AC2 — 224's three-lane rubric survives the extension (no regression)
# ---------------------------------------------------------------------------

def test_three_lane_rubric_still_intact():
    p = build_assessment_prompt("do a thing", 0, "")
    assert "Routing lanes" in p
    assert "delegate" in p and "assign" in p and "mission" in p
    assert "answers THIS conversation" in p
    assert "off-thread" in p.lower()
    # 224's named-agent + defer signals remain
    assert "my accountant agent" in p and "role possessive" in p
    for phrase in ("queue it", "later", "when free"):
        assert phrase in p
    assert '"action": "respond|delegate|assign|mission"' in p
