"""F204 — a skill shows what it holds, and says when it only points at a document.

Night 6 (05:16Z): the owner opened skill #358 and saw "a name, one line… no
files, 0 tokens". It held 475 characters: a skill made with
platform_create_workspace_skill keeps its SKILL.md in prompt_template and has no
file rows, and the Skills list showed only file rows. Its body also pointed at
'harbourline-brand-voice.md' instead of carrying the guide's rules. The content
is now listed as its SKILL.md, and a skill that points at a document is created
with advice to carry the rules instead.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

BODY_358 = ("You are a content creator for Harbourline. Your primary goal is to maintain the brand's distinct voice "
            "in all communications. Refer to the \"harbourline-brand-voice.md\" document for specific guidelines on "
            "tone, language, and style. Always prioritize authenticity, warmth, and a slightly sophisticated yet "
            "approachable demeanor.")


@pytest.fixture
def cafe(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    return NS(db=db_session, ws=ws, ctx=NS(workspace_id=ws, user=NS(id=None, email="owner@cafe.test",
                                                                     is_super_admin=False, role=None)))


def _skill_358(cafe):
    from core.models.core import Skill

    skill = Skill(name="Harbourline voice", description="Ensures all content adheres to the Harbourline brand voice.",
                  skill_type="cognitive", prompt_template=BODY_358, skill_source="workspace-user",
                  skill_version="1.0.0", workspace_id=cafe.ws, is_active=True, skill_metadata={})
    cafe.db.add(skill)
    cafe.db.flush()
    return skill


def test_a_skill_kept_in_its_prompt_is_listed_with_its_skill_md(cafe):
    from api.skills import get_skill_details

    skill = _skill_358(cafe)

    shown = asyncio.run(get_skill_details(skill.id, ctx=cafe.ctx, db=cafe.db))

    files = [(f.file_path, f.file_size_bytes, f.estimated_tokens) for f in shown.files]
    assert files and files[0][0] == "SKILL.md"                            # night: no files, 0 tokens
    assert files[0][1] == len(BODY_358.encode("utf-8")) and files[0][2] > 0


def test_a_skill_with_file_rows_is_listed_as_before():
    from api.skills import _skill_files

    row = NS(file_path="SKILL.md", file_type="markdown", content_summary="s", file_size_bytes=10,
             estimated_tokens=3, load_level=2)
    listed = _skill_files(NS(files=[row], prompt_template="also here", description="d"))
    assert [(f.file_path, f.estimated_tokens) for f in listed] == [("SKILL.md", 3)]
    assert _skill_files(NS(files=[], prompt_template="  ", description="d")) == []


def test_a_skill_that_points_at_a_document_is_made_with_advice(cafe):
    from modules.tools.discovery.handlers_skills import create_workspace_skill

    made = asyncio.run(create_workspace_skill(cafe.db, cafe.ws, {"name": "Harbourline voice", "content": BODY_358}))

    assert made["success"] is True                                          # a pointer is advice, never a refusal
    assert made.get("advice") == ("This skill points at 'harbourline-brand-voice.md' instead of carrying its rules. "
                                  "If those rules don't change often, put them in the skill, so agents don't "
                                  "depend on a search each time.")          # night: no word of it
    assert made["advice"] in made["message"]


def test_a_skill_that_carries_its_rules_is_made_without_advice(cafe):
    from modules.tools.discovery.handlers_skills import create_workspace_skill

    made = asyncio.run(create_workspace_skill(cafe.db, cafe.ws, {
        "name": "Harbourline voice", "content": "Write warmly and plainly. No jargon. Name the café. Sign off 'The Harbourline team'."}))

    assert made["success"] is True and "advice" not in made


def test_a_long_run_of_hyphens_is_read_at_once():
    """Review CRITICAL: 'Notes' then a run of hyphens (a divider) made the pointer
    search backtrack exponentially, holding the event loop for every workspace."""
    import time

    from modules.tools.discovery.handlers_skills import points_at_documents_advice

    started = time.monotonic()
    assert points_at_documents_advice("Notes" + "-" * 36 + "x\nWrite warmly.") is None
    assert time.monotonic() - started < 0.5                                 # before: seconds, and doubling per hyphen
    assert "'guide.md'" in points_at_documents_advice("See guide.md" + "-" * 36)
