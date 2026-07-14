"""PRD-202 S1 (P2-21) — spec-conformant SKILL.md schema + import/export.

Pins:
1. The canonical ``skill_source`` provenance scheme resolves every value —
   canonical ``scheme:ref`` AND legacy tags (``builtin-core``, bare git ids).
2. A standard-conformant folder imports to a ``Skill`` (+ ``SkillFile``) rows
   with the frontmatter ``description`` persisted as the L1 trigger text.
3. A ``Skill`` row exports back to a standard folder an external runner accepts.
4. import(export(skill)) round-trips name / description / body.

All pure — fixture folders on a tmp dir, an in-memory fake session; no git,
no network, no LLM (the ``workspace`` scheme is trusted → static-only scan on
clean fixtures returns no findings).
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from types import SimpleNamespace

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.agents.services.skill_source_scheme import (  # noqa: E402
    CANONICAL_SKILL_SOURCE_SCHEMES,
    canonicalize_skill_source,
    is_external_source,
    parse_skill_source,
    scheme_of,
)


# ---------------------------------------------------------------------------
# 1. Canonical provenance scheme resolves
# ---------------------------------------------------------------------------

def test_skill_source_scheme_canonical_resolves_legacy_and_canonical():
    # Legacy exact tags
    assert parse_skill_source("builtin-core") == ("builtin", "core")
    assert parse_skill_source("builtin-seeds") == ("builtin", "seeds")
    assert parse_skill_source("workspace-user") == ("workspace", "user")
    assert parse_skill_source("workspace-fork") == ("workspace", "fork")
    # Legacy git shape: a bare numeric SkillSource id
    assert parse_skill_source("5") == ("git", "5")
    # Already-canonical scheme:ref
    assert parse_skill_source("plugin:jira-admin") == ("plugin", "jira-admin")
    assert parse_skill_source("git:anthropic-official") == ("git", "anthropic-official")
    # Unknown / empty
    assert parse_skill_source("") == ("unknown", "")
    assert parse_skill_source(None) == ("unknown", "")


def test_canonicalize_round_trips_through_parse():
    for scheme in CANONICAL_SKILL_SOURCE_SCHEMES:
        value = canonicalize_skill_source(scheme, "some-ref")
        assert value == f"{scheme}:some-ref"
        assert parse_skill_source(value) == (scheme, "some-ref")
        assert scheme_of(value) == scheme


def test_canonicalize_rejects_unknown_scheme():
    try:
        canonicalize_skill_source("ftp", "x")
        assert False, "expected ValueError for an unknown scheme"
    except ValueError:
        pass


def test_is_external_source_only_git_and_plugin():
    assert is_external_source("git:anthropic-official") is True
    assert is_external_source("plugin:jira") is True
    assert is_external_source("5") is True  # legacy git
    assert is_external_source("builtin-core") is False
    assert is_external_source("workspace:user") is False
    assert is_external_source("workspace-fork") is False


# ---------------------------------------------------------------------------
# In-memory fake session (no DB)
# ---------------------------------------------------------------------------

class _FakeQuery:
    def filter(self, *a, **k):
        return self

    def first(self):
        return None  # every import here is a fresh insert

    def delete(self):
        return 0

    def all(self):
        return []


class _FakeSession:
    def __init__(self):
        self.added = []
        self._next_id = 0

    def query(self, *a, **k):
        return _FakeQuery()

    def add(self, obj):
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            self._next_id += 1
            obj.id = self._next_id

    def flush(self):
        for o in self.added:
            if getattr(o, "id", None) is None:
                self._next_id += 1
                o.id = self._next_id

    def commit(self):
        pass

    def skills(self):
        return [o for o in self.added if type(o).__name__ == "Skill"]

    def skill_files(self):
        return [o for o in self.added if type(o).__name__ == "SkillFile"]


def _write_standard_folder(root: pathlib.Path, *, name="my-skill", description="Does a specific thing well", body="# My Skill\n\nStep one, step two.") -> pathlib.Path:
    folder = root / name
    (folder / "scripts").mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nversion: 2.1.0\ntags:\n  - alpha\n---\n\n{body}\n",
        encoding="utf-8",
    )
    (folder / "scripts" / "run.py").write_text("print('hello from skill script')\n", encoding="utf-8")
    (folder / "reference.md").write_text("# Reference\n\nExtra detail.\n", encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------
# 2. Import: standard folder -> Skill (+ SkillFile) with description as L1
# ---------------------------------------------------------------------------

def test_import_standard_skill_folder_persists_description_as_l1(tmp_path):
    from modules.agents.services.skill_portability import import_standard_skill_folder

    folder = _write_standard_folder(tmp_path)
    db = _FakeSession()

    result = asyncio.run(import_standard_skill_folder(
        db, str(folder),
        source_scheme="workspace", source_ref="user",
        workspace_id=None,
    ))

    assert result["success"] is True, result
    skills = db.skills()
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "my-skill"
    # description is the L1 trigger text — on the column AND in the JSONB the
    # L1 loader reads.
    assert skill.description == "Does a specific thing well"
    assert skill.skill_metadata["description"] == "Does a specific thing well"
    # L2 body lives in prompt_template
    assert "# My Skill" in skill.prompt_template
    # provenance is canonical
    assert skill.skill_source == "workspace:user"
    assert parse_skill_source(skill.skill_source) == ("workspace", "user")

    # L3 bundle indexed: the script + the resource, not skipped
    files = {f.file_path: f for f in db.skill_files()}
    assert any(p.endswith("run.py") for p in files), files
    script = next(f for p, f in files.items() if p.endswith("run.py"))
    assert script.load_level == 3 and script.file_type == "script"


def test_import_rejects_folder_without_description(tmp_path):
    from modules.agents.services.skill_portability import import_standard_skill_folder

    folder = tmp_path / "bad-skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: bad-skill\n---\n\n# Body\n", encoding="utf-8")
    db = _FakeSession()

    result = asyncio.run(import_standard_skill_folder(
        db, str(folder), source_scheme="workspace", source_ref="user",
    ))
    assert result["success"] is False
    assert "description" in result["error"].lower()


# ---------------------------------------------------------------------------
# 3. Export: Skill row -> standard folder
# ---------------------------------------------------------------------------

def test_export_skill_emits_standard_folder(tmp_path):
    from modules.agents.services.skill_portability import export_skill_to_folder

    # A skill with an on-disk bundle (scripts/) to copy out.
    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "run.py").write_text("print('x')\n", encoding="utf-8")

    skill = SimpleNamespace(
        name="exported-skill",
        description="An exportable capability",
        prompt_template="# Exported\n\nBody text.",
        skill_version="1.4.0",
        tags=["beta"],
        category="analytics",
        skill_metadata={"description": "An exportable capability", "author": "vector"},
        filesystem_path=str(src),
    )

    out = export_skill_to_folder(None, skill, str(tmp_path / "out"))
    out_dir = pathlib.Path(out)

    skill_md = (out_dir / "SKILL.md").read_text(encoding="utf-8")
    assert skill_md.startswith("---")
    assert "name: exported-skill" in skill_md
    assert "description: An exportable capability" in skill_md
    assert "# Exported" in skill_md  # body present
    # internal bookkeeping is NOT exported
    assert "security_scan" not in skill_md
    # bundled script copied into the standard scripts/ dir
    assert (out_dir / "scripts" / "run.py").read_text(encoding="utf-8") == "print('x')\n"


# ---------------------------------------------------------------------------
# 4. Round-trip: import(export(skill)) preserves name / description / body
# ---------------------------------------------------------------------------

def test_import_export_roundtrips(tmp_path):
    from modules.agents.services.skill_portability import (
        export_skill_to_folder,
        import_standard_skill_folder,
    )

    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "run.py").write_text("print('roundtrip')\n", encoding="utf-8")
    original = SimpleNamespace(
        name="roundtrip-skill",
        description="Round trips cleanly",
        prompt_template="# Roundtrip\n\nContent.",
        skill_version="3.0.0",
        tags=["x"],
        category="general",
        skill_metadata={"description": "Round trips cleanly"},
        filesystem_path=str(src),
    )

    exported_dir = export_skill_to_folder(None, original, str(tmp_path / "out"))

    db = _FakeSession()
    result = asyncio.run(import_standard_skill_folder(
        db, exported_dir, source_scheme="workspace", source_ref="fork",
    ))
    assert result["success"] is True, result
    reimported = db.skills()[0]
    assert reimported.name == original.name
    assert reimported.description == original.description
    assert "# Roundtrip" in reimported.prompt_template
    # the exported bundle survived the round trip and was re-indexed
    assert any(f.file_path.endswith("run.py") for f in db.skill_files())
