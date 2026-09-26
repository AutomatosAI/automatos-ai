"""PRD-251 Wave 1 US-119 — built-in skills load from the platform, synced from automatos-skills.

One manifest (``core/seeds/skills/manifest.json``, read through
``core/builtin_skills.py``) lists the built-in skills. The boot seeder
(``core/seeds/seed_builtin_skills.py``) creates a missing row, the loader
refreshes a builtin row by content hash, and ``scripts/sync-skills.py`` writes
ONLY the seed files it is told to. Pins:

1. platform-management is created and refreshed exactly as before: the same
   row, at the same path, under the same lock key, through Auto's seeding.
2. A fixture built-in skill in a test manifest is created at seed time and
   refreshed at load after its file changes; an entry not synced yet is skipped.
3. sync-skills.py syncs only the named fixture skills into the seeds folder with
   the banner, never touches an unnamed entry (platform-management included),
   and refuses a source with no frontmatter (pure file IO).
4. A git-imported row with a built-in skill's name is left untouched.

SQLite (StaticPool) with a portable copy of ``skills``; no Postgres, no network.
The sync-script tests write only under pytest's tmp_path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
_ROOT = _ORCH.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.seeds.seed_auto_agent as seed_auto_agent  # noqa: E402
import core.seeds.seed_builtin_skills as seeder  # noqa: E402
from core.builtin_skills import (  # noqa: E402
    MANIFEST_PATH,
    ManifestError,
    builtin_skill_paths,
    load_manifest,
    read_seed,
)
from core.models.core import Skill  # noqa: E402
from modules.agents.services.skill_loader import SkillLoader, SkillLoaderConfig  # noqa: E402
from modules.agents.services.skill_source_scheme import scheme_of  # noqa: E402
from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT  # noqa: E402
from tests.test_prd226_doctrine import DOCTRINE_ANCHORS  # noqa: E402

SEEDS_DIR = _ORCH / "core" / "seeds"
PLATFORM_SEED = SEEDS_DIR / "platform-management-skill.md"
SYNC_SCRIPT = _ROOT / "scripts" / "sync-skills.py"
BANNER_HEAD = "<!-- GENERATED FILE — DO NOT EDIT IN THIS REPO. Source of truth:"

# The row _upsert_platform_management_skill wrote before US-119, column by column
# (prompt_template, content_hash and skill_version come from the seed file).
LEGACY_PLATFORM_ROW = {
    "name": "platform-management",
    "description": (
        "Complete platform operations — marketplace, agents, playbooks, heartbeats, "
        "board, governance, LLMs, workspace setup"
    ),
    "skill_type": "technical",
    "category": "agent-role",
    "skill_source": "builtin-core",
    "tags": ["platform", "admin", "marketplace", "agents", "playbooks", "governance"],
    "is_active": True,
    "workspace_id": None,
    "skill_metadata": None,
    "tools_schema": None,
}

# The skills US-120's two marketplace agents carry (its notes): all built in.
SOCIALS_SKILLS = (
    "social-video-director", "social-ops", "social-production-workflow", "social-template-payloads",
    "social-media-strategist", "social-brand-voice", "carousel-design-system", "image-prompt-engineer",
    "short-video-editing-coach", "brand-kit-builder", "visual-storyteller",
)
NEVER_BUILT_IN = ("instagram-curator", "twitter-engager", "linkedin-content-creator", "html-to-png")


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


@pytest.fixture
def skills_table():
    copies = sa.MetaData()
    table = sa.Table(
        Skill.__table__.name,
        copies,
        *[sa.Column(c.name, _portable(c.type), primary_key=c.primary_key) for c in Skill.__table__.columns],
    )
    # Prod's partial unique index (dedupe_skills_unique_workspace_name): global names are unique.
    sa.Index("uq_skills_marketplace_name", table.c.name, unique=True, sqlite_where=table.c.workspace_id.is_(None))
    return table


@pytest.fixture
def db(skills_table, monkeypatch, tmp_path):
    engine = sa.create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    skills_table.metadata.create_all(engine)
    monkeypatch.setattr(SkillLoaderConfig, "SKILLS_BASE_DIR", str(tmp_path / "skill-cache"))
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def _legacy_seed(path: Path):
    """The pre-US-119 reader, verbatim: (body, content_hash, skill_version)."""
    raw = path.read_text(encoding="utf-8").strip()
    version = re.search(r'^version:\s*"?([\d.]+)"?', raw, re.M)
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        body = parts[2].strip() if len(parts) > 2 else raw
    else:
        body = raw
    return body, hashlib.sha256(body.encode("utf-8")).hexdigest(), version.group(1) if version else "1.0.0"


def _legacy_columns(row) -> dict:
    return {column: getattr(row, column) for column in LEGACY_PLATFORM_ROW}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _manifest(core_dir: Path, skills: dict) -> Path:
    """A test manifest at <core_dir>/seeds/skills/manifest.json (seeds root <core_dir>/seeds)."""
    return _write(core_dir / "seeds" / "skills" / "manifest.json", json.dumps({"skills": skills}))


def _source(name: str, *, version: str = "2.0.0", body: str = "Draft three posts from the week's reports.") -> str:
    """A SKILL.md as the automatos-skills repo carries it."""
    return (
        f"---\nname: {name}\n"
        f"description: Fixture skill {name} for the built-in skill tests\n"
        f'version: "{version}"\n'
        "tags: [fixture, social-media]\n"
        "category: agent-role\n"
        "---\n\n"
        f"# {name}\n\n{body}\n"
    )


def _seed(name: str, *, version: str = "1.2.0", body: str = "Draft three posts from the week's reports.") -> str:
    """A seed as sync-skills.py writes it: frontmatter intact, the banner first in the body."""
    frontmatter, source_body = _source(name, version=version, body=body).split("---", 2)[1:]
    banner = (
        f"{BANNER_HEAD}\n     automatos-skills/test/{name}/SKILL.md (v{version}). "
        f"Re-sync: python3 scripts/sync-skills.py {name} -->\n"
    )
    return f"---{frontmatter}---\n\n{banner}\n{source_body.lstrip()}"


# ---------------------------------------------------------------------------
# 1. platform-management: created and refreshed exactly as before
# ---------------------------------------------------------------------------


def test_platform_management_stays_at_its_path():
    assert SkillLoader._BUILTIN_PATHS["platform-management"] == PLATFORM_SEED
    assert load_manifest()["platform-management"].seed_path == PLATFORM_SEED
    assert load_manifest()["platform-management"].source == "team/auto/SKILL.md"


def test_platform_management_row_is_created_exactly_as_before(db):
    body, digest, version = _legacy_seed(PLATFORM_SEED)

    created = seeder.ensure_builtin_skill(db, "platform-management")
    db.commit()

    row = db.query(Skill).one()
    assert created.id == row.id
    assert _legacy_columns(row) == LEGACY_PLATFORM_ROW
    assert (row.prompt_template, row.content_hash, row.skill_version) == (body, digest, version)
    # Create-only: a second call returns the same row and writes nothing.
    assert seeder.ensure_builtin_skill(db, "platform-management").id == row.id
    assert db.query(Skill).count() == 1


def test_the_boot_seeder_creates_the_same_platform_management_row(db):
    outcome = seeder.seed_builtin_skills(db)
    db.commit()

    # Socials seeds are created too once the owner has synced them; until then they wait.
    assert "platform-management" in outcome["created"]
    assert set(outcome["created"]) | set(outcome["not_synced"]) == set(load_manifest())
    row = db.query(Skill).filter(Skill.name == "platform-management").one()
    assert _legacy_columns(row) == LEGACY_PLATFORM_ROW
    again = seeder.seed_builtin_skills(db)
    assert again["created"] == [] and "platform-management" in again["present"]


def test_platform_management_is_refreshed_at_load_exactly_as_before(db):
    seeder.ensure_builtin_skill(db, "platform-management")
    db.commit()
    row = db.query(Skill).one()
    row.prompt_template, row.content_hash = "an older body", "0" * 64
    db.commit()
    body, digest, version = _legacy_seed(PLATFORM_SEED)

    loaded = SkillLoader(db).load_skill_core("platform-management", db=db)

    assert loaded == body
    db.expire_all()
    row = db.query(Skill).one()
    assert (row.prompt_template, row.content_hash) == (body, digest)
    # Only the body and its hash move; the rest of the row is untouched.
    assert _legacy_columns(row) == LEGACY_PLATFORM_ROW
    assert row.skill_version == version


def test_auto_seeding_goes_through_the_builtin_seeder(monkeypatch):
    requested, assigned = [], []
    platform_row = MagicMock(id=7)
    monkeypatch.setattr(seed_auto_agent, "ensure_builtin_skill", lambda db, name: requested.append(name) or platform_row)
    monkeypatch.setattr(seed_auto_agent, "_assign_skill_to_agent", lambda db, agent, skill: assigned.append(skill))
    monkeypatch.setattr(
        seed_auto_agent, "_get_default_model_config",
        lambda: {"provider": "test", "model_id": "test", "max_tokens": 4000},
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None  # no Auto row yet

    seed_auto_agent.seed_auto_agent(db, uuid4())

    assert requested == ["platform-management"]
    assert assigned == [platform_row]


def test_the_seed_lock_key_is_the_one_platform_management_always_used():
    db = MagicMock()
    db.get_bind.return_value.dialect.name = "postgresql"
    db.query.return_value.filter.return_value.first.return_value = MagicMock(skill_source="builtin-core")

    seeder.ensure_builtin_skill(db, "platform-management")

    statement, params = db.execute.call_args.args
    assert "pg_advisory_xact_lock(hashtext(:key))" in str(statement)
    assert params == {"key": "seed:platform-management"}


# ---------------------------------------------------------------------------
# 2. A fixture built-in skill: created at seed time, refreshed at load
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_manifest(tmp_path):
    manifest = _manifest(tmp_path / "core", {
        "fixture-skill": {"seed": "fixture-skill.md", "source": "test/fixture-skill/SKILL.md"},
        "not-synced-yet": {"seed": "not-synced-yet.md", "source": "test/not-synced-yet/SKILL.md"},
    })
    _write(manifest.parent / "fixture-skill.md", _seed("fixture-skill"))
    return manifest


def test_a_fixture_builtin_is_created_at_seed_time(db, fixture_manifest):
    outcome = seeder.seed_builtin_skills(db, manifest_path=fixture_manifest)
    db.commit()

    assert outcome["created"] == ["fixture-skill"]
    assert outcome["not_synced"] == ["not-synced-yet"]
    row = db.query(Skill).one()
    seed = read_seed(fixture_manifest.parent / "fixture-skill.md")
    assert (row.name, row.workspace_id, row.skill_source, row.is_active) == (
        "fixture-skill", None, "builtin:fixture-skill", True,
    )
    assert scheme_of(row.skill_source) == "builtin"
    assert (row.description, row.skill_type, row.category, row.tags, row.skill_version) == (
        "Fixture skill fixture-skill for the built-in skill tests", "technical", "agent-role",
        ["fixture", "social-media"], "1.2.0",
    )
    assert (row.prompt_template, row.content_hash) == (seed.body, seed.content_hash)
    assert row.prompt_template.startswith(BANNER_HEAD)

    again = seeder.seed_builtin_skills(db, manifest_path=fixture_manifest)
    assert (again["created"], again["present"]) == ([], ["fixture-skill"])
    assert db.query(Skill).count() == 1


def test_a_fixture_builtin_is_refreshed_at_load_after_its_file_changes(db, fixture_manifest, monkeypatch):
    seed_path = fixture_manifest.parent / "fixture-skill.md"
    seeder.seed_builtin_skills(db, manifest_path=fixture_manifest)
    db.commit()
    monkeypatch.setattr(SkillLoader, "_BUILTIN_PATHS", builtin_skill_paths(fixture_manifest))
    assert SkillLoader(db).load_skill_core("fixture-skill", db=db) == read_seed(seed_path).body

    _write(seed_path, _seed("fixture-skill", version="1.3.0", body="Draft five posts, each with its source."))
    loaded = SkillLoader(db).load_skill_core("fixture-skill", db=db)  # a new loader: L2 is cached per loader

    changed = read_seed(seed_path)
    assert loaded == changed.body and "Draft five posts, each with its source." in loaded
    db.expire_all()
    row = db.query(Skill).one()
    assert (row.prompt_template, row.content_hash) == (changed.body, changed.content_hash)


def test_a_seed_without_a_description_is_not_seeded(db, tmp_path):
    manifest = _manifest(tmp_path / "core", {"bare-skill": {"seed": "bare-skill.md", "source": "test/bare/SKILL.md"}})
    _write(manifest.parent / "bare-skill.md", "---\nname: bare-skill\n---\n\nNo trigger text.\n")

    assert seeder.seed_builtin_skills(db, manifest_path=manifest)["invalid"] == ["bare-skill"]
    assert db.query(Skill).count() == 0


# ---------------------------------------------------------------------------
# 4. A git-imported row with a built-in skill's name is left untouched
# ---------------------------------------------------------------------------


def _git_row(**overrides) -> Skill:
    values = {
        "name": "fixture-skill",
        "description": "imported from a git skill source",
        "skill_type": "custom",
        "category": "general",
        "skill_version": "0.9.0",
        "skill_source": "7",  # the legacy git shape: a SkillSource id
        "prompt_template": "GIT BODY",
        "tags": ["git"],
        "skill_metadata": {"name": "fixture-skill"},
        "is_active": True,
        "workspace_id": None,
    }
    return Skill(**{**values, **overrides})


def _snapshot(row) -> dict:
    return {column.name: getattr(row, column.name) for column in Skill.__table__.columns}


def test_a_git_imported_row_of_the_same_name_is_left_untouched(db, fixture_manifest, monkeypatch):
    db.add(_git_row())
    db.commit()
    git_row = db.query(Skill).one()
    before = _snapshot(git_row)

    outcome = seeder.seed_builtin_skills(db, manifest_path=fixture_manifest)
    db.commit()

    assert outcome["left_alone"] == ["fixture-skill"] and outcome["created"] == []
    assert seeder.ensure_builtin_skill(db, "fixture-skill", manifest_path=fixture_manifest) is None
    db.expire_all()
    assert db.query(Skill).count() == 1
    assert _snapshot(db.query(Skill).one()) == before

    # Loading it never refreshes it from the built-in seed, whose body differs.
    monkeypatch.setattr(SkillLoader, "_BUILTIN_PATHS", builtin_skill_paths(fixture_manifest))
    assert SkillLoader(db).load_skill_core("fixture-skill", db=db) == "GIT BODY"
    db.expire_all()
    assert _snapshot(db.query(Skill).one()) == before


def test_a_workspace_git_row_stays_untouched_beside_the_builtin(db, skills_table, fixture_manifest):
    workspace = uuid4()
    db.execute(skills_table.insert().values(
        name="fixture-skill", description="a workspace's git import", skill_type="custom",
        skill_source="git:acme-skills", prompt_template="WORKSPACE BODY", is_active=True,
        workspace_id=workspace,
    ))
    db.commit()
    before = _snapshot(db.query(Skill).filter(Skill.workspace_id.isnot(None)).one())

    outcome = seeder.seed_builtin_skills(db, manifest_path=fixture_manifest)
    db.commit()

    assert outcome["created"] == ["fixture-skill"]
    db.expire_all()
    builtin = db.query(Skill).filter(Skill.workspace_id.is_(None)).one()
    assert builtin.skill_source == "builtin:fixture-skill"
    assert _snapshot(db.query(Skill).filter(Skill.workspace_id.isnot(None)).one()) == before


# ---------------------------------------------------------------------------
# 3. scripts/sync-skills.py: the named skills only (pure file IO)
# ---------------------------------------------------------------------------


def _sync_module():
    spec = importlib.util.spec_from_file_location("sync_skills_under_test", SYNC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _auto_source(*, drop: str = "") -> str:
    """team/auto/SKILL.md carrying what Auto's self-checks look for (less ``drop``)."""
    doctrine = "\n".join(f"{n}. **{anchor}.**" for n, anchor in enumerate(DOCTRINE_ANCHORS, 1) if anchor != drop)
    return (
        '---\nname: platform-management\ndescription: Workspace OS charter for Auto\nversion: "2.4.0"\n---\n\n'
        f"# Auto\n\n## H. The Manager's Doctrine\n\n{doctrine}\n\n{DISPATCH_CONTRACT_FRAGMENT}\n"
    )


@pytest.fixture
def world(tmp_path):
    repo = tmp_path / "automatos-skills"
    _write(repo / "social" / "social-a" / "SKILL.md", _source("social-a"))
    _write(repo / "social" / "social-b" / "SKILL.md", _source("social-b"))
    _write(repo / "team" / "auto" / "SKILL.md", _auto_source())
    manifest = _manifest(tmp_path / "orchestrator" / "core", {
        "platform-management": {
            "seed": "../platform-management-skill.md", "source": "team/auto/SKILL.md", "skill_source": "builtin-core",
        },
        "social-a": {"seed": "social-a.md", "source": "social/social-a/SKILL.md"},
        "social-b": {"seed": "social-b.md", "source": "social/social-b/SKILL.md"},
    })
    platform_seed = _write(manifest.parent.parent / "platform-management-skill.md", "---\nname: platform-management\n---\n\nv2.3.0\n")
    return SimpleNamespace(repo=repo, manifest=manifest, seeds=manifest.parent, platform_seed=platform_seed)


def _sync(world, *names) -> int:
    return _sync_module().main([*names, "--skills-repo", str(world.repo), "--manifest", str(world.manifest)])


def test_sync_writes_only_the_named_skill_with_the_banner(world):
    platform_before = world.platform_seed.read_bytes()

    assert _sync(world, "social-a") == 0

    written = (world.seeds / "social-a.md").read_text(encoding="utf-8")
    source = (world.repo / "social" / "social-a" / "SKILL.md").read_text(encoding="utf-8")
    frontmatter, body = source.split("---", 2)[1:]
    assert written.startswith(f"---{frontmatter}---\n\n")  # frontmatter intact
    seed = read_seed(world.seeds / "social-a.md")
    assert seed.body.splitlines()[:2] == [
        BANNER_HEAD,
        "     automatos-skills/social/social-a/SKILL.md (v2.0.0). Re-sync: python3 scripts/sync-skills.py social-a -->",
    ]
    assert seed.body.endswith(body.strip())
    assert not (world.seeds / "social-b.md").exists()
    assert world.platform_seed.read_bytes() == platform_before


def test_sync_writes_several_named_skills_and_platform_management_only_when_named(world):
    assert _sync(world, "social-a", "social-b", "social-a") == 0
    assert {path.name for path in world.seeds.glob("*.md")} == {"social-a.md", "social-b.md"}
    assert "v2.3.0" in world.platform_seed.read_text(encoding="utf-8")

    assert _sync(world, "platform-management") == 0
    synced = world.platform_seed.read_text(encoding="utf-8")
    assert "automatos-skills/team/auto/SKILL.md (v2.4.0). Re-sync: python3 scripts/sync-skills.py platform-management" in synced


def test_platform_management_keeps_autos_doctrine_checks(world):
    _write(world.repo / "team" / "auto" / "SKILL.md", _auto_source(drop="Board as ledger"))
    before = world.platform_seed.read_bytes()

    assert _sync(world, "platform-management") == 1
    assert world.platform_seed.read_bytes() == before


def test_sync_refuses_a_source_without_frontmatter_and_writes_nothing(world):
    _write(world.repo / "social" / "social-b" / "SKILL.md", "# social-b\n\nNo frontmatter here.\n")

    assert _sync(world, "social-a", "social-b") == 1
    assert list(world.seeds.glob("*.md")) == []


def test_sync_refuses_a_source_that_names_another_skill(world):
    _write(world.repo / "social" / "social-b" / "SKILL.md", _source("social-a"))

    assert _sync(world, "social-b") == 1
    assert list(world.seeds.glob("*.md")) == []


def test_sync_refuses_a_name_the_manifest_does_not_list(world):
    assert _sync(world, "social-a", "instagram-curator") == 1
    assert list(world.seeds.glob("*.md")) == []


def test_sync_has_no_sync_everything_mode(world):
    with pytest.raises(SystemExit) as refused:
        _sync_module().main(["--skills-repo", str(world.repo), "--manifest", str(world.manifest)])
    assert refused.value.code == 2
    assert list(world.seeds.glob("*.md")) == []


def test_sync_skills_replaces_the_auto_only_script():
    assert SYNC_SCRIPT.is_file()
    assert not (_ROOT / "scripts" / "sync-auto-skill.py").exists()


# ---------------------------------------------------------------------------
# The committed manifest, the seeds folder, and the boot wiring
# ---------------------------------------------------------------------------


def test_the_manifest_lists_the_socials_skills_as_builtins():
    manifest = load_manifest()
    assert set(SOCIALS_SKILLS) <= set(manifest)
    assert not set(NEVER_BUILT_IN) & set(manifest)
    for name in SOCIALS_SKILLS:
        entry = manifest[name]
        assert entry.seed_path == MANIFEST_PATH.parent / f"{name}.md"
        assert entry.skill_source == f"builtin:{name}" and not entry.row
    width = Skill.__table__.c.skill_source.type.length
    for entry in manifest.values():
        assert scheme_of(entry.skill_source) == "builtin", entry.name
        assert len(entry.skill_source) <= width, entry.name


def test_every_file_in_the_skills_seeds_folder_is_a_generated_manifest_seed():
    seeds = set(builtin_skill_paths().values())
    for path in MANIFEST_PATH.parent.iterdir():
        if path == MANIFEST_PATH:
            continue
        assert path in seeds, f"{path.name} is not a built-in skill's seed"
        assert BANNER_HEAD in path.read_text(encoding="utf-8"), f"{path.name} was not written by sync-skills.py"


@pytest.mark.parametrize("skills, problem", [
    ({"Bad Name": {"seed": "x.md", "source": "a/SKILL.md"}}, "lowercase slug"),
    ({"x": {"seed": "../../x.md", "source": "a/SKILL.md"}}, "outside"),
    ({"x": {"seed": "x.txt", "source": "a/SKILL.md"}}, "'seed'"),
    ({"x": {"seed": "x.md", "source": "a/README.md"}}, "'source'"),
    ({"x": {"seed": "x.md", "source": "../a/SKILL.md"}}, "'source'"),
    ({"x": {"seed": "x.md", "source": "a/SKILL.md", "row": {"prompt_template": "y"}}}, "may pin only"),
    ({"x": {"seed": "x.md", "source": "a/SKILL.md", "extra": 1}}, "unknown keys"),
    ({}, "non-empty"),
])
def test_a_malformed_manifest_fails_loud(tmp_path, skills, problem):
    with pytest.raises(ManifestError, match=re.escape(problem)):
        load_manifest(_manifest(tmp_path / "core", skills))


def test_the_boot_leader_seeds_the_builtin_skills_before_the_marketplace():
    src = (_ORCH / "main.py").read_text(encoding="utf-8")
    start = src.index("async def _boot_phase_1_core")
    body = src[start:src.index("async def ", start + 1)]
    assert body.index("if not is_leader") < body.index("seed_builtin_skills(db)")
    assert body.index("seed_builtin_skills(db)") < body.index("seed_packages(db, create_only=True)")
