"""PRD-231 US-002 — the two-file sync script + --check drift mode.

Pure, LLM-free, Postgres-free. The sync script (``scripts/sync-auto-skill.py``)
is loaded from its file path (its name has a hyphen, so it is not importable by
name). Every test that needs a "source" reconstructs one from the COMMITTED seed
by stripping its generated banner — so the suite carries no sibling
``automatos-skills`` checkout and still exercises real content through the real
self-checks.
"""

import importlib.util
import pathlib
import re
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SYNC_PATH = _REPO_ROOT / "scripts" / "sync-auto-skill.py"


def _load_sync():
    spec = importlib.util.spec_from_file_location("prd231_sync_auto_skill", _SYNC_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass(Source) resolves annotations via
    # sys.modules[cls.__module__]; an unregistered module makes that None.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sync_mod = _load_sync()

# The two committed seeds, addressed through the module's own SOURCES table so a
# path change there can never silently orphan this suite.
CHARTER_SEED = next(s for s in sync_mod.SOURCES if "management" in s.target.name)
OPS_SEED = next(s for s in sync_mod.SOURCES if "operations" in s.target.name)


def _reconstruct_source(seed_text: str) -> str:
    """Recover an authored SKILL.md from a generated seed: same frontmatter, body
    with the banner stripped. render_seed() over this reproduces the seed byte
    for byte (round-trip stable), which is what lets the tests run without the
    sibling repo."""
    frontmatter, body = sync_mod.split_frontmatter(seed_text)
    return f"---{frontmatter}---\n\n{sync_mod.strip_banner(body)}"


def _write_tmp_repo(tmp_path) -> pathlib.Path:
    """A tmp automatos-skills checkout whose two SKILL.md sources are recovered
    from the committed seeds (valid content → self-checks pass)."""
    repo = tmp_path / "automatos-skills"
    for src in sync_mod.SOURCES:
        dest = repo.joinpath(*src.src_subpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_reconstruct_source(src.target.read_text()), encoding="utf-8")
    return repo


def _tmp_sources(tmp_seeds) -> tuple:
    """SOURCES clones whose targets point at a tmp seeds dir (so tests never
    write the real seeds), keeping the real self-check callables."""
    return (
        sync_mod.Source(
            key="charter",
            src_rel=CHARTER_SEED.src_rel,
            src_subpath=CHARTER_SEED.src_subpath,
            target=tmp_seeds / CHARTER_SEED.target.name,
            check=sync_mod._check_charter,
        ),
        sync_mod.Source(
            key="ops",
            src_rel=OPS_SEED.src_rel,
            src_subpath=OPS_SEED.src_subpath,
            target=tmp_seeds / OPS_SEED.target.name,
            check=sync_mod._check_ops,
        ),
    )


# ── AC1: SOURCES table + per-file self-checks (tampered fixtures) ─────────────

def test_sources_table_has_both_pairs():
    subpaths = {s.src_subpath for s in sync_mod.SOURCES}
    targets = {s.target.name for s in sync_mod.SOURCES}
    assert ("team", "auto", "SKILL.md") in subpaths
    assert ("team", "auto-ops", "SKILL.md") in subpaths
    assert "platform-management-skill.md" in targets
    assert "platform-operations-skill.md" in targets


def test_charter_self_check_passes_on_real_content():
    fm, body = sync_mod.split_frontmatter(CHARTER_SEED.target.read_text())
    assert sync_mod._check_charter(fm, body) == []


def test_charter_self_check_flags_missing_doctrine_anchor():
    fm, body = sync_mod.split_frontmatter(CHARTER_SEED.target.read_text())
    tampered = body.replace("Board as ledger", "Board as diary")
    problems = sync_mod._check_charter(fm, tampered)
    assert any("doctrine anchors missing" in p and "Board as ledger" in p for p in problems)


def test_charter_self_check_flags_missing_contract_opener():
    fm, body = sync_mod.split_frontmatter(CHARTER_SEED.target.read_text())
    tampered = body.replace(sync_mod.CONTRACT_OPENER, "A dispatch is roughly:")
    problems = sync_mod._check_charter(fm, tampered)
    assert any("dispatch-contract" in p for p in problems)


def test_charter_self_check_flags_leaked_ops_header():
    fm, body = sync_mod.split_frontmatter(CHARTER_SEED.target.read_text())
    tampered = body + "\n\n# Platform Operations Reference\n"
    problems = sync_mod._check_charter(fm, tampered)
    assert any(sync_mod.OPS_REFERENCE_HEADER in p for p in problems)


def test_ops_self_check_passes_on_real_content():
    fm, body = sync_mod.split_frontmatter(OPS_SEED.target.read_text())
    assert sync_mod._check_ops(fm, body) == []


def test_ops_self_check_flags_missing_section_head():
    fm, body = sync_mod.split_frontmatter(OPS_SEED.target.read_text())
    tampered = body.replace("## 19.", "## nineteen")
    problems = sync_mod._check_ops(fm, tampered)
    assert any("## 19." in p for p in problems)


def test_ops_self_check_flags_wrong_name():
    body_only = sync_mod.split_frontmatter(OPS_SEED.target.read_text())[1]
    problems = sync_mod._check_ops("name: platform-management\n", body_only)
    assert any("platform-operations" in p for p in problems)


def test_sync_refuses_to_write_when_self_checks_fail(tmp_path, monkeypatch):
    """A tampered source (doctrine anchor removed) must fail the self-check and
    leave the target UNWRITTEN — the refusal path, end to end."""
    repo = tmp_path / "automatos-skills"
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    # Charter source with a doctrine anchor removed.
    good = _reconstruct_source(CHARTER_SEED.target.read_text())
    dest = repo.joinpath(*CHARTER_SEED.src_subpath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(good.replace("Board as ledger", "Board as diary"), encoding="utf-8")

    one = sync_mod.Source(
        key="charter", src_rel=CHARTER_SEED.src_rel, src_subpath=CHARTER_SEED.src_subpath,
        target=seeds / CHARTER_SEED.target.name, check=sync_mod._check_charter,
    )
    monkeypatch.setattr(sync_mod, "SOURCES", (one,))
    rc = sync_mod.sync(repo)
    assert rc == 1
    assert not one.target.exists()  # refused: nothing written


# ── AC2: --check exits 0 fresh, non-zero (naming the file) after a byte-flip ──

def test_check_zero_on_fresh_sync(tmp_path, monkeypatch, capsys):
    repo = _write_tmp_repo(tmp_path)
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    monkeypatch.setattr(sync_mod, "SOURCES", _tmp_sources(seeds))
    assert sync_mod.sync(repo) == 0
    assert sync_mod.check(repo) == 0  # fresh → clean


def test_check_nonzero_naming_file_after_byte_flip(tmp_path, monkeypatch, capsys):
    repo = _write_tmp_repo(tmp_path)
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    sources = _tmp_sources(seeds)
    monkeypatch.setattr(sync_mod, "SOURCES", sources)
    assert sync_mod.sync(repo) == 0
    capsys.readouterr()  # clear

    victim = sources[1].target  # the ops seed
    victim.write_text(victim.read_text() + "X", encoding="utf-8")  # one-byte hand-edit
    rc = sync_mod.check(repo)
    err = capsys.readouterr().err
    assert rc != 0
    assert victim.name in err  # the drift message NAMES the drifted seed


def test_check_via_main_skips_cleanly_without_sibling(monkeypatch, capsys):
    monkeypatch.setattr(sync_mod, "find_skills_repo", lambda: None)
    rc = sync_mod.main(["--check"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIP" in out and "automatos-skills" in out


# ── AC3: banner carries path + version + sha12; reader hash symmetry ──────────

@pytest.mark.parametrize("seed", [CHARTER_SEED, OPS_SEED], ids=lambda s: s.target.name)
def test_banner_carries_source_path_version_and_sha12(seed):
    _fm, body = sync_mod.split_frontmatter(seed.target.read_text())
    banner = body[: body.index("-->") + 3]
    assert seed.src_rel in banner          # source path
    assert re.search(r"\(v[\d.]+\)", banner)  # source version
    assert re.search(r"source-body-sha256\[:12\]=[0-9a-f]{12}", banner)  # body sha12


def _reader_body(raw: str) -> str:
    """The frontmatter-stripped body EXACTLY as both platform readers compute it:
    seed_auto_agent._upsert_platform_management_skill and
    skill_loader._refresh_builtin_if_stale both do ``raw.split('---', 2)[2].strip()``.
    """
    return raw.split("---", 2)[2].strip()


@pytest.mark.parametrize("seed", [CHARTER_SEED, OPS_SEED], ids=lambda s: s.target.name)
def test_reader_frontmatter_strip_hash_symmetry(seed):
    import hashlib

    raw = seed.target.read_text()
    # Mirror both readers' split; they must agree, and the banner must ride
    # INSIDE the hashed body (so the two content_hash computations stay equal).
    upsert_body = _reader_body(raw)
    refresh_body = _reader_body(raw)
    assert upsert_body == refresh_body
    assert sync_mod._BANNER_OPEN in upsert_body  # banner is inside the hashed body
    h1 = hashlib.sha256(upsert_body.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(refresh_body.encode("utf-8")).hexdigest()
    assert h1 == h2


@pytest.mark.parametrize("seed", [CHARTER_SEED, OPS_SEED], ids=lambda s: s.target.name)
def test_banner_recorded_sha_matches_body(seed):
    recorded, actual = sync_mod.seed_drift(seed.target.read_text())
    assert recorded is not None
    assert recorded == actual


# ── AC4: both seeds regenerated with the right split ──────────────────────────

def test_charter_seed_has_no_ops_reference_header():
    body = CHARTER_SEED.target.read_text()
    assert "\n# Platform Operations Reference" not in body
    # but still carries the charter pins
    assert sync_mod.CONTRACT_OPENER in body
    assert "Board as ledger" in body


def test_ops_seed_exists_with_the_cookbook():
    assert OPS_SEED.target.exists()
    body = OPS_SEED.target.read_text()
    assert "# Platform Operations Reference" in body
    assert "## 0." in body and "## 19." in body
    assert "name: platform-operations" in body


def test_round_trip_render_is_byte_stable():
    """Recovering a source from a seed and re-rendering reproduces the seed
    exactly — the invariant the sibling-free tests rely on."""
    for seed in (CHARTER_SEED, OPS_SEED):
        original = seed.target.read_text()
        rebuilt = sync_mod.render_seed(_reconstruct_source(original), seed.src_rel)
        assert rebuilt == original
