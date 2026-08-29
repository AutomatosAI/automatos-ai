"""PRD-231 US-005 — the seed drift guard, two layers.

Layer 1 (always-on, no sibling repo): every generated seed carries a
sha256[:12] of its source body in its banner; recomputing the body sha and
comparing to the recorded one catches a hand-edit to a live seed — the "never
edit the platform copy" rule made structural. This layer runs in CI unchanged.

Layer 2 (freshness, sibling-gated): the same check invokes
``sync-auto-skill.py --check`` when the ``automatos-skills`` sibling is present
(developer machines) and skips cleanly with a printed reason where it is absent
(CI). Layer 1 is the guarantee; layer 2 is the bonus when the source is on disk.

Pure, LLM-free, Postgres-free. Tampering is proven on a tmp COPY — the real
committed seeds are never mutated.
"""

import importlib.util
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SYNC_PATH = _REPO_ROOT / "scripts" / "sync-auto-skill.py"


def _load_sync():
    spec = importlib.util.spec_from_file_location("prd231_sync_drift", _SYNC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (dataclass annotation resolution)
    spec.loader.exec_module(mod)
    return mod


sync_mod = _load_sync()

SEEDS = [s.target for s in sync_mod.SOURCES]


def _guard_message(seed_path: pathlib.Path) -> str:
    """The failure message the guard raises — it must point a confused developer
    at the source of truth and the one sanctioned way to regenerate."""
    recorded, actual = sync_mod.seed_drift(seed_path.read_text(encoding="utf-8"))
    return (
        f"{seed_path.name} was hand-edited in the platform repo "
        f"(recorded body sha {recorded} != actual {actual}). Seeds under "
        f"orchestrator/core/seeds/ are GENERATED — never edit the live copy. "
        f"Author in automatos-skills and run: python3 scripts/sync-auto-skill.py"
    )


def _assert_untampered(seed_path: pathlib.Path) -> None:
    recorded, actual = sync_mod.seed_drift(seed_path.read_text(encoding="utf-8"))
    assert recorded is not None, (
        f"{seed_path.name}: banner records no source sha — regenerate via "
        f"scripts/sync-auto-skill.py"
    )
    assert recorded == actual, _guard_message(seed_path)


# ── Layer 1: the always-on sha guard ─────────────────────────────────────────

@pytest.mark.parametrize("seed", SEEDS, ids=lambda p: p.name)
def test_committed_seed_matches_its_recorded_sha(seed):
    """The real committed seeds are in sync — no hand-edit has slipped in."""
    _assert_untampered(seed)


@pytest.mark.parametrize("seed", SEEDS, ids=lambda p: p.name)
def test_hand_edit_to_a_seed_body_fails_the_guard(seed, tmp_path):
    """A one-byte edit to a COPY of a seed body breaks the sha match, and the
    guard message names the file and the rule (automatos-skills + the sync)."""
    copy = tmp_path / seed.name
    copy.write_text(seed.read_text(encoding="utf-8") + "X", encoding="utf-8")  # body edit

    recorded, actual = sync_mod.seed_drift(copy.read_text(encoding="utf-8"))
    assert recorded is not None
    assert recorded != actual  # drift detected

    with pytest.raises(AssertionError) as exc:
        _assert_untampered(copy)
    msg = str(exc.value)
    assert seed.name in msg
    assert "automatos-skills" in msg
    assert "sync-auto-skill.py" in msg


def test_banner_edit_is_also_caught(tmp_path):
    """Editing the recorded sha in the banner (instead of the body) is caught
    too — the recorded value no longer matches the untouched body."""
    seed = SEEDS[0]
    raw = seed.read_text(encoding="utf-8")
    recorded, _actual = sync_mod.seed_drift(raw)
    # corrupt one hex digit of the recorded sha
    flipped = "0" if recorded[0] != "0" else "1"
    tampered = raw.replace(recorded, flipped + recorded[1:], 1)
    copy = tmp_path / seed.name
    copy.write_text(tampered, encoding="utf-8")

    new_recorded, new_actual = sync_mod.seed_drift(copy.read_text(encoding="utf-8"))
    assert new_recorded != new_actual  # banner tamper detected


# ── Layer 2: the sibling-gated --check freshness guard ───────────────────────

def test_check_skips_cleanly_when_sibling_absent(monkeypatch, capsys):
    """CI path: no automatos-skills sibling → --check skips with a printed reason
    and exits 0 (layer 1 is the always-on guarantee)."""
    monkeypatch.setattr(sync_mod, "find_skills_repo", lambda: None)
    rc = sync_mod.main(["--check"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIP" in out
    assert "automatos-skills" in out


def test_check_runs_when_sibling_present(monkeypatch, capsys, tmp_path):
    """Developer path: a present sibling makes --check actually run (not skip).

    Pointed at an existing-but-empty dir, the check RUNS and reports the missing
    sources as a FAIL — proving the path-exists branch was taken rather than the
    clean skip. (A real, fresh sibling would instead print OK/rc 0.)"""
    monkeypatch.setattr(sync_mod, "find_skills_repo", lambda: tmp_path)
    rc = sync_mod.main(["--check"])
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "SKIP" not in out.out  # did NOT take the skip branch
    assert "FAIL" in combined     # ran the check against the (empty) sibling
    assert rc != 0


def test_check_reports_fresh_when_real_sibling_present(monkeypatch, capsys):
    """If a real automatos-skills checkout is on this machine, --check confirms
    the committed seeds match it (rc 0). Skipped where the sibling is absent —
    layer 1 already guarantees integrity there."""
    real = sync_mod.find_skills_repo()
    if real is None:
        pytest.skip("automatos-skills sibling not present — layer-1 sha guard covers integrity")
    rc = sync_mod.main(["--check"])
    out = capsys.readouterr().out
    assert rc == 0, f"real-sibling --check reported drift:\n{out}"
