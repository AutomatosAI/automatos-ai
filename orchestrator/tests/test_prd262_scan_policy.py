"""PRD-262 US-013: the vulnerability scan blocks (D1, FR-3).

``scripts/ci/image_scan_policy.py`` judges each images.yml build's Grype JSON: a fixable
Critical, or a match on CISA's KEV list at any severity, blocks unless an unexpired
``.grype.yaml`` entry covers it. The fixture ``tests/fixtures/prd262_grype_report.json``
is laid out exactly as Grype 0.118's JSON presenter writes a report
(``grype/presenter/models`` at v0.118.0). Its curl and starlette rows carry the values
images run 38915020775 reported on 16 Oct: two fixable Critical curl rows, the KEV-listed
starlette Medium and a High that does not block. The ``wont-fix`` zlib Critical is an
illustrative row that must not block either.

Also static checks on images.yml (every build is scanned from its registry, the
database downloads while the image builds, the policy step reads the scan and the
allowlist), on the daily scan-allowlist workflow, and on the committed allowlist.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prd262_grype_report.json"
TODAY = date(2026, 10, 18)
CURL_CRITICALS = {"CVE-2026-8924", "CVE-2026-8927"}
STARLETTE_KEV = "GHSA-86qp-5c8j-p5mr"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = _load("image_scan_policy")
REPORT = json.loads(FIXTURE.read_text(encoding="utf-8"))
IMAGES_WORKFLOW = yaml.safe_load((WORKFLOWS / "images.yml").read_text(encoding="utf-8"))
BUILD_STEPS = IMAGES_WORKFLOW["jobs"]["build"]["steps"]
ALLOWLIST_WORKFLOW = yaml.safe_load((WORKFLOWS / "scan-allowlist.yml").read_text(encoding="utf-8"))


def _step(name: str) -> dict:
    return next(step for step in BUILD_STEPS if step.get("name") == name)


def _index(predicate) -> int:
    return next(position for position, step in enumerate(BUILD_STEPS) if predicate(step))


def _entry(vulnerability: str, expires: date = TODAY + timedelta(days=30), **extra) -> dict:
    return {"vulnerability": vulnerability, "reason": "accepted for the test", "expires": expires.isoformat(), **extra}


def _files(tmp_path: Path, entries: list, report: dict = REPORT) -> tuple[Path, Path]:
    allowlist = tmp_path / ".grype.yaml"
    allowlist.write_text(yaml.safe_dump({"ignore": entries}), encoding="utf-8")
    scan = tmp_path / "report.json"
    scan.write_text(json.dumps(report), encoding="utf-8")
    return scan, allowlist


def _check(tmp_path: Path, entries: list, mode: str = "fail", report: dict = REPORT) -> int:
    scan, allowlist = _files(tmp_path, entries, report)
    summary = tmp_path / "summary.md"
    return policy.main(["check", "--report", str(scan), "--allowlist", str(allowlist), "--mode", mode,
                        "--name", "api", "--arch", "amd64", "--summary", str(summary), "--today", TODAY.isoformat()])


def _blocking(entries: list, report: dict = REPORT) -> set[str]:
    parsed = [policy.parse_entry(raw, f"ignore[{i}]", TODAY) for i, raw in enumerate(entries)]
    return {row.vulnerability for row in policy.evaluate(policy.rows(report), parsed, TODAY).blocking}


COVERED = [
    _entry("CVE-2026-8924", package={"version": "8.14.1-2+deb13u5", "type": "deb"}),
    _entry("CVE-2026-8927", package={"version": "8.14.1-2+deb13u5", "type": "deb"}),
    _entry(STARLETTE_KEV, package={"name": "starlette", "version": "0.41.3", "type": "python"}),
]


# --------------------------------------------------------------------------- the policy


def test_a_fixable_critical_and_a_kev_medium_fail_the_build(tmp_path, capsys):
    assert _check(tmp_path, []) == 1
    out = capsys.readouterr().out
    assert "::error title=Scan policy::api (amd64): curl 8.14.1-2+deb13u5 (deb) CVE-2026-8924 Critical" in out
    assert f"{STARLETTE_KEV} Medium, fixed in 1.0.1: KEV" in out
    assert _blocking([]) == CURL_CRITICALS | {STARLETTE_KEV}


def test_the_same_rows_allowlisted_pass(tmp_path, capsys):
    assert _check(tmp_path, COVERED) == 0
    assert "::error" not in capsys.readouterr().out
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "nothing blocks, 3 allowlisted" in summary
    assert "Grype 0.118.0: 3 Critical, 1 High, 1 Medium." in summary


def test_a_high_and_an_unfixed_critical_never_block():
    rows = {row.vulnerability: row for row in policy.rows(REPORT)}
    assert rows["GHSA-7f5h-v6xp-fcq8"].blocks == ()
    assert rows["CVE-2023-45853"].severity == "Critical" and rows["CVE-2023-45853"].blocks == ()
    assert rows[STARLETTE_KEV].blocks == ("KEV",)
    assert rows["CVE-2026-8924"].blocks == ("fixable Critical",)


def test_report_mode_warns_and_passes(tmp_path, capsys):
    assert _check(tmp_path, [], mode="report") == 0
    out = capsys.readouterr().out
    assert "::warning title=Scan policy (report only)::api (amd64): curl" in out and "::error" not in out
    assert "3 row(s) would block; report only" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_a_kev_listed_alias_blocks_a_row_rated_low():
    report = copy.deepcopy(REPORT)
    high = next(m for m in report["matches"] if m["vulnerability"]["id"] == "GHSA-7f5h-v6xp-fcq8")
    high["vulnerability"]["severity"] = "Low"
    high["relatedVulnerabilities"][0]["knownExploited"] = [{"cve": "CVE-2025-62727", "knownRansomwareCampaignUse": "Unknown"}]
    assert "GHSA-7f5h-v6xp-fcq8" in _blocking(COVERED, report)


def test_an_expired_entry_covers_nothing(tmp_path, capsys):
    stale = [*COVERED[:2], _entry(STARLETTE_KEV, expires=TODAY - timedelta(days=1))]
    assert _check(tmp_path, stale) == 1
    out = capsys.readouterr().out
    assert f"the {STARLETTE_KEV} entry expired on {TODAY - timedelta(days=1)}" in out
    assert _blocking([*COVERED[:2], _entry(STARLETTE_KEV, expires=TODAY)]) == set(), "an entry applies on its last day"


def test_an_entry_covers_only_the_package_it_names():
    other_version = _entry("CVE-2026-8924", package={"version": "8.14.1-2+deb13u7", "type": "deb"})
    assert "CVE-2026-8924" in _blocking([other_version])
    other_name = _entry("CVE-2026-8927", package={"name": "curl"})
    assert "CVE-2026-8927" in _blocking([other_name]), "the CVE-2026-8927 row is libcurl4t64, not curl"
    assert "CVE-2026-8927" not in _blocking([_entry("CVE-2026-8927", package={"name": "libcurl4t64"})])


def test_a_cve_entry_covers_a_ghsa_row_only_with_include_aliases():
    assert STARLETTE_KEV in _blocking([_entry("CVE-2026-48710")])
    assert STARLETTE_KEV not in _blocking([_entry("CVE-2026-48710", **{"include-aliases": True})])


def test_a_file_that_is_not_a_grype_report_fails(tmp_path, capsys):
    assert _check(tmp_path, [], mode="report", report={"results": []}) == 1
    assert "not a Grype JSON report" in capsys.readouterr().out


# --------------------------------------------------------------------------- the allowlist


@pytest.mark.parametrize("raw,problem", [
    ({"vulnerability": "CVE-1", "expires": "2026-11-01"}, "`reason` must be a non-empty string"),
    ({"vulnerability": "CVE-1", "reason": "x"}, "needs `expires`"),
    ({"vulnerability": "CVE-1", "reason": "x", "expires": "2027-01-17"}, "more than 90 days after 2026-10-18"),
    ({"vulnerability": "CVE-1", "reason": "x", "expires": "next month"}, "is not a YYYY-MM-DD date"),
    ({"reason": "x", "expires": "2026-11-01"}, "`vulnerability` must be a non-empty string"),
    ({"vulnerability": "CVE-1", "reason": "x", "expires": "2026-11-01", "namespace": "nvd:cpe"},
     "keys the policy does not read: namespace"),
    ({"vulnerability": "CVE-1", "reason": "x", "expires": "2026-11-01", "package": {"location": "/usr/**"}},
     "`package` keys the policy does not read: location"),
    ({"vulnerability": "CVE-1", "reason": "x", "expires": "2026-11-01", "package": {"version": 1.8}},
     "`version` must be a non-empty string"),
])
def test_an_entry_the_policy_cannot_read_is_refused(raw, problem):
    with pytest.raises(policy.AllowlistError, match=problem.replace("(", r"\(").replace(")", r"\)")):
        policy.parse_entry(raw, "ignore[0]", TODAY)


def test_ninety_days_out_is_the_limit():
    entry = policy.parse_entry(_entry("CVE-1", expires=TODAY + timedelta(days=90)), "ignore[0]", TODAY)
    assert entry.expires == TODAY + timedelta(days=90)


def test_a_bad_entry_fails_the_check_in_either_mode(tmp_path, capsys):
    assert _check(tmp_path, [*COVERED, {"vulnerability": "CVE-1", "expires": "2026-11-01"}], mode="report") == 1
    assert "::error title=Scan policy::" in capsys.readouterr().out


def test_a_grype_setting_in_the_allowlist_is_refused(tmp_path):
    path = tmp_path / ".grype.yaml"
    path.write_text("only-fixed: true\nignore: []\n", encoding="utf-8")
    with pytest.raises(policy.AllowlistError, match="holds only an `ignore:` list"):
        policy.load_allowlist(path, TODAY)


def test_the_daily_check_fails_once_an_entry_has_expired(tmp_path, capsys):
    _, allowlist = _files(tmp_path, [_entry("CVE-1", expires=TODAY - timedelta(days=1)), _entry("CVE-2")])
    args = ["allowlist", "--allowlist", str(allowlist), "--today", TODAY.isoformat()]
    assert policy.main(args) == 1
    assert "::error title=Allowlist entry expired::CVE-1 expired on 2026-10-17" in capsys.readouterr().out


def test_the_daily_check_passes_and_warns_in_an_entrys_last_week(tmp_path, capsys):
    _, allowlist = _files(tmp_path, [_entry("CVE-1", expires=TODAY + timedelta(days=3)), _entry("CVE-2")])
    assert policy.main(["allowlist", "--allowlist", str(allowlist), "--today", TODAY.isoformat()]) == 0
    assert "::warning title=Allowlist entry expires soon::CVE-1 expires on 2026-10-21" in capsys.readouterr().out


def test_the_committed_allowlist_is_readable_and_its_curl_rows_name_the_cached_version():
    """Every entry has a reason and an expiry at most 90 days out (expiry itself is the daily lane's job)."""
    entries = policy.load_allowlist(ROOT / ".grype.yaml", date.today())
    curl = [entry for entry in entries if entry.vulnerability in CURL_CRITICALS]
    assert {entry.vulnerability for entry in curl} == CURL_CRITICALS
    for entry in curl:
        assert dict(entry.package) == {"version": "8.14.1-2+deb13u5", "type": "deb"}
        assert "US-011" in entry.reason


# --------------------------------------------------------------------------- images.yml


def test_every_build_is_scanned_from_the_registry_it_pushed_to():
    scan = _step("Vulnerability scan")
    assert "if" not in scan and "continue-on-error" not in scan
    assert '"registry:${IMAGE}@${DIGEST}" --platform "linux/${ARCH}"' in scan["run"]
    assert "-o json=report.json" in scan["run"]
    assert scan["env"]["GRYPE_REGISTRY_INSECURE_USE_HTTP"] == "${{ matrix.registry != '' }}"
    assert scan["env"]["IMAGE"] == "${{ matrix.image }}" and scan["env"]["DIGEST"] == "${{ steps.build.outputs.digest }}"
    assert scan["working-directory"] == "${{ runner.temp }}/grype", "outside the checkout: Grype applies no allowlist"


def test_no_build_step_is_allowed_to_fail_and_the_report_only_scan_action_is_gone():
    assert not [step.get("name") for step in BUILD_STEPS if "continue-on-error" in step]
    for step in BUILD_STEPS:
        assert "fail-build" not in (step.get("with") or {}), step.get("name")
        assert not step.get("uses", "").startswith("anchore/scan-action@"), step.get("name")


def test_grype_is_pinned_and_its_database_downloads_while_the_image_builds():
    install = _step("Install Grype")
    assert install["uses"].startswith("anchore/scan-action/download-grype@")
    assert install["with"]["grype-version"] == "v0.118.0", "the policy reads Grype 0.118's knownExploited"
    database = _step("Download the Grype database (background)")
    assert '"$GRYPE" db update' in database["run"] and "nohup" in database["run"] and database["run"].rstrip().endswith("&")
    build = _index(lambda step: step.get("id") == "build")
    assert _index(lambda step: step is database) < build < _index(lambda step: step.get("name") == "Vulnerability scan")
    scan = _step("Vulnerability scan")
    assert "until [ -f db.status ]" in scan["run"] and scan["env"]["GRYPE_DB_AUTO_UPDATE"] == "false"


def test_the_policy_step_judges_the_scan_against_the_allowlist():
    upload = _step("Upload the scan report")
    assert upload["with"]["name"] == "grype-${{ matrix.name }}-${{ matrix.arch }}"
    step = _step("Scan policy")
    assert "if" not in step and step["env"]["POLICY_MODE"] in policy.MODES
    assert "scripts/ci/image_scan_policy.py check" in step["run"]
    for flag in ('--report "$REPORT"', "--allowlist .grype.yaml", '--mode "$POLICY_MODE"', '--summary "$GITHUB_STEP_SUMMARY"'):
        assert flag in step["run"], flag
    assert _index(lambda s: s is upload) < _index(lambda s: s is step), "the report is kept even when the policy fails"


def test_the_merge_job_waits_for_every_scanned_build():
    assert "build" in IMAGES_WORKFLOW["jobs"]["merge"]["needs"]


def test_a_change_to_the_allowlist_or_the_policy_rebuilds_and_rescans_every_image():
    triggers = IMAGES_WORKFLOW.get("on", IMAGES_WORKFLOW.get(True))
    assert ".grype.yaml" in triggers["pull_request"]["paths"]
    pick = next(step for step in IMAGES_WORKFLOW["jobs"]["changes"]["steps"] if step.get("id") == "pick")
    assert ".grype.yaml" in json.loads(pick["env"]["SHARED_INPUTS"])
    assert "scripts/ci/image_*.py" in json.loads(pick["env"]["SHARED_INPUTS"])


# --------------------------------------------------------------------------- the daily lane


def test_the_allowlist_lane_runs_every_day_and_on_allowlist_changes():
    triggers = ALLOWLIST_WORKFLOW.get("on", ALLOWLIST_WORKFLOW.get(True))
    assert triggers["schedule"][0]["cron"].split()[2:] == ["*", "*", "*"], "every day"
    assert {".grype.yaml", "scripts/ci/image_scan_policy.py"} <= set(triggers["pull_request"]["paths"])
    run = "\n".join(step.get("run", "") for step in ALLOWLIST_WORKFLOW["jobs"]["allowlist"]["steps"])
    assert "scripts/ci/image_scan_policy.py allowlist --allowlist .grype.yaml" in run
