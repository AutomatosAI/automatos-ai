"""The vulnerability scan policy for images.yml (PRD-262 US-013, D1, FR-3).

Two commands:

* ``check`` (each images.yml build job, after its Grype scan) reads Grype's JSON report
  for one image and arch, and the allowlist ``.grype.yaml``. A row blocks when it is a
  fixable Critical (severity Critical, ``fix.state`` ``fixed``), or when it is on CISA's
  Known Exploited Vulnerabilities list, whatever its severity, and no unexpired
  allowlist entry matches it. In ``report`` mode the blocking rows are warnings and the
  step passes; in ``fail`` mode they fail it.
* ``allowlist`` (the daily scan-allowlist workflow) fails once an entry has expired,
  and warns a week before.

The KEV flag is the ``knownExploited`` list Grype 0.118 writes on a match's
``vulnerability`` (``grype/presenter/models/vulnerability_metadata.go`` at v0.118.0;
its table marks the same rows ``(kev)``). The policy also reads it on the match's
``relatedVulnerabilities``, so a GHSA row whose CVE alias is listed blocks too.

The allowlist is Grype's own ignore-rule format plus ``expires``::

    ignore:
      - vulnerability: CVE-2026-8924
        package: {version: 8.14.1-2+deb13u5, type: deb}   # optional: name, version, type
        include-aliases: true      # optional: also match a row whose related ids hold it
        fix-state: fixed           # optional
        reason: why it is accepted
        expires: 2026-11-17        # the last day (UTC) it applies, at most 90 days out

Both commands refuse an entry without ``vulnerability``, ``reason`` or ``expires``, one
that expires more than 90 days out, and any key the policy does not read, so an entry
never matches more, or less, than it says. images.yml runs Grype outside the checkout,
so Grype never applies the file itself: this script is the one place an entry takes
effect, and Grype's table in the log still shows every row.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

MODES = ("report", "fail")
MAX_DAYS = 90
WARN_DAYS = 7
FIXED = "fixed"
CRITICAL = "critical"
ENTRY_KEYS = frozenset({"vulnerability", "reason", "expires", "include-aliases", "fix-state", "package"})
PACKAGE_KEYS = ("name", "version", "type")
SEVERITIES = ("Critical", "High", "Medium", "Low", "Negligible", "Unknown")


class AllowlistError(ValueError):
    """The allowlist, or an entry of it, that the policy refuses to read."""


class ReportError(ValueError):
    """A file that is not a Grype JSON report."""


@dataclass(frozen=True)
class Row:
    """One Grype match, reduced to what the policy reads."""

    vulnerability: str
    aliases: Tuple[str, ...]
    severity: str
    fix_state: str
    fixed_in: Tuple[str, ...]
    kev: bool
    name: str
    version: str
    type: str

    @property
    def blocks(self) -> Tuple[str, ...]:
        """Why the policy blocks this row: a fixable Critical, a KEV match, or both."""
        why = []
        if self.severity.lower() == CRITICAL and self.fix_state == FIXED:
            why.append("fixable Critical")
        if self.kev:
            why.append("KEV")
        return tuple(why)


@dataclass(frozen=True)
class Entry:
    """One allowlist entry: what it matches, why, and until when."""

    vulnerability: str
    reason: str
    expires: date
    include_aliases: bool = False
    fix_state: str = ""
    package: Tuple[Tuple[str, str], ...] = ()

    def matches(self, row: Row) -> bool:
        """Every field the entry sets equals the row's, as Grype's ignore rules match."""
        ids = (row.vulnerability, *row.aliases) if self.include_aliases else (row.vulnerability,)
        if self.vulnerability not in ids:
            return False
        if self.fix_state and self.fix_state != row.fix_state:
            return False
        return all(getattr(row, key) == value for key, value in self.package)

    def expired(self, today: date) -> bool:
        """True after the entry's last day."""
        return today > self.expires


@dataclass(frozen=True)
class Verdict:
    """The rows the policy blocks, the ones the allowlist covers, and expired cover."""

    blocking: Tuple[Row, ...]
    allowlisted: Tuple[Tuple[Row, Entry], ...]
    expired: Tuple[Tuple[Row, Entry], ...]


# --------------------------------------------------------------------------- allowlist


def _text(raw: dict, key: str, where: str, required: bool = True) -> str:
    value = raw.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise AllowlistError(f"{where}: `{key}` must be a non-empty string (quote a value YAML reads as a number)")
    return value.strip()


def _expiry(value: object, where: str) -> date:
    if value is None:
        raise AllowlistError(f"{where}: needs `expires`, the last day the entry applies (YYYY-MM-DD)")
    if isinstance(value, datetime):
        raise AllowlistError(f"{where}: `expires` is a day (YYYY-MM-DD), not a time")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as error:
        raise AllowlistError(f"{where}: `expires` {value!r} is not a YYYY-MM-DD date") from error


def _package(value: object, where: str) -> Tuple[Tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise AllowlistError(f"{where}: `package` is a mapping of {', '.join(PACKAGE_KEYS)}")
    unknown = sorted(set(value) - set(PACKAGE_KEYS))
    if unknown:
        raise AllowlistError(f"{where}: `package` keys the policy does not read: {', '.join(unknown)}")
    return tuple((key, _text(value, key, f"{where} package")) for key in PACKAGE_KEYS if key in value)


def parse_entry(raw: object, where: str, today: date) -> Entry:
    """One ``ignore:`` item as an Entry; AllowlistError says what is wrong with it."""
    if not isinstance(raw, dict):
        raise AllowlistError(f"{where}: an entry is a mapping")
    unknown = sorted(set(raw) - ENTRY_KEYS)
    if unknown:
        raise AllowlistError(f"{where}: keys the policy does not read: {', '.join(unknown)}")
    vulnerability = _text(raw, "vulnerability", where)
    where = f"{where} ({vulnerability})"
    expires = _expiry(raw.get("expires"), where)
    if (expires - today).days > MAX_DAYS:
        raise AllowlistError(f"{where}: expires {expires}, more than {MAX_DAYS} days after {today}")
    aliases = raw.get("include-aliases", False)
    if not isinstance(aliases, bool):
        raise AllowlistError(f"{where}: `include-aliases` is true or false")
    return Entry(
        vulnerability=vulnerability,
        reason=_text(raw, "reason", where),
        expires=expires,
        include_aliases=aliases,
        fix_state=_text(raw, "fix-state", where, required=False),
        package=_package(raw.get("package"), where),
    )


def _ignore_list(path: Path) -> list:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise AllowlistError(f"{path.name}: not YAML ({error})") from error
    data = {} if data is None else data
    if not isinstance(data, dict) or set(data) - {"ignore"}:
        raise AllowlistError(f"{path.name}: holds only an `ignore:` list. images.yml runs Grype outside the "
                             "checkout, so a Grype setting here would do nothing")
    items = data.get("ignore") or []
    if not isinstance(items, list):
        raise AllowlistError(f"{path.name}: `ignore` is a list of entries")
    return items


def load_allowlist(path: Path, today: date) -> List[Entry]:
    """Every entry, or AllowlistError listing every entry the policy refuses."""
    entries, problems = [], []
    for index, raw in enumerate(_ignore_list(path)):
        try:
            entries.append(parse_entry(raw, f"{path.name} ignore[{index}]", today))
        except AllowlistError as error:
            problems.append(str(error))
    if problems:
        raise AllowlistError("\n".join(problems))
    return entries


# --------------------------------------------------------------------------- the report


def _known_exploited(match: dict) -> bool:
    related = match.get("relatedVulnerabilities") or []
    listed = (match.get("vulnerability") or {}).get("knownExploited")
    return bool(listed) or any(item.get("knownExploited") for item in related if isinstance(item, dict))


def to_row(match: dict) -> Row:
    """A Grype 0.118 ``matches[]`` item as a Row."""
    vulnerability = match.get("vulnerability") or {}
    artifact = match.get("artifact") or {}
    fix = vulnerability.get("fix") or {}
    related = match.get("relatedVulnerabilities") or []
    return Row(
        vulnerability=str(vulnerability.get("id") or ""),
        aliases=tuple(str(item["id"]) for item in related if isinstance(item, dict) and item.get("id")),
        severity=str(vulnerability.get("severity") or "Unknown"),
        fix_state=str(fix.get("state") or ""),
        fixed_in=tuple(str(version) for version in fix.get("versions") or []),
        kev=_known_exploited(match),
        name=str(artifact.get("name") or ""),
        version=str(artifact.get("version") or ""),
        type=str(artifact.get("type") or ""),
    )


def rows(report: object) -> List[Row]:
    """Every match of a Grype JSON report, once each, most severe first."""
    matches = report.get("matches") if isinstance(report, dict) else None
    if not isinstance(matches, list):
        raise ReportError("not a Grype JSON report: it has no `matches` list")
    order = {severity.lower(): rank for rank, severity in enumerate(SEVERITIES)}
    found = {to_row(match) for match in matches if isinstance(match, dict)}
    return sorted(found, key=lambda row: (order.get(row.severity.lower(), len(order)), row.name, row.vulnerability))


def evaluate(found: Sequence[Row], entries: Sequence[Entry], today: date) -> Verdict:
    """Split the rows the policy blocks into allowlisted and blocking ones."""
    blocking, allowlisted, expired = [], [], []
    for row in (item for item in found if item.blocks):
        live = next((entry for entry in entries if not entry.expired(today) and entry.matches(row)), None)
        if live:
            allowlisted.append((row, live))
            continue
        stale = next((entry for entry in entries if entry.matches(row)), None)
        if stale:
            expired.append((row, stale))
        blocking.append(row)
    return Verdict(tuple(blocking), tuple(allowlisted), tuple(expired))


# --------------------------------------------------------------------------- output


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def describe(row: Row) -> str:
    """One line naming the package, the vulnerability and why it blocks."""
    fixed = f", fixed in {', '.join(row.fixed_in)}" if row.fix_state == FIXED and row.fixed_in else ""
    return f"{row.name} {row.version} ({row.type}) {row.vulnerability} {row.severity}{fixed}: {' and '.join(row.blocks)}"


def annotate(verdict: Verdict, mode: str, label: str) -> None:
    """Workflow annotations: an error per blocking row in fail mode, a warning in report mode."""
    level, title = ("error", "Scan policy") if mode == "fail" else ("warning", "Scan policy (report only)")
    for row in verdict.blocking:
        print(f"::{level} title={title}::{label}: {describe(row)}")
    for row, entry in verdict.expired:
        print(f"::warning title=Allowlist entry expired::{label}: the {entry.vulnerability} entry expired on "
              f"{entry.expires}, so {row.name} {row.version} is no longer allowlisted")


def _rows_table(verdict: Verdict) -> List[str]:
    lines = []
    if verdict.blocking:
        lines += ["", "| Package | Version | Type | Vulnerability | Severity | Fixed in | Blocks as |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {row.name} | {_cell(row.version)} | {row.type} | {row.vulnerability} | {row.severity} | "
                  f"{_cell(', '.join(row.fixed_in)) or '–'} | {' and '.join(row.blocks)} |" for row in verdict.blocking]
    if verdict.allowlisted:
        lines += ["", "Allowlisted in `.grype.yaml`:", "", "| Package | Version | Vulnerability | Until | Reason |",
                  "|---|---|---|---|---|"]
        lines += [f"| {row.name} | {_cell(row.version)} | {row.vulnerability} | {entry.expires} | {_cell(entry.reason)} |"
                  for row, entry in verdict.allowlisted]
    return lines


def summary_markdown(found: Sequence[Row], verdict: Verdict, mode: str, label: str, grype: str) -> str:
    """The step summary: severity counts, the policy's verdict, and the rows behind it."""
    counts = Counter(row.severity.capitalize() for row in found)
    tally = ", ".join(f"{counts[severity]} {severity}" for severity in SEVERITIES if counts[severity]) or "no matches"
    blocking = len(verdict.blocking)
    if not blocking:
        outcome = "nothing blocks"
    elif mode == "fail":
        outcome = f"**{blocking} row(s) block the build**"
    else:
        outcome = f"{blocking} row(s) would block; report only, so the build carries on"
    lines = [f"### Vulnerability scan: {label}", "", f"Grype {grype}: {tally}.", "",
             f"**Scan policy** ({mode} mode; fixable Critical, or KEV at any severity): {outcome}, "
             f"{len(verdict.allowlisted)} allowlisted."]
    return "\n".join(lines + _rows_table(verdict)) + "\n"


def _append(path: Optional[str], text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as summary:
            summary.write(text + "\n")


def _refuse(error: ValueError, title: str) -> int:
    for line in str(error).splitlines():
        print(f"::error title={title}::{line}")
    return 1


def _today(value: Optional[str]) -> date:
    return date.fromisoformat(value) if value else datetime.now(timezone.utc).date()


# --------------------------------------------------------------------------- commands


def check(args: argparse.Namespace) -> int:
    """Judge one image's scan; exit 1 only in fail mode with a blocking row, or on bad input."""
    today = _today(args.today)
    label = f"{args.name} ({args.arch})"
    try:
        entries = load_allowlist(Path(args.allowlist), today)
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        found = rows(report)
    except (AllowlistError, ReportError, json.JSONDecodeError) as error:
        return _refuse(error, "Scan policy")
    verdict = evaluate(found, entries, today)
    annotate(verdict, args.mode, label)
    grype = str((report.get("descriptor") or {}).get("version") or "(version not reported)")
    _append(args.summary, summary_markdown(found, verdict, args.mode, label, grype))
    print(f"{label}: {len(found)} matches, {len(verdict.blocking)} blocking, "
          f"{len(verdict.allowlisted)} allowlisted ({args.mode} mode)")
    return 1 if verdict.blocking and args.mode == "fail" else 0


def allowlist(args: argparse.Namespace) -> int:
    """Fail once an entry has expired; warn in its last week."""
    today = _today(args.today)
    try:
        entries = load_allowlist(Path(args.allowlist), today)
    except AllowlistError as error:
        return _refuse(error, "Scan allowlist")
    expired = [entry for entry in entries if entry.expired(today)]
    for entry in expired:
        print(f"::error title=Allowlist entry expired::{entry.vulnerability} expired on {entry.expires}. "
              f"Fix the image, or renew the entry with a current reason ({entry.reason})")
    for entry in entries:
        if not entry.expired(today) and (entry.expires - today).days < WARN_DAYS:
            print(f"::warning title=Allowlist entry expires soon::{entry.vulnerability} expires on {entry.expires}")
    lines = [f"### Scan allowlist on {today}", "", "| Vulnerability | Until | Status | Reason |", "|---|---|---|---|"]
    lines += [f"| {entry.vulnerability} | {entry.expires} | {'expired' if entry.expired(today) else 'live'} | "
              f"{_cell(entry.reason)} |" for entry in entries]
    _append(args.summary, "\n".join(lines) + "\n")
    print(f"{len(entries)} entries, {len(expired)} expired")
    return 1 if expired else 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)
    one = commands.add_parser("check", help="judge one image's Grype JSON report")
    for flag in ("--report", "--allowlist", "--name", "--arch"):
        one.add_argument(flag, required=True)
    one.add_argument("--mode", required=True, choices=MODES)
    every = commands.add_parser("allowlist", help="fail once an allowlist entry has expired")
    every.add_argument("--allowlist", required=True)
    for command in (one, every):
        command.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
        command.add_argument("--today", help="YYYY-MM-DD in place of today's UTC date (tests)")
    return root


def main(argv: List[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return {"check": check, "allowlist": allowlist}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
