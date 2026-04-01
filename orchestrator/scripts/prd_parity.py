#!/usr/bin/env python3
"""
PRD Parity Audit Script — PRD-123 Pattern #9
=============================================

Checks PRDs against the codebase to verify feature completeness.

Usage:
    python scripts/prd_parity.py docs/PRDS/ --output parity_report.json

Parses PRD markdown files, extracts "What Ships" table entries, and checks
if corresponding files, endpoints, or models exist in the codebase.
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParityEntry:
    """Single component from a PRD's What Ships table."""
    component: str
    expected: str
    found: bool = False
    coverage_note: str = ""


@dataclass
class PrdResult:
    """Parity result for a single PRD."""
    prd_id: str
    prd_title: str
    entries: list[ParityEntry] = field(default_factory=list)

    @property
    def coverage_percent(self) -> float:
        if not self.entries:
            return 0.0
        found = sum(1 for e in self.entries if e.found)
        return round((found / len(self.entries)) * 100, 1)


@dataclass
class ParityReport:
    """Aggregate parity report across all PRDs."""
    prds: list[PrdResult] = field(default_factory=list)

    @property
    def aggregate_coverage(self) -> float:
        total = sum(len(p.entries) for p in self.prds)
        found = sum(1 for p in self.prds for e in p.entries if e.found)
        return round((found / total) * 100, 1) if total else 0.0

    def as_dict(self) -> dict:
        return {
            "aggregate_coverage_percent": self.aggregate_coverage,
            "prd_count": len(self.prds),
            "prds": [
                {
                    "prd_id": p.prd_id,
                    "prd_title": p.prd_title,
                    "coverage_percent": p.coverage_percent,
                    "entries": [asdict(e) for e in p.entries],
                }
                for p in self.prds
            ],
        }


def _extract_prd_id(filename: str) -> str:
    """Extract PRD number from filename like '123-HARNESS-PATTERN-ADOPTION.md'."""
    match = re.match(r"(\d+)", filename)
    return f"PRD-{match.group(1)}" if match else filename


def _extract_title(content: str) -> str:
    """Extract first H1 title from markdown."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"


def _extract_what_ships(content: str) -> list[tuple[str, str]]:
    """
    Extract rows from 'What Ships' table in PRD markdown.

    Returns list of (component, description) tuples.
    """
    entries = []

    # Look for "What Ships" section
    ships_match = re.search(
        r"(?:What Ships|Deliverables|Components).*?\n\|.*?\n\|[-\s|]+\n((?:\|.+\n)+)",
        content,
        re.IGNORECASE,
    )
    if not ships_match:
        return entries

    table_rows = ships_match.group(1)
    for row in table_rows.strip().split("\n"):
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if len(cells) >= 2:
            entries.append((cells[0], cells[1]))

    return entries


def _check_component_exists(component: str, base_path: Path) -> tuple[bool, str]:
    """
    Check if a component exists in the codebase.

    Looks for:
    - File paths mentioned in the component
    - Python class/function names
    - API endpoint patterns
    """
    # Check for file path patterns
    path_match = re.search(r"[\w/]+\.py", component)
    if path_match:
        filepath = base_path / "orchestrator" / path_match.group()
        if filepath.exists():
            return True, f"File exists: {filepath.name}"
        return False, f"File not found: {path_match.group()}"

    # Check for class/function names (CamelCase or snake_case)
    name_match = re.search(r"[A-Z]\w+(?:Service|Section|Model|Router|Guard)", component)
    if name_match:
        name = name_match.group()
        # Search in orchestrator directory
        for root, _dirs, files in os.walk(base_path / "orchestrator"):
            for f in files:
                if f.endswith(".py"):
                    try:
                        with open(os.path.join(root, f)) as fh:
                            if f"class {name}" in fh.read():
                                return True, f"Class found in {f}"
                    except Exception:
                        continue
        return False, f"Class {name} not found"

    # Check for endpoint patterns
    endpoint_match = re.search(r"(GET|POST|PUT|DELETE)\s+(/[\w/{}\-]+)", component)
    if endpoint_match:
        method, path = endpoint_match.groups()
        for root, _dirs, files in os.walk(base_path / "orchestrator" / "api"):
            for f in files:
                if f.endswith(".py"):
                    try:
                        with open(os.path.join(root, f)) as fh:
                            content = fh.read()
                            # Normalize path for matching
                            normalized = path.replace("{", "").replace("}", "")
                            if normalized in content or path in content:
                                return True, f"Endpoint found in {f}"
                    except Exception:
                        continue
        return False, f"Endpoint {method} {path} not found"

    # Default: can't determine
    return False, "Could not verify automatically"


def audit_prd(prd_path: Path, base_path: Path) -> PrdResult:
    """Audit a single PRD against the codebase."""
    content = prd_path.read_text()
    prd_id = _extract_prd_id(prd_path.name)
    title = _extract_title(content)

    result = PrdResult(prd_id=prd_id, prd_title=title)

    what_ships = _extract_what_ships(content)
    for component, description in what_ships:
        found, note = _check_component_exists(component, base_path)
        result.entries.append(ParityEntry(
            component=component,
            expected=description,
            found=found,
            coverage_note=note,
        ))

    return result


def audit_directory(prd_dir: Path, base_path: Path) -> ParityReport:
    """Audit all PRDs in a directory."""
    report = ParityReport()

    for prd_file in sorted(prd_dir.glob("*.md")):
        if prd_file.name.startswith("_") or prd_file.name.lower() == "readme.md":
            continue
        try:
            result = audit_prd(prd_file, base_path)
            if result.entries:  # Only include PRDs with What Ships tables
                report.prds.append(result)
        except Exception as exc:
            logger.warning("Failed to audit %s: %s", prd_file.name, exc)

    return report


def main():
    parser = argparse.ArgumentParser(description="PRD Parity Audit")
    parser.add_argument("prd_dir", type=Path, help="Directory containing PRD markdown files")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON report path")
    parser.add_argument("--base", type=Path, default=None, help="Project base directory")

    args = parser.parse_args()

    base_path = args.base or args.prd_dir.parent.parent
    report = audit_directory(args.prd_dir, base_path)

    output = report.as_dict()

    if args.output:
        args.output.write_text(json.dumps(output, indent=2))
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(output, indent=2))

    print(f"\nAggregate coverage: {report.aggregate_coverage}%")
    for prd in report.prds:
        print(f"  {prd.prd_id}: {prd.coverage_percent}% ({prd.prd_title})")

    return 0 if report.aggregate_coverage >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
