#!/usr/bin/env python3
"""Code shape on CHANGED Python code (AGENTS.md → Code shape).

Linters count statements or complexity; the repository's rule is about size a reader
can hold in their head. This check holds every Python function a change touches to:

* **Length.** At most ``MAX_FUNCTION_LINES`` code lines. Blank lines, comment-only
  lines and the docstring don't count.
* **Nesting.** At most ``MAX_NESTING`` levels of compound statements (``if``, ``for``,
  ``while``, ``try``, ``with``, ``match``). A nested function or class starts again
  at zero.
* **File size.** A NEW file has at most ``MAX_FILE_LINES`` lines. An existing file
  already over that size may still change, but growing it prints a warning:
  split it instead.

Only the change is judged. A function counts as touched when any line of it is in
the diff against the base (``git diff <merge-base>``). Existing code meets the rule
the next time someone edits it, the same ratchet the TypeScript baseline uses.
Tests and migrations are exempt from length and nesting.

Output is GitHub annotations (``::error file=…,line=…::``), so failures show inline
on the pull request. Exit 1 on any error; warnings alone exit 0.

    python scripts/ci/check_changed_code_shape.py [--base origin/main]
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

MAX_FUNCTION_LINES = 50
MAX_NESTING = 4
MAX_FILE_LINES = 800

NESTING_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
if hasattr(ast, "TryStar"):  # Python 3.11+
    NESTING_NODES = NESTING_NODES + (ast.TryStar,)
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
EXEMPT_PATTERNS = (
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)test_[^/]*\.py$"),
    re.compile(r"_test\.py$"),
    re.compile(r"(^|/)conftest\.py$"),
    re.compile(r"(^|/)alembic/versions/"),
)
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Finding:
    level: str  # "error" or "warning"
    path: str
    line: int
    message: str

    def annotation(self) -> str:
        return f"::{self.level} file={self.path},line={self.line}::{self.message}"


def changed_lines(diff_text: str) -> Dict[str, Set[int]]:
    """New-side line numbers per file, from ``git diff -U0`` output."""
    result: Dict[str, Set[int]] = {}
    current: Optional[str] = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            current = target[2:] if target.startswith("b/") else None
            if current is not None:
                result.setdefault(current, set())
            continue
        match = HUNK.match(line)
        if match and current is not None:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            result[current].update(range(start, start + count))
    return result


def is_exempt(path: str) -> bool:
    return any(pattern.search(path) for pattern in EXEMPT_PATTERNS)


def _docstring_lines(node: ast.AST) -> Set[int]:
    body = getattr(node, "body", None) or []
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        return set(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return set()


def code_line_count(node: ast.AST, source_lines: List[str]) -> int:
    """Lines of the function that hold code: not blank, not comment-only, not its docstring."""
    skip = _docstring_lines(node)
    count = 0
    for number in range(node.lineno, (node.end_lineno or node.lineno) + 1):
        text = source_lines[number - 1].strip() if number - 1 < len(source_lines) else ""
        if number in skip or not text or text.startswith("#"):
            continue
        count += 1
    return count


def max_nesting(node: ast.AST) -> int:
    """Deepest chain of compound statements inside ``node``; a nested scope starts again."""
    def walk(current: ast.AST, depth: int) -> int:
        deepest = depth
        for child in ast.iter_child_nodes(current):
            if isinstance(child, SCOPE_NODES):
                continue
            child_depth = depth + 1 if isinstance(child, NESTING_NODES) else depth
            deepest = max(deepest, walk(child, child_depth))
        return deepest
    return walk(node, 0)


def function_findings(path: str, source: str, touched: Set[int]) -> List[Finding]:
    """Length and nesting findings for the functions of ``source`` that ``touched`` reaches."""
    tree = ast.parse(source, filename=path)
    lines = source.splitlines()
    findings: List[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        span = range(node.lineno, (node.end_lineno or node.lineno) + 1)
        if not any(number in touched for number in span):
            continue
        length = code_line_count(node, lines)
        if length > MAX_FUNCTION_LINES:
            findings.append(Finding("error", path, node.lineno,
                                    f"{node.name}() has {length} code lines (max {MAX_FUNCTION_LINES}). "
                                    "Split it into smaller functions."))
        depth = max_nesting(node)
        if depth > MAX_NESTING:
            findings.append(Finding("error", path, node.lineno,
                                    f"{node.name}() nests {depth} levels deep (max {MAX_NESTING}). "
                                    "Return early or extract a helper."))
    return findings


def file_size_finding(path: str, is_new: bool, lines_before: int, lines_after: int) -> Optional[Finding]:
    """A new file over the limit is an error; an existing oversized file that grew is a warning."""
    if lines_after <= MAX_FILE_LINES:
        return None
    if is_new:
        return Finding("error", path, 1, f"new file has {lines_after} lines (max {MAX_FILE_LINES}). Split it by concern.")
    if lines_after > lines_before:
        return Finding("warning", path, 1,
                       f"file grew to {lines_after} lines (limit {MAX_FILE_LINES}); split it instead of growing it.")
    return None


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _line_count(ref: str, path: str) -> int:
    try:
        return len(_git("show", f"{ref}:{path}").splitlines())
    except subprocess.CalledProcessError:
        return 0


def check(base: str) -> List[Finding]:
    root = _git("rev-parse", "--show-toplevel").strip()
    merge_base = _git("merge-base", "HEAD", base).strip()
    added = set(_git("diff", "--name-only", "--diff-filter=A", merge_base, "HEAD", "--", "*.py").split())
    touched_by_file = changed_lines(_git("diff", "-U0", "--diff-filter=AMR", merge_base, "HEAD", "--", "*.py"))
    findings: List[Finding] = []
    for path, touched in sorted(touched_by_file.items()):
        with open(f"{root}/{path}", encoding="utf-8") as handle:
            source = handle.read()
        size = file_size_finding(path, path in added, _line_count(merge_base, path), len(source.splitlines()))
        if size is not None:
            findings.append(size)
        if not is_exempt(path):
            findings.extend(function_findings(path, source, touched))
    return findings


def report(findings: Iterable[Finding]) -> int:
    errors = 0
    for finding in findings:
        print(finding.annotation())
        errors += finding.level == "error"
    print(f"code shape: {errors} error(s) on changed Python code "
          f"(functions ≤{MAX_FUNCTION_LINES} code lines, nesting ≤{MAX_NESTING}, new files ≤{MAX_FILE_LINES} lines)")
    return 1 if errors else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main", help="branch or commit the change is measured against")
    args = parser.parse_args(argv)
    return report(check(args.base))


if __name__ == "__main__":
    sys.exit(main())
