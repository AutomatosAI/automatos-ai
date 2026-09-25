"""The code-shape check on changed Python code (``scripts/ci/check_changed_code_shape.py``).

The check is pure apart from its git calls, so these guards feed it fixture source and
fixture diffs: a touched long function fails, an untouched one passes, a docstring and
blank lines don't count, deep nesting fails and a nested scope starts again, a new big
file fails and a grown old one only warns. Same loading posture as the schema-drift
guards (``test_prd209_schema_drift.py``).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import textwrap

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CHECK_PATH = _REPO_ROOT / "scripts" / "ci" / "check_changed_code_shape.py"


def _load_check():
    spec = importlib.util.spec_from_file_location("check_changed_code_shape", _CHECK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Register before exec so dataclasses can resolve the module during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shape = _load_check()


def _function(name: str, body_lines: int, docstring: bool = False, blanks: int = 0) -> str:
    lines = [f"def {name}():"]
    if docstring:
        lines += ['    """A docstring', "", "    that spans lines.", '    """']
    lines += [f"    x{i} = {i}" for i in range(body_lines)]
    lines += [""] * blanks
    lines.append("    return None")
    return "\n".join(lines) + "\n"


def test_touched_long_function_fails():
    source = _function("too_long", body_lines=shape.MAX_FUNCTION_LINES)  # + def + return
    findings = shape.function_findings("pkg/mod.py", source, touched={3})
    assert [f.level for f in findings] == ["error"]
    assert "too_long()" in findings[0].message


def test_untouched_long_function_passes():
    source = _function("old_and_long", body_lines=shape.MAX_FUNCTION_LINES + 20) + "\n\ndef other():\n    return 1\n"
    last = source.count("\n")
    assert shape.function_findings("pkg/mod.py", source, touched={last}) == []


def test_docstring_blank_and_comment_lines_do_not_count():
    source = _function("documented", body_lines=shape.MAX_FUNCTION_LINES - 2, docstring=True, blanks=5)
    source = source.replace("    x0 = 0\n", "    # a comment\n    x0 = 0\n")
    assert shape.function_findings("pkg/mod.py", source, touched={2}) == []


def test_nesting_over_the_limit_fails_and_at_the_limit_passes():
    deep = textwrap.dedent("""\
        def deep(items):
            for a in items:
                if a:
                    while a:
                        try:
                            with open(a) as f:
                                return f
                        except OSError:
                            return None
    """)
    findings = shape.function_findings("pkg/mod.py", deep, touched={1})
    assert any("nests 5 levels" in f.message for f in findings)
    at_limit = textwrap.dedent("""\
        def at_limit(items):
            for a in items:
                if a:
                    while a:
                        try:
                            return a
                        except OSError:
                            return None
    """)
    assert not any("nests" in f.message for f in shape.function_findings("pkg/mod.py", at_limit, touched={1}))


def test_nested_function_starts_a_new_scope():
    source = textwrap.dedent("""\
        def outer(items):
            for a in items:
                if a:
                    def inner(b):
                        if b:
                            for c in b:
                                if c:
                                    return c
                    return inner
    """)
    findings = shape.function_findings("pkg/mod.py", source, touched={1})
    assert not any("outer()" in f.message and "nests" in f.message for f in findings)


def test_changed_lines_reads_new_side_hunks():
    diff = textwrap.dedent("""\
        diff --git a/pkg/mod.py b/pkg/mod.py
        --- a/pkg/mod.py
        +++ b/pkg/mod.py
        @@ -10,0 +11,3 @@ def f():
        @@ -40 +44 @@ def g():
        diff --git a/pkg/gone.py b/pkg/gone.py
        --- a/pkg/gone.py
        +++ /dev/null
        @@ -1,5 +0,0 @@
    """)
    assert shape.changed_lines(diff) == {"pkg/mod.py": {11, 12, 13, 44}}


def test_file_size_new_file_errors_and_grown_file_warns():
    limit = shape.MAX_FILE_LINES
    assert shape.file_size_finding("new.py", True, 0, limit + 1).level == "error"
    assert shape.file_size_finding("old.py", False, limit + 10, limit + 20).level == "warning"
    assert shape.file_size_finding("old.py", False, limit + 20, limit + 10) is None
    assert shape.file_size_finding("small.py", True, 0, limit) is None


def test_tests_and_migrations_are_exempt_from_length_and_nesting():
    assert shape.is_exempt("orchestrator/tests/test_x.py")
    assert shape.is_exempt("orchestrator/alembic/versions/abc_rev.py")
    assert shape.is_exempt("services/media-render/tests/test_server.py")
    assert not shape.is_exempt("orchestrator/modules/socials/service.py")


def test_report_exits_1_on_errors_and_0_on_warnings_only(capsys):
    warning = shape.Finding("warning", "a.py", 1, "grew")
    error = shape.Finding("error", "b.py", 3, "too long")
    assert shape.report([warning]) == 0
    assert shape.report([warning, error]) == 1
    assert "::error file=b.py,line=3::too long" in capsys.readouterr().out
