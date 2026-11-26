"""PRD-282 US-303: the chart stops a render on exactly the log settings the code cannot use.

``charts/automatos/templates/_logs.tpl`` lists the formats and the level names a render
accepts (``automatos.logFormats``, ``automatos.logLevels``), after trimming and in any
case. The API and the worker read the same settings with ``log_settings``. This test reads
both lists, so the chart can neither refuse a value the code takes nor pass one it does not.

Pure file reads: no helm.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.monitoring import log_settings

ROOT = Path(__file__).resolve().parents[2]
LOGS_TEMPLATE = (ROOT / "charts" / "automatos" / "templates" / "_logs.tpl").read_text(encoding="utf-8")


def _listed(name: str) -> list[str]:
    found = re.search(r'\{\{- define "' + re.escape(name) + r'" -\}\}\n(.*?)\n\{\{- end \}\}', LOGS_TEMPLATE, re.S)
    assert found, f"{name} is not defined in _logs.tpl"
    return found.group(1).split()


def test_the_chart_accepts_the_formats_the_code_writes():
    assert _listed("automatos.logFormats") == list(log_settings.LOG_FORMATS)


def test_the_chart_accepts_the_levels_the_code_reads():
    assert _listed("automatos.logLevels") == list(log_settings.LOG_LEVELS)


def test_the_chart_reads_them_as_the_code_does():
    assert "LOG_FORMAT\" | toString | trim | lower" in LOGS_TEMPLATE
    assert "LOG_LEVEL\" | toString | trim | upper" in LOGS_TEMPLATE
    assert "NOTSET" not in _listed("automatos.logLevels")
