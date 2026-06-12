"""PRD-154 S10 — honest pilot surfaces: agent-stats fabrication removed.

BINDING D10 (pilots see Studio). The agent-statistics endpoints fabricated a
flattering ``average_performance: 85.5`` plus zeroed execution placeholders and
a frozen ``2025-08-01`` timestamp. Two byte-identical copies existed:

  * orchestrator/api/agents.py   :: get_agent_stats         (/api/agents/stats — orphaned)
  * orchestrator/api/system.py   :: get_agent_statistics    (/api/system/agent-statistics — the one the Agents page renders)

Both are de-fabricated: the placeholder metrics are dropped (the UI already
degrades to "No data") and the timestamp is now a real UTC value.

Source-guard test — reads the two modules' text, so no ``modules.*`` import and
no collection-order guard is required.
"""

from pathlib import Path

_ORCH = Path(__file__).resolve().parent.parent
_AGENTS = _ORCH / "api" / "agents.py"
_SYSTEM = _ORCH / "api" / "system.py"

# ``85.5`` and ``average_performance`` are unique to the fabricated stats block;
# they appear nowhere else in either module (confirmed by grep), so whole-file
# absence is a faithful, scoped guard. (The frozen 2025-08-01 timestamp also
# lives in unrelated status/execution stubs outside S10's scope, so it is NOT
# asserted globally here.)
def test_agents_py_stats_drops_fabricated_performance():
    text = _AGENTS.read_text()
    assert "85.5" not in text, "agents.py still fabricates the 85.5 performance number"
    assert '"average_performance"' not in text


def test_system_py_stats_drops_fabricated_performance():
    text = _SYSTEM.read_text()
    assert "85.5" not in text, "system.py still fabricates the 85.5 performance number"
    assert '"average_performance"' not in text


def test_both_stats_endpoints_use_a_real_timestamp():
    for path in (_AGENTS, _SYSTEM):
        text = path.read_text()
        # The stats blocks now derive their timestamp from a real UTC clock.
        assert "datetime.now(timezone.utc).isoformat()" in text, (
            f"{path.name} stats must use a real UTC timestamp, not a frozen literal"
        )
