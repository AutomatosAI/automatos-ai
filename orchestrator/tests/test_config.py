"""
Tests for orchestrator/config.py — PRD-138 US-001
==================================================

Verifies SEMANTIC_TOOL_ROUTING_TOP_K is exposed via the canonical Config
module, defaults to 15 when unset, and honours environment overrides.

The config module reads env vars at class definition time, so each test
reloads it after manipulating os.environ.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


def _reload_config(monkeypatch, env_value):
    """Set/clear SEMANTIC_TOOL_ROUTING_TOP_K and reload the config module."""
    if env_value is None:
        monkeypatch.delenv("SEMANTIC_TOOL_ROUTING_TOP_K", raising=False)
    else:
        monkeypatch.setenv("SEMANTIC_TOOL_ROUTING_TOP_K", env_value)

    # If already imported, drop it so the class-level os.getenv re-runs
    sys.modules.pop("config", None)
    import config  # noqa: WPS433 — intentional re-import after env change
    importlib.reload(config)
    return config


def test_semantic_tool_routing_top_k_default(monkeypatch):
    """When SEMANTIC_TOOL_ROUTING_TOP_K is unset, config exposes default 15."""
    config_module = _reload_config(monkeypatch, None)

    assert hasattr(config_module.config, "SEMANTIC_TOOL_ROUTING_TOP_K")
    assert config_module.config.SEMANTIC_TOOL_ROUTING_TOP_K == 15
    assert isinstance(config_module.config.SEMANTIC_TOOL_ROUTING_TOP_K, int)


def test_semantic_tool_routing_top_k_override(monkeypatch):
    """When SEMANTIC_TOOL_ROUTING_TOP_K=20 in env, config returns 20."""
    config_module = _reload_config(monkeypatch, "20")

    assert config_module.config.SEMANTIC_TOOL_ROUTING_TOP_K == 20
    assert isinstance(config_module.config.SEMANTIC_TOOL_ROUTING_TOP_K, int)
