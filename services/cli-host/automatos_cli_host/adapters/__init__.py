"""The adapters — one per CLI with behaviour the preset cannot express.

``adapter_for(cli_id, binaries)`` is the only entry point the session, the host
and the terminal use. A preset without an adapter here is *known* (the registry,
the backend mirror, the picker) but not *served*: the host announces it with
``served: false`` and the claim filter keeps its tickets for a host that can.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

from ..presets import REGISTRY, CliPreset, UnknownCli, preset_for
from .base import LaunchContext, Prepared, PresetAdapter, Reply, ToolClass, ToolIntent, hook_command
from .claude import ClaudeAdapter

_ADAPTERS: Dict[str, Callable[[CliPreset, Optional[str]], PresetAdapter]] = {
    "claude": ClaudeAdapter,
}


class NotServed(RuntimeError):
    """A CLI the registry knows but this host has no adapter for (yet)."""


def has_adapter(cli_id: str) -> bool:
    return cli_id in _ADAPTERS


def adapter_for(cli_id: Optional[str], binaries: Optional[Mapping[str, str]] = None) -> PresetAdapter:
    """The adapter for a CLI id (``None`` = the default CLI). ``UnknownCli`` for a
    name the registry has never heard of; ``NotServed`` for a known CLI without an
    adapter — both are honest results on a ticket, never a silent fallback to
    another CLI."""
    preset = preset_for(cli_id)
    factory = _ADAPTERS.get(preset.id)
    if factory is None:
        raise NotServed(f"{preset.label} is not served by this host yet (no adapter for {preset.id!r})")
    binary = (binaries or {}).get(preset.id)
    return factory(preset, binary)


def adapters(binaries: Optional[Mapping[str, str]] = None) -> Dict[str, PresetAdapter]:
    """Every served CLI, for the capabilities announce."""
    return {cli_id: adapter_for(cli_id, binaries) for cli_id in REGISTRY if cli_id in _ADAPTERS}


__all__ = [
    "LaunchContext", "NotServed", "Prepared", "PresetAdapter", "Reply", "ToolClass", "ToolIntent",
    "UnknownCli", "adapter_for", "adapters", "has_adapter", "hook_command",
]
