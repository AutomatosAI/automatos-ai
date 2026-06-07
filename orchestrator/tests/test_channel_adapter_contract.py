"""PRD-142 Wave 3 · WS-J · W3-S13 — Channels primitive contract test.

The Channels primitive's `BRAIN §3.x` contract says: *every adapter
implements the same shape; a new channel = a new adapter file with
zero core change; in/out counts tracked.* The §H DoD adds: *parametrized
contract test across all 11 adapters; failure path visible; restart
durable (no in-memory state lost); tenant isolated; one source of truth
for the in/out tally; heartbeat finding observable.*

This file pins those contracts under the Wave 2 net so a refactor
cannot silently drop a method, leak across workspaces, or forget to
update the activity counter.

What this proves (matching W3-S13 §AC):

1. **Parametrized contract.** Every one of the 11 ``BaseChannelAdapter``
   subclasses implements the same 5-method surface
   (``start``, ``stop``, ``send_message``, ``test_connection``,
   ``_to_envelope``), all five are coroutines / methods of the
   expected shape, and the constructor accepts
   ``(connection_id, workspace_id, config)``. Adding the 12th adapter
   only requires the same 5 methods — no core change.
2. **In/out counts tracked.** ``BaseChannelAdapter._update_activity_stats``
   runs a single SQL UPDATE that increments ``message_count`` and
   sets ``last_activity_at`` on the OWN ``channel_connections`` row
   (scoped by ``id = :conn_id``). No other adapter or workspace
   touches that row.
3. **Tenant isolation.** ``workspace_id`` is locked at adapter
   construction and never reassignable from a platform message.
   ``handle_message`` passes ``self.workspace_id`` into the envelope
   and the execute context — never reads it from the inbound payload.
   ``_update_activity_stats`` is scoped to ``self.connection_id`` (the
   adapter's own row only — workspace A's adapter cannot touch
   workspace B's row).
4. **Heartbeat.** A tiny stateless helper ``_emit_channels_primitive``
   calls ``emit_primitive_finding`` with primitive='channels' and the
   correct status — green on a clean in→route→exec→reply turn, down on
   a caught exception. Skip when no ``workspace_id`` (A4 honest gap),
   swallow emit failures, never raise back to the adapter.
5. **Wire-up.** ``base.handle_message`` imports the helper and calls
   the wrapper at both the success boundary AND in the outer except,
   so the tile reflects real-time channel health.

The tests deliberately operate at the *unit* level via importlib +
source-text + AST inspection — a full adapter spin-up would drag
python-telegram-bot, slack-bolt, discord.py, matrix-nio, etc into the
unit suite. Mirrors the W3-S6 / W3-S8 / W3-S9 / W3-S10 / W3-S11
patterns.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Paths to the surfaces we pin without importing them through heavy
# package __init__ chains (each adapter pulls in its own SDK).
# ---------------------------------------------------------------------------

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

CHANNELS_DIR = ORCH_ROOT / "channels"
BASE_PY = CHANNELS_DIR / "base.py"
MANAGER_PY = CHANNELS_DIR / "manager.py"
PRIMITIVE_HEARTBEAT_PY = CHANNELS_DIR / "primitive_heartbeat.py"

# The 11 BaseChannelAdapter subclasses (verified 2026-06-06).
# (module_filename, class_name) — derived from
# channels/manager.py::_ADAPTER_MAP plus the imessage adapter.
ADAPTERS: list[tuple[str, str]] = [
    ("telegram_adapter", "TelegramAdapter"),
    ("slack_adapter", "SlackAdapter"),
    ("discord_adapter", "DiscordAdapter"),
    ("teams_adapter", "TeamsAdapter"),
    ("google_chat_adapter", "GoogleChatAdapter"),
    ("signal_adapter", "SignalAdapter"),
    ("imessage_adapter", "IMessageAdapter"),
    ("irc_adapter", "IRCAdapter"),
    ("matrix_adapter", "MatrixAdapter"),
    ("line_adapter", "LINEAdapter"),
    ("whatsapp_adapter", "WhatsAppAdapter"),
]


# ---------------------------------------------------------------------------
# Lightweight Postgres env so any module that touches config doesn't crash
# at import time. Setdefault — a real .env still wins.
# ---------------------------------------------------------------------------
import os

for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)


# ===========================================================================
# Source-text view of the adapter files. Used by the AC1 contract checks so
# we never need to install the platform SDKs to verify the shape.
# ===========================================================================


@pytest.fixture(scope="module")
def adapter_source_by_class() -> dict[str, str]:
    """{class_name: source_text} for each of the 11 adapters."""
    out: dict[str, str] = {}
    for module_file, class_name in ADAPTERS:
        path = CHANNELS_DIR / f"{module_file}.py"
        assert path.exists(), f"missing adapter source: {path}"
        out[class_name] = path.read_text()
    return out


@pytest.fixture(scope="module")
def adapter_ast_by_class(adapter_source_by_class) -> dict[str, ast.ClassDef]:
    """{class_name: ClassDef AST node} for each of the 11 adapters."""
    out: dict[str, ast.ClassDef] = {}
    for class_name, src in adapter_source_by_class.items():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                out[class_name] = node
                break
        else:
            raise AssertionError(f"class {class_name!r} not found in source")
    return out


# ===========================================================================
# 1. AC1 — ALL 11 ADAPTERS IMPLEMENT THE BaseChannelAdapter CONTRACT.
#
# Parametrized over (module_filename, class_name) — adding the 12th adapter
# means adding ONE tuple to ADAPTERS, no core change.
# ===========================================================================


class TestAdapterContractSurface:
    """The 5-method shape every adapter must implement.

    Source-level checks only — never instantiates the adapter (each
    drags a different SDK). The ABC `abstractmethod` markers in
    ``BaseChannelAdapter`` already enforce instance-time, but a refactor
    that drops a method WITH the @abstractmethod decorator from the
    base would silently break every subclass at runtime. This test
    catches that drift at static-check time."""

    @pytest.fixture(scope="class")
    def base_src(self) -> str:
        return BASE_PY.read_text()

    def test_base_declares_five_abstract_methods(self, base_src):
        """The base ABC MUST declare exactly the 5 abstract methods
        we expect — any drift here is a contract change."""
        tree = ast.parse(base_src)
        abstracts = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "BaseChannelAdapter":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for dec in item.decorator_list:
                            if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
                                abstracts.add(item.name)
        assert abstracts == {
            "start", "stop", "send_message", "test_connection", "_to_envelope",
        }, (
            f"BaseChannelAdapter MUST declare exactly 5 abstract methods — "
            f"got {sorted(abstracts)}. A new channel = a new adapter file "
            f"with these 5 methods, ZERO core change."
        )

    @pytest.mark.parametrize("module_file,class_name", ADAPTERS)
    def test_adapter_inherits_base(self, adapter_ast_by_class, module_file, class_name):
        cls = adapter_ast_by_class[class_name]
        bases = [
            (b.id if isinstance(b, ast.Name) else getattr(b, "attr", None))
            for b in cls.bases
        ]
        assert "BaseChannelAdapter" in bases, (
            f"{class_name} MUST inherit from BaseChannelAdapter — got bases={bases}"
        )

    @pytest.mark.parametrize("module_file,class_name", ADAPTERS)
    def test_adapter_implements_start_as_coroutine(
        self, adapter_ast_by_class, module_file, class_name,
    ):
        cls = adapter_ast_by_class[class_name]
        for item in cls.body:
            if isinstance(item, ast.AsyncFunctionDef) and item.name == "start":
                return
        pytest.fail(f"{class_name} MUST implement async def start(self)")

    @pytest.mark.parametrize("module_file,class_name", ADAPTERS)
    def test_adapter_implements_stop_as_coroutine(
        self, adapter_ast_by_class, module_file, class_name,
    ):
        cls = adapter_ast_by_class[class_name]
        for item in cls.body:
            if isinstance(item, ast.AsyncFunctionDef) and item.name == "stop":
                return
        pytest.fail(f"{class_name} MUST implement async def stop(self)")

    @pytest.mark.parametrize("module_file,class_name", ADAPTERS)
    def test_adapter_implements_send_message_as_coroutine(
        self, adapter_ast_by_class, module_file, class_name,
    ):
        cls = adapter_ast_by_class[class_name]
        for item in cls.body:
            if isinstance(item, ast.AsyncFunctionDef) and item.name == "send_message":
                arg_names = [a.arg for a in item.args.args]
                assert arg_names[:3] == ["self", "channel_id", "text"], (
                    f"{class_name}.send_message MUST accept (channel_id, text) — "
                    f"got {arg_names}"
                )
                return
        pytest.fail(
            f"{class_name} MUST implement "
            f"async def send_message(self, channel_id, text, **kwargs) -> bool"
        )

    @pytest.mark.parametrize("module_file,class_name", ADAPTERS)
    def test_adapter_implements_test_connection_as_coroutine(
        self, adapter_ast_by_class, module_file, class_name,
    ):
        cls = adapter_ast_by_class[class_name]
        for item in cls.body:
            if isinstance(item, ast.AsyncFunctionDef) and item.name == "test_connection":
                return
        pytest.fail(
            f"{class_name} MUST implement async def test_connection(self) -> Dict[str, Any]"
        )

    @pytest.mark.parametrize("module_file,class_name", ADAPTERS)
    def test_adapter_implements_to_envelope(
        self, adapter_ast_by_class, module_file, class_name,
    ):
        cls = adapter_ast_by_class[class_name]
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "_to_envelope":
                return
        pytest.fail(
            f"{class_name} MUST implement def _to_envelope(self, platform_message)"
        )

    @pytest.mark.parametrize("module_file,class_name", ADAPTERS)
    def test_adapter_constructor_accepts_canonical_triple(
        self, adapter_ast_by_class, module_file, class_name,
    ):
        """Every adapter MUST be constructable with
        ``(connection_id, workspace_id, config)`` so the ``ChannelManager``
        factory can spin them up uniformly. The base ABC enforces this
        positionally — an adapter that drops or reorders an arg breaks
        the manager."""
        cls = adapter_ast_by_class[class_name]
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                arg_names = [a.arg for a in item.args.args]
                assert arg_names[:4] == ["self", "connection_id", "workspace_id", "config"], (
                    f"{class_name}.__init__ MUST accept "
                    f"(connection_id, workspace_id, config) — got {arg_names}"
                )
                return
        # No __init__ override is fine — the base provides one with the
        # right shape (assertion: the base __init__ has the triple).
        base_src = BASE_PY.read_text()
        base_tree = ast.parse(base_src)
        for node in ast.walk(base_tree):
            if isinstance(node, ast.ClassDef) and node.name == "BaseChannelAdapter":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        arg_names = [a.arg for a in item.args.args]
                        assert arg_names[:4] == [
                            "self", "connection_id", "workspace_id", "config",
                        ], (
                            f"BaseChannelAdapter.__init__ canonical shape changed — "
                            f"breaks the ChannelManager factory contract"
                        )
                        return


# ===========================================================================
# 2. AC2 — IN/OUT MESSAGE COUNTS ARE TRACKED (OBSERVABLE).
#
# Pinned via source inspection of BaseChannelAdapter._update_activity_stats:
# every successful handle_message runs the canonical UPDATE — so a refactor
# that drops the counter from the success path fails this test.
# ===========================================================================


class TestActivityCountsObservable:
    """Pin the in/out counter mechanism so the tile/admin UI keeps
    showing real numbers."""

    @pytest.fixture(scope="class")
    def base_src(self) -> str:
        return BASE_PY.read_text()

    def test_update_activity_stats_increments_message_count(self, base_src):
        """The canonical UPDATE MUST increment ``message_count`` — the
        column the admin UI reads to show 'X messages handled by this
        connection'."""
        assert "message_count = COALESCE(message_count, 0) + 1" in base_src, (
            "_update_activity_stats MUST atomically increment "
            "message_count (in/out counts observable)"
        )

    def test_update_activity_stats_writes_last_activity_at(self, base_src):
        """Pin that ``last_activity_at`` is set on every successful turn
        — the tile/admin UI surfaces this as 'last heard from X seconds
        ago'."""
        assert "last_activity_at = :now" in base_src, (
            "_update_activity_stats MUST set last_activity_at on every "
            "successful turn (observable freshness)"
        )

    def test_handle_message_calls_update_activity_stats(self, base_src):
        """The success path of handle_message MUST call
        _update_activity_stats — otherwise the counter never moves."""
        tree = ast.parse(base_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "handle_message"):
                body_src = ast.get_source_segment(base_src, node) or ""
                assert "_update_activity_stats(db)" in body_src, (
                    "handle_message MUST call _update_activity_stats(db) "
                    "on the success path — otherwise message_count "
                    "never increments"
                )
                return
        pytest.fail("handle_message not found in base.py")


# ===========================================================================
# 3. AC3 — CROSS-WORKSPACE ISOLATION.
#
# Two layers:
#   (a) workspace_id is locked at construction; handle_message NEVER
#       reads it from the inbound platform_message.
#   (b) _update_activity_stats is scoped to self.connection_id only.
# ===========================================================================


class TestCrossWorkspaceIsolation:
    """A4 — one workspace's channel adapter cannot read or mutate
    another's. Pin both the envelope and the SQL boundary."""

    @pytest.fixture(scope="class")
    def base_src(self) -> str:
        return BASE_PY.read_text()

    def test_workspace_id_locked_at_construction(self, base_src):
        """``BaseChannelAdapter.__init__`` MUST store workspace_id as an
        instance attr — never re-read it from platform_message."""
        tree = ast.parse(base_src)
        init_body = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "BaseChannelAdapter":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_body = ast.get_source_segment(base_src, item) or ""
                        break
        assert init_body is not None
        assert "self.workspace_id = workspace_id" in init_body, (
            "BaseChannelAdapter MUST lock workspace_id at construction "
            "(self.workspace_id = workspace_id)"
        )

    def test_handle_message_does_not_read_workspace_from_payload(self, base_src):
        """The platform_message is UNTRUSTED — an attacker could put
        any workspace_id in it. Pin that handle_message never reads
        it from the inbound payload."""
        tree = ast.parse(base_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "handle_message"):
                body_src = ast.get_source_segment(base_src, node) or ""
                # Negative assertion — workspace_id MUST come from
                # envelope.workspace_id (which the adapter populates
                # from self.workspace_id) or self.workspace_id directly.
                # NEVER from platform_message.get('workspace_id').
                assert "platform_message.get('workspace_id'" not in body_src
                assert 'platform_message.get("workspace_id"' not in body_src
                assert 'platform_message["workspace_id"' not in body_src
                # Positive assertion — the execute context passes
                # envelope.workspace_id (the trusted one).
                assert "str(envelope.workspace_id)" in body_src, (
                    "handle_message MUST forward envelope.workspace_id "
                    "(the locked-at-construction value) into the execute "
                    "context — never trust the inbound payload"
                )
                return
        pytest.fail("handle_message not found in base.py")

    def test_update_activity_stats_scoped_to_own_connection(self, base_src):
        """The UPDATE clause MUST filter by self.connection_id — workspace
        A's adapter cannot bump workspace B's row."""
        assert "WHERE id = :conn_id" in base_src, (
            "_update_activity_stats UPDATE MUST be scoped by "
            "WHERE id = :conn_id — otherwise it could mutate another "
            "workspace's connection row"
        )
        tree = ast.parse(base_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_update_activity_stats"):
                body_src = ast.get_source_segment(base_src, node) or ""
                # The bound param MUST come from self.connection_id —
                # never from any inbound message field.
                assert '"conn_id": self.connection_id' in body_src, (
                    "_update_activity_stats MUST bind self.connection_id "
                    "(no payload-controlled conn_id)"
                )
                return
        pytest.fail("_update_activity_stats not found in base.py")

    def test_envelope_workspace_id_is_adapter_workspace_id(
        self, adapter_source_by_class,
    ):
        """For every adapter's ``_to_envelope``: the ``workspace_id`` on
        the produced RequestEnvelope MUST come from ``self.workspace_id``
        — never from the platform_message dict. Pin per-adapter."""
        for class_name, src in adapter_source_by_class.items():
            tree = ast.parse(src)
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "_to_envelope":
                    body = ast.get_source_segment(src, node) or ""
                    # MUST reference self.workspace_id at least once.
                    assert "self.workspace_id" in body, (
                        f"{class_name}._to_envelope MUST set "
                        f"workspace_id from self.workspace_id, not from "
                        f"the inbound payload"
                    )
                    # MUST NOT read workspace_id from the platform_message
                    # dict — the inbound payload is untrusted.
                    assert "platform_message.get('workspace_id'" not in body
                    assert 'platform_message.get("workspace_id"' not in body
                    assert 'platform_message["workspace_id"' not in body
                    found = True
                    break
            assert found, f"{class_name}._to_envelope not found"


# ===========================================================================
# 4. AC4 — CHANNELS PRIMITIVE HEARTBEAT (W3-S1 WIRING).
# ===========================================================================


def _load_primitive_heartbeat():
    """Load ``channels/primitive_heartbeat.py`` directly so the channels
    package __init__ doesn't fire (which would drag in ChannelManager
    + DB)."""
    # First stub the ``channels`` package so the relative-ish import path
    # of the helper resolves without firing the real __init__ side
    # effects.
    if "channels" not in sys.modules:
        stub = types.ModuleType("channels")
        stub.__path__ = [str(CHANNELS_DIR)]
        sys.modules["channels"] = stub
    if "services" not in sys.modules:
        services_stub = types.ModuleType("services")
        services_stub.__path__ = [str(ORCH_ROOT / "services")]
        sys.modules["services"] = services_stub
    spec = importlib.util.spec_from_file_location(
        "channels.primitive_heartbeat", str(PRIMITIVE_HEARTBEAT_PY)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["channels.primitive_heartbeat"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestChannelsHeartbeatHelper:
    """The W3-S1 helper plumb for the channels primitive."""

    def test_helper_file_exists(self):
        assert PRIMITIVE_HEARTBEAT_PY.exists(), (
            "channels/primitive_heartbeat.py MUST exist — it's the W3-S1 "
            "emit surface for the channels tile"
        )

    def test_helper_emits_green_on_success(self, monkeypatch):
        mod = _load_primitive_heartbeat()
        calls: list[tuple] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda ws, prim, status, detail: calls.append(
                (ws, prim, status, detail)
            ) or True,
        )
        mod._emit_channels_primitive(
            "ws-A", success=True, detail="ok",
        )
        assert calls == [("ws-A", "channels", "green", "ok")]

    def test_helper_emits_down_on_failure(self, monkeypatch):
        mod = _load_primitive_heartbeat()
        calls: list[tuple] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda ws, prim, status, detail: calls.append(
                (ws, prim, status, detail)
            ) or True,
        )
        mod._emit_channels_primitive(
            "ws-A", success=False, detail="RuntimeError: boom",
        )
        assert calls == [("ws-A", "channels", "down", "RuntimeError: boom")]

    def test_helper_skips_when_workspace_id_missing(self, monkeypatch):
        """A4 — no workspace_id MUST result in NO emit (honest gap over
        fabricated default). Mirrors W3-S6/S8/S9/S10/S11 helpers."""
        mod = _load_primitive_heartbeat()
        calls: list[tuple] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda *a, **k: calls.append(a) or True,
        )
        mod._emit_channels_primitive(None, success=True)
        mod._emit_channels_primitive("", success=False)
        assert calls == [], (
            "no workspace_id MUST mean no emit — never default to an "
            "anonymous workspace"
        )

    def test_helper_swallows_emit_failures(self, monkeypatch):
        """A busted heartbeat MUST NOT raise back into the channel
        pipeline. Pin the best-effort contract."""
        mod = _load_primitive_heartbeat()

        def _boom(*_a, **_k):
            raise RuntimeError("heartbeat_results write failed")

        monkeypatch.setattr(mod, "emit_primitive_finding", _boom)
        # Must not raise.
        mod._emit_channels_primitive("ws-A", success=True, detail="ok")
        mod._emit_channels_primitive("ws-A", success=False, detail="boom")

    def test_helper_truncates_long_detail(self, monkeypatch):
        """500-char cap mirrors the W3-S1 ``emit_primitive_finding`` API.
        Pin the helper does not blow past it."""
        mod = _load_primitive_heartbeat()
        captured: list[str] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda ws, prim, status, detail: captured.append(detail) or True,
        )
        long_detail = "x" * 9000
        mod._emit_channels_primitive("ws-A", success=False, detail=long_detail)
        assert len(captured) == 1
        assert len(captured[0]) <= 500

    def test_helper_uses_canonical_primitive_name(self, monkeypatch):
        """The primitive key MUST be exactly 'channels' (the W3-S1
        ``PRIMITIVE_NAMES`` set rejects anything else)."""
        mod = _load_primitive_heartbeat()
        seen: list[str] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda ws, prim, status, detail: seen.append(prim) or True,
        )
        mod._emit_channels_primitive("ws-A", success=True)
        assert seen == ["channels"]


# ===========================================================================
# 5. AC5 — base.handle_message WIRES THE HEARTBEAT EMIT.
#
# Static grep on base.py — handle_message MUST call the wrapper on BOTH
# the success boundary and inside the outer except. A refactor that
# drops the wire-up MUST fail this test.
# ===========================================================================


class TestBaseHandleMessageWiresHeartbeat:
    """Pin the wire-up between base.handle_message and the W3-S1 emit."""

    @pytest.fixture(scope="class")
    def base_src(self) -> str:
        return BASE_PY.read_text()

    def test_base_references_channels_heartbeat_helper(self, base_src):
        """Either the inline emit method calls the helper directly or
        the file imports it. Pin both call sites use the helper name."""
        assert "_emit_channels_primitive" in base_src or "_emit_channel_heartbeat" in base_src, (
            "base.py MUST reference the heartbeat emit (either via the "
            "wrapper method name or the helper directly)"
        )

    def test_handle_message_emits_on_success_path(self, base_src):
        """The success boundary MUST emit success=True so the tile
        reflects a clean turn."""
        tree = ast.parse(base_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "handle_message"):
                body_src = ast.get_source_segment(base_src, node) or ""
                # Need at least one success=True emit inside the body.
                assert re.search(r"_emit_channel_heartbeat\([^)]*success=True", body_src), (
                    "handle_message MUST emit success=True on the success "
                    "boundary so the channels tile can flip green"
                )
                return
        pytest.fail("handle_message not found in base.py")

    def test_handle_message_emits_on_failure_path(self, base_src):
        """The outer except MUST emit success=False so the tile reflects
        a real failure."""
        tree = ast.parse(base_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "handle_message"):
                body_src = ast.get_source_segment(base_src, node) or ""
                assert re.search(r"_emit_channel_heartbeat\([^)]*success=False", body_src), (
                    "handle_message MUST emit success=False inside the "
                    "outer except so the channels tile can flip down on "
                    "a real failure"
                )
                return
        pytest.fail("handle_message not found in base.py")

    def test_heartbeat_wrapper_uses_self_workspace_id(self, base_src):
        """The wrapper MUST forward ``self.workspace_id`` (the trusted
        construction-time value) — never an inbound payload value."""
        tree = ast.parse(base_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_emit_channel_heartbeat"):
                body_src = ast.get_source_segment(base_src, node) or ""
                assert "self.workspace_id" in body_src, (
                    "_emit_channel_heartbeat MUST forward self.workspace_id "
                    "(the locked-at-construction value)"
                )
                # No reading from any 'platform_message' inside the
                # wrapper — it is only called with success/detail kwargs.
                assert "platform_message" not in body_src, (
                    "_emit_channel_heartbeat MUST NOT read from "
                    "platform_message — workspace_id is locked at "
                    "construction"
                )
                return
        pytest.fail("_emit_channel_heartbeat helper not found in base.py")


# ===========================================================================
# 6. AC6 — END-TO-END SHAPE under a stubbed pipeline.
#
# Drive BaseChannelAdapter.handle_message with the minimum doubles needed
# to traverse the success path and the failure path; confirm the emit is
# called with the right (workspace_id, success) — never another workspace's.
# ===========================================================================


def _make_test_adapter(workspace_id: str = "ws-X", connection_id: str = "conn-1"):
    """Create a concrete BaseChannelAdapter subclass for the end-to-end
    shape tests — minimal implementations of the 5 abstract methods so
    we can instantiate it without an SDK."""
    # Stub the channels package so importing base doesn't trigger
    # channels/__init__ (which pulls in ChannelManager + the DB).
    if "channels" not in sys.modules:
        stub = types.ModuleType("channels")
        stub.__path__ = [str(CHANNELS_DIR)]
        sys.modules["channels"] = stub

    # Load base.py directly via importlib so the abstract class is
    # constructable without dragging the heavy manager import path.
    spec = importlib.util.spec_from_file_location(
        "channels.base", str(BASE_PY)
    )
    base_mod = importlib.util.module_from_spec(spec)
    sys.modules["channels.base"] = base_mod
    spec.loader.exec_module(base_mod)

    class _TestAdapter(base_mod.BaseChannelAdapter):
        def __init__(self, connection_id, workspace_id, config):
            super().__init__(connection_id, workspace_id, config)
            self.sent: list[tuple[str, str]] = []

        async def start(self): self.is_running = True
        async def stop(self): self.is_running = False

        async def send_message(self, channel_id, text, **kwargs) -> bool:
            self.sent.append((channel_id, text))
            return True

        async def test_connection(self):
            return {"ok": True, "detail": "stub"}

        def _to_envelope(self, platform_message):
            return SimpleNamespace(
                content=platform_message.get("text", ""),
                source=SimpleNamespace(value="test"),
                workspace_id=self.workspace_id,
            )

    return _TestAdapter(connection_id, workspace_id, {})


def _install_handle_message_stubs(
    *,
    router_cls,
    factory_cls,
    session_local,
):
    """Pre-install sys.modules stubs for the lazy imports inside
    ``BaseChannelAdapter.handle_message``.

    ``handle_message`` does ``from core.routing.engine import
    UniversalRouter`` / ``from core.database.database import SessionLocal``
    / ``from modules.agents.factory.agent_factory import AgentFactory``
    at call time. Earlier tests in the suite may leave a *path-only*
    ``core.llm`` stub in sys.modules — that breaks the real engine import
    (cf. ``from core.llm.manager import …`` raises ModuleNotFoundError
    because the parent stub has no manager attribute).

    Installing module stubs ourselves bypasses the real import entirely
    and gives the lazy-import statements something to bind to. Returns
    the list of installed module names so the test can pop them after.
    """
    # ``core`` and ``core.routing`` and ``core.database`` and ``modules``
    # may already be in sys.modules; keep what's there. We only install
    # the leaves the handle_message lazy imports actually need.
    parents = [
        ("core", []),
        ("core.routing", ["core"]),
        ("core.database", ["core"]),
        ("modules", []),
        ("modules.agents", ["modules"]),
        ("modules.agents.factory", ["modules", "modules.agents"]),
    ]
    installed: list[str] = []
    for name, _ in parents:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []
            sys.modules[name] = stub
            installed.append(name)

    # Leaf stubs with the actual symbols handle_message imports.
    routing_engine = types.ModuleType("core.routing.engine")
    routing_engine.UniversalRouter = router_cls  # type: ignore[attr-defined]
    sys.modules["core.routing.engine"] = routing_engine
    installed.append("core.routing.engine")

    db_mod = types.ModuleType("core.database.database")
    db_mod.SessionLocal = session_local  # type: ignore[attr-defined]
    sys.modules["core.database.database"] = db_mod
    installed.append("core.database.database")

    agent_mod = types.ModuleType("modules.agents.factory.agent_factory")
    agent_mod.AgentFactory = factory_cls  # type: ignore[attr-defined]
    sys.modules["modules.agents.factory.agent_factory"] = agent_mod
    installed.append("modules.agents.factory.agent_factory")

    return installed


def _restore_modules(installed: list[str], saved: dict[str, types.ModuleType | None]):
    """Remove module stubs we installed (or restore the originals if any)."""
    for name in installed:
        if saved.get(name) is not None:
            sys.modules[name] = saved[name]  # type: ignore[assignment]
        else:
            sys.modules.pop(name, None)


class TestEndToEndHeartbeatShape:
    """Drive handle_message with stubbed router/factory/db and pin the
    heartbeat emit shape on both the success and failure paths."""

    def test_success_path_emits_green_with_own_workspace_id(self, monkeypatch):
        """The success path MUST emit (self.workspace_id, success=True)
        — never another workspace's id."""
        adapter = _make_test_adapter(workspace_id="ws-OWN", connection_id="conn-1")

        # Stub the lazy imports performed inside handle_message.
        stub_decision = SimpleNamespace(agent_id="agent-1")

        class _StubRouter:
            def __init__(self, db=None): pass
            async def route(self, env): return stub_decision

        class _StubFactory:
            def __init__(self, db_session=None): pass
            async def execute_with_prompt(self, **kw): return {"response": "reply"}

        stub_db = MagicMock()
        stub_db.close = MagicMock()

        # Snapshot affected sys.modules entries, install our stubs, and
        # restore on test exit. Bypasses cross-test contamination from
        # other suites that left path-only parent stubs in sys.modules.
        saved = {
            name: sys.modules.get(name)
            for name in (
                "core", "core.routing", "core.database",
                "core.routing.engine", "core.database.database",
                "modules", "modules.agents", "modules.agents.factory",
                "modules.agents.factory.agent_factory",
            )
        }
        installed = _install_handle_message_stubs(
            router_cls=_StubRouter,
            factory_cls=_StubFactory,
            session_local=lambda: stub_db,
        )
        try:
            # Capture heartbeat calls. Pre-load the helper module so it's
            # registered in sys.modules under the canonical 'channels.
            # primitive_heartbeat' name — base.py's lazy `from channels.
            # primitive_heartbeat import _emit_channels_primitive` resolves
            # against this loaded module, so patching its attribute is
            # the right interception point.
            hb_mod = _load_primitive_heartbeat()
            emit_calls: list[tuple] = []
            monkeypatch.setattr(
                hb_mod,
                "_emit_channels_primitive",
                lambda ws_id, *, success, detail="": emit_calls.append(
                    (ws_id, success, detail)
                ),
            )

            # No-op the activity stats update so we don't hit a real DB.
            async def _noop_stats(_db):
                return None
            adapter._update_activity_stats = _noop_stats

            platform_msg = {
                "channel_id": "c-1",
                "reply_channel_id": "c-1",
                "text": "hello",
                # Adversarial: try to inject a different workspace_id.
                "workspace_id": "ws-ATTACKER",
            }
            asyncio.run(adapter.handle_message(platform_msg))

            # Heartbeat MUST have been called with the adapter's own ws id,
            # NEVER the attacker's.
            assert len(emit_calls) == 1
            ws_emitted, success_emitted, _ = emit_calls[0]
            assert ws_emitted == "ws-OWN", (
                f"heartbeat MUST emit self.workspace_id (ws-OWN), got {ws_emitted!r}"
            )
            assert success_emitted is True, (
                "clean pipeline traversal MUST emit success=True"
            )
            # The adapter MUST have sent the reply to the channel.
            assert adapter.sent == [("c-1", "reply")]
        finally:
            _restore_modules(installed, saved)

    def test_failure_path_emits_down_with_own_workspace_id(self, monkeypatch):
        """A caught exception MUST emit (self.workspace_id, success=False)
        — and the original error MUST be surfaced (not silently
        swallowed)."""
        adapter = _make_test_adapter(workspace_id="ws-OWN", connection_id="conn-2")

        # Router raises — simulates an outage anywhere in the pipeline.
        class _BoomRouter:
            def __init__(self, db=None): pass
            async def route(self, envelope):
                raise RuntimeError("router exploded")

        class _StubFactory:
            def __init__(self, db_session=None): pass
            async def execute_with_prompt(self, **kw): return {}

        stub_db = MagicMock()
        stub_db.close = MagicMock()

        saved = {
            name: sys.modules.get(name)
            for name in (
                "core", "core.routing", "core.database",
                "core.routing.engine", "core.database.database",
                "modules", "modules.agents", "modules.agents.factory",
                "modules.agents.factory.agent_factory",
            )
        }
        installed = _install_handle_message_stubs(
            router_cls=_BoomRouter,
            factory_cls=_StubFactory,
            session_local=lambda: stub_db,
        )
        try:
            hb_mod = _load_primitive_heartbeat()
            emit_calls: list[tuple] = []
            monkeypatch.setattr(
                hb_mod,
                "_emit_channels_primitive",
                lambda ws_id, *, success, detail="": emit_calls.append(
                    (ws_id, success, detail)
                ),
            )

            platform_msg = {
                "channel_id": "c-2",
                "text": "hello",
            }
            # Must NOT raise — handle_message catches and emits.
            asyncio.run(adapter.handle_message(platform_msg))

            assert len(emit_calls) == 1
            ws_emitted, success_emitted, detail_emitted = emit_calls[0]
            assert ws_emitted == "ws-OWN"
            assert success_emitted is False, (
                "caught exception MUST emit success=False on the channels tile"
            )
            assert "router exploded" in detail_emitted, (
                "the original error MUST be surfaced in the detail "
                "(visible failure, not silent)"
            )
        finally:
            _restore_modules(installed, saved)

    def test_emit_failure_does_not_break_handle_message(self, monkeypatch):
        """If the heartbeat helper itself raises (the W3-S1 'best-effort'
        contract), handle_message MUST still complete the turn — the
        adapter MUST NOT propagate a heartbeat failure to the platform.
        """
        adapter = _make_test_adapter(workspace_id="ws-OWN", connection_id="conn-3")

        stub_decision = SimpleNamespace(agent_id="agent-1")

        class _StubRouter:
            def __init__(self, db=None): pass
            async def route(self, env): return stub_decision

        class _StubFactory:
            def __init__(self, db_session=None): pass
            async def execute_with_prompt(self, **kw): return {"response": "reply"}

        stub_db = MagicMock()
        stub_db.close = MagicMock()

        saved = {
            name: sys.modules.get(name)
            for name in (
                "core", "core.routing", "core.database",
                "core.routing.engine", "core.database.database",
                "modules", "modules.agents", "modules.agents.factory",
                "modules.agents.factory.agent_factory",
            )
        }
        installed = _install_handle_message_stubs(
            router_cls=_StubRouter,
            factory_cls=_StubFactory,
            session_local=lambda: stub_db,
        )
        try:
            # Heartbeat raises — the wrapper must swallow it.
            hb_mod = _load_primitive_heartbeat()

            def _boom(*_a, **_k):
                raise RuntimeError("heartbeat dead")

            monkeypatch.setattr(hb_mod, "_emit_channels_primitive", _boom)

            async def _noop_stats(_db):
                return None
            adapter._update_activity_stats = _noop_stats

            platform_msg = {
                "channel_id": "c-3", "reply_channel_id": "c-3", "text": "hi",
            }
            # Must NOT raise even though the heartbeat raised.
            asyncio.run(adapter.handle_message(platform_msg))
            # Reply still went out.
            assert adapter.sent == [("c-3", "reply")]
        finally:
            _restore_modules(installed, saved)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


async def _aio_return(value):
    return value
