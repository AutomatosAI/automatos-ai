"""Codex — the hooks tier (design §3, §6). Codex's hook payload and response
contract are Claude-shaped, so translation is identity and the shim is reused
verbatim; the adapter's weight is *where things live*:

* **one config home per agent per host** (``CODEX_HOME`` = ``<state>/agents/
  <agent>/.codex``, §6.1) — the session index lives in that home, so a home per
  ticket would break ``codex resume``; the operator's ``~/.codex`` is never written;
* the operator's login carried in by **symlinking ``~/.codex/auth.json``** —
  never read for its values, never copied into our state dir, never on a
  command line (the subscription invariant, spelled for Codex);
* hooks in ``config.toml`` ``[[hooks.<Event>]]`` tables (``hooks.json`` is
  plugin-scoped and never fires, §6.2), seeded from the operator's own config
  so their model/provider/trust carry over, ``timeout = 30`` (seconds — ``0``
  floors to one second, §6.3), regenerated per spawn;
* the folder-trust decision written into OUR config (``[projects."<cwd>"]``),
  the analogue of Claude's ``hasTrustDialogAccepted``;
* the rollout as the record (§6.8): the model on ``turn_context``, cumulative
  ``total_token_usage`` on the last ``token_count``, the final ``agent_message``.

Every fact here is from Codex 0.154.0 on the operator's machine or munder's
production bridge (``hive.ts:2060``); the six §6.10 checks are what a live run
still has to confirm.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .base import LaunchContext, Prepared, PresetAdapter, Refusal, ToolClass, ToolIntent, hook_command

SHELL_TOOLS = frozenset({"exec_command", "unified_exec", "shell", "container.exec"})
PATCH_TOOLS = frozenset({"apply_patch"})
READ_TOOLS = frozenset({"view_image", "read_file"})
WEB_TOOLS = frozenset({"web_search"})
# write_stdin feeds a shell the gate already judged (exec_command); plans and
# questions touch nothing.
BENIGN_TOOLS = frozenset({"update_plan", "request_user_input", "write_stdin"})

_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+?)\s*$", re.M)
_PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+?)\s*$", re.M)
_HOOKS_MARK = "# --- automatos-cli-host lifecycle hooks (auto-generated; do not edit) ---"
_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def patch_paths(patch: str) -> Tuple[str, ...]:
    """Every file an ``apply_patch`` would touch, from its header lines."""
    seen = []
    for m in list(_PATCH_FILE_RE.finditer(patch or "")) + list(_PATCH_MOVE_RE.finditer(patch or "")):
        p = m.group(1).strip()
        if p and p not in seen:
            seen.append(p)
    return tuple(seen)


def shell_command_of(tool_input: Mapping[str, Any]) -> str:
    """Codex spells the command as a string (``cmd``) or an argv list (``command``)."""
    raw = tool_input.get("cmd", tool_input.get("command"))
    if isinstance(raw, (list, tuple)):
        return shlex.join(str(x) for x in raw)
    return str(raw or "")


# ── the rollout (design §6.8) ────────────────────────────────────────────────

def _records(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                yield rec


def _n(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and v == v else 0


def read_rollout_usage(path: Path) -> Dict[str, Any]:
    """Token totals of one rollout, in the host's normalized shape: the LAST
    cumulative ``total_token_usage`` (never a sum of cumulative records), the
    model from the last ``turn_context``, per-model from each turn's own
    ``last_token_usage``. ``cached_input_tokens`` → ``cache_read_input_tokens``;
    ``reasoning_output_tokens`` reported additionally. Tokens, never a price."""
    totals: Dict[str, Any] = {k: 0 for k in _USAGE_KEYS}
    totals.update({"reasoning_output_tokens": 0, "assistant_messages": 0, "model": None, "per_model": {}, "total_tokens": 0})
    model: Optional[str] = None
    last_total: Optional[Dict[str, Any]] = None
    per_model: Dict[str, Dict[str, int]] = {}
    try:
        for rec in _records(path):
            kind = rec.get("type")
            payload = rec.get("payload") or {}
            if kind == "turn_context":
                model = payload.get("model") or model
            elif kind == "event_msg" and payload.get("type") == "token_count":
                info = payload.get("info") or {}
                if isinstance(info.get("total_token_usage"), dict):
                    last_total = info["total_token_usage"]
                turn = info.get("last_token_usage") if isinstance(info.get("last_token_usage"), dict) else {}
                bucket = per_model.setdefault(model or "unknown", {k: 0 for k in _USAGE_KEYS})
                bucket["input_tokens"] += _n(turn.get("input_tokens"))
                bucket["output_tokens"] += _n(turn.get("output_tokens"))
                bucket["cache_read_input_tokens"] += _n(turn.get("cached_input_tokens"))
            elif kind == "event_msg" and payload.get("type") == "agent_message":
                totals["assistant_messages"] += 1
    except OSError:
        return totals
    if last_total:
        totals["input_tokens"] = _n(last_total.get("input_tokens"))
        totals["output_tokens"] = _n(last_total.get("output_tokens"))
        totals["cache_read_input_tokens"] = _n(last_total.get("cached_input_tokens"))
        totals["reasoning_output_tokens"] = _n(last_total.get("reasoning_output_tokens"))
    totals["model"] = model
    totals["per_model"] = {m: b for m, b in per_model.items() if any(b.values())}
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


def last_agent_message(path: Path) -> Optional[str]:
    text: Optional[str] = None
    try:
        for rec in _records(path):
            payload = rec.get("payload") or {}
            if rec.get("type") == "event_msg" and payload.get("type") == "agent_message":
                candidate = str(payload.get("message") or payload.get("text") or "").strip()
                if candidate:
                    text = candidate
    except OSError:
        return None
    return text


def find_rollout(home: Path, session_id: str) -> Optional[Path]:
    """``<home>/sessions/YYYY/MM/DD/rollout-<ts>-<session_id>.jsonl``, wherever the
    date folders put it."""
    if not session_id or not (home / "sessions").is_dir():
        return None
    matches = sorted((home / "sessions").rglob(f"rollout-*-{session_id}.jsonl"))
    return matches[-1] if matches else None


# ── the adapter ─────────────────────────────────────────────────────────────

class CodexAdapter(PresetAdapter):
    def __init__(self, preset, binary: Optional[str] = None, home: Optional[Path] = None) -> None:
        super().__init__(preset, binary)
        self._home = home   # the operator's home (tests give a fake one); None = Path.home()

    # ── identity ────────────────────────────────────────────────────────────
    def operator_home(self) -> Path:
        return (self._home or Path.home()) / ".codex"

    def _auth_state(self) -> Optional[Dict[str, Any]]:
        """The login file's SHAPE — which keys exist and the mode label. Values are
        never read out of here (the file may hold a key; we only need to know
        whether it is the plan or a key)."""
        try:
            data = json.loads((self.operator_home() / "auth.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else {}

    def login_mode(self) -> Optional[str]:
        """``chatgpt`` | ``apikey`` | ``unknown``; None = not logged in at all."""
        state = self._auth_state()
        if state is None:
            return None
        mode = str(state.get("auth_mode") or "").strip().lower()
        tokens = state.get("tokens")
        if mode == "apikey" or (not (isinstance(tokens, dict) and tokens) and "OPENAI_API_KEY" in state):
            return "apikey"
        if mode == "chatgpt" or (isinstance(tokens, dict) and tokens):
            return "chatgpt"
        return mode or "unknown"

    def logged_in(self) -> Optional[Refusal]:
        mode = self.login_mode()
        if mode is None:
            return Refusal("codex_not_logged_in", "Codex is not logged in on this machine. Run `codex login` (ChatGPT) and retry.")
        if mode != "chatgpt":
            return super().logged_in()   # the preset's refusal: an API key is not the operator's plan
        return None

    def detect(self) -> Dict[str, Any]:
        out = super().detect()
        out["login_mode"] = self.login_mode() if out["path"] else None
        return out

    # ── the config home ─────────────────────────────────────────────────────
    def agent_home(self, ctx: LaunchContext) -> Path:
        root = ctx.state_dir or ctx.session_dir.parent.parent
        return root / "agents" / (ctx.agent_id or "shared") / ".codex"

    def _link(self, src: Path, dest: Path) -> None:
        if dest.exists() or dest.is_symlink():
            return
        if src.exists():
            os.symlink(src, dest)   # the operator's own login, in place; never a copy

    def _config_text(self, cwd: Path) -> str:
        operator = self.operator_home() / "config.toml"
        try:
            base = operator.read_text(encoding="utf-8") if operator.exists() else ""
        except OSError:
            base = ""
        base = base.split(_HOOKS_MARK)[0].rstrip() + "\n" if _HOOKS_MARK in base else base
        cmd = hook_command()
        lines = ["", _HOOKS_MARK]
        for event in sorted(self.preset.hook_events):
            lines += [f"[[hooks.{event}]]", f"[[hooks.{event}.hooks]]", 'type = "command"',
                      f"command = {json.dumps(cmd)}", f"timeout = {self.preset.hook_timeout(event)}", ""]
        trust_header = f"[projects.{json.dumps(str(cwd))}]"
        if trust_header not in base:
            lines += [trust_header, 'trust_level = "trusted"', ""]
        return base.rstrip("\n") + "\n" + "\n".join(lines)

    def prepare(self, ctx: LaunchContext) -> Prepared:
        home = self.agent_home(ctx)
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        (home / "sessions").mkdir(exist_ok=True)
        operator = self.operator_home()
        self._link(operator / "auth.json", home / "auth.json")
        self._link(operator / "packages", home / "packages")
        config = home / "config.toml"
        config.write_text(self._config_text(ctx.cwd), encoding="utf-8")
        os.chmod(config, 0o600)
        return Prepared(env={"CODEX_HOME": str(home)})

    def record_trust(self, cwd: Path, home: Optional[Path] = None) -> bool:
        return False   # trust lives in OUR config.toml (prepare); the operator's is never written

    # ── tools ───────────────────────────────────────────────────────────────
    def tool_intent(self, tool_name: str, tool_input: Mapping[str, Any]) -> ToolIntent:
        ti = tool_input if isinstance(tool_input, Mapping) else {}
        if tool_name in SHELL_TOOLS:
            return ToolIntent(tool=tool_name, cls=ToolClass.SHELL, command=shell_command_of(ti))
        if tool_name in PATCH_TOOLS:
            patch = tool_input if isinstance(tool_input, str) else str(ti.get("input") or ti.get("patch") or "")
            return ToolIntent(tool=tool_name, cls=ToolClass.FILE_WRITE, paths=patch_paths(patch))
        if tool_name in READ_TOOLS:
            paths = tuple(str(ti[k]) for k in ("path", "file_path") if ti.get(k))
            return ToolIntent(tool=tool_name, cls=ToolClass.FILE_READ, paths=paths)
        if tool_name in WEB_TOOLS:
            return ToolIntent(tool=tool_name, cls=ToolClass.WEB, paths=tuple(str(ti[k]) for k in ("query", "url") if ti.get(k)))
        if tool_name in BENIGN_TOOLS:
            return ToolIntent(tool=tool_name, cls=ToolClass.BENIGN)
        return ToolIntent(tool=tool_name, cls=ToolClass.UNKNOWN)

    # ── the record ──────────────────────────────────────────────────────────
    def read_usage(self, transcript: Path) -> Dict[str, Any]:
        return read_rollout_usage(transcript)

    def last_text(self, transcript: Path) -> Optional[str]:
        return last_agent_message(transcript)

    def transcript_path(self, cwd: str, session_id: str, home: Optional[Path] = None) -> Optional[Path]:
        """Without an agent home in hand (the Canvas terminal), the operator's own
        sessions are the only place to look."""
        root = (home / ".codex") if home is not None else self.operator_home()
        return find_rollout(root, session_id)


__all__ = ["CodexAdapter", "find_rollout", "last_agent_message", "patch_paths", "read_rollout_usage", "shell_command_of"]
