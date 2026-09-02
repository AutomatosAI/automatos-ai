"""The transcript is the record: Claude Code writes every session to
``~/.claude/projects/<cwd-key>/<session_id>.jsonl``. Hooks tell us the exact
path (``transcript_path``); this module reads the parts a result needs:

* the final assistant text (the ``Stop`` hook also carries it — this is the
  fallback and the cross-check);
* per-model token usage (input / output / cache read / cache write) summed over
  the session's assistant messages — reported as tokens, never as a price:
  on a subscription there is no dollar figure to invent.

Ported from munder-difflin ``transcript.ts`` (key rule + tail reading).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


def project_key(cwd: str) -> str:
    """Claude Code's project directory key: every non-alphanumeric → ``-``."""
    return re.sub(r"[^a-zA-Z0-9]", "-", cwd)


def project_dir(cwd: str, home: Optional[Path] = None) -> Path:
    return (home or Path.home()) / ".claude" / "projects" / project_key(cwd)


def transcript_path(cwd: str, session_id: str, home: Optional[Path] = None) -> Path:
    return project_dir(cwd, home) / f"{session_id}.jsonl"


def _num(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and v == v else 0


def read_usage(path: Path) -> Dict[str, Any]:
    """Token totals over the assistant messages of one transcript."""
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "assistant_messages": 0, "model": None}
    per_model: Dict[str, Dict[str, int]] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") != "assistant":
                    continue
                msg = rec.get("message") or {}
                usage = msg.get("usage") or {}
                if not usage:
                    continue
                model = msg.get("model") or "unknown"
                bucket = per_model.setdefault(model, {"input_tokens": 0, "output_tokens": 0,
                                                       "cache_read_input_tokens": 0,
                                                       "cache_creation_input_tokens": 0})
                for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                    n = _num(usage.get(key))
                    bucket[key] += n
                    totals[key] += n
                totals["assistant_messages"] += 1
                totals["model"] = model
    except OSError:
        return {**totals, "per_model": {}, "total_tokens": 0}
    totals["per_model"] = per_model
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


def last_assistant_text(path: Path, tail_bytes: int = 512 * 1024) -> Optional[str]:
    """The last assistant text block, tail-read so long transcripts stay cheap."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
            lines = fh.read().decode("utf-8", "replace").split("\n")
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # the first tail line may be cut mid-record
        if rec.get("type") != "assistant":
            continue
        content = (rec.get("message") or {}).get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        return text
    return None
