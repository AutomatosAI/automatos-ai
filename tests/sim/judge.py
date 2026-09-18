"""A standalone judge for the quality and usefulness rows (PRD-247 S0.4).

Independent of the platform on purpose: it calls OpenRouter directly with the
operator's own key from ``~/.automatos-sim/env`` and a model of its own, so a
bad night for Auto cannot grade itself kindly. The rubric is the six-dimension
``run_verdict`` idea reduced to the two numbers the scorecard needs, with the
reasons kept so a grade can be argued with.

No key → no judge: the scorer falls back to the pack's cheap checks and says
so. Verdicts are cached by content hash so a re-score costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import JUDGE_CACHE_DIR, Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_S = 120
OUTPUT_CAP = 24_000
RUBRIC = """You grade the work an AI agent delivered to a small-business customer who is not technical.
Score two things from 1 (unacceptable) to 5 (excellent):
- quality: is the output accurate, complete for the brief, well organised, free of filler and invented facts?
- usefulness: could the customer act on this today without redoing it? Does it answer what was actually asked?
Be strict: a polished document that ignores the brief is a 2. A short, correct answer to the brief is a 4 or 5.
Reply with JSON only: {"quality": <1-5>, "usefulness": <1-5>, "reasons": ["...", "..."]}"""


def judge_available(settings: Settings) -> bool:
    return bool(settings.judge and settings.openrouter_api_key)


def cache_key(model: str, brief: str, expect: str, output: str) -> str:
    return hashlib.sha256("\x1e".join((model, brief, expect, output)).encode("utf-8")).hexdigest()


def parse_verdict(text: str) -> dict[str, Any] | None:
    """The JSON object in a reply, fences or prose around it tolerated; scores clamped to 1..5."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    verdict: dict[str, Any] = {}
    for key in ("quality", "usefulness"):
        try:
            verdict[key] = max(1, min(5, int(round(float(raw.get(key))))))
        except (TypeError, ValueError):
            return None
    reasons = raw.get("reasons")
    verdict["reasons"] = [str(r) for r in reasons][:6] if isinstance(reasons, list) else []
    return verdict


def _prompt(brief: str, expect: str, output: str) -> str:
    return (f"BRIEF (what the customer asked for):\n{brief.strip()}\n\n"
            f"ACCEPTANCE (what the pack says a good result contains):\n{expect.strip() or '(none given)'}\n\n"
            f"OUTPUT (what the agent delivered):\n{output.strip()[:OUTPUT_CAP]}")


def judge_output(settings: Settings, *, brief: str, expect: str, output: str,
                 cache_dir: Path = JUDGE_CACHE_DIR) -> dict[str, Any] | None:
    """Grade one output. ``None`` when no judge is configured; a dict with ``error`` when the call failed."""
    if not judge_available(settings):
        return None
    if not output.strip():
        return {"quality": 1, "usefulness": 1, "reasons": ["nothing was delivered"], "source": "rule"}
    key = cache_key(settings.judge_model_id, brief, expect, output)
    cached = cache_dir / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    body = {
        "model": settings.judge_model_id, "temperature": 0,
        "messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": _prompt(brief, expect, output)}],
        "response_format": {"type": "json_object"}, "usage": {"include": True},
    }
    request = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode("utf-8"), method="POST", headers={
        "Authorization": f"Bearer {settings.openrouter_api_key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/automatos-ai", "X-Title": "automatos-sim judge",
    })
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"error": f"judge HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}", "source": "judge"}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"error": f"judge call failed: {type(exc).__name__}: {exc}", "source": "judge"}
    content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    verdict = parse_verdict(content)
    if verdict is None:
        return {"error": f"judge reply was not a verdict: {content[:300]!r}", "source": "judge"}
    usage = payload.get("usage") or {}
    verdict.update({"model": settings.judge_model_id, "source": "judge",
                    "cost_usd": float(usage.get("cost") or 0.0), "tokens": usage.get("total_tokens")})
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict
