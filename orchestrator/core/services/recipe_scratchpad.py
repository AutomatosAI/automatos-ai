"""
Recipe Scratchpad - Ephemeral Inter-Step Context
=================================================

Redis-backed hash per execution that replaces verbose text dumps between
recipe steps.  Falls back to an in-memory dict when Redis is unavailable.

Key layout (Redis HASH):
    recipe_exec:{execution_id}
        _input:{key}              -> trigger/user input values
        _meta:recipe_id           -> recipe PK
        _meta:total_steps         -> step count
        step_{N}:tool_results     -> JSON array of auto-extracted tool summaries
        step_{N}:output_summary   -> first 500 chars of agent response
        step_{N}:exports          -> JSON of agent's explicit scratchpad_write calls
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)

# URL regex for auto-extraction
_URL_RE = re.compile(r'https?://[^\s\'"<>]+')
# KEY: value pattern (e.g. "Branch: fix/PILOT-15")
_KV_RE = re.compile(r'^([A-Z][A-Za-z_ ]{1,30}):\s+(.+)$', re.MULTILINE)


class RecipeScratchpad:
    """
    Ephemeral scratchpad for a single recipe execution.

    Provides structured context to downstream steps instead of dumping
    full agent output text (80-90% token savings).
    """

    def __init__(self, execution_id: str, redis_client=None):
        self.execution_id = execution_id
        self._redis = None
        self._fallback: Dict[str, str] = {}

        # Try to get a Redis connection
        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                from core.redis.client import get_redis_client
                rc = get_redis_client()
                if rc:
                    self._redis = rc.get_redis()
            except Exception:
                pass

        self._key = f"recipe_exec:{execution_id}"

        if self._redis:
            logger.info("[scratchpad] Using Redis for execution %s", execution_id)
        else:
            logger.info("[scratchpad] Redis unavailable — using in-memory fallback for %s", execution_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hset(self, field: str, value: str) -> None:
        try:
            if self._redis:
                self._redis.hset(self._key, field, value)
            else:
                self._fallback[field] = value
        except Exception as exc:
            logger.warning("[scratchpad] hset failed (%s): %s", field, exc)
            self._fallback[field] = value

    def _hget(self, field: str) -> Optional[str]:
        try:
            if self._redis:
                return self._redis.hget(self._key, field)
            return self._fallback.get(field)
        except Exception:
            return self._fallback.get(field)

    def _hgetall(self) -> Dict[str, str]:
        try:
            if self._redis:
                return self._redis.hgetall(self._key) or {}
            return dict(self._fallback)
        except Exception:
            return dict(self._fallback)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_inputs(self, input_data: Optional[dict]) -> None:
        """Populate _input:* fields at execution start."""
        if not input_data:
            return
        for key, value in input_data.items():
            if key == "metadata":
                continue
            self._hset(f"_input:{key}", str(value))

    def write_meta(self, recipe_id: int, total_steps: int) -> None:
        """Store recipe metadata."""
        self._hset("_meta:recipe_id", str(recipe_id))
        self._hset("_meta:total_steps", str(total_steps))

    def write_step_results(
        self,
        step_order: int,
        tool_calls: List[Dict[str, Any]],
        agent_output: str,
        agent_exports: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Auto-extract key data from a completed step and store in scratchpad.

        Called after each step completes. Extracts structured summaries from
        tool results and agent output without any LLM calls.
        """
        prefix = f"step_{step_order}"

        # 1. Auto-extract from tool calls
        extracted = self._extract_tool_summaries(tool_calls)
        if extracted:
            self._hset(f"{prefix}:tool_results", json.dumps(extracted))

        # 2. Agent output summary (first 500 chars)
        if agent_output:
            summary = agent_output[:500]
            if len(agent_output) > 500:
                summary += "..."
            self._hset(f"{prefix}:output_summary", summary)

        # 3. Exports from scratchpad_write tool calls
        if agent_exports:
            self._hset(f"{prefix}:exports", json.dumps(agent_exports))

    def write_export(self, step_order: int, key: str, value: str) -> None:
        """
        Store an explicit export from the scratchpad_write tool.

        Called when an agent uses the scratchpad_write tool during execution.
        """
        prefix = f"step_{step_order}"
        field = f"{prefix}:exports"
        existing_raw = self._hget(field)
        existing: dict = {}
        if existing_raw:
            try:
                existing = json.loads(existing_raw)
            except (json.JSONDecodeError, TypeError):
                pass
        existing[key] = value
        self._hset(field, json.dumps(existing))

    def get_exports(self) -> Dict[str, str]:
        """Return all exports across all steps (flat dict)."""
        all_data = self._hgetall()
        exports: Dict[str, str] = {}
        for field, value in all_data.items():
            if field.endswith(":exports"):
                try:
                    exports.update(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    pass
        return exports

    def format_context_for_step(self, target_step_order: int) -> Optional[str]:
        """
        Build compact structured context for the next step's system message.

        Returns None if no prior data exists (first step).
        """
        all_data = self._hgetall()
        if not all_data:
            return None

        completed_steps = set()
        for field in all_data:
            if field.startswith("step_") and ":" in field:
                step_num_str = field.split(":")[0].replace("step_", "")
                try:
                    step_num = int(step_num_str)
                    if step_num < target_step_order:
                        completed_steps.add(step_num)
                except ValueError:
                    pass

        if not completed_steps:
            return None

        parts = [
            "=" * 40,
            f"EXECUTION CONTEXT (Steps 1-{max(completed_steps)} completed)",
            "=" * 40,
        ]

        # Inputs section
        input_lines = []
        for field, value in sorted(all_data.items()):
            if field.startswith("_input:"):
                key = field.replace("_input:", "")
                input_lines.append(f"- {key}: {value}")
        if input_lines:
            parts.append("\n## Inputs")
            parts.extend(input_lines)

        # Per-step sections
        for step_num in sorted(completed_steps):
            prefix = f"step_{step_num}"
            step_parts = [f"\n## Step {step_num}"]

            # Tool results
            raw_tools = all_data.get(f"{prefix}:tool_results")
            if raw_tools:
                try:
                    tool_items = json.loads(raw_tools)
                    for item in tool_items:
                        action = item.get("action", "unknown")
                        kd = item.get("key_data", {})
                        if kd:
                            kd_str = ", ".join(f"{k}: {v}" for k, v in kd.items())
                            step_parts.append(f"- {action} -> {kd_str}")
                        else:
                            step_parts.append(f"- {action} (completed)")
                except (json.JSONDecodeError, TypeError):
                    pass

            # Output summary
            summary = all_data.get(f"{prefix}:output_summary")
            if summary:
                step_parts.append(f"- Agent: \"{summary}\"")

            # Exports
            raw_exports = all_data.get(f"{prefix}:exports")
            if raw_exports:
                try:
                    exports = json.loads(raw_exports)
                    if exports:
                        exports_str = ", ".join(f"{k}: {v}" for k, v in exports.items())
                        step_parts.append(f"- Exports: {{{exports_str}}}")
                except (json.JSONDecodeError, TypeError):
                    pass

            if len(step_parts) > 1:  # More than just the header
                parts.extend(step_parts)

        parts.append("=" * 40)
        return "\n".join(parts)

    def cleanup(self) -> None:
        """
        Set TTL on the Redis hash (don't delete immediately for debugging).
        No-op for in-memory fallback.
        """
        ttl = getattr(config, "RECIPE_SCRATCHPAD_TTL", 3600)
        try:
            if self._redis:
                self._redis.expire(self._key, ttl)
                logger.info("[scratchpad] Set TTL %ds on %s", ttl, self._key)
        except Exception as exc:
            logger.warning("[scratchpad] Failed to set TTL: %s", exc)

    # ------------------------------------------------------------------
    # Auto-extraction (no LLM calls)
    # ------------------------------------------------------------------

    def _extract_tool_summaries(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract structured summaries from tool call results.

        Pulls out action name + key response fields from Composio tool
        results, search_codebase results, etc.
        """
        summaries = []
        for tc in tool_calls:
            action = tc.get("action", "unknown")
            result = tc.get("result", "")

            key_data: Dict[str, Any] = {}

            # Try to parse result as JSON dict
            result_dict = None
            if isinstance(result, dict):
                result_dict = result
            elif isinstance(result, str):
                # Try parsing as JSON
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        result_dict = parsed
                except (json.JSONDecodeError, TypeError):
                    pass

            if result_dict:
                key_data = self._extract_key_fields(action, result_dict)
            elif isinstance(result, str) and result:
                # llm_context format often has headers then JSON:
                #   "Tool: composio_execute\nStatus: success\n...\n[\n  {...}\n]"
                # Try to find embedded JSON array or object
                json_extracted = self._try_extract_json_from_text(result)
                if json_extracted:
                    key_data = self._extract_key_fields(action, json_extracted)

                # Also extract URLs and KV patterns from text
                urls = _URL_RE.findall(result[:2000])
                if urls:
                    key_data["urls"] = urls[:3]
                kv_matches = _KV_RE.findall(result[:2000])
                for k, v in kv_matches[:5]:
                    key_data[k.strip().lower().replace(" ", "_")] = v.strip()

            summaries.append({
                "action": action,
                "key_data": key_data,
            })

        return summaries

    @staticmethod
    def _try_extract_json_from_text(text: str) -> Optional[dict]:
        """
        Try to find and parse an embedded JSON object or array from formatted
        tool result text (e.g. llm_context from Composio tool executor).

        Returns the first JSON object found (unwraps single-element arrays),
        or None if no JSON found.
        """
        # Look for first [ or { in the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            idx = text.find(start_char)
            if idx == -1:
                continue
            # Find matching end — try from the end of string backwards
            end_idx = text.rfind(end_char)
            if end_idx <= idx:
                continue
            try:
                parsed = json.loads(text[idx:end_idx + 1])
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                    return parsed[0]  # Unwrap single-element array
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    # Merge key fields from all items
                    merged: dict = {}
                    for item in parsed[:3]:
                        if isinstance(item, dict):
                            merged.update(item)
                    return merged
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    def _extract_key_fields(
        self, action: str, data: dict
    ) -> Dict[str, Any]:
        """Extract key fields from a tool result dict based on action type."""
        key_data: Dict[str, Any] = {}

        # Common high-value fields across Composio actions
        important_keys = {
            "key", "issue_key", "id", "pr_url", "url", "html_url",
            "status", "state", "name", "title", "assignee",
            "branch", "ref", "sha", "number",
            "channel_id", "channel", "message_ts",
            "created", "updated",
        }

        for k, v in data.items():
            k_lower = k.lower()
            if k_lower in important_keys or k in important_keys:
                if isinstance(v, (str, int, float, bool)):
                    key_data[k] = v
                elif isinstance(v, dict):
                    # Nested: try to get name/key/id
                    for sub in ("name", "key", "id", "login", "displayName"):
                        if sub in v:
                            key_data[f"{k}_{sub}"] = v[sub]
                            break

        # Composio-specific: look inside nested 'data' or 'details'
        for wrapper in ("data", "details", "response_data"):
            if wrapper in data and isinstance(data[wrapper], dict):
                nested = self._extract_key_fields(action, data[wrapper])
                key_data.update(nested)

        # search_codebase: extract file paths
        action_lower = action.lower()
        if "search" in action_lower and "codebase" in action_lower:
            results = data.get("results", data.get("matches", []))
            if isinstance(results, list):
                files = []
                for r in results[:5]:
                    if isinstance(r, dict):
                        fp = r.get("file_path") or r.get("path") or r.get("filename")
                        if fp:
                            files.append(fp)
                if files:
                    key_data["files"] = files

        # Cap size: drop fields with very long values
        cleaned = {}
        for k, v in key_data.items():
            if isinstance(v, str) and len(v) > 200:
                continue
            cleaned[k] = v

        return cleaned
