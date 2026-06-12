"""Converged tool-loop executor.

PRD-142 W3-S4 (G6): the chat surface (``_stream_tool_loop``) and the agent
``execute_with_prompt`` inner loop both delegate here. The loop pins:

- Iteration cap via ``max_iterations``.
- Dedup via :class:`ToolExecutionTracker` (exact + search-spiral + per-tool
  retry limits, prefix-aware defaults, dispatcher-aware counting).
- ``finish_reason == "length"`` recovery: when the LLM truncates mid tool
  call (JSON malformed), inject a "use shorter content" system message
  and re-invoke the LLM instead of repeatedly failing the bad call.
- Tenant pass-through: the workspace_id given to :meth:`run` is forwarded
  to every tool callback invocation — every endpoint stays workspace-scoped.
- Optional streaming via an ``on_event`` callback — chat passes the SSE
  emitter, agents leave it ``None``.
- Optional ``on_tool_result`` post-tool hook so callers (chat) can layer
  Composio recovery / fatal_error short-circuit on top without re-implementing
  the spine. Returns ``ToolPostResult`` to signal "force-final" or "continue".

Stdlib-only at import time so unit tests can load it without dragging the
``modules.tools`` package init (asyncpg/pgvector heavy).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .tool_execution_tracker import ToolExecutionTracker

logger = logging.getLogger(__name__)


# Public types — callers can import these for static typing.
ToolCall = Dict[str, Any]            # OpenAI-format tool_call dict
Message = Dict[str, Any]             # OpenAI-format chat message
LLMResponse = Any                    # SimpleNamespace-like w/ content, tool_calls, finish_reason, usage
LLMCallback = Callable[[List[Message], Optional[List[Dict[str, Any]]]], Awaitable[LLMResponse]]
ToolCallback = Callable[[str, Dict[str, Any], str, Optional[Any]], Awaitable[Any]]
EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class ToolPostResult:
    """Caller hook return type for :meth:`ToolLoopExecutor.run` ``on_tool_result``.

    - ``force_final``: stop the loop immediately and synthesize a final
      response with ``final_content`` (chat uses this for Composio
      action-not-mapped + fatal_error short-circuits).
    - ``followup_messages``: messages to append BEFORE the next LLM call
      (chat uses this to inject Composio retry hints).
    - ``llm_context_override``: if set, replaces the tool's ``content``
      that goes back to the LLM (chat uses this for direct-action Composio).
    """
    force_final: bool = False
    final_content: Optional[str] = None
    followup_messages: List[Message] = field(default_factory=list)
    llm_context_override: Optional[str] = None


@dataclass
class ToolLoopResult:
    """The return value of :meth:`ToolLoopExecutor.run`.

    Mirrors the SimpleNamespace shape both legacy loops expose downstream:
    the surviving final ``LLMResponse`` plus the iteration count.
    """
    response: LLMResponse
    iterations: int
    forced_final: bool = False
    max_iterations_reached: bool = False


@dataclass
class RoundState:
    """Snapshot of one tool-execution round, passed to ``on_round_end``.

    Lets callers (chat) decide whether to force-synthesize after dedup skips
    or per-tool cap exhaustion — the behaviour the legacy chat loop had.
    Agents can leave ``on_round_end`` unset and ignore this entirely.
    """
    iteration: int
    had_skips: bool = False
    had_fatal_errors: bool = False
    tool_attempts: Dict[str, int] = field(default_factory=dict)


_LENGTH_RECOVERY_MSG = (
    "Your previous response was truncated (output token limit reached) "
    "while writing tool call arguments. The JSON was incomplete and could "
    "not be parsed. Please retry with SHORTER content — use concise text, "
    "fewer sections, or summarise instead of writing full prose in the "
    "tool arguments."
)


class ToolLoopExecutor:
    """Shared tool-execution loop for chat + agent paths.

    Usage:
        executor = ToolLoopExecutor(
            llm_callback=lambda msgs, tools: llm_manager.generate_response(msgs, tools=tools),
            tool_callback=lambda name, args, call_id, ws: my_tool_executor(name, args, ...),
            max_iterations=config.CHATBOT_MAX_TOOL_ITERATIONS,
            content_truncate_tokens=2000,
        )
        result = await executor.run(
            initial_response=response,
            messages=llm_messages,   # mutated in place
            tools=tool_schemas,
            workspace_id=ws_id,
            on_event=my_sse_emitter,     # optional (chat)
            on_tool_result=my_recovery,  # optional (chat)
        )
    """

    def __init__(
        self,
        *,
        llm_callback: LLMCallback,
        tool_callback: ToolCallback,
        max_iterations: int = 10,
        content_truncate_tokens: int = 2000,
        tracker: Optional[ToolExecutionTracker] = None,
    ) -> None:
        self._llm = llm_callback
        self._tool = tool_callback
        self.max_iterations = max(1, int(max_iterations))
        self.content_truncate_tokens = max(0, int(content_truncate_tokens))
        self.tracker = tracker if tracker is not None else ToolExecutionTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        initial_response: LLMResponse,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        workspace_id: Optional[Any] = None,
        on_event: Optional[EventCallback] = None,
        on_tool_result: Optional[
            Callable[[str, Dict[str, Any], Any], Awaitable[Optional[ToolPostResult]]]
        ] = None,
        on_round_end: Optional[
            Callable[[RoundState], Awaitable[Optional[ToolPostResult]]]
        ] = None,
    ) -> ToolLoopResult:
        """Run the tool loop. Returns the final LLM response + iteration count.

        ``messages`` is mutated in place: the loop appends one assistant
        ``tool_calls`` message per round plus the matching ``tool`` results,
        mirroring what both legacy loops did.

        ``on_round_end`` (optional) fires after each round's tool results
        have been appended but before the next LLM call. If it returns a
        ``ToolPostResult`` with ``force_final=True`` the executor synthesizes
        a final answer with ``tools=None`` and returns — preserving the chat
        loop's force-synth-on-skip / force-synth-on-fatal-error behaviour.
        """
        current = initial_response
        iteration = 0
        max_reached = False

        if not _has_tool_calls(current):
            return ToolLoopResult(response=current, iterations=0)

        while _has_tool_calls(current) and iteration < self.max_iterations:
            iteration += 1
            logger.info(
                "[tool-loop] iteration %d: %d tool call(s)",
                iteration, len(current.tool_calls or []),
            )

            # finish_reason=length recovery: if the LLM ran out of output mid
            # tool_call, the JSON arguments are likely malformed. Don't burn
            # an attempt running the bad call — inject a "shorter content"
            # system message and ask the LLM to retry.
            recovery_resp = await self._maybe_recover_truncated_args(
                current, messages, tools,
            )
            if recovery_resp is not None:
                current = recovery_resp
                if not _has_tool_calls(current):
                    return ToolLoopResult(response=current, iterations=iteration)

            tool_results, forced, round_state = await self._execute_round(
                current.tool_calls or [],
                workspace_id=workspace_id,
                iteration=iteration,
                on_event=on_event,
                on_tool_result=on_tool_result,
            )
            if forced is not None:
                # Caller asked us to short-circuit — return the synthesized response.
                return ToolLoopResult(
                    response=forced, iterations=iteration, forced_final=True,
                )

            # Append the assistant tool-call message + matching tool results.
            messages.append(_build_assistant_tool_message(current.tool_calls or []))
            messages.extend(tool_results)

            # Per-round hook: caller may force a final synthesis (chat's
            # dedup/fatal-error short-circuit).
            if on_round_end is not None:
                post = await on_round_end(round_state)
                if post is not None and post.force_final:
                    synth = await self._force_synthesize(messages, post.final_content)
                    return ToolLoopResult(
                        response=synth, iterations=iteration, forced_final=True,
                    )

            # Re-invoke the LLM with the appended history.
            current = await self._llm(messages, tools)

        if iteration >= self.max_iterations and _has_tool_calls(current):
            max_reached = True

        return ToolLoopResult(
            response=current,
            iterations=iteration,
            max_iterations_reached=max_reached,
        )

    async def _force_synthesize(
        self,
        messages: List[Message],
        explicit_content: Optional[str],
    ) -> LLMResponse:
        """Synthesize a final response with tools disabled.

        Used by ``on_round_end`` callers that want to halt the loop early
        (dedup detected, fatal error). If ``explicit_content`` is given,
        wrap it in a SimpleNamespace response without re-calling the LLM.
        """
        from types import SimpleNamespace
        if explicit_content is not None:
            return SimpleNamespace(
                content=explicit_content, tool_calls=None, usage=None,
                finish_reason="stop", model="tool-loop", provider="tool-loop",
            )
        messages.append({
            "role": "system",
            "content": (
                "You now have the tool results needed. "
                "Do NOT call any more tools. "
                "Write the final answer for the user using the tool output above."
            ),
        })
        return await self._llm(messages, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _execute_round(
        self,
        tool_calls: List[ToolCall],
        *,
        workspace_id: Optional[Any],
        iteration: int,
        on_event: Optional[EventCallback],
        on_tool_result: Optional[
            Callable[[str, Dict[str, Any], Any], Awaitable[Optional[ToolPostResult]]]
        ],
    ) -> Tuple[List[Message], Optional[LLMResponse], RoundState]:
        """Execute a single round of tool calls.

        Returns (tool_result_messages, force_final_response_or_None, round_state).
        ``round_state`` carries per-round signals (had_skips, had_fatal_errors,
        per-tool attempts) that the caller's ``on_round_end`` hook can consult.
        """
        tool_results: List[Message] = []
        state = RoundState(iteration=iteration)
        attempts: Dict[str, int] = {}

        for tool_call in tool_calls:
            name = _tc_name(tool_call)
            call_id = _tc_id(tool_call)
            args = _tc_args(tool_call)
            post: Optional[ToolPostResult] = None

            await _emit(on_event, {
                "type": "tool-start",
                "tool_call_id": call_id,
                "tool_name": name,
                "tool_input": args,
            })

            # Dedup + per-tool cap check
            should_skip, reason = self.tracker.should_skip_execution(name, args)
            if should_skip:
                logger.info("[tool-loop] dedup skip %s: %s", name, reason)
                tool_results.append(_tool_msg(call_id, name, f"Skipped: {reason}"))
                state.had_skips = True
                await _emit(on_event, {
                    "type": "tool-end",
                    "tool_call_id": call_id,
                    "tool_name": name,
                    "success": False,
                    "skipped": True,
                    "reason": reason,
                })
                continue

            self.tracker.record_execution(name, args)
            attempts[name] = attempts.get(name, 0) + 1

            start = time.time()
            try:
                result = await self._tool(name, args, call_id, workspace_id)
                content = _result_to_llm_context(result, self.content_truncate_tokens)
                success = True
            except Exception as exc:  # noqa: BLE001 — surface as tool error to LLM
                logger.error("[tool-loop] %s raised: %s", name, exc, exc_info=True)
                result = {"success": False, "error": str(exc)}
                content = f"Error executing {name}: {exc}"
                success = False

            # Caller post-hook (chat: Composio recovery, fatal_error)
            if on_tool_result is not None:
                post = await on_tool_result(name, args, result)
                if post is not None:
                    if post.force_final:
                        # Caller wants us to stop now and return a synthesized response.
                        from types import SimpleNamespace
                        forced = SimpleNamespace(
                            content=post.final_content or "",
                            tool_calls=None,
                            usage=None,
                            finish_reason="stop",
                            model=getattr(result, "model", "tool-loop"),
                            provider=getattr(result, "provider", "tool-loop"),
                        )
                        state.tool_attempts = attempts
                        return tool_results, forced, state
                    if post.llm_context_override is not None:
                        content = post.llm_context_override

            tool_results.append(_tool_msg(call_id, name, content))

            # Per-tool result inspection: fatal-error short-circuit signal.
            if isinstance(result, dict) and result.get("fatal_error"):
                state.had_fatal_errors = True

            duration_ms = int((time.time() - start) * 1000)
            await _emit(on_event, {
                "type": "tool-end",
                "tool_call_id": call_id,
                "tool_name": name,
                "success": success,
                "duration_ms": duration_ms,
            })

            # Followup messages from the caller post-hook (chat: Composio retry hints)
            if on_tool_result is not None and post is not None and post.followup_messages:
                tool_results.extend(post.followup_messages)

        state.tool_attempts = attempts
        return tool_results, None, state

    async def _maybe_recover_truncated_args(
        self,
        response: LLMResponse,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[LLMResponse]:
        """If finish_reason='length' AND any tool_call arguments are malformed
        JSON, inject the recovery system message and re-invoke the LLM.

        Returns the new response (caller continues with it), or None to
        signal no recovery was needed.
        """
        if getattr(response, "finish_reason", None) != "length":
            return None
        tool_calls = response.tool_calls or []
        if not tool_calls:
            return None

        has_bad = False
        for tc in tool_calls:
            args_str = tc.get("function", {}).get("arguments", "{}")
            if isinstance(args_str, str):
                try:
                    json.loads(args_str)
                except json.JSONDecodeError:
                    has_bad = True
                    break
        if not has_bad:
            return None

        logger.warning(
            "[tool-loop] LLM truncated (finish_reason=length) with malformed tool-call JSON — recovering"
        )
        messages.append({"role": "system", "content": _LENGTH_RECOVERY_MSG})
        return await self._llm(messages, tools)


# ---------------------------------------------------------------------------
# Helpers — pure, stdlib only.
# ---------------------------------------------------------------------------


def _has_tool_calls(response: Any) -> bool:
    tcs = getattr(response, "tool_calls", None)
    return bool(tcs)


def _tc_name(tc: ToolCall) -> str:
    return tc.get("function", {}).get("name", "unknown")


def _tc_id(tc: ToolCall) -> str:
    return tc.get("id") or f"call_{int(time.time() * 1000)}"


def _tc_args(tc: ToolCall) -> Dict[str, Any]:
    args_str = tc.get("function", {}).get("arguments", "{}")
    if isinstance(args_str, str):
        try:
            return json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            return {}
    return dict(args_str or {})


def _result_to_llm_context(result: Any, truncate_tokens: int) -> str:
    """Coerce the tool callback's return into a string the LLM can consume.

    PRD-157 S3: oversized tool results are truncated on a *token* boundary
    (model-aware), not by a raw char slice.
    """
    if isinstance(result, dict):
        ctx = result.get("llm_context")
        if ctx is None:
            raw = result.get("raw_result", result)
            ctx = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    else:
        ctx = str(result)
    if truncate_tokens:
        from modules.rag.budget import truncate_to_token_budget

        ctx = truncate_to_token_budget(ctx, truncate_tokens)
    return ctx


def _tool_msg(call_id: str, name: str, content: str) -> Message:
    return {
        "tool_call_id": call_id,
        "role": "tool",
        "name": name,
        "content": content,
    }


def _build_assistant_tool_message(tool_calls: List[ToolCall]) -> Message:
    """Build the assistant message that carries the tool_calls (OpenAI shape)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": _tc_id(tc),
                "type": "function",
                "function": {
                    "name": _tc_name(tc),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                },
            }
            for tc in tool_calls
        ],
    }


async def _emit(cb: Optional[EventCallback], event: Dict[str, Any]) -> None:
    if cb is None:
        return
    try:
        await cb(event)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tool-loop] event callback raised: %s", exc)


__all__ = [
    "ToolLoopExecutor",
    "ToolLoopResult",
    "ToolPostResult",
    "ToolExecutionTracker",
]
