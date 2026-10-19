"""
The platform's messages and tools in Bedrock's Converse shape, and its answer back (#1170).

Converse is one request shape for every Bedrock model that takes messages: Claude
(the 5.x models included), Nova, Llama, Mistral, DeepSeek, gpt-oss. It replaces the
per-family ``invoke_model`` bodies, which sent a Claude-shaped body to any model they
didn't know (Nova, Titan) and never sent a tool turn back.

The conversion follows the Anthropic client's (F186): an assistant's ``tool_calls``
become ``toolUse`` blocks, tool results become ``toolResult`` blocks in the next user
turn, the first system message is the system prompt and a later one is a user turn.
Converse wants turns that alternate, so consecutive turns of one role are merged.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .image_blocks import decoded_image

logger = logging.getLogger(__name__)

# Converse's stop reasons, named as the other providers name them (the tool loop reads "length").
_STOP_REASONS = {
    "end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
    "guardrail_intervened": "content_filter", "content_filtered": "content_filter",
}
# The media types Converse takes, as its image ``format``.
_IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}


def _image_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A base64 image or a data URL (``image_blocks``) as a Converse image; None, logged,
    for a format Converse doesn't take or an image whose bytes aren't in the message."""
    image = decoded_image(block)
    if image is None:
        return None
    media_type, data = image
    if media_type not in _IMAGE_FORMATS:
        logger.warning("[bedrock] an image of type %r has no Converse form; left out", media_type)
        return None
    return {"image": {"format": _IMAGE_FORMATS[media_type], "source": {"bytes": data}}}


def _tool_result(tool_use_id: Any, content: Any) -> Dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return {"toolResult": {"toolUseId": str(tool_use_id), "content": [{"text": text or "(empty)"}]}}


def _content_blocks(content: Any) -> List[Dict[str, Any]]:
    """A message's content as Converse blocks: text, images, and Anthropic-style tool results."""
    if content is None:
        return []
    if not isinstance(content, list):
        text = str(content)
        return [{"text": text}] if text.strip() else []
    blocks: List[Dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and str(block.get("text", "")).strip():
            blocks.append({"text": str(block["text"])})
        elif block.get("type") == "tool_result":
            blocks.append(_tool_result(block.get("tool_use_id"), block.get("content", "")))
        elif (image := _image_block(block)) is not None:
            blocks.append(image)
        else:
            logger.warning("[bedrock] a %r content block has no Converse form; left out", block.get("type"))
    return blocks


def _tool_uses(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """An assistant's OpenAI-style ``tool_calls`` as Converse ``toolUse`` blocks."""
    uses = []
    for call in tool_calls:
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if not isinstance(parsed, dict):
            logger.warning("[bedrock] tool call %s had arguments that aren't a JSON object; sent as {}",
                           call.get("id"))
        uses.append({"toolUse": {"toolUseId": str(call.get("id")), "name": function.get("name"),
                                 "input": parsed if isinstance(parsed, dict) else {}}})
    return uses


def _turn(message: Dict[str, Any]) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """One platform message as a (role, blocks) Converse turn; None when it has nothing to send."""
    role, content = message.get("role"), message.get("content")
    if role == "tool":
        return "user", [_tool_result(message.get("tool_call_id"), content)]
    if role == "assistant":
        blocks = _content_blocks(content) + _tool_uses(message.get("tool_calls") or [])
        return ("assistant", blocks) if blocks else None
    blocks = _content_blocks(content)
    return ("user", blocks) if blocks else None


def _results_first(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A merged user turn with its tool results first, as Converse wants them right after
    the assistant's tool calls; the other blocks keep their order after them."""
    return [b for b in blocks if "toolResult" in b] + [b for b in blocks if "toolResult" not in b]


def converse_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Platform messages as Converse ``(system, messages)``. Turns of one role in a row
    are merged into one, as Converse wants turns that alternate."""
    system: List[Dict[str, Any]] = []
    turns: List[Dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system" and not system:
            system = _content_blocks(message.get("content"))
            continue
        turn = _turn({**message, "role": "user"} if message.get("role") == "system" else message)
        if turn is None:
            continue
        role, blocks = turn
        if turns and turns[-1]["role"] == role:
            turns[-1] = {"role": role, "content": _results_first([*turns[-1]["content"], *blocks])}
        else:
            turns.append({"role": role, "content": blocks})
    return system, turns


def converse_tools(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """OpenAI-style function tools (already sanitised) as Converse's ``toolConfig``."""
    specs = []
    for tool in tools:
        function = tool.get("function") or tool
        spec = {"name": function.get("name"), "inputSchema": {"json": function.get("parameters") or {"type": "object"}}}
        if function.get("description"):
            spec["description"] = function["description"]
        specs.append({"toolSpec": spec})
    return {"tools": specs}


def history_tools(turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A ``toolConfig`` naming the tools a conversation already used, for a call that
    passes no tools of its own: Converse refuses ``toolUse`` and ``toolResult`` blocks
    unless their tools are defined (a tool loop's closing, tool-free call)."""
    names = sorted({block["toolUse"]["name"] for turn in turns for block in turn["content"]
                    if "toolUse" in block and block["toolUse"].get("name")})
    return converse_tools([{"name": name, "parameters": {"type": "object"}} for name in names]) if names else None


def answer(response: Dict[str, Any]) -> Dict[str, Any]:
    """Converse's answer as the platform's response fields: content, tool calls, reasoning,
    usage and why it stopped."""
    blocks = ((response.get("output") or {}).get("message") or {}).get("content") or []
    text = "".join(block["text"] for block in blocks if "text" in block)
    tool_calls = [{
        "id": use.get("toolUseId"), "type": "function",
        "function": {"name": use.get("name"), "arguments": json.dumps(use.get("input") or {})},
    } for use in (block["toolUse"] for block in blocks if "toolUse" in block)]
    reasoning = "\n\n".join(
        (block["reasoningContent"].get("reasoningText") or {}).get("text", "")
        for block in blocks if "reasoningContent" in block).strip()
    usage = response.get("usage") or {}
    stop = response.get("stopReason")
    return {
        "content": text,
        "tool_calls": tool_calls or None,
        "reasoning": reasoning or None,
        "usage": {"prompt_tokens": usage.get("inputTokens", 0), "completion_tokens": usage.get("outputTokens", 0),
                  "total_tokens": usage.get("totalTokens", 0)},
        "finish_reason": _STOP_REASONS.get(stop, stop),
    }
