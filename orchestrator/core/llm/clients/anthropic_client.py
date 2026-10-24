"""
Anthropic Provider Implementation
=================================

Anthropic Claude models provider.

Images in a message go as Anthropic image blocks: an OpenAI-style data URL (as
``modules/attachments/resolver.py`` builds for an image attachment) as base64, read the
way Bedrock and Gemini read it (:mod:`.image_blocks`), and an https link as a URL source
Anthropic fetches. The Messages API answers 400 to an ``image_url`` block, and to a text
block that is empty or only whitespace, so a blank text block is left out.
"""

import base64
import json
import logging
from typing import Dict, Any, List, Optional

from config import config
from .base import BaseLLMProvider, LLMResponse, accepts_sampling_params, request_max_tokens, run_blocking
from .image_blocks import decoded_image
from core.llm.anthropic_workspace import workspace_for_key, workspace_headers

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)

# The image types the Messages API reads (platform.claude.com/docs/en/build-with-claude/vision).
_ANTHROPIC_IMAGES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
# The content blocks that carry an image: OpenAI's ``image_url`` and Anthropic's own ``image``.
_IMAGE_BLOCKS = frozenset({"image", "image_url"})
# The only links sent as a URL source: Anthropic fetches the image from outside.
_HTTPS = "https://"
# What the model reads where an image couldn't be sent: it says so, never guesses (PRD-223 S0.3).
_IMAGE_LEFT_OUT = ("[IMAGE UNAVAILABLE] An image was attached here, but it could not be shown to you ({why}). "
                   "Do not guess what it shows; say you could not see it.")


def _left_out(why: str) -> Dict[str, Any]:
    """The text block that stands where an image couldn't be sent."""
    return {"type": "text", "text": _IMAGE_LEFT_OUT.format(why=why)}


def _base64_image(media_type: str, data: bytes) -> Dict[str, Any]:
    """An image's bytes as an Anthropic base64 image block; a note, logged, for a type the
    Messages API doesn't read (HEIC, SVG, BMP, TIFF…)."""
    if media_type not in _ANTHROPIC_IMAGES:
        logger.warning("[anthropic] an image of type %r has no Anthropic form; a note stands in for it", media_type)
        return _left_out(f"its type, {media_type or 'unknown'}, is not supported")
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}}


def _is_https(url: str) -> bool:
    return url.lower().startswith(_HTTPS)


def _unfetchable() -> Dict[str, Any]:
    """The note, logged, for an image given by a link Anthropic can't fetch (http, localhost)."""
    logger.warning("[anthropic] an image that is neither inline data nor an https link can't be sent; "
                   "a note stands in for it")
    return _left_out("it is neither inline data nor an https link")


def _linked_image(block: Dict[str, Any]) -> Dict[str, Any]:
    """An OpenAI-style image whose bytes aren't in the message: an https link as a URL
    source; a note, logged, for anything else."""
    link = block.get("image_url")
    url = str(link.get("url", "")) if isinstance(link, dict) else ""
    return {"type": "image", "source": {"type": "url", "url": url}} if _is_https(url) else _unfetchable()


def _referenced_image(block: Dict[str, Any]) -> Dict[str, Any]:
    """An Anthropic image block whose bytes the API fetches itself: a Files API id, or an
    https URL source, as it is; a note, logged, for any other source."""
    source = block.get("source")
    kind = source.get("type") if isinstance(source, dict) else None
    if kind == "file" or (kind == "url" and _is_https(str(source.get("url", "")))):
        return block
    if kind == "url":
        return _unfetchable()
    logger.warning("[anthropic] an image block with no source the API reads; a note stands in for it")
    return _left_out("it has no readable source")


def _anthropic_block(block: Any) -> Any:
    """One content block in the Messages API's shape: an image (OpenAI-style or Anthropic's
    own) as an Anthropic image block, or a note where it can't be sent; any other block as it is."""
    if not isinstance(block, dict) or block.get("type") not in _IMAGE_BLOCKS:
        return block
    try:
        image = decoded_image(block)
    except ValueError:
        logger.warning("[anthropic] an image that isn't valid base64 can't be sent; a note stands in for it")
        return _left_out("its data isn't valid base64")
    if image is not None:
        return _base64_image(*image)
    return _linked_image(block) if block["type"] == "image_url" else _referenced_image(block)


def _is_blank_text(block: Any) -> bool:
    """A text block the Messages API refuses: empty, or only whitespace (a photo sent with
    no caption)."""
    return isinstance(block, dict) and block.get("type") == "text" and not str(block.get("text") or "").strip()


def _anthropic_content(content: List[Any]) -> List[Any]:
    """A content list with its images in the Messages API's shape, in the order they came,
    and its blank text blocks left out."""
    return [_anthropic_block(block) for block in content if not _is_blank_text(block)]


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _tool_results_in(text: str) -> Optional[List[Dict[str, Any]]]:
    """The blocks of a string that is a JSON array of ``tool_result`` blocks (an older way
    to send tool results); None for any other string, a JSON array a user typed included."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, list) and parsed and all(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in parsed):
        return parsed
    return None


def _user_turn(content: Any) -> Optional[Dict[str, Any]]:
    """A user message. A content list goes with its images as Anthropic image blocks and
    its blank text left out (nothing when nothing is left); a string that is a JSON array
    of ``tool_result`` blocks goes as those blocks; any other string as it came."""
    if isinstance(content, list):
        blocks = _anthropic_content(content)
        return {"role": "user", "content": blocks} if blocks else None
    if not isinstance(content, str):
        return {"role": "user", "content": str(content)}
    results = _tool_results_in(content)
    return {"role": "user", "content": content if results is None else results}


def _tool_use(call: Dict[str, Any]) -> Dict[str, Any]:
    """An OpenAI-style tool call as a ``tool_use`` block; arguments that aren't a JSON object go as {}."""
    function = call.get("function") or {}
    arguments = function.get("arguments") or "{}"
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return {"type": "tool_use", "id": call.get("id"), "name": function.get("name"),
            "input": parsed if isinstance(parsed, dict) else {}}


def _assistant_turn(content: Any, tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """An assistant message: its text and its tool calls as ``tool_use`` blocks. A content
    of None (or a list with nothing left) and no tool calls sends nothing, never the text
    "None" (F186)."""
    if tool_calls:
        blocks = _anthropic_content(content) if isinstance(content, list) else (
            [{"type": "text", "text": content}] if isinstance(content, str) and content.strip() else [])
        return {"role": "assistant", "content": [*blocks, *(_tool_use(call) for call in tool_calls)]}
    if isinstance(content, list):
        blocks = _anthropic_content(content)
        return {"role": "assistant", "content": blocks} if blocks else None
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    return None if content is None else {"role": "assistant", "content": str(content)}


def _ends_on_tool_results(turns: List[Dict[str, Any]]) -> bool:
    last = turns[-1] if turns else None
    return bool(last and last["role"] == "user" and isinstance(last["content"], list)
                and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in last["content"]))


def _with_message(turns: List[Dict[str, Any]], msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """``turns`` with one more message (not the system prompt) in the Messages API's shape.
    A tool result joins the user turn of results right before it; a later system message
    (a nudge, a forced synthesis) is a user turn; any other role is left out."""
    role, content = msg.get("role"), msg.get("content", "")
    if role == "tool":
        block = {"type": "tool_result", "tool_use_id": msg.get("tool_call_id"),
                 "content": content if isinstance(content, str) else json.dumps(content)}
        if _ends_on_tool_results(turns):
            return [*turns[:-1], {"role": "user", "content": [*turns[-1]["content"], block]}]
        return [*turns, {"role": "user", "content": [block]}]
    if role == "system":
        turn = {"role": "user", "content": _text(content)}
    elif role == "user":
        turn = _user_turn(content)
    else:
        turn = _assistant_turn(content, msg.get("tool_calls") or []) if role == "assistant" else None
    return turns if turn is None else [*turns, turn]


def _usage_dict(usage: Any) -> Dict[str, int]:
    """Anthropic's accounting: ``input_tokens`` EXCLUDES the cached parts. The
    platform's ``prompt_tokens`` is the FULL prompt (fresh + cache read + cache
    write) with the cache parts reported beside it — the tracker re-prices those
    at the vendor's cache rates (2026-09-09)."""
    fresh = int(getattr(usage, "input_tokens", 0) or 0)
    output = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    prompt = fresh + cache_read + cache_write
    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": prompt + output,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
    }


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation"""
    
    def _initialize_client(self):
        if anthropic is None:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        # Try multiple sources for API key
        api_key = self.config.api_key or config.ANTHROPIC_API_KEY
        
        # BOOTSTRAP STRATEGY: Don't require key at initialization
        # Only fail when actually making API calls
        if not api_key:
            logger.warning(
                "Anthropic API key not configured. "
                "LLM features will fail until key is added. "
                "Configure 'development_anthropic' credential or set ANTHROPIC_API_KEY env var."
            )
            self.client = None  # Will fail gracefully on first use
        else:
            client_kwargs = {"api_key": api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            # An organization-level key names the workspace it bills on every request (9 Oct 2026).
            headers = workspace_headers(workspace_for_key(api_key))
            if headers:
                client_kwargs["default_headers"] = headers

            self.client = anthropic.Anthropic(**client_kwargs)
            logger.info(f"Initialized Anthropic client with model: {self.config.model}")
    
    def _convert_messages_to_anthropic_format(self, messages: List[Dict[str, Any]]) -> tuple:
        """
        Convert OpenAI-style messages to Anthropic format.

        Handles:
        - System messages: the first becomes the system parameter; a later one
          (a nudge, a forced synthesis) is a user turn
        - User messages (can contain text or tool results)
        - Assistant messages: text, and their tool_calls as tool_use blocks
        - Tool results (role "tool"): tool_result blocks in the next user turn
        - Images in a user or assistant content list: Anthropic image blocks
          (base64 for a data URL, a URL source for an https link), or a text
          note where one can't be sent; a blank text block (a photo sent with
          no caption) left out; every other block as it came

        F186 (night 6): an assistant message's tool_calls and the tool results
        were dropped, a content of None went out as the text "None", and a later
        system message replaced the system prompt. A tool loop on this client
        never saw its own calls, their results, or what it said before them.

        23 Oct 2026: a list content went out as it came, so a photo sent from the
        app reached Anthropic as an OpenAI ``image_url`` block, which it answers
        with a 400 ("Input tag 'image_url' found using 'type' does not match").

        Anthropic tool result format:
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "...", "content": "..."}
            ]
        }
        """
        system_message, seen_system = "", False
        turns: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system" and not seen_system:
                system_message, seen_system = _text(msg.get("content", "")), True
            else:
                turns = _with_message(turns, msg)
        return system_message, turns

    def _convert_tools_to_anthropic_format(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI-style tools to Anthropic format.
        
        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "...",
                "parameters": {...}
            }
        }
        
        Anthropic format:
        {
            "name": "tool_name",
            "description": "...",
            "input_schema": {...}  # Note: Anthropic uses input_schema, not parameters
        }
        """
        anthropic_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Handle OpenAI format: {"type": "function", "function": {...}}
                if tool.get("type") == "function" and "function" in tool:
                    func_def = tool["function"]
                    anthropic_tools.append({
                        "name": func_def.get("name"),
                        "description": func_def.get("description", ""),
                        "input_schema": func_def.get("parameters", {})
                    })
                # Handle direct function format: {"name": ..., "description": ..., "parameters": ...}
                elif "name" in tool and ("parameters" in tool or "input_schema" in tool):
                    anthropic_tools.append({
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters") or tool.get("input_schema", {})
                    })
        return anthropic_tools

    @staticmethod
    def _extract_cache_prefix(messages: List[Dict[str, Any]]) -> Optional[str]:
        """PRD-201 S4: the cache-stable prefix hint the assembler stamped on the
        system message (``cache_prefix`` key). Non-Anthropic providers ignore the
        key; this is the only reader. Returns the last non-empty one (there is one
        system message)."""
        prefix: Optional[str] = None
        for msg in messages or []:
            if isinstance(msg, dict) and msg.get("role") == "system" and msg.get("cache_prefix"):
                prefix = msg["cache_prefix"]
        return prefix

    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using Anthropic API without blocking the event loop"""
        if self.client is None:
            raise ValueError(
                "Anthropic API key not configured. Cannot generate response. "
                "Please configure 'development_anthropic' credential or set ANTHROPIC_API_KEY env var."
            )

        from core.llm.request_scope import is_headless_run

        try:
            system_message, user_messages = self._convert_messages_to_anthropic_format(messages)
            # PRD-201 S4: the assembler's cache-stable prefix, if the caller
            # stamped it on the system message. PRD-201 S5: is this a headless
            # agent run? Both are read once, before the executor hop.
            cache_prefix = self._extract_cache_prefix(messages)
            headless = is_headless_run()

            def _call():
                kwargs = self._request_kwargs(system_message, user_messages, cache_prefix, tools, headless)
                if headless:
                    return self.client.beta.messages.create(betas=["context-management-2025-06-27"], **kwargs)
                return self.client.messages.create(**kwargs)

            return self._llm_response(await run_blocking(_call))
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _request_kwargs(self, system_message: Any, user_messages: List[Dict[str, Any]],
                        cache_prefix: Optional[str], tools: Optional[List[Dict]], headless: bool) -> Dict[str, Any]:
        """The Messages API request: model, budget, cached system prompt, messages, tools."""
        from core.llm.prompt_cache import build_cached_system
        from modules.memory.memory_tool import memory_tool_definition

        kwargs = {
            "model": self.config.model,
            "max_tokens": request_max_tokens(self.config),
            # PRD-201 S4: emit cache_control on the stable prefix. This IS
            # the Anthropic client, so the marker is inherently on the
            # Anthropic route only (gate-by-provider, §8-Q1). Render order
            # tools → system → messages caches tools+system together.
            "system": build_cached_system(system_message, cache_prefix),
            "messages": user_messages,
        }
        if accepts_sampling_params(self.config.model):
            kwargs["temperature"] = self.config.temperature
        # PRD-17: Add tools if provided (convert to Anthropic format)
        anthropic_tools = self._convert_tools_to_anthropic_format(tools) if tools else []
        if headless:
            # PRD-201 S5: context editing (clear stale tool results,
            # cache-preservingly) + the memory tool on the Anthropic-routed
            # headless loop only. Beta strings confirmed via the claude-api
            # skill; flag for build-time re-confirm (§8-Q2). Chat never
            # sets the headless scope, so it is unaffected.
            anthropic_tools.append(memory_tool_definition())
            kwargs["context_management"] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        return kwargs

    @staticmethod
    def _llm_response(response: Any) -> LLMResponse:
        """PRD-17: a response's text and tool calls, read by block type (a thinking
        model's first block is its thinking, not text)."""
        tool_calls = [
            {
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(block.input)},
            }
            for block in response.content
            if block.type == "tool_use"
        ]
        return LLMResponse(
            content="".join(block.text for block in response.content if block.type == "text"),
            usage=_usage_dict(response.usage),
            model=response.model,
            provider="anthropic",
            tool_calls=tool_calls or None,
            finish_reason=response.stop_reason,
        )

    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Anthropic API (synchronous)"""
        if self.client is None:
            raise ValueError(
                "Anthropic API key not configured. Cannot generate response. "
                "Please configure 'development_anthropic' credential or set ANTHROPIC_API_KEY env var."
            )
        
        try:
            system_message, user_messages = self._convert_messages_to_anthropic_format(messages)
            
            kwargs = {
                "model": self.config.model,
                "max_tokens": request_max_tokens(self.config),
                "system": system_message,
                "messages": user_messages,
            }
            if accepts_sampling_params(self.config.model):
                kwargs["temperature"] = self.config.temperature
            return self._llm_response(self.client.messages.create(**kwargs))
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

