"""
Prompt Analyzer - Message Processing and Tool Intent Detection
==============================================================

Handles:
- Converting chat messages to LLM format
- Detecting tool intent from user messages
"""

import logging
from typing import List, Dict, Any, Optional

from core.attachment_refs import render_unresolved_file_part

logger = logging.getLogger(__name__)

class PromptAnalyzer:
    """
    Analyzes user prompts for tool intent and complexity.
    Used by StreamingChatService to determine how to process messages.
    """
    
    def is_fresh_start_request(self, text: str) -> bool:
        """Detect user requests to ignore prior chat context/memory."""
        if not text:
            return False
        text_lower = text.lower()
        patterns = [
            "clear memory",
            "reset memory",
            "start fresh",
            "fresh run",
            "new session",
            "ignore previous",
            "ignore earlier",
            "forget previous",
            "forget earlier"
        ]
        return any(p in text_lower for p in patterns)
    
    def convert_to_llm_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        resolved_attachment_ids: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Convert chat messages to LLM format with system prompt.

        Args:
            messages: List of message dicts with role and parts
            system_prompt: Optional custom system prompt
            resolved_attachment_ids: PRD-223 S0.4 — attachment ids the caller
                will inject into this prompt via AttachmentResolver. Their
                file parts are left for the resolver to render; every other
                file part gets an explicit unavailable marker.

        Returns:
            List of LLM-formatted messages
        """
        llm_messages = []
        msg_count = len(messages)
        
        # Default system prompt - friendly and helpful personality
        if system_prompt is None:
            system_prompt = f"""You are Automatos - a friendly, capable AI assistant who genuinely enjoys helping people.

## Who I Am

Hey! I'm Automatos, your AI partner. Think of me as a knowledgeable friend who happens to have access to databases, documents, and all sorts of useful tools. I'm here to help you get things done, not just answer questions.

**My vibe:**
- Warm and conversational - I chat like a helpful colleague, not a corporate robot
- I remember you! I have memory of our conversations ({msg_count} messages so far)
- Action-oriented - if you ask me to do something, I'll actually do it
- Honest - I'll tell you when I don't know something instead of making stuff up
- Genuinely interested in helping you succeed

## How I Work

**For casual chat:** I'll just talk naturally! No need for fancy tools to say hi or discuss ideas.

**When you need data or actions:** I have some great tools:
- **Database queries** - I can search your data, run analytics, get metrics
- **Knowledge search** - Find docs, guides, uploaded files
- **External apps** - Email, Slack, GitHub via integrations
- **File operations** - Create reports, save documents

**My approach:**
1. I figure out what you actually need (not just what you literally asked)
2. I use tools when they help, skip them when they don't
3. I give you real answers with the data, not just instructions
4. I keep it conversational - no corporate jargon

## Quick Tips

- If you've told me your name, I'll use it - feels nicer that way!
- Ask me to "remember" things and I will (for real, I have memory)
- If you want a report, I'll make one and save it so you can download it
- Questions about data? I'll query the database and give you actual numbers

## What I Won't Do

- Pretend I don't have memory (I do!)
- Make up data or statistics (I'll look it up)
- Give you generic corporate-speak (I'll be real with you)
- Use tools when a simple friendly answer will do

Let's get stuff done! What can I help you with?"""

        # If tool metadata is available, reinforce tool usage and list tool names
        tool_names = []
        if available_tools:
            for tool in available_tools:
                if not isinstance(tool, dict):
                    continue
                tool_name = tool.get("function", {}).get("name")
                if tool_name:
                    tool_names.append(tool_name)
            tool_names = sorted(set(tool_names))

        if tool_names and system_prompt:
            max_list = 30
            listed = ", ".join(tool_names[:max_list])
            extra = "" if len(tool_names) <= max_list else f" (+{len(tool_names) - max_list} more)"
            system_prompt += (
                f"\n\n## My Toolkit\n"
                f"I have access to these tools: {listed}{extra}\n\n"
                "I'll use these when they're helpful - like querying databases for real data, "
                "searching docs for information, or connecting to external apps. "
                "But for regular conversation? I'll just chat naturally, no tools needed!"
            )

        # Only add system prompt if non-empty (orchestrated paths build their own)
        if system_prompt:
            llm_messages.append({"role": "system", "content": system_prompt})
        
        # Convert each message
        for msg in messages:
            content = self._extract_message_content(msg, resolved_attachment_ids)
            llm_messages.append({
                'role': msg['role'],
                'content': content
            })

        return llm_messages

    def _extract_message_content(
        self,
        msg: Dict[str, Any],
        resolved_attachment_ids: Optional[List[str]] = None
    ) -> str:
        """Extract text content from a message.

        Legacy ``document://`` file parts are already resolved to text by
        _resolve_file_parts in the service layer. Ephemeral attachments are
        rendered by AttachmentResolver *after* this runs, so file parts it
        owns are skipped here rather than described (PRD-223 S0.4).
        """
        if msg.get('parts'):
            text_parts = []
            for p in msg['parts']:
                if p.get('type') == 'text':
                    text_value = p.get('text')
                    if text_value is not None:
                        text_parts.append(str(text_value))
                elif p.get('type') == 'file':
                    marker = render_unresolved_file_part(p, resolved_attachment_ids)
                    if marker:
                        text_parts.append(marker)
            return '\n'.join(text_parts)
        return msg.get('content', '')
    
    def extract_latest_user_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the latest user message text: the person's own words.

        #1444: the page-context preamble part is left out. This text is what the
        turn remembers and distils, and what its tool and knowledge searches use.
        """
        for msg in reversed(messages):
            if msg.get('role') != 'user':
                continue
            content = self._extract_message_content(_without_page_context(msg))
            if content:
                return content
        return ''


def _without_page_context(msg: Dict[str, Any]) -> Dict[str, Any]:
    """The message without the page-context preamble part (a new dict; ``msg`` is untouched)."""
    from services.page_context import is_page_context_part

    parts = msg.get('parts')
    if not parts:
        return msg
    return {**msg, 'parts': [part for part in parts if not is_page_context_part(part)]}


# Module-level instance for easy access
_prompt_analyzer = None

def get_prompt_analyzer() -> PromptAnalyzer:
    """Get or create the global PromptAnalyzer instance."""
    global _prompt_analyzer
    if _prompt_analyzer is None:
        _prompt_analyzer = PromptAnalyzer()
    return _prompt_analyzer

