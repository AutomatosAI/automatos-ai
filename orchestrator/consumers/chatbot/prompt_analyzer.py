"""
Prompt Analyzer - Message Processing and Tool Intent Detection
==============================================================

Handles:
- Converting chat messages to LLM format
- Extracting search terms from conversational queries
- Detecting tool intent from user messages
- Identifying simple vs complex prompts
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Simple message patterns that don't need tools
SIMPLE_PATTERNS = [
    'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'ok', 'yes', 'no',
    'what model', 'who are you', 'what are you'
]

# Explicit tool call syntax (e.g., "Use tool foo with params {...}")
EXPLICIT_TOOL_CALL_RE = re.compile(
    r"^\s*use\s+tool\s+([a-zA-Z0-9_\-\.]+)\s*(?:with\s+params|params|with\s+arguments|arguments)?\s*(\{.*\})\s*$",
    re.IGNORECASE | re.DOTALL,
)


class PromptAnalyzer:
    """
    Analyzes user prompts for tool intent and complexity.
    Used by StreamingChatService to determine how to process messages.
    """
    
    def __init__(self):
        self.simple_patterns = SIMPLE_PATTERNS
    
    def is_simple_message(self, text: str) -> bool:
        """Check if message is a simple greeting/acknowledgment."""
        text_lower = (text or "").lower().strip()
        if len(text_lower) >= 50:
            return False

        # Normalize punctuation -> spaces so we can do safe token/phrase checks.
        cleaned = re.sub(r"[^a-z0-9\\s]", " ", text_lower)
        cleaned = re.sub(r"\\s+", " ", cleaned).strip()
        if not cleaned:
            return True

        # IMPORTANT:
        # - Single-word patterns must match as WHOLE WORDS (avoid "hi" matching "this")
        # - Multi-word patterns can match as phrases
        words = set(cleaned.split())
        for p in self.simple_patterns:
            p = (p or "").strip().lower()
            if not p:
                continue
            if " " in p:
                if p in cleaned:
                    return True
            else:
                if p in words:
                    return True

        return False

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
    
    def extract_search_terms(self, query: str) -> str:
        """
        Extract meaningful search terms from conversational queries.
        
        Converts: "Can you access my code and show me how agents work?"
        To: "agents AgentFactory agent_factory"
        """
        # Remove common conversational phrases
        stop_phrases = [
            r"can you (please )?",
            r"could you (please )?",
            r"please ",
            r"show me (how )?(to )?",
            r"tell me (about )?(how )?(to )?",
            r"i want to (know|see|understand) (about )?(how )?(to )?",
            r"help me (understand|with) ",
            r"what is (the )?",
            r"what are (the )?",
            r"how (does|do|can|to) ",
            r"where (is|are|can) ",
            r"access my ",
            r"look at (my )?",
            r"find (me )?",
            r"search (for )?",
            r"query (the )?",
        ]
        
        cleaned = query.lower()
        for phrase in stop_phrases:
            cleaned = re.sub(phrase, "", cleaned, flags=re.IGNORECASE)
        
        # Remove punctuation
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        
        # Split into words and filter
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'my', 'your', 'our', 'their', 'its', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
            'work', 'works', 'working', 'about', 'just', 'also', 'only', 'some',
            'code', 'codebase'  # 'code' too generic
        }
        
        words = cleaned.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add variations for common terms
        expanded = []
        term_expansions = {
            'agent': ['AgentFactory', 'agent_factory', 'agent_execution'],
            'agents': ['AgentFactory', 'agent_factory', 'agent_execution'],
            'workflow': ['workflow_execution', 'WorkflowExecution'],
            'workflows': ['workflow_execution', 'WorkflowExecution'],
            'memory': ['memory_system', 'HierarchicalMemorySystem'],
            'memories': ['memory_system', 'HierarchicalMemorySystem'],
            'tool': ['ToolRegistry', 'tool_executor'],
            'tools': ['ToolRegistry', 'tool_executor'],
        }
        
        for kw in keywords:
            expanded.append(kw)
            if kw in term_expansions:
                expanded.extend(term_expansions[kw])
        
        result = " ".join(expanded[:8])  # Limit to 8 terms
        return result if result.strip() else query[:50]

    def parse_explicit_tool_call(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Parse explicit tool call syntax:
        "Use tool <tool_name> with params {...}"
        """
        match = EXPLICIT_TOOL_CALL_RE.match(query or "")
        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2) or "{}"
        try:
            tool_args = json.loads(args_str)
        except Exception as exc:
            logger.warning(f"Failed to parse explicit tool params for {tool_name}: {exc}")
            return {
                "tool_name": tool_name,
                "tool_args": {},
                "parse_error": f"Invalid JSON params: {exc}"
            }

        return {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "parse_error": None
        }

    def convert_to_llm_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """
        Convert chat messages to LLM format with system prompt.
        
        Args:
            messages: List of message dicts with role and parts
            system_prompt: Optional custom system prompt
            
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
            content = self._extract_message_content(msg)
            llm_messages.append({
                'role': msg['role'],
                'content': content
            })
        
        return llm_messages
    
    def _extract_message_content(self, msg: Dict[str, Any]) -> str:
        """Extract text content from a message."""
        if msg.get('parts'):
            text_parts = []
            for p in msg['parts']:
                if p.get('type') != 'text':
                    continue
                text_value = p.get('text')
                if text_value is not None:
                    text_parts.append(str(text_value))
            return '\n'.join(text_parts)
        return msg.get('content', '')
    
    def extract_latest_user_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the latest user message text."""
        for msg in reversed(messages):
            if msg.get('role') != 'user':
                continue
            content = self._extract_message_content(msg)
            if content:
                return content
        return ''


# Module-level instance for easy access
_prompt_analyzer = None

def get_prompt_analyzer() -> PromptAnalyzer:
    """Get or create the global PromptAnalyzer instance."""
    global _prompt_analyzer
    if _prompt_analyzer is None:
        _prompt_analyzer = PromptAnalyzer()
    return _prompt_analyzer

