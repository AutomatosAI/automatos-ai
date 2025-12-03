"""
Prompt Analyzer - Message Processing and Tool Intent Detection
==============================================================

Handles:
- Converting chat messages to LLM format
- Extracting search terms from conversational queries
- Detecting tool intent from user messages
- Identifying simple vs complex prompts
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Simple message patterns that don't need tools
SIMPLE_PATTERNS = [
    'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'ok', 'yes', 'no',
    'what model', 'who are you', 'what are you'
]

# Tool intent indicators
TOOL_INDICATORS = {
    'search_codebase': [
        'code', 'function', 'class', 'method', 'implement', 'how does', 'works',
        'agentfactory', 'factory', 'service', 'model'
    ],
    'search_knowledge': [
        'document', 'doc', 'guide', 'how to', 'tutorial', 'architecture',
        'design', 'readme', 'help'
    ],
    'query_database': [
        'database', 'data', 'count', 'how many', 'statistics', 'workflow',
        'agent', 'execution', 'list', 'show', 'query'
    ],
    'search_multimodal': [
        'image', 'diagram', 'table', 'chart', 'formula', 'picture', 'screenshot'
    ]
}

# Explicit tool request patterns (for models without native tool calling)
EXPLICIT_TOOL_PATTERNS = {
    'search_knowledge': ['search doc', 'find doc', 'show me doc', 'in the doc'],
    'search_codebase': ['show code', 'show me code', 'search code', 'find code'],
    'query_database': ['query database', 'from database', 'sql', 'how many']
}


class PromptAnalyzer:
    """
    Analyzes user prompts for tool intent and complexity.
    Used by StreamingChatService to determine how to process messages.
    """
    
    def __init__(self):
        self.simple_patterns = SIMPLE_PATTERNS
        self.tool_indicators = TOOL_INDICATORS
        self.explicit_patterns = EXPLICIT_TOOL_PATTERNS
    
    def is_simple_message(self, text: str) -> bool:
        """Check if message is a simple greeting/acknowledgment."""
        text_lower = text.lower().strip()
        return len(text_lower) < 50 and any(p in text_lower for p in self.simple_patterns)
    
    def detect_tool_intent(self, query: str) -> List[str]:
        """
        Detect which tools should be triggered based on user query.
        Returns list of tool names to execute (can be multiple).
        """
        query_lower = query.lower()
        tools_to_trigger = []
        
        for tool_name, indicators in self.tool_indicators.items():
            if any(ind in query_lower for ind in indicators):
                tools_to_trigger.append(tool_name)
        
        return tools_to_trigger
    
    def detect_explicit_tool_requests(self, query: str) -> List[str]:
        """
        Detect EXPLICIT tool requests for models without native tool calling.
        More restrictive than detect_tool_intent - only triggers on clear requests.
        """
        query_lower = query.lower()
        tools_to_trigger = []
        
        for tool_name, patterns in self.explicit_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                tools_to_trigger.append(tool_name)
        
        return tools_to_trigger
    
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
    
    def convert_to_llm_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
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
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = f"""You are the Automatos AI Assistant - a custom AI platform with PERSISTENT MEMORY.

⚠️ CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. You HAVE memory. The conversation history ({msg_count} messages) is provided below.
2. If someone tells you their name, REMEMBER IT and use it.
3. NEVER say "I don't have memory", "I can't remember", "as an AI I don't...", or any similar phrases.
4. When asked "do you remember me?" or "what's my name?" - CHECK the conversation history AND the memory context provided.
5. If the name IS in the conversation or memory, say it. If it's NOT there, say "You haven't told me your name yet - what should I call you?"

You have tools: search_codebase, search_knowledge, query_database (natural language to SQL).

Be helpful, direct, and friendly. You're part of the Automatos family."""
        
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

