"""
Smart Intent Classifier
========================

Intelligently determines whether a user message requires:
- Simple conversation (no tools needed)
- Memory retrieval (personal context)
- Tool execution (database, search, external apps)
- Multi-step reasoning (complex tasks)

This is the "brain" that decides HOW to respond, not just WHAT to respond.
"""

import re
import logging
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Primary intent categories."""
    GREETING = "greeting"           # Hi, hello, how are you
    CHITCHAT = "chitchat"           # General conversation, feelings, opinions
    MEMORY_RECALL = "memory_recall" # What's my name, what did we discuss
    FACTUAL = "factual"             # General knowledge questions
    DATA_QUERY = "data_query"       # Database, analytics, metrics
    SEARCH = "search"               # Find documents, knowledge base
    EXTERNAL_ACTION = "external"    # Email, Slack, GitHub actions
    CREATION = "creation"           # Create file, write report, generate
    MULTI_STEP = "multi_step"       # Complex task requiring planning


@dataclass
class IntentResult:
    """Result of intent classification."""
    primary_intent: Intent
    confidence: float
    requires_tools: bool
    requires_memory: bool
    suggested_tools: List[str]
    reasoning: str
    is_simple: bool  # Can be answered without tools


class SmartIntentClassifier:
    """
    Classifies user intent to route messages appropriately.

    Philosophy:
    - Default to CONVERSATION (chitchat) unless clear tool signals
    - Memory is for personal context, not general knowledge
    - Tools are for ACTIONS, not opinions or feelings
    """

    # Greeting patterns - definite conversation
    GREETINGS = {
        "hi", "hello", "hey", "howdy", "hiya", "yo",
        "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "thx", "ty",
        "bye", "goodbye", "see you", "later",
        "ok", "okay", "sure", "cool", "nice", "great", "awesome",
        "yes", "no", "yep", "nope", "yeah", "nah"
    }

    # Memory recall patterns - need to check user's personal context
    MEMORY_PATTERNS = [
        r"\bmy name\b", r"\bwho am i\b", r"\bdo you know me\b",
        r"\bremember\b.*\b(me|my|i)\b", r"\bwhat did (we|i|you)\b",
        r"\bearlier\b.*\b(said|told|discussed)\b",
        r"\blast time\b", r"\bpreviously\b", r"\bbefore\b.*\b(you|we|i)\b",
        r"\bwhat do you know about me\b", r"\bmy (preference|favorite)\b"
    ]

    # Data/Analytics patterns - need database tools
    DATA_PATTERNS = [
        r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bstatistics\b",
        r"\bmetrics\b", r"\banalytics\b", r"\breport\b", r"\btrends?\b",
        r"\bquery\b.*\b(database|db|data)\b", r"\bsql\b",
        r"\blist (all|the|my)\b", r"\bshow (me )?(all|the|my)\b.*\b(agent|workflow|user|recipe)\b",
        r"\bfrom (the )?database\b", r"\bin (the )?database\b",
        # PRD-64: Platform self-awareness queries
        r"\bmy agents?\b", r"\bmy recipes?\b", r"\bmy workflows?\b",
        r"\busage\b", r"\bcosts?\b", r"\btoken (usage|costs?|spend)\b",
        r"\bconnected apps?\b", r"\bmy documents?\b",
        # PRD-64: Broader platform entity queries (catch "what agents do I have?" etc.)
        r"\b(what|which|show|list|get|display|tell)\b.*\bagents?\b",
        r"\b(what|which|show|list|get|display|tell)\b.*\brecipes?\b",
        r"\b(what|which|show|list|get|display|tell)\b.*\bworkflows?\b",
        r"\b(what|which|show|list|get|display|tell)\b.*\bdocuments?\b.*\b(uploaded|have|stored|exist)\b",
        r"\b(what|which|show|list|get|display|tell)\b.*\bworkspace\b",
        r"\b(what|which|show|list|get|display|tell)\b.*\bintegrations?\b",
        r"\b(what|which|show|list|get|display|tell)\b.*\bconnected\b",
        r"\bagents?\b.*\b(do i have|exist|available|set up|configured|running)\b",
        r"\brecipes?\b.*\b(do i have|exist|available|set up|configured|running)\b",
        r"\bmemory\b.*\b(stats?|stored|count|how many)\b",
        r"\bhow much\b.*\b(spend|cost|spent|spending)\b",
    ]

    # Search patterns - need knowledge base tools
    SEARCH_PATTERNS = [
        r"\bfind\b.*\b(doc|document|file|info)\b",
        r"\bsearch\b.*\b(for|about|in)\b", r"\blook up\b",
        r"\bin the (docs?|knowledge|files?)\b",
        r"\bdocumentation\b", r"\bhow (do|does|to)\b.*\b(work|use)\b",
        r"\bguide\b", r"\btutorial\b"
    ]

    # External action patterns - need Composio/external tools
    EXTERNAL_PATTERNS = [
        r"\b(send|write|compose|draft)\b.*\b(email|message|slack)\b",
        r"\b(post|share|publish)\b.*\b(to|on)\b",
        r"\bgithub\b", r"\bgitlab\b", r"\bjira\b", r"\bslack\b", r"\bgmail\b",
        r"\bcreate\b.*\b(issue|pr|pull request|ticket)\b",
        r"\bcheck my\b.*\b(email|inbox|notifications)\b"
    ]

    # Creation patterns - need file/document tools
    CREATION_PATTERNS = [
        r"\bcreate\b.*\b(file|report|document|pdf|docx|xlsx|spreadsheet)\b",
        r"\bwrite\b.*\b(to|a)\b.*\b(file|report|document)\b",
        r"\bgenerate\b.*\b(report|analysis|summary|document|pdf|invoice)\b",
        r"\bmake\b.*\b(a|me|the)\b.*\b(report|document|pdf|spreadsheet)\b",
        r"\bexport\b.*\b(as|to)\b",
        r"\bsave\b.*\b(as|to)\b",
        # PRD-64: Platform creation actions
        r"\bcreate\b.*\b(agent|recipe|workflow)\b",
        r"\bmake\b.*\b(agent|recipe|workflow)\b",
    ]

    # Workspace code patterns - need workspace file/exec/git tools
    WORKSPACE_PATTERNS = [
        # File operations
        r"\bread\b.*\b(file|code|source|script|config)\b",
        r"\bshow\b.*\b(file|code|source|content|line)\b",
        r"\bopen\b.*\b(file|code)\b",
        r"\bcat\b.*\.\w{1,5}\b",  # cat main.py, cat config.yml
        r"\bview\b.*\b(file|code|source)\b",
        # Edit/fix operations
        r"\b(fix|edit|update|change|modify|patch|refactor)\b.*\b(file|code|bug|error|typo|line|function|class|method)\b",
        r"\b(bug|error|typo|issue)\b.*\b(in|on|at)\b.*\.\w{1,5}\b",  # bug in main.py
        r"\bwrite\b.*\b(code|function|class|method|test)\b",
        # Search/grep operations
        r"\b(search|grep|find)\b.*\b(code|function|class|def |import|variable|string|pattern|error|todo)\b",
        r"\bwhere\b.*\b(defined|declared|imported|used|called)\b",
        r"\bfind\b.*\b(in the|in my|across)\b.*\b(repo|code|project|codebase)\b",
        # Execution
        r"\b(run|execute)\b.*\b(test|tests|pytest|jest|npm|script|command|build|lint)\b",
        r"\bpytest\b", r"\bnpm test\b", r"\bnpm run\b",
        # Git operations
        r"\b(commit|push|pull|diff|blame|stash)\b",
        r"\bgit\b.*\b(status|log|add|commit|push|pull|diff|branch|checkout)\b",
        # File references (explicit paths)
        r"\b\w+\.(py|js|ts|jsx|tsx|json|yaml|yml|md|html|css|sql|go|rs|rb|java|sh)\b",
        # General workspace/repo context
        r"\b(codebase|source code|repository|repo)\b",
        r"\bfiles?\b.*\b(in the|in my)\b.*\b(workspace|repo|project)\b",
        r"\bwhat files\b",
        r"\bproject structure\b",
        r"\blist\b.*\b(files|directory|folder)\b",
    ]

    # Chitchat/Opinion patterns - NO tools needed
    CHITCHAT_PATTERNS = [
        r"\bwhat do you think\b", r"\byour opinion\b",
        r"\bhow (are|do) you\b", r"\bdo you (like|feel|think)\b",
        r"\btell me (about yourself|a joke|something)\b",
        r"\bwho (are|made) you\b", r"\bwhat (are|is) you\b",
        r"\bcan you\b.*\b(help|assist)\b",  # Offers of help
        r"\bwhat can you do\b", r"\byour capabilities\b"
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._memory_re = [re.compile(p, re.IGNORECASE) for p in self.MEMORY_PATTERNS]
        self._data_re = [re.compile(p, re.IGNORECASE) for p in self.DATA_PATTERNS]
        self._search_re = [re.compile(p, re.IGNORECASE) for p in self.SEARCH_PATTERNS]
        self._external_re = [re.compile(p, re.IGNORECASE) for p in self.EXTERNAL_PATTERNS]
        self._creation_re = [re.compile(p, re.IGNORECASE) for p in self.CREATION_PATTERNS]
        self._chitchat_re = [re.compile(p, re.IGNORECASE) for p in self.CHITCHAT_PATTERNS]
        self._workspace_re = [re.compile(p, re.IGNORECASE) for p in self.WORKSPACE_PATTERNS]

    def classify(self, query: str, conversation_context: Optional[List[Dict]] = None) -> IntentResult:
        """
        Classify the user's intent.

        Args:
            query: The user's message
            conversation_context: Recent conversation history (for context)

        Returns:
            IntentResult with classification details
        """
        if not query:
            return IntentResult(
                primary_intent=Intent.CHITCHAT,
                confidence=1.0,
                requires_tools=False,
                requires_memory=False,
                suggested_tools=[],
                reasoning="Empty query",
                is_simple=True
            )

        query_lower = query.lower().strip()
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)
        words = set(query_clean.split())

        # 1. Check for greetings (highest confidence, simplest case)
        if self._is_greeting(query_lower, words):
            return IntentResult(
                primary_intent=Intent.GREETING,
                confidence=0.95,
                requires_tools=False,
                requires_memory=False,
                suggested_tools=[],
                reasoning="Greeting or acknowledgment detected",
                is_simple=True
            )

        # 2. Check for workspace/code operations (needs workspace tools)
        if self._matches_patterns(query, self._workspace_re):
            suggested = self._get_workspace_tool_hints(query_lower)
            return IntentResult(
                primary_intent=Intent.MULTI_STEP,
                confidence=0.85,
                requires_tools=True,
                requires_memory=False,
                suggested_tools=suggested or ["workspace_list_dir", "workspace_read_file"],
                reasoning="Workspace code operation detected",
                is_simple=False
            )

        # 3. Check for chitchat/opinion (no tools needed)
        if self._matches_patterns(query, self._chitchat_re):
            return IntentResult(
                primary_intent=Intent.CHITCHAT,
                confidence=0.85,
                requires_tools=False,
                requires_memory=False,
                suggested_tools=[],
                reasoning="Conversational/opinion question",
                is_simple=True
            )

        # 3. Check for memory recall (personal context needed)
        if self._matches_patterns(query, self._memory_re):
            return IntentResult(
                primary_intent=Intent.MEMORY_RECALL,
                confidence=0.9,
                requires_tools=False,
                requires_memory=True,
                suggested_tools=[],
                reasoning="User asking about personal context or past conversations",
                is_simple=True
            )

        # 4. Check for data/analytics queries
        if self._matches_patterns(query, self._data_re):
            suggested = ["smart_query_database", "query_database"]
            # PRD-64: Suggest platform actions for platform-specific queries
            suggested += self._get_platform_tool_hints(query_lower)
            return IntentResult(
                primary_intent=Intent.DATA_QUERY,
                confidence=0.85,
                requires_tools=True,
                requires_memory=False,
                suggested_tools=suggested,
                reasoning="Data/analytics query detected",
                is_simple=False
            )

        # 5-7. Check for compound intents BEFORE returning single intents.
        # "Search docs and create a PDF" is MULTI_STEP, not just SEARCH.
        is_search = self._matches_patterns(query, self._search_re)
        is_creation = self._matches_patterns(query, self._creation_re)
        is_external = self._matches_patterns(query, self._external_re)

        if is_search and is_creation:
            return IntentResult(
                primary_intent=Intent.MULTI_STEP,
                confidence=0.9,
                requires_tools=True,
                requires_memory=False,
                suggested_tools=["search_knowledge", "generate_document"],
                reasoning="Multi-step: search knowledge then generate document",
                is_simple=False
            )

        if is_search:
            return IntentResult(
                primary_intent=Intent.SEARCH,
                confidence=0.85,
                requires_tools=True,
                requires_memory=False,
                suggested_tools=["search_knowledge", "semantic_search"],
                reasoning="Knowledge search query detected",
                is_simple=False
            )

        # External + creation combo (already have is_external, is_creation from above)
        if is_external and is_creation:
            return IntentResult(
                primary_intent=Intent.MULTI_STEP,
                confidence=0.85,
                requires_tools=True,
                requires_memory=False,
                suggested_tools=["generate_document", "composio_execute"],
                reasoning="Multi-step: create document + external action",
                is_simple=False
            )

        if is_external:
            return IntentResult(
                primary_intent=Intent.EXTERNAL_ACTION,
                confidence=0.85,
                requires_tools=True,
                requires_memory=False,
                suggested_tools=["composio_execute"],
                reasoning="External app action requested",
                is_simple=False
            )

        if is_creation:
            suggested = ["generate_document", "write_file"]
            suggested += self._get_platform_tool_hints(query_lower)
            return IntentResult(
                primary_intent=Intent.CREATION,
                confidence=0.8,
                requires_tools=True,
                requires_memory=False,
                suggested_tools=suggested,
                reasoning="Content creation requested",
                is_simple=False
            )

        # 8. Check for complex multi-step (long queries with multiple verbs)
        if self._is_complex_query(query):
            return IntentResult(
                primary_intent=Intent.MULTI_STEP,
                confidence=0.75,
                requires_tools=True,
                requires_memory=True,  # May need context
                suggested_tools=[],  # LLM decides
                reasoning="Complex multi-step task detected",
                is_simple=False
            )

        # 9. Default: Factual/Conversational (let LLM handle without tools)
        # This is the KEY change - we default to NO tools
        return IntentResult(
            primary_intent=Intent.FACTUAL,
            confidence=0.7,
            requires_tools=False,  # Important: don't force tools
            requires_memory=True,  # Check if user context helps
            suggested_tools=[],
            reasoning="General question - attempting conversational response first",
            is_simple=True
        )

    def _is_greeting(self, query_lower: str, words: set) -> bool:
        """Check if query is a simple greeting."""
        # Direct match
        if query_lower in self.GREETINGS:
            return True

        # Check if starts with greeting word and is short
        if len(words) <= 5:
            for word in words:
                if word in self.GREETINGS:
                    return True

        return False

    def _matches_patterns(self, query: str, patterns: List[re.Pattern]) -> bool:
        """Check if query matches any pattern in list."""
        for pattern in patterns:
            if pattern.search(query):
                return True
        return False

    def _get_platform_tool_hints(self, query_lower: str) -> List[str]:
        """PRD-64: Return platform action tool hints based on query keywords."""
        hints = []

        # Agent queries
        if any(w in query_lower for w in ["agent", "agents"]):
            if any(w in query_lower for w in ["create", "make", "build", "add"]):
                hints.append("platform_create_agent")
            else:
                hints.append("platform_list_agents")

        # Recipe/workflow queries
        if any(w in query_lower for w in ["recipe", "recipes", "workflow", "workflows"]):
            if any(w in query_lower for w in ["create", "make", "build"]):
                hints.append("platform_create_recipe")
            else:
                hints.append("platform_list_recipes")

        # Analytics/usage queries
        if any(w in query_lower for w in ["usage", "token", "tokens", "cost", "costs", "spend"]):
            hints.append("platform_get_llm_usage")
            hints.append("platform_get_cost_breakdown")

        # Document queries
        if any(w in query_lower for w in ["document", "documents", "uploaded"]):
            hints.append("platform_list_documents")

        # Workspace queries
        if any(w in query_lower for w in ["workspace", "connected app", "connected apps", "integration"]):
            hints.append("platform_get_workspace_info")
            hints.append("platform_list_connected_apps")

        return hints

    def _get_workspace_tool_hints(self, query_lower: str) -> List[str]:
        """Return workspace tool hints based on query keywords."""
        hints = []

        # File read/view
        if any(w in query_lower for w in ["read", "show", "open", "view", "cat", "content"]):
            hints.append("workspace_read_file")

        # File edit/write/fix
        if any(w in query_lower for w in ["fix", "edit", "update", "change", "modify", "patch", "write", "refactor"]):
            hints.append("workspace_read_file")
            hints.append("workspace_write_file")

        # Search/grep
        if any(w in query_lower for w in ["search", "grep", "find", "where"]):
            hints.append("workspace_grep")

        # Execution
        if any(w in query_lower for w in ["run", "execute", "test", "pytest", "npm", "build", "lint"]):
            hints.append("workspace_exec")

        # Git
        if any(w in query_lower for w in ["commit", "push", "pull", "diff", "blame", "stash", "git"]):
            hints.append("workspace_git")

        # Directory listing
        if any(w in query_lower for w in ["files", "directory", "folder", "structure", "list"]):
            hints.append("workspace_list_dir")

        return hints

    def _is_complex_query(self, query: str) -> bool:
        """Detect complex multi-step queries."""
        # Multiple action verbs
        action_verbs = ["create", "find", "search", "get", "send", "write",
                       "analyze", "compare", "generate", "summarize"]
        verb_count = sum(1 for v in action_verbs if v in query.lower())

        # Long query with multiple sentences
        sentences = query.split('.')

        return verb_count >= 2 or (len(sentences) >= 3 and len(query) > 150)

    def should_use_tools(self, query: str) -> Tuple[bool, str]:
        """
        Quick check: should we even consider tools for this query?

        Returns:
            (should_use_tools, reason)
        """
        result = self.classify(query)
        return (result.requires_tools, result.reasoning)

    def get_tool_hint(self, query: str) -> Optional[str]:
        """
        Get a hint about which tool category might be needed.
        Used to filter available tools before sending to LLM.
        """
        result = self.classify(query)
        if result.suggested_tools:
            return result.suggested_tools[0]
        return None


# Module-level singleton
_classifier = None

def get_intent_classifier() -> SmartIntentClassifier:
    """Get the global intent classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = SmartIntentClassifier()
    return _classifier
