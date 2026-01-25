"""
Action Classifier
=================

Heuristic-first classifier for Composio actions.
Uses pattern matching for 90% of actions, falls back to LLM for ambiguous cases.

Design Principles (per GPT-5.2 recommendation):
- Heuristics first: Cheaper, faster, more predictable
- LLM fallback: Only for ambiguous cases
- Deterministic: Same input = same output
- Maintainable: Easy to add new patterns
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .taxonomy import (
    CAPABILITY_TAXONOMY,
    DESTRUCTIVE_CAPABILITIES,
    REQUIRES_CONFIRMATION,
    get_risk_level,
)

logger = logging.getLogger(__name__)


@dataclass
class ActionClassification:
    """Result of classifying a Composio action."""

    # Core classification
    capabilities: List[str] = field(default_factory=list)
    intent_keywords: List[str] = field(default_factory=list)

    # Safety classification
    risk_level: str = "safe"  # safe, cautious, dangerous
    destructive: bool = False
    requires_confirmation: bool = False

    # Parameter hints
    required_params: Optional[List[str]] = None
    optional_params: Optional[List[str]] = None

    # Metadata
    normalized_name: Optional[str] = None
    short_description: Optional[str] = None

    # Classification info
    method: str = "heuristic"  # heuristic or llm
    confidence: float = 1.0
    model: Optional[str] = None


class ActionClassifier:
    """
    Classifies Composio actions using heuristics first, LLM only for ambiguous cases.

    This approach:
    - Reduces LLM API costs
    - Provides faster classification
    - Ensures deterministic results for most actions
    - Allows manual override for edge cases
    """

    # =========================================================================
    # HEURISTIC PATTERNS
    # =========================================================================
    # Format: capability -> (name_patterns, description_patterns, confidence)

    CAPABILITY_HEURISTICS: Dict[str, Dict[str, Any]] = {
        # -----------------------------------------------------------------
        # Message operations
        # -----------------------------------------------------------------
        "message.send": {
            "name_patterns": [
                r"send.*message", r"post.*message", r"chat.*post",
                r"send.*dm", r"direct.*message", r"sends?_a_message",
                r"post_to", r"notify", r"send_notification"
            ],
            "desc_patterns": [
                r"send a message", r"post a message", r"sends? message",
                r"notify", r"direct message"
            ],
            "confidence": 0.95
        },
        "message.read": {
            "name_patterns": [
                r"get.*message", r"list.*message", r"read.*message",
                r"fetch.*message", r"retrieve.*message", r"messages?_list"
            ],
            "desc_patterns": [
                r"retrieve message", r"get message", r"list message",
                r"read message", r"fetch message"
            ],
            "confidence": 0.95
        },
        "message.reply": {
            "name_patterns": [
                r"reply", r"respond", r"thread", r"reply_to"
            ],
            "desc_patterns": [
                r"reply to", r"respond to", r"in thread"
            ],
            "confidence": 0.90
        },
        "message.delete": {
            "name_patterns": [
                r"delete.*message", r"remove.*message"
            ],
            "desc_patterns": [
                r"delete a message", r"remove message"
            ],
            "confidence": 0.98
        },

        # -----------------------------------------------------------------
        # Channel operations
        # -----------------------------------------------------------------
        "channel.list": {
            "name_patterns": [
                r"list.*channel", r"get.*channel", r"channels?_list",
                r"list.*conversation", r"conversations?_list"
            ],
            "desc_patterns": [
                r"list channel", r"get channel", r"list conversation"
            ],
            "confidence": 0.95
        },
        "channel.create": {
            "name_patterns": [
                r"create.*channel", r"new.*channel", r"channels?_create"
            ],
            "desc_patterns": [
                r"create a channel", r"new channel"
            ],
            "confidence": 0.95
        },
        "channel.archive": {
            "name_patterns": [
                r"archive.*channel", r"close.*channel"
            ],
            "desc_patterns": [
                r"archive a channel", r"close channel"
            ],
            "confidence": 0.98
        },
        "channel.invite": {
            "name_patterns": [
                r"invite.*channel", r"add.*channel", r"join.*channel"
            ],
            "desc_patterns": [
                r"invite to channel", r"add to channel"
            ],
            "confidence": 0.90
        },

        # -----------------------------------------------------------------
        # File operations
        # -----------------------------------------------------------------
        "file.read": {
            "name_patterns": [
                r"get.*file", r"download.*file", r"read.*file",
                r"files?_get", r"files?_download"
            ],
            "desc_patterns": [
                r"download file", r"get file", r"read file"
            ],
            "confidence": 0.95
        },
        "file.write": {
            "name_patterns": [
                r"upload.*file", r"create.*file", r"write.*file",
                r"files?_upload", r"files?_create"
            ],
            "desc_patterns": [
                r"upload file", r"create file", r"write file"
            ],
            "confidence": 0.95
        },
        "file.delete": {
            "name_patterns": [
                r"delete.*file", r"remove.*file"
            ],
            "desc_patterns": [
                r"delete file", r"remove file"
            ],
            "confidence": 0.98
        },
        "file.list": {
            "name_patterns": [
                r"list.*file", r"files?_list"
            ],
            "desc_patterns": [
                r"list file"
            ],
            "confidence": 0.95
        },

        # -----------------------------------------------------------------
        # Issue operations (GitHub, Jira, etc.)
        # -----------------------------------------------------------------
        "issue.create": {
            "name_patterns": [
                r"create.*issue", r"open.*issue", r"new.*issue",
                r"issues?_create", r"file.*issue"
            ],
            "desc_patterns": [
                r"create an issue", r"open issue", r"new issue"
            ],
            "confidence": 0.95
        },
        "issue.read": {
            "name_patterns": [
                r"get.*issue", r"issues?_get", r"read.*issue"
            ],
            "desc_patterns": [
                r"get issue", r"read issue"
            ],
            "confidence": 0.95
        },
        "issue.list": {
            "name_patterns": [
                r"list.*issue", r"issues?_list"
            ],
            "desc_patterns": [
                r"list issue"
            ],
            "confidence": 0.95
        },
        "issue.update": {
            "name_patterns": [
                r"update.*issue", r"edit.*issue", r"issues?_update"
            ],
            "desc_patterns": [
                r"update issue", r"edit issue"
            ],
            "confidence": 0.95
        },
        "issue.close": {
            "name_patterns": [
                r"close.*issue", r"resolve.*issue"
            ],
            "desc_patterns": [
                r"close issue", r"resolve issue"
            ],
            "confidence": 0.95
        },
        "issue.delete": {
            "name_patterns": [
                r"delete.*issue", r"remove.*issue"
            ],
            "desc_patterns": [
                r"delete issue"
            ],
            "confidence": 0.98
        },
        "issue.comment": {
            "name_patterns": [
                r"comment.*issue", r"add.*comment", r"issues?_comment"
            ],
            "desc_patterns": [
                r"comment on issue", r"add comment"
            ],
            "confidence": 0.90
        },

        # -----------------------------------------------------------------
        # Pull Request operations
        # -----------------------------------------------------------------
        "pr.create": {
            "name_patterns": [
                r"create.*pull", r"open.*pull", r"new.*pull",
                r"pull.*create", r"create.*pr"
            ],
            "desc_patterns": [
                r"create pull request", r"open pull request"
            ],
            "confidence": 0.95
        },
        "pr.read": {
            "name_patterns": [
                r"get.*pull", r"pull.*get", r"read.*pull"
            ],
            "desc_patterns": [
                r"get pull request"
            ],
            "confidence": 0.95
        },
        "pr.list": {
            "name_patterns": [
                r"list.*pull", r"pulls?_list"
            ],
            "desc_patterns": [
                r"list pull request"
            ],
            "confidence": 0.95
        },
        "pr.merge": {
            "name_patterns": [
                r"merge.*pull", r"pull.*merge"
            ],
            "desc_patterns": [
                r"merge pull request"
            ],
            "confidence": 0.98
        },
        "pr.review": {
            "name_patterns": [
                r"review.*pull", r"approve.*pull", r"pull.*review"
            ],
            "desc_patterns": [
                r"review pull request", r"approve pull request"
            ],
            "confidence": 0.95
        },

        # -----------------------------------------------------------------
        # Repository operations
        # -----------------------------------------------------------------
        "repo.create": {
            "name_patterns": [
                r"create.*repo", r"new.*repo", r"repos?_create"
            ],
            "desc_patterns": [
                r"create repository", r"new repository"
            ],
            "confidence": 0.95
        },
        "repo.read": {
            "name_patterns": [
                r"get.*repo", r"repos?_get"
            ],
            "desc_patterns": [
                r"get repository"
            ],
            "confidence": 0.95
        },
        "repo.list": {
            "name_patterns": [
                r"list.*repo", r"repos?_list"
            ],
            "desc_patterns": [
                r"list repository", r"list repositories"
            ],
            "confidence": 0.95
        },
        "repo.delete": {
            "name_patterns": [
                r"delete.*repo", r"remove.*repo"
            ],
            "desc_patterns": [
                r"delete repository"
            ],
            "confidence": 0.98
        },

        # -----------------------------------------------------------------
        # Email operations
        # -----------------------------------------------------------------
        "email.send": {
            "name_patterns": [
                r"send.*email", r"send.*mail", r"emails?_send"
            ],
            "desc_patterns": [
                r"send email", r"send mail"
            ],
            "confidence": 0.95
        },
        "email.read": {
            "name_patterns": [
                r"get.*email", r"read.*email", r"emails?_get"
            ],
            "desc_patterns": [
                r"get email", r"read email"
            ],
            "confidence": 0.95
        },
        "email.list": {
            "name_patterns": [
                r"list.*email", r"emails?_list"
            ],
            "desc_patterns": [
                r"list email"
            ],
            "confidence": 0.95
        },
        "email.delete": {
            "name_patterns": [
                r"delete.*email", r"remove.*email"
            ],
            "desc_patterns": [
                r"delete email"
            ],
            "confidence": 0.98
        },

        # -----------------------------------------------------------------
        # Calendar/Event operations
        # -----------------------------------------------------------------
        "event.create": {
            "name_patterns": [
                r"create.*event", r"schedule.*event", r"new.*event",
                r"events?_create", r"create.*meeting"
            ],
            "desc_patterns": [
                r"create event", r"schedule event", r"create meeting"
            ],
            "confidence": 0.95
        },
        "event.read": {
            "name_patterns": [
                r"get.*event", r"events?_get"
            ],
            "desc_patterns": [
                r"get event"
            ],
            "confidence": 0.95
        },
        "event.list": {
            "name_patterns": [
                r"list.*event", r"events?_list", r"calendar.*list"
            ],
            "desc_patterns": [
                r"list event", r"list calendar"
            ],
            "confidence": 0.95
        },
        "event.delete": {
            "name_patterns": [
                r"delete.*event", r"cancel.*event"
            ],
            "desc_patterns": [
                r"delete event", r"cancel event"
            ],
            "confidence": 0.98
        },

        # -----------------------------------------------------------------
        # User operations
        # -----------------------------------------------------------------
        "user.lookup": {
            "name_patterns": [
                r"get.*user", r"find.*user", r"users?_get", r"lookup.*user"
            ],
            "desc_patterns": [
                r"get user", r"find user", r"lookup user"
            ],
            "confidence": 0.95
        },
        "user.list": {
            "name_patterns": [
                r"list.*user", r"users?_list"
            ],
            "desc_patterns": [
                r"list user"
            ],
            "confidence": 0.95
        },
        "user.invite": {
            "name_patterns": [
                r"invite.*user", r"add.*user"
            ],
            "desc_patterns": [
                r"invite user"
            ],
            "confidence": 0.90
        },

        # -----------------------------------------------------------------
        # Data/Record operations (generic)
        # -----------------------------------------------------------------
        "data.query": {
            "name_patterns": [
                r"search", r"query", r"find", r"filter"
            ],
            "desc_patterns": [
                r"search", r"query", r"find", r"filter"
            ],
            "confidence": 0.80
        },
        "data.create": {
            "name_patterns": [
                r"create", r"add", r"insert", r"new"
            ],
            "desc_patterns": [
                r"create", r"add", r"insert"
            ],
            "confidence": 0.75
        },
        "data.update": {
            "name_patterns": [
                r"update", r"edit", r"modify", r"change"
            ],
            "desc_patterns": [
                r"update", r"edit", r"modify"
            ],
            "confidence": 0.75
        },
        "data.delete": {
            "name_patterns": [
                r"delete", r"remove", r"destroy", r"purge"
            ],
            "desc_patterns": [
                r"delete", r"remove", r"destroy"
            ],
            "confidence": 0.90
        },

        # -----------------------------------------------------------------
        # Spreadsheet operations
        # -----------------------------------------------------------------
        "sheet.read": {
            "name_patterns": [
                r"get.*sheet", r"read.*sheet", r"sheets?_get",
                r"get.*spreadsheet", r"get.*cells?"
            ],
            "desc_patterns": [
                r"get spreadsheet", r"read sheet", r"get cells"
            ],
            "confidence": 0.95
        },
        "sheet.write": {
            "name_patterns": [
                r"update.*sheet", r"write.*sheet", r"sheets?_update",
                r"update.*cells?", r"append.*sheet"
            ],
            "desc_patterns": [
                r"update spreadsheet", r"write to sheet", r"update cells"
            ],
            "confidence": 0.95
        },

        # -----------------------------------------------------------------
        # Auth/Admin operations
        # -----------------------------------------------------------------
        "auth.token": {
            "name_patterns": [
                r"create.*token", r"generate.*token", r"tokens?_create"
            ],
            "desc_patterns": [
                r"create token", r"generate token"
            ],
            "confidence": 0.98
        },
        "auth.revoke": {
            "name_patterns": [
                r"revoke", r"invalidate.*token"
            ],
            "desc_patterns": [
                r"revoke", r"invalidate"
            ],
            "confidence": 0.98
        },
    }

    # Destructive action patterns (always flag as destructive)
    DESTRUCTIVE_PATTERNS = [
        r"delete", r"remove", r"destroy", r"purge", r"drop",
        r"revoke", r"terminate", r"disable", r"archive", r"close"
    ]

    def __init__(self, llm_client=None):
        """
        Initialize the classifier.

        Args:
            llm_client: Optional LLM client for fallback classification
        """
        self.llm_client = llm_client
        self._compiled_patterns: Dict[str, Dict[str, Any]] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        for capability, config in self.CAPABILITY_HEURISTICS.items():
            self._compiled_patterns[capability] = {
                "name": [re.compile(p, re.IGNORECASE) for p in config.get("name_patterns", [])],
                "desc": [re.compile(p, re.IGNORECASE) for p in config.get("desc_patterns", [])],
                "confidence": config.get("confidence", 0.8)
            }

        self._destructive_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DESTRUCTIVE_PATTERNS
        ]

    def classify(
        self,
        action_id: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        force_llm: bool = False
    ) -> ActionClassification:
        """
        Classify a Composio action.

        Args:
            action_id: The action identifier (e.g., "SLACK_SENDS_A_MESSAGE")
            description: Action description from Composio
            parameters: Action parameters schema
            force_llm: Force LLM classification even if heuristics match

        Returns:
            ActionClassification with capabilities and safety info
        """
        if not force_llm:
            heuristic_result = self._classify_by_heuristics(
                action_id, description, parameters
            )
            if heuristic_result.confidence >= 0.85:
                return heuristic_result

        # Fall back to LLM if available and heuristics are uncertain
        if self.llm_client:
            return self._classify_by_llm(action_id, description, parameters)

        # Return heuristic result even with lower confidence if no LLM
        return self._classify_by_heuristics(action_id, description, parameters)

    def _classify_by_heuristics(
        self,
        action_id: str,
        description: Optional[str],
        parameters: Optional[Dict[str, Any]]
    ) -> ActionClassification:
        """Classify using pattern matching heuristics."""

        action_lower = action_id.lower()
        desc_lower = (description or "").lower()

        matched_capabilities: List[Tuple[str, float]] = []

        # Match against capability patterns
        for capability, patterns in self._compiled_patterns.items():
            name_match = any(p.search(action_lower) for p in patterns["name"])
            desc_match = any(p.search(desc_lower) for p in patterns["desc"])

            if name_match or desc_match:
                # Higher confidence for name matches
                confidence = patterns["confidence"]
                if name_match and desc_match:
                    confidence = min(1.0, confidence + 0.05)
                elif desc_match and not name_match:
                    confidence = confidence - 0.1

                matched_capabilities.append((capability, confidence))

        # Sort by confidence and take top matches
        matched_capabilities.sort(key=lambda x: x[1], reverse=True)

        capabilities = [cap for cap, _ in matched_capabilities[:3]]
        max_confidence = matched_capabilities[0][1] if matched_capabilities else 0.5

        # Check for destructive patterns
        destructive = any(p.search(action_lower) for p in self._destructive_patterns)

        # If no specific matches, try generic categorization
        if not capabilities:
            capabilities = self._infer_generic_capability(action_lower, desc_lower)
            max_confidence = 0.6

        # Generate intent keywords from action name
        intent_keywords = self._extract_intent_keywords(action_id)

        # Determine risk level
        risk_level = get_risk_level(capabilities)
        if destructive and risk_level == "safe":
            risk_level = "dangerous"

        # Check if confirmation required
        requires_confirmation = destructive or any(
            cap in REQUIRES_CONFIRMATION for cap in capabilities
        )

        # Extract parameters
        required_params, optional_params = self._extract_params(parameters)

        # Generate normalized name
        normalized_name = self._normalize_action_name(action_id)

        # Generate short description
        short_description = self._generate_short_description(
            action_id, description, capabilities
        )

        return ActionClassification(
            capabilities=capabilities,
            intent_keywords=intent_keywords,
            risk_level=risk_level,
            destructive=destructive,
            requires_confirmation=requires_confirmation,
            required_params=required_params,
            optional_params=optional_params,
            normalized_name=normalized_name,
            short_description=short_description,
            method="heuristic",
            confidence=max_confidence,
        )

    def _classify_by_llm(
        self,
        action_id: str,
        description: Optional[str],
        parameters: Optional[Dict[str, Any]]
    ) -> ActionClassification:
        """Classify using LLM for ambiguous cases."""

        # Build taxonomy string for prompt
        taxonomy_str = "\n".join(
            f"- {cap}: {desc}" for cap, desc in CAPABILITY_TAXONOMY.items()
        )

        prompt = f"""Classify this API action for a tool routing system.

ACTION: {action_id}
DESCRIPTION: {description or 'No description provided'}
PARAMETERS: {parameters or 'No parameters'}

CAPABILITY TAXONOMY:
{taxonomy_str}

Respond in JSON only:
{{
    "capabilities": ["capability.name", ...],  // 1-3 most relevant from taxonomy
    "intent_keywords": ["keyword", ...],       // 3-5 words users might say
    "destructive": boolean,                    // Does it delete/remove/revoke?
    "risk_level": "safe|cautious|dangerous",
    "short_description": "..."                 // <100 chars, action-oriented
}}"""

        try:
            # This is a placeholder - actual implementation depends on LLM client
            response = self.llm_client.complete(
                prompt=prompt,
                model="gpt-4o-mini",  # Use fast model for classification
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response)

            return ActionClassification(
                capabilities=result.get("capabilities", ["data.query"]),
                intent_keywords=result.get("intent_keywords", []),
                risk_level=result.get("risk_level", "safe"),
                destructive=result.get("destructive", False),
                requires_confirmation=result.get("destructive", False),
                normalized_name=self._normalize_action_name(action_id),
                short_description=result.get("short_description"),
                method="llm",
                confidence=0.9,
                model="gpt-4o-mini",
            )

        except Exception as e:
            logger.warning(f"LLM classification failed for {action_id}: {e}")
            # Fall back to heuristics
            return self._classify_by_heuristics(action_id, description, parameters)

    def _infer_generic_capability(
        self,
        action_lower: str,
        desc_lower: str
    ) -> List[str]:
        """Infer generic capability when specific patterns don't match."""

        text = f"{action_lower} {desc_lower}"

        if any(w in text for w in ["list", "get", "fetch", "read", "show"]):
            return ["data.query"]
        if any(w in text for w in ["create", "add", "new", "insert"]):
            return ["data.create"]
        if any(w in text for w in ["update", "edit", "modify", "change"]):
            return ["data.update"]
        if any(w in text for w in ["delete", "remove", "destroy"]):
            return ["data.delete"]
        if any(w in text for w in ["send", "post", "notify"]):
            return ["message.send"]
        if any(w in text for w in ["search", "find", "query"]):
            return ["search.general"]

        return ["data.query"]  # Default to safe read operation

    def _extract_intent_keywords(self, action_id: str) -> List[str]:
        """Extract intent keywords from action name."""

        # Split on underscores and common separators
        words = re.split(r'[_\-\s]+', action_id.lower())

        # Filter out common noise words
        noise_words = {
            'a', 'an', 'the', 'to', 'from', 'in', 'on', 'for', 'with',
            'by', 'of', 'or', 'and', 'is', 'are', 'using', 'via'
        }

        keywords = [w for w in words if w not in noise_words and len(w) > 2]

        # Take first 5 meaningful words
        return keywords[:5]

    def _extract_params(
        self,
        parameters: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Extract required and optional parameters from schema."""

        if not parameters:
            return None, None

        required = []
        optional = []

        properties = parameters.get("properties", {})
        required_list = parameters.get("required", [])

        for param_name in properties.keys():
            if param_name in required_list:
                required.append(param_name)
            else:
                optional.append(param_name)

        return required if required else None, optional if optional else None

    def _normalize_action_name(self, action_id: str) -> str:
        """Convert action ID to a normalized, readable name."""

        # Remove common prefixes
        name = re.sub(r'^(COMPOSIO_|[A-Z]+_)', '', action_id)

        # Convert to lowercase with underscores
        name = re.sub(r'([A-Z])', r'_\1', name).lower()
        name = re.sub(r'_+', '_', name).strip('_')

        # Limit length
        if len(name) > 50:
            name = name[:50]

        return name

    def _generate_short_description(
        self,
        action_id: str,
        description: Optional[str],
        capabilities: List[str]
    ) -> str:
        """Generate a short, action-oriented description."""

        if description and len(description) < 100:
            return description

        if description:
            # Take first sentence
            first_sentence = description.split('.')[0]
            if len(first_sentence) < 100:
                return first_sentence

        # Generate from action name and capabilities
        action_words = ' '.join(self._extract_intent_keywords(action_id))
        return f"{action_words.title()}"[:100]
