"""
Intent Classifier (Phase 3)
============================

Classifies user queries to determine:
1. Category (EMAIL, CALENDAR, CODE, COMMUNICATION, KNOWLEDGE, DATABASE, etc.)
2. Action Type (FETCH, CREATE, UPDATE, DELETE, SEARCH)
3. Confidence score

Uses rule-based pattern matching for speed (< 100ms).
Can be enhanced with ML models later.
"""

import re
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IntentClassification:
    """Result of intent classification."""
    category: str
    action_type: str
    confidence: float
    reasoning: str
    method: str = "rule_based"


class IntentClassifier:
    """
    Rule-based intent classifier for query routing.
    
    Fast (< 100ms) pattern matching for tool selection.
    """
    
    # Category patterns (regular expressions)
    CATEGORY_PATTERNS = {
        "EMAIL": [
            r"\b(email|mail|gmail|outlook|inbox|message|send.*email|compose|reply|forward)\b",
            r"\b(recipient|subject line|attach|cc|bcc)\b"
        ],
        "CALENDAR": [
            r"\b(calendar|meeting|schedule|appointment|event|invite|reschedule)\b",
            r"\b(book.*time|set.*reminder|agenda|time slot)\b"
        ],
        "CODE": [
            r"\b(github|repository|repo|pull request|pr|issue|commit|branch|merge)\b",
            r"\b(code|programming|function|class|method|variable|api)\b",
            r"\b(codegraph|analyze.*code|code.*structure)\b"
        ],
        "COMMUNICATION": [
            r"\b(slack|chat|dm|message.*channel|teams|discord)\b",
            r"\b(post.*slack|send.*slack|notify.*team)\b"
        ],
        "KNOWLEDGE": [
            r"\b(search.*knowledge|rag|documents|files|wiki)\b",
            r"\b(find.*document|look up.*info|research|reference)\b",
            r"\b(knowledge base|documentation|manual)\b"
        ],
        "DATABASE": [
            r"\b(database|query|table|sql|postgres|mysql)\b",
            r"\b(fetch.*from.*database|get.*data|retrieve.*record)\b",
            r"\b(nl2sql|natural language.*sql)\b"
        ],
        "STORAGE": [
            r"\b(drive|dropbox|box|file.*storage|cloud.*storage)\b",
            r"\b(upload|download|sync.*file|share.*file)\b"
        ],
        "PROJECT_MANAGEMENT": [
            r"\b(jira|asana|trello|monday|notion|task|ticket)\b",
            r"\b(create.*task|update.*ticket|assign.*issue)\b"
        ],
        "ANALYTICS": [
            r"\b(analytics|metrics|dashboard|report|chart|graph)\b",
            r"\b(visualize|analyze.*data|statistics)\b"
        ],
        "AI": [
            r"\b(generate|summarize|analyze|translate|explain)\b",
            r"\b(chatgpt|gpt|claude|llm|ai.*model)\b"
        ],
        "MEMORY": [
            r"\b(remember|recall|memory|save.*for.*later|store.*information)\b",
            r"\b(what.*remember|remind.*me|past.*conversation)\b"
        ]
    }
    
    # Action type patterns
    ACTION_PATTERNS = {
        "FETCH": [
            r"\b(get|fetch|retrieve|show|list|find|search|look.*up)\b",
            r"\b(what|which|where|who|when)\b"
        ],
        "CREATE": [
            r"\b(create|add|new|make|compose|write|post|send)\b",
            r"\b(schedule|book|set.*up)\b"
        ],
        "UPDATE": [
            r"\b(update|edit|modify|change|revise|amend)\b",
            r"\b(reschedule|reassign)\b"
        ],
        "DELETE": [
            r"\b(delete|remove|cancel|discard|unschedule)\b"
        ],
        "SEARCH": [
            r"\b(search|find|look for|locate|discover)\b"
        ],
        "ANALYZE": [
            r"\b(analyze|examine|investigate|review|inspect)\b",
            r"\b(summarize|explain|breakdown)\b"
        ]
    }
    
    def __init__(self):
        # Pre-compile regex patterns for performance
        self.compiled_categories = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.CATEGORY_PATTERNS.items()
        }
        
        self.compiled_actions = {
            action: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for action, patterns in self.ACTION_PATTERNS.items()
        }
    
    def classify(self, query: str) -> IntentClassification:
        """
        Classify a user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            IntentClassification with category, action_type, confidence
        """
        logger.debug(f"[intent] Classifying query: {query[:100]}...")
        
        query_lower = query.lower()
        
        # Step 1: Detect category
        category_scores = {}
        for category, patterns in self.compiled_categories.items():
            score = 0
            matches = []
            for pattern in patterns:
                if pattern.search(query_lower):
                    score += 1
                    matches.append(pattern.pattern)
            
            if score > 0:
                category_scores[category] = {
                    "score": score,
                    "matches": matches
                }
        
        # Get best category
        if not category_scores:
            # No category matched - default to GENERAL
            best_category = "GENERAL"
            category_confidence = 0.3
            category_reasoning = "No specific category patterns matched"
        else:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k]["score"])
            max_score = category_scores[best_category]["score"]
            total_patterns = len(self.CATEGORY_PATTERNS.get(best_category, []))
            category_confidence = min(0.85, 0.5 + (max_score / total_patterns) * 0.5)
            category_reasoning = f"Matched {max_score} patterns: {category_scores[best_category]['matches'][:2]}"
        
        # Step 2: Detect action type
        action_scores = {}
        for action, patterns in self.compiled_actions.items():
            score = 0
            for pattern in patterns:
                if pattern.search(query_lower):
                    score += 1
            
            if score > 0:
                action_scores[action] = score
        
        # Get best action
        if not action_scores:
            best_action = "UNKNOWN"
            action_confidence = 0.3
        else:
            best_action = max(action_scores.keys(), key=lambda k: action_scores[k])
            action_confidence = min(0.9, 0.6 + (action_scores[best_action] * 0.15))
        
        # Combined confidence (weighted average)
        overall_confidence = (category_confidence * 0.7) + (action_confidence * 0.3)
        
        reasoning = f"Category: {best_category} ({category_confidence:.2f}), Action: {best_action} ({action_confidence:.2f})"
        
        result = IntentClassification(
            category=best_category,
            action_type=best_action,
            confidence=overall_confidence,
            reasoning=reasoning,
            method="rule_based"
        )
        
        logger.info(f"[intent] Classified as {result.category}/{result.action_type} (confidence={result.confidence:.2f})")
        
        return result
    
    def classify_with_context(
        self,
        query: str,
        conversation_history: Optional[list] = None,
        agent_assignments: Optional[list] = None
    ) -> IntentClassification:
        """
        Classify with additional context.
        
        Args:
            query: User's query
            conversation_history: Recent messages for context
            agent_assignments: Apps assigned to this agent
            
        Returns:
            IntentClassification
        """
        # Base classification
        result = self.classify(query)
        
        # Boost confidence if agent has matching tools
        if agent_assignments:
            matching_apps = [
                app for app in agent_assignments
                if result.category.lower() in [c.lower() for c in app.get("categories", [])]
            ]
            if matching_apps:
                result.confidence = min(0.95, result.confidence + 0.1)
                result.reasoning += f" (Agent has {len(matching_apps)} matching tools)"
        
        # Consider conversation history (simple keyword persistence)
        if conversation_history:
            recent_text = " ".join([msg.get("content", "") for msg in conversation_history[-3:]])
            if result.category.lower() in recent_text.lower():
                result.confidence = min(0.95, result.confidence + 0.05)
                result.reasoning += " (Context from conversation)"
        
        return result
