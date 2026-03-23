"""
Tone detection and adaptive response styling.

Reads the user's emotional signals from their message and returns
a style directive that gets injected into the system prompt so
the agent adapts *how* it responds, not just *what* it responds.

This is what makes Automatos agents feel human — they read the room.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ToneSignal:
    """Detected tone from a user message."""
    tone: str            # frustrated, urgent, casual, formal, curious, playful
    confidence: float    # 0.0–1.0
    signals: tuple       # what triggered the detection
    style: str           # instruction injected into system prompt


# ── Signal patterns ─────────────────────────────────────────────
# Each pattern: (compiled regex, tone label, confidence boost, signal name)

_PATTERNS = [
    # Frustration — swearing, caps, repeated punctuation, dismissals
    (re.compile(r'\b(fuck|shit|damn|wtf|ffs|crap|bloody|jesus christ)\b', re.I), 'frustrated', 0.4, 'profanity'),
    (re.compile(r'[A-Z]{4,}'), 'frustrated', 0.25, 'shouting'),
    (re.compile(r'[!?]{2,}'), 'frustrated', 0.2, 'repeated_punctuation'),
    (re.compile(r'\b(bla|blah|whatever|useless|worthless|broken)\b', re.I), 'frustrated', 0.3, 'dismissal'),
    (re.compile(r'\b(doesnt work|not working|still broken|cant|can\'t)\b', re.I), 'frustrated', 0.2, 'broken_report'),

    # Urgency — time pressure, imperatives
    (re.compile(r'\b(asap|urgent|now|immediately|hurry|deadline|by friday|by tomorrow)\b', re.I), 'urgent', 0.35, 'time_pressure'),
    (re.compile(r'\b(just do it|just fix|stop talking|get on with)\b', re.I), 'urgent', 0.3, 'imperative'),

    # Curiosity — questions, exploration
    (re.compile(r'\b(why|how|what if|explain|curious|wondering|tell me about)\b', re.I), 'curious', 0.2, 'question'),
    (re.compile(r'\b(could we|would it|is it possible|what about)\b', re.I), 'curious', 0.2, 'exploration'),

    # Casual — informal, short, relaxed
    (re.compile(r'\b(lol|haha|cool|nice|cheers|ta|yeah|nah|yep|nope|mate)\b', re.I), 'casual', 0.3, 'informal_language'),
    (re.compile(r'^.{1,15}$'), 'casual', 0.15, 'short_message'),

    # Formal — structured, polite
    (re.compile(r'\b(please|kindly|could you|would you|I would appreciate|regards)\b', re.I), 'formal', 0.25, 'polite_language'),
    (re.compile(r'\b(requirements?|deliverables?|stakeholders?|pursuant)\b', re.I), 'formal', 0.3, 'business_language'),

    # Playful — jokes, emojis, banter
    (re.compile(r'[\U0001F600-\U0001F64F]'), 'playful', 0.3, 'emoji'),
    (re.compile(r'\b(heh|lmao|rofl|joke|kidding|banter)\b', re.I), 'playful', 0.25, 'humor'),
]

# ── Style directives per tone ───────────────────────────────────
# These get injected into the system prompt to shape the response.

_STYLES = {
    'frustrated': (
        "## How to respond right now\n"
        "The user is frustrated. Be direct and short. No filler, no preamble, "
        "no apologies. Lead with the fix or the answer. Don't explain what "
        "you're about to do — just do it. Don't ask clarifying questions "
        "unless absolutely necessary. Match their energy — be blunt, not gentle."
    ),
    'urgent': (
        "## How to respond right now\n"
        "The user is under time pressure. Skip context and background. "
        "Give the answer or action first, details after. Use short sentences. "
        "If there are options, pick the best one and do it — don't present a menu."
    ),
    'curious': (
        "## How to respond right now\n"
        "The user is exploring and wants to understand. Explain the 'why' not "
        "just the 'what'. Use examples. It's OK to go deeper — they're asking "
        "because they want to learn, not because they need a quick fix."
    ),
    'casual': (
        "## How to respond right now\n"
        "Keep it relaxed and conversational. Short responses. "
        "Match their tone — if they're brief, be brief. No corporate speak."
    ),
    'formal': (
        "## How to respond right now\n"
        "The user is being professional. Match their register. Use clear "
        "structure, complete sentences, and proper formatting. "
        "Be thorough but not verbose."
    ),
    'playful': (
        "## How to respond right now\n"
        "The user is in a good mood. It's OK to be warm and have personality. "
        "Keep the energy up but stay useful — don't sacrifice substance for style."
    ),
}

# Fallback when no strong signal is detected
_DEFAULT_STYLE = ""


def detect_tone(message: str, history: Optional[List[str]] = None) -> ToneSignal:
    """
    Detect the user's tone from their message.

    Args:
        message: The current user message.
        history: Optional list of recent user messages (newest last)
                 to detect escalation patterns.

    Returns:
        ToneSignal with the dominant tone and a style directive.
    """
    scores: dict[str, float] = {}
    signals: dict[str, list[str]] = {}

    for pattern, tone, boost, signal_name in _PATTERNS:
        if pattern.search(message):
            scores[tone] = scores.get(tone, 0.0) + boost
            signals.setdefault(tone, []).append(signal_name)

    # Escalation detection — if recent history also shows frustration,
    # boost the score. This catches the "asked 3 times" pattern.
    if history and len(history) >= 2:
        prior_frustration = 0
        for prev_msg in history[-3:]:
            for pattern, tone, boost, _ in _PATTERNS:
                if tone == 'frustrated' and pattern.search(prev_msg):
                    prior_frustration += boost
        if prior_frustration > 0.3:
            scores['frustrated'] = scores.get('frustrated', 0.0) + 0.2
            signals.setdefault('frustrated', []).append('escalation')

    if not scores:
        return ToneSignal(
            tone='neutral',
            confidence=0.0,
            signals=(),
            style=_DEFAULT_STYLE,
        )

    # Pick the dominant tone
    dominant = max(scores, key=scores.get)
    confidence = min(scores[dominant], 1.0)

    return ToneSignal(
        tone=dominant,
        confidence=confidence,
        signals=tuple(signals.get(dominant, [])),
        style=_STYLES.get(dominant, _DEFAULT_STYLE),
    )


def get_tone_directive(message: str, history: Optional[List[str]] = None,
                       threshold: float = 0.3) -> str:
    """
    Convenience function: returns the style directive string to inject
    into the system prompt, or empty string if confidence is below threshold.

    Usage in the chat pipeline:
        tone_block = get_tone_directive(user_message, recent_history)
        system_prompt = f"{base_prompt}\n{memory_block}\n{tone_block}"
    """
    signal = detect_tone(message, history)
    if signal.confidence >= threshold:
        return signal.style
    return _DEFAULT_STYLE
