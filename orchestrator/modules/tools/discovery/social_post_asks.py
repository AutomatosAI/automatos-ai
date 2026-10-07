"""F380 (night 11, 7 Oct): which tickets ask for a Socials post, and which answers say one was made.

Night 11's Social Media Director tickets were "September numbers card for Instagram",
"Quote card: Rosa at Lantern Kitchen", "15-second video: September top three cafés",
"Wholesale offer post for LinkedIn" and "Before and after: the Guji re-roast" (an
Instagram before/after post). A brief asks for a post when it names a channel (or
Socials) near the thing to make (a post, card, carousel, reel, story, video), or names
one of the Socials templates' kinds (a quote, stats, offer or fact card, a carousel).

It stays narrow, because a ticket about posts is not always a ticket for one:
a title that opens on reading or planning work ("Summarize last week's Instagram
posts", "Plan next week's posts", "Upload the Guji photos", "Align Socials visuals
with the documents") and a brief that says the post is not wanted ("caption only",
"storyboard", "don't make a post") or asks for samples on the brand board are not.

An answer says a post was made when one of its sentences says something was created,
drafted, saved, submitted or is waiting, and names the Socials tab or a channel's post
(#2160: "This draft is now waiting for your approval in the Socials tab.").

Pure: no database.
"""
from __future__ import annotations

import re
from typing import Iterable

_CHANNEL = r"(?:instagram|insta|linkedin|twitter|tiktok|facebook|youtube|socials?|social media)"
_THING = r"(?:posts?|cards?|carousels?|reels?|stor(?:y|ies)|videos?|tweets?|graphics?|infographics?|shorts)"
# How far apart, in characters within one sentence, the channel and the thing may sit.
NEAR_CHARS = 60
_CHANNEL_AND_THING = re.compile(
    rf"\b{_CHANNEL}\b[^.\n]{{0,{NEAR_CHARS}}}?\b{_THING}\b|\b{_THING}\b[^.\n]{{0,{NEAR_CHARS}}}?\b{_CHANNEL}\b",
    re.IGNORECASE)
_TEMPLATE_KIND = re.compile(
    r"\b(?:quote|stats?|numbers|offer|fact|review|announcement|before[- ]and[- ]after|before/after)\s+"
    r"(?:cards?|posts?)\b|\bcarousels?\b|\b\d{1,3}[- ]second (?:video|reel)\b", re.IGNORECASE)
_NOT_A_POST = re.compile(
    r"\b(?:captions? only|text only|no post|storyboard|brand board|samples?|mock-?ups?)\b"
    r"|\bdon't (?:make|create|draft|render) (?:a |the |any )?posts?\b", re.IGNORECASE)
_READS_OR_PLANS = re.compile(
    r"^\s*(?:summari[sz]e|analy[sz]e|review(?!\s+cards?\b)|audit|plan|brainstorm|report|research|list|find|check"
    r"|count|upload|schedule|compare|align|match|restyle|redesign)\b", re.IGNORECASE)

_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_MADE = re.compile(r"\b(?:created|drafted|saved|submitted|made|ready|waiting|awaiting|sent for)\b", re.IGNORECASE)
_SOCIALS_TAB = re.compile(r"\bSocials\b")  # the product's own word, capitalised
# Where a generate_document file is served: a Deliverable, not a Socials post (#2160's link).
_GENERATED_FILE = re.compile(r"generated-documents/|/documents/generated/")
DOCUMENT_ACTION = "generate_document"


def mentions_social_posts(text: object) -> bool:
    """The text speaks of making social posts: a channel near a post, or a Socials template's kind."""
    words = str(text or "")
    return bool(_CHANNEL_AND_THING.search(words) or _TEMPLATE_KIND.search(words))


def asks_for_a_social_post(title: object, description: object) -> bool:
    """The ticket's own brief asks for a Socials post to be made (see the module note)."""
    head = str(title or "")
    brief = f"{head}\n{description or ''}"
    if _READS_OR_PLANS.match(head) or _NOT_A_POST.search(brief):
        return False
    return mentions_social_posts(brief)


def claims_a_social_post(answer: object) -> bool:
    """One sentence of the answer says a post was made or is waiting, naming Socials or a channel's post."""
    for sentence in _SENTENCE.findall(str(answer or "").replace("\u2019", "'")):
        if _MADE.search(sentence) and (_SOCIALS_TAB.search(sentence) or mentions_social_posts(sentence)):
            return True
    return False


def made_a_document_image(answer: object, ran: Iterable[str]) -> bool:
    """The run made a generate_document file (its action ran, or the answer links one)."""
    return DOCUMENT_ACTION in set(ran or ()) or bool(_GENERATED_FILE.search(str(answer or "")))


__all__ = ["asks_for_a_social_post", "claims_a_social_post", "made_a_document_image", "mentions_social_posts"]
