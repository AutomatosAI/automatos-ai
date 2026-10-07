"""F379 / F381 (night 11, 7 Oct): a turn about social posts gets how Socials works today, and its templates.

Night 11, in the owner's chats with Auto:

- "rendering happens later, usually after approval" and "I don't have a direct tool to create video
  content" (chats 508b4e05, c28f8e28): a post renders when it is saved, and Socials has four video
  templates (B10, B-I5-1);
- "there's no instagram-carousel template", "Instagram Image": names that don't exist (friction 6);
- asked for "Headline, Closing Title, Point 1 Title": the template's field names, read out to the
  owner (iteration 5);
- "There isn't a direct 'click here to upload photos to a post' button" (02:26), and an agent given
  Dropbox to take the photos (B-I4-1, B-I4-2): the post editor's Look panel takes a photo per spot.

Auto's always-on platform skill still describes the retired automatos-social renderer
(workspace_html_to_png, "the only renderer") and says nothing of the Socials tools. That skill is
generated from the automatos-skills repo, which is where its wording is changed. Until it is, a
turn whose owner talks about social posts, a channel's post, a carousel, a reel or a video, a
social template by name, or photos for a post or an agent gets this note last: the post tools, the
formats, this workspace's own social template names, that a post renders when it is saved, that a
video is a post too, how a refused render is fixed, how the owner adds a photo, and that Auto says
a post exists only after the call that made it. A social template the owner names gets its
fields, for a post: never ``generate_document``'s note.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

_SOCIAL = re.compile(
    r"\b(?:social(?:\s+media)?|socials|instagram|insta|linkedin|tiktok|facebook|twitter|tweets?|reels?|carousels?|"
    r"(?:x|ig)\s+posts?|stor(?:y|ies)\s+posts?)\b", re.I)
_A_POST = re.compile(r"\b(?:a|an|the|my|our|this|that|these|those|\d+|two|three|four|five)\s+(?:[\w'-]+\s+){0,2}?"
                     r"(?<!blog )posts?\b(?![\s-]*(?:office|code|box|it)\b)", re.I)
_A_VIDEO = re.compile(r"\b(?:make|create|draft|do|film|put together|render)\b[^.?!\n]{0,40}\bvideos?\b"
                      r"|\b\d+[- ]?(?:second|sec|s)\s+videos?\b", re.I)
_PHOTOS = re.compile(r"\b(?:photos?|pictures?|pics?|images?)\b", re.I)
_FOR_A_POST = re.compile(r"\b(?:posts?|agents?|cards?|socials?|director)\b"
                         r"|\b(?:add|upload|give|attach|put|send|share|drop)\b", re.I)
# Photos for somewhere else: a shop's products, the website, the documents.
_ELSEWHERE = re.compile(r"\b(?:shopify|products?|website|knowledge|documents?|invoices?)\b", re.I)

SOCIALS_NOTE = (
    "Socials, as it works today. It replaces the automatos-social renderer and the workspace_html_to_png "
    "pipeline in your platform skill, which are retired: never use them for a post. A post is made with "
    "platform_create_social_post and changed with platform_update_social_post; platform_list_social_posts "
    "lists them with what each still needs. Its format is video, image, carousel, fact_card, infographic or "
    "text; social_image and social_video are the formats of templates, never of a post. Choose its template "
    "from this workspace's own, by its exact name: {names}. A video is a post too: format video with a video "
    "template, and you make it. Fill the template's own fields by their exact names (platform_get_template_"
    "schema lists them: bare names, no \"data.\"), with the owner's facts only; a fact with no field goes in "
    "the copy. Ask the owner only for facts they haven't given, in plain words, never by field names. A post "
    "with a template renders as soon as it is saved, then waits for the owner's approval in the Socials tab: "
    "rendering never waits for approval. When the call fails because the render was refused, the post is a "
    "draft: fill the fields it names with platform_update_social_post and render true, and never make it "
    "again. Say a post was made, changed or waits in the Socials tab only after the call that did it "
    "succeeded in this turn, with its id. Photos: the owner adds their own in the Socials tab: open the post, "
    "and in its Look panel choose Upload (or Library, for a picture already in Deliverables), pick the spot "
    "under \"Where your picture goes\" (one photo per spot, such as Before photo and After photo), then drop "
    "the file or press Browse files, and render the post again. It needs no other app: never use Dropbox, "
    "Google Drive or another tool for it, and never change an agent's tools or settings to answer how."
)
NAMED = (" The owner named the social template \"{name}\" (template_id {id}): make the post on it, "
         "platform_create_social_post with template \"{name}\"; its fields (* needs a value): {fields}.")


def about_socials(texts: Sequence[str]) -> bool:
    """Whether the owner's latest message (``texts[0]``) is about social posts: a channel or a social
    word, a post, a video to make, or photos to add or give (not a shop's, the website's or a
    document's), or photos in a chat about Socials."""
    latest = str(texts[0] if texts else "")
    if _SOCIAL.search(latest) or _A_POST.search(latest) or _A_VIDEO.search(latest):
        return True
    if not _PHOTOS.search(latest) or _ELSEWHERE.search(latest):
        return False
    return bool(_FOR_A_POST.search(latest)) or any(_SOCIAL.search(str(text)) for text in texts[1:])


def _named_social(texts: Sequence[str], socials: Sequence[Any]) -> Optional[Any]:
    """The social template the owner names in these turns, if any (``named_template``'s matching)."""
    from consumers.chatbot.named_template import named_in_conversation

    named = named_in_conversation(texts, socials) if socials else None
    return getattr(named, "row", None) if named is not None else None


def socials_note(texts: Sequence[str], rows: Sequence[Any]) -> Optional[str]:
    """The Socials note for these owner turns (latest first), or None when they are not about social
    posts and name no social template. A turn given to the Social Media Director gets what its ticket
    must say instead. ``rows``: the workspace's templates, every format."""
    from consumers.chatbot.socials_assign_lane import ticket_note
    from modules.tools.discovery.social_post_checks import field_list, social_rows, template_names

    ticket = ticket_note()
    if ticket is not None:   # the Director's ticket: what goes on it, never how to make the post here
        return ticket
    socials = social_rows(rows)
    named = _named_social(texts, socials)
    if named is None and not about_socials(texts):
        return None
    note = SOCIALS_NOTE.format(names=template_names(socials))
    if named is not None:
        note += NAMED.format(name=named.name, id=named.id, fields=field_list(named))
    return note


__all__ = ["NAMED", "SOCIALS_NOTE", "about_socials", "socials_note"]
