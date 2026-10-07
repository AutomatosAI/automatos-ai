"""F379 (night 11, 7 Oct): a post Auto drafts is checked the way its render will check it, before it is saved.

Night 11, Auto's Socials path, each a second call or a post that never rendered:

- both of Auto's post creations first failed with ``format: social_image`` (04da700e, 4a842a85):
  the Director's persona and the template field speak of social_image and social_video, the
  TEMPLATE formats, and the model sent one as the post's own (B-I5-3);
- "there's no instagram-carousel template" (5347dc49): the template is called "Carousel", and
  a name was looked up exactly, case and all;
- carousel 39ee6ae3 was saved with ``slide1_heading … slide5_body``, ``sale_date`` and ``price``,
  none of them the Carousel's fields: the save took them, and the render answered 422 (B-i3-4);
- three posts were saved with no template at all (668d7948, 6ba38a0b, 7f2f7d51) and never
  rendered (B-I5-2, B-i3-6).

So, before anything is saved: a template format sent as the post's format is taken as the post
format it names (social_image → image, social_video → video), and any other unknown format is
refused with the post formats; a template is found by its name ignoring case, spacing and a
trailing "template", or by the one social template name the words hold, and a miss lists the
workspace's social templates; a field the template doesn't have is refused with the template's
own fields (name, label, and which need a value); and a post to render with no template is
refused with the templates to choose from. A render that names empty fields is told in their
labels (``missing_in_words``).

Stdlib at import: the template service and the model constants are read when a call needs them.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# A template's format, sent as a post's: the one post format it can mean.
POST_FORMAT_ALIASES = {"social_image": "image", "social_video": "video"}
TEXT_POST, VIDEO_POST = "text", "video"
IMAGE_TEMPLATE, VIDEO_TEMPLATE = "social_image", "social_video"
TEMPLATE_KINDS = ((IMAGE_TEMPLATE, "image"), (VIDEO_TEMPLATE, "video"))
# How many of a template's fields a refusal lists; the rest are counted.
FIELDS_LISTED = 40
NEEDS_A_VALUE = "*"
_TEMPLATE_WORD = re.compile(r"\s*\btemplates?\b\s*$", re.I)
_QUOTES = "\"'“”‘’`"
_MISSING = re.compile(r"^fill in (?P<names>.+) before rendering$")

NO_SUCH_TEMPLATE = "No template {ref!r} in this workspace: platform_list_templates lists them (format social_video or social_image)."
ITS_TEMPLATES = " This workspace's social templates: {names}."
NOT_A_FORMAT = ("format {value!r} is not a post format: a post is one of {formats}. social_image and social_video "
                "are the formats of templates (platform_list_templates), not of posts. Nothing was saved.")
UNKNOWN_FIELDS = ("The {template} template has no field {unknown}. Nothing was saved. Its fields, by these exact "
                  "names (* needs a value): {fields}. Put each of the owner's facts in the field it fits; a fact "
                  "with no field goes in the post's copy, never in a field the template doesn't have.")
NEEDS_A_TEMPLATE = ("A {format} post renders on a template, and none was named. Nothing was saved. Choose one of "
                    "this workspace's {kind} templates: {names} (platform_get_template_schema gives its fields), "
                    "or send render false to keep a draft without one, or format text for a post of copy alone.")
NO_TEMPLATES = "none yet"
MORE_FIELDS = ", and {count} more (platform_get_template_schema lists them)"


def post_format(value: Any) -> Tuple[Any, Optional[str]]:
    """The post format ``value`` names, and None; or ``value`` and why it is refused.
    None stays None (the post takes no format)."""
    from core.models.socials import SOCIAL_POST_FORMATS

    if value is None:
        return None, None
    said = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    said = POST_FORMAT_ALIASES.get(said, said)
    if said in SOCIAL_POST_FORMATS:
        return said, None
    return value, NOT_A_FORMAT.format(value=value, formats=", ".join(SOCIAL_POST_FORMATS))


def _latest_by_name(rows: Iterable[Any]) -> List[Any]:
    """One row per name: the service lists a name's versions newest first."""
    seen: Dict[str, Any] = {}
    for row in rows:
        seen.setdefault(str(row.name or ""), row)
    return list(seen.values())


def social_rows(rows: Iterable[Any]) -> List[Any]:
    """The social templates among ``rows`` (as the service lists them), one per name, image ones first."""
    from core.social_templates import is_social_format

    socials = [row for row in rows if is_social_format(getattr(row, "format", None))]
    return sorted(_latest_by_name(socials), key=lambda row: (row.format != IMAGE_TEMPLATE, str(row.name or "")))


def social_templates(db: Any, workspace_id: Any) -> List[Any]:
    """The workspace's social templates, one per name (the latest version), image ones first."""
    from modules.documents.template_service import DocumentTemplateService

    return social_rows(DocumentTemplateService(db).list_templates(workspace_id))


def template_names(rows: Sequence[Any], fmt: Optional[str] = None) -> str:
    """'image: Carousel, Quote card; video: UI story promo', or one kind's names when ``fmt`` is given."""
    if fmt is not None:
        names = [row.name for row in rows if row.format == fmt]
        return ", ".join(names) if names else NO_TEMPLATES
    kinds = [f"{word}: {', '.join(row.name for row in rows if row.format == fmt_)}"
             for fmt_, word in TEMPLATE_KINDS if any(row.format == fmt_ for row in rows)]
    return "; ".join(kinds) if kinds else NO_TEMPLATES


def _plain_name(text: Any) -> str:
    return " ".join(str(text or "").strip().strip(_QUOTES).split()).casefold()


def _named_inside(said: str, rows: Sequence[Any]) -> Optional[Any]:
    """The one social template whose whole name the words hold ("instagram carousel" → Carousel)."""
    hits = [row for row in rows if re.search(r"(?<!\w)" + re.escape(_plain_name(row.name)) + r"(?!\w)", said)]
    return hits[0] if len(hits) == 1 else None


def template_by_name(db: Any, workspace_id: Any, ref: str) -> Tuple[Optional[Any], str]:
    """The template ``ref`` names, and ""; or None and the refusal, listing the social templates.
    Its exact name first (any format, so a document template is refused as one), then ignoring
    case and spacing, then without a trailing "template", then the one social name it holds."""
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    exact = service.get_template_by_name(workspace_id, ref)
    if exact is not None:
        return exact, ""
    said = _plain_name(ref)
    every = _latest_by_name(service.list_templates(workspace_id))
    same = next((row for row in every if _plain_name(row.name) == said), None)
    if same is not None:
        return same, ""
    socials = social_templates(db, workspace_id)
    bare = _TEMPLATE_WORD.sub("", said)
    found = next((row for row in socials if _plain_name(row.name) == bare), None) or _named_inside(bare, socials)
    if found is not None:
        return found, ""
    return None, NO_SUCH_TEMPLATE.format(ref=ref) + ITS_TEMPLATES.format(names=template_names(socials))


def _schema(template: Any) -> Mapping[str, Any]:
    blocks = getattr(template, "blocks", None)
    schema = blocks.get("variables_schema") if isinstance(blocks, dict) else None
    return schema if isinstance(schema, dict) else {}


def _needs_a_value(spec: Any) -> bool:
    return not isinstance(spec, dict) or spec.get("default") is None


def field_list(template: Any, limit: int = FIELDS_LISTED) -> str:
    """The template's fields, those that need a value first: 'headline* (Headline), eyebrow (Eyebrow)'."""
    schema = _schema(template)
    ordered = sorted(schema.items(), key=lambda item: not _needs_a_value(item[1]))
    words = [f"{name}{NEEDS_A_VALUE if _needs_a_value(spec) else ''}"
             + (f" ({spec['label']})" if isinstance(spec, dict) and spec.get("label") else "")
             for name, spec in ordered]
    more = MORE_FIELDS.format(count=len(words) - limit) if len(words) > limit else ""
    return ", ".join(words[:limit]) + more


def unknown_fields(template: Any, variables: Any) -> Optional[str]:
    """Why ``variables`` can't be saved on ``template`` (a field it doesn't have, given a value),
    or None. A field sent as null clears it, so it is never refused."""
    if template is None or not isinstance(variables, dict):
        return None
    schema = _schema(template)
    unknown = [name for name, spec in variables.items() if spec is not None and name not in schema]
    if not unknown:
        return None
    return UNKNOWN_FIELDS.format(template=repr(getattr(template, "name", "")), unknown=", ".join(unknown),
                                 fields=field_list(template))


def needs_a_template(db: Any, workspace_id: Any, fmt: Optional[str]) -> str:
    """The refusal for a post to render with no template: the templates of its kind to choose from."""
    kind = VIDEO_POST if fmt == VIDEO_POST else "image"
    names = template_names(social_templates(db, workspace_id), VIDEO_TEMPLATE if fmt == VIDEO_POST else IMAGE_TEMPLATE)
    return NEEDS_A_TEMPLATE.format(format=fmt or kind, kind=kind, names=names)


def missing_in_words(error: str, template: Any) -> str:
    """A render's "fill in a, b before rendering" said with the fields' labels; any other reason as it is."""
    found = _MISSING.match(str(error or "").strip())
    if not found:
        return str(error or "")
    schema = _schema(template)
    names = [name.strip() for name in found.group("names").split(",") if name.strip()]
    said = [f"{schema[name]['label']} ({name})" if isinstance(schema.get(name), dict) and schema[name].get("label")
            else name for name in names]
    return f"these fields are empty: {', '.join(said)}"


def missing_names(error: str) -> List[str]:
    """The field names a render's "fill in … before rendering" names; [] for any other reason."""
    found = _MISSING.match(str(error or "").strip())
    return [name.strip() for name in found.group("names").split(",") if name.strip()] if found else []


__all__ = ["IMAGE_TEMPLATE", "NO_SUCH_TEMPLATE", "POST_FORMAT_ALIASES", "TEXT_POST", "VIDEO_POST", "VIDEO_TEMPLATE",
           "field_list", "missing_in_words", "missing_names", "needs_a_template",
           "post_format", "social_rows", "social_templates", "template_by_name", "template_names", "unknown_fields"]
