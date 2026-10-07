"""The tools that read and change the workspace brand kit (night 9b: the voice says who signs).

PRD-251 US-115: both call the functions the REST routes call
(modules/documents/brand_kit.py). The logo, its variants, the logo mark and the font
FILES are uploaded by a person; neither tool uploads anything.

It left actions_documents.py, whose one register function is past the length rule,
when the kit's voice gained ``sign_off`` (services.brand_rules fills a placeholder
signature with it): the schema lists every BrandVoice field, so an agent sees it.

PRD-255 (US-008): the schema is every field of the v2 kit an agent may set (the stored
logo files stay uploads), and says in plain words what each means, so "less orange"
reaches ``accent_use`` / ``palette.accent`` and "more space" ``spacing_unit_pt``.
"""

import copy

from core.brand_palette import ROLE_DERIVED, ROLE_SET
from modules.documents.brand_system import ACCENT_USES, DATE_STYLE_FROM_COUNTRY, DATE_STYLES, ROLE_JOBS, TYPE_STEPS

from .action_registry import ActionDefinition, ActionRegistry

_TYPE_STEP = {
    "type": "object",
    "description": "One type step; a field left out keeps its value.",
    "properties": {
        "size_pt": {"type": "number", "description": "Size in points (5 to 96)."},
        "line_pt": {"type": "number", "description": "Line height in points: from size_pt to 3 times it."},
        "weight": {"type": "integer", "description": "100 to 900, in hundreds."},
    },
}

_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The brand's name."},
        "tagline": {"type": "string", "description": "The brand's tagline."},
        "primary_color": {"type": "string", "description": "Hex colour, such as #1a1a2e or #abc."},
        "secondary_color": {"type": "string", "description": "Hex colour."},
        "text_color": {"type": "string", "description": "Hex colour of body text."},
        "font_family": {
            "type": "string",
            "description": (
                "The body font as a CSS font stack ending in a generic family, such as Geist, Inter, "
                "sans-serif. Inter, Geist and Newsreader ship with the platform; any other font prints "
                "only once its woff2 is uploaded. A stack without a generic family keeps the one it replaces."
            ),
        },
        "heading_font": {
            "type": "string",
            "description": (
                "The headings' font stack ending in a generic family, such as Newsreader, Georgia, serif. "
                "Empty: the body font."
            ),
        },
        "logo_url": {
            "type": "string",
            "description": "An http(s) URL of the logo. An uploaded logo is used instead when there is one.",
        },
        "logo_mark_url": {
            "type": "string",
            "description": "An http(s) URL of the square logo mark (the icon beside or instead of the logo).",
        },
        "company": {
            "type": "object",
            "description": "Contact details; merged key by key.",
            "properties": {
                "name": {"type": "string"},
                "address": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "website": {"type": "string"},
            },
        },
        "social_handles": {
            "type": "object",
            "description": (
                "The brand's handle per network, keyed by toolkit name (twitter, "
                "instagram, linkedin, tiktok, youtube or another connected toolkit), "
                "with or without the @, such as {\"twitter\": \"@acme\", \"linkedin\": "
                "\"acme-inc\"}. Replaces the whole map."
            ),
        },
        "voice": {
            "type": "object",
            "description": "How the brand sounds; merged key by key.",
            "properties": {
                "tone": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "description": "The tone word (up to 32 characters)."},
                            "meaning": {
                                "type": "string",
                                "description": "One line on what the word means for this brand (up to 120 characters).",
                            },
                        },
                        "required": ["word"],
                    },
                    "description": (
                        "3 to 5 tone words, each with an optional one-line meaning, such as "
                        "{\"word\": \"warm\", \"meaning\": \"friendly, never gushing\"}. Replaces the "
                        "whole list; an empty list clears them."
                    ),
                },
                "banned_phrases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Phrases the brand never uses (up to 50).",
                },
                "sign_off": {
                    "type": "string",
                    "description": ("Who signs the brand's letters, emails and documents, one line (up to 120 "
                                    "characters), such as the owner's name and the business's. It signs every "
                                    "letter that names no signer of its own (a document's data.signer wins). An "
                                    "empty string clears it."),
                },
            },
        },
        "palette": {
            "type": "object",
            "description": (
                "The colour roles (PRD-255), merged key by key. Each is a hex colour; a role "
                "not set is derived from the four kit colours, and an empty string returns a "
                "set role to derived. A derived role sent at the colour it has now stays "
                "derived. ink, heading and muted must read at 4.5:1 on paper (the page) and "
                "surface_2 (table header fills); accent and accent_2 at 3:1. A role that does "
                "not is refused with its ratio and what it sits on."
            ),
            "properties": {
                role: {"type": "string", "description": f"{job[:1].upper()}{job[1:]}."} for role, job in ROLE_JOBS.items()
            },
        },
        "palette_source": {
            "type": ["object", "string"],
            "description": (
                "Which colour roles are set and which follow the kit's colours, as "
                "platform_get_brand_kit answers it (it may be sent back as it is). A role "
                "\"derived\" goes back to following the kit's colours unless palette changes "
                "its colour; \"set\" pins it at its colour. The string \"derived\" returns "
                "every role to derived."
            ),
            "properties": {role: {"type": "string", "enum": [ROLE_SET, ROLE_DERIVED]} for role in ROLE_JOBS},
        },
        "accent_use": {
            "type": "string",
            "enum": list(ACCENT_USES),
            "description": (
                "How far the accent goes: sparing (the default) keeps it to highlights; bold "
                "also fills table headers with it. \"Less orange\" or \"the orange as an accent "
                "only\" is sparing (and, to change the colour itself, palette.accent)."
            ),
        },
        "type_scale": {
            "type": "object",
            "description": (
                "The type scale (PRD-255), merged step by step and a step field by field: "
                "display, h1, h2, h3, body, small and caption. A step not sent keeps its value."
            ),
            "properties": {step: copy.deepcopy(_TYPE_STEP) for step in TYPE_STEPS},
        },
        "spacing_unit_pt": {
            "type": "number",
            "description": (
                "The spacing grid's unit in points (2 to 12; 4 by default): every gap between "
                "sections, paragraphs and table cells is a multiple of it. \"More space\" is a larger unit."
            ),
        },
        "page_margin_mm": {"type": "number", "description": "The page margin in millimetres (6 to 50; 18 by default)."},
        "logo_rules": {
            "type": "object",
            "description": "How the logo is placed; merged key by key.",
            "properties": {
                "letterhead_mm": {"type": "number", "description": "The letterhead logo's height in mm (6 to 60)."},
                "clear_space": {"type": "number", "description": "Clear space round the logo, in logo heights (0 to 2)."},
                "min_mm": {"type": "number", "description": "The least height the logo prints at, in mm (4 to 40)."},
            },
        },
        "currency": {
            "type": "string",
            "description": (
                "The brand's currency as a three-letter ISO 4217 code, such as GBP. Empty: the "
                "country's currency, and with no country amounts print with no currency."
            ),
        },
        "date_style": {
            "type": "string",
            "enum": [*DATE_STYLES, DATE_STYLE_FROM_COUNTRY],
            "description": (
                "How dates print: d MMMM yyyy (5 October 2026) or MMMM d, yyyy (October 5, 2026). "
                "Empty: the country's style, and with no country d MMMM yyyy."
            ),
        },
        "country": {
            "type": "string",
            "description": (
                "Where the business is, as a two-letter ISO 3166-1 code such as GB, IE or US. "
                "An empty currency or date_style takes the country's (GB: GBP and 5 October 2026; "
                "IE and the other euro countries: EUR; US: USD and October 5, 2026), so set the "
                "country when the owner says where they are or which currency they bill in. "
                "Empty: no country."
            ),
        },
    },
    "required": [],
}


def register_brand_kit_get_action(registry: ActionRegistry) -> None:
    """Register platform_get_brand_kit (PRD-251 US-115; PRD-255 moved it here from
    actions_documents.py, whose one register function is past the length rule)."""
    registry.register(ActionDefinition(
        name="platform_get_brand_kit",
        description=(
            "Read the workspace brand kit: name, tagline, hex colours, body and heading "
            "fonts, logo and logo mark, company contact details, the brand's social handle "
            "per network, and its voice (tone words with their meanings, banned phrases, "
            "who signs). Also its design system: every colour role (palette) with "
            "palette_source saying whether the owner set it or it is derived from the "
            "colours, accent_use, the type scale, the spacing unit, the page margin, the "
            "logo's rules and uploaded variants, the country, the currency and the date style. Branded "
            "documents and Socials posts render with it. Also returns suggestions: values the "
            "workspace already knows (its business profile and name) to fill empty fields "
            "with. Read it before drafting on-brand copy or changing the kit."
        ),
        category="documents",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["documents", "brand", "brand kit", "colours", "fonts", "logo", "voice", "socials"],
        examples=[
            "what's our brand kit?",
            "which colours and fonts does our brand use?",
            "what tone of voice should our posts have?",
        ],
    ))


def register_brand_kit_update_action(registry: ActionRegistry) -> None:
    """Register platform_update_brand_kit."""
    registry.register(ActionDefinition(
        name="platform_update_brand_kit",
        description=(
            "Change fields of the workspace brand kit. Send only the fields to change; "
            "every other field keeps its value. company and voice merge key by key; "
            "social_handles replaces the whole map, so send every handle to keep (a "
            "network left out, or given an empty handle, is removed). The kit is "
            "validated first: an invalid value is refused with the reason and nothing "
            "is saved. palette, type_scale and logo_rules merge key by key too; a colour "
            "role sent empty, or marked \"derived\" in palette_source, goes back to being "
            "derived from the kit's colours; what platform_get_brand_kit answers can be "
            "sent back unchanged. The logo, its dark-background and one-colour versions, the logo mark "
            "and the font files are uploaded by a person in the brand kit settings; this tool sets text, colours, fonts and http(s) "
            "logo URLs only."
        ),
        category="documents",
        parameters=copy.deepcopy(_PARAMETERS),
        permission_level="write",
        requires_confirmation=False,
        admin_only=True,  # F151: REST PUT /brand-kit is workspace:manage
        tags=["documents", "brand", "brand kit", "colours", "fonts", "voice", "socials", "setup",
              "accent", "spacing", "type scale", "currency", "country"],
        examples=[
            "set our primary brand colour to #0055aa",
            "our tone of voice is warm, plain and confident",
            "add our instagram handle to the brand kit",
            "make the orange an accent only",
            "more space between sections",
        ],
    ))
