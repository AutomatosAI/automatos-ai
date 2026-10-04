"""The tool that changes the workspace brand kit (night 9b: the voice says who signs).

It left actions_documents.py, whose one register function is past the length rule,
when the kit's voice gained ``sign_off`` (services.brand_rules fills a placeholder
signature with it): the schema lists every BrandVoice field, so an agent sees it.
"""

import copy

from .action_registry import ActionDefinition, ActionRegistry

_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The brand's name."},
        "tagline": {"type": "string", "description": "The brand's tagline."},
        "primary_color": {"type": "string", "description": "Hex colour, such as #1a1a2e or #abc."},
        "secondary_color": {"type": "string", "description": "Hex colour."},
        "accent_color": {"type": "string", "description": "Hex colour."},
        "text_color": {"type": "string", "description": "Hex colour of body text."},
        "font_family": {
            "type": "string",
            "description": "The body font as a CSS font stack, such as Inter, sans-serif.",
        },
        "heading_font": {
            "type": "string",
            "description": "The headings' font stack, such as \"Brand Display\", serif. Empty: the body font.",
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
                    "items": {"type": "string"},
                    "description": "3 to 5 tone words, such as warm, plain, confident. An empty list clears them.",
                },
                "banned_phrases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Phrases the brand never uses (up to 50).",
                },
                "sign_off": {
                    "type": "string",
                    "description": ("Who signs the brand's letters, emails and documents, one line (up to 120 "
                                    "characters), such as the owner's name and the business's. An empty string "
                                    "clears it."),
                },
            },
        },
    },
    "required": [],
}


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
            "is saved. The logo, logo mark and font files are uploaded by a person in "
            "the brand kit settings; this tool sets text, colours, fonts and http(s) "
            "logo URLs only."
        ),
        category="documents",
        parameters=copy.deepcopy(_PARAMETERS),
        permission_level="write",
        requires_confirmation=False,
        admin_only=True,  # F151: REST PUT /brand-kit is workspace:manage
        tags=["documents", "brand", "brand kit", "colours", "fonts", "voice", "socials", "setup"],
        examples=[
            "set our primary brand colour to #0055aa",
            "our tone of voice is warm, plain and confident",
            "add our instagram handle to the brand kit",
        ],
    ))
