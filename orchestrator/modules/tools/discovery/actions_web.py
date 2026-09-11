"""Web ActionDefinitions (PRD-240) — every agent can read and search the web.

A platform capability, not a marketplace app: nothing to connect, nothing to
assign. ``platform_web_fetch`` needs no key at all. ``platform_web_search`` answers through the
first search engine the deployment has — the OpenRouter key, the Composio key,
or a SearXNG container — and says plainly which options exist when it has none.
Schema truth from birth: each ``required`` names exactly the param its handler
hard-fails without.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_web_actions(registry: ActionRegistry) -> None:
    """Register platform_web_fetch + platform_web_search (PRD-240 S2/S3)."""

    registry.register(ActionDefinition(
        name="platform_web_fetch",
        description=(
            "Read a public web page or document by URL and get its text back "
            "(title, headings, paragraphs, links) — no key needed. Use it to open "
            "a page you already know, follow a link from a search result, or read "
            "documentation, articles and READMEs. Private and internal addresses "
            "are refused. Returns {url, final_url, title, content, truncated}; when "
            "web access is switched off on this server it returns "
            "{available:false, reason} — say so, never retry."
        ),
        category="web",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full http(s) URL to read.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Cap on returned text (default 20000, max 60000).",
                },
            },
            "required": ["url"],
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["web", "research", "browse", "fetch", "read", "url"],
        examples=[
            "read https://docs.python.org/3/whatsnew/3.13.html",
            "open the page the search found and summarise it",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_web_search",
        description=(
            "Search the web for pages about a topic and get back "
            "[{title, url, snippet}]. Use it for anything current, changing or "
            "outside your training data — then platform_web_fetch the results worth "
            "reading. Answers through whichever search engine this server has "
            "(OpenRouter, Composio or SearXNG); when it has none it returns "
            "{available:false, reason, options} naming how to enable one — tell "
            "the user, never retry."
        ),
        category="web",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, in plain language.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Results to return (default 5, max 10).",
                },
            },
            "required": ["query"],
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["web", "research", "search", "find", "news", "current"],
        examples=[
            "research the current state of EU AI Act enforcement",
            "find the latest release notes for fastapi",
        ],
    ))
