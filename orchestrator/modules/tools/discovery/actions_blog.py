"""Blog ActionDefinitions (publish post, list posts)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_blog_actions(registry: ActionRegistry) -> None:
    """Register blog-related platform actions."""

    registry.register(ActionDefinition(
        name="platform_publish_blog_post",
        description=(
            "Write and publish a blog post to the workspace blog. Content should be "
            "in markdown format. The post will be visible on any website using the "
            "Automatos blog widget. Use this after completing research or writing tasks "
            "to share findings publicly."
        ),
        category="blog",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the blog post.",
                },
                "content": {
                    "type": "string",
                    "description": "Full blog post content in markdown format.",
                },
                "excerpt": {
                    "type": "string",
                    "description": "Short excerpt/summary (max 300 chars). Auto-generated from content if omitted.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorisation (e.g. ['ai', 'research', 'automation']).",
                },
                "category": {
                    "type": "string",
                    "description": "Post category (e.g. 'Research', 'Engineering', 'News').",
                },
                "cover_image_url": {
                    "type": "string",
                    "description": "URL to a cover image for the post.",
                },
                "publish_immediately": {
                    "type": "boolean",
                    "description": "If true (default), post is published immediately. If false, saved as draft.",
                },
            },
            "required": ["title", "content"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["blog", "write", "publish", "content"],
        examples=[
            "publish a blog post about our findings",
            "write an article about AI automation",
            "create a blog post summarising the research",
            "publish article about market trends",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_blog_posts",
        description=(
            "List existing blog posts in the workspace. Returns titles, slugs, "
            "statuses, and publish dates. Use to check what content has already "
            "been published before writing new posts."
        ),
        category="blog",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["draft", "published", "archived"],
                    "description": "Filter by status. Defaults to 'published'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of posts to return. Defaults to 10.",
                },
                "category": {
                    "type": "string",
                    "description": "Filter by category.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["blog", "read", "list", "content"],
        examples=[
            "list my blog posts",
            "what blog posts have been published?",
            "show me draft blog posts",
            "check what articles exist",
        ],
    ))
