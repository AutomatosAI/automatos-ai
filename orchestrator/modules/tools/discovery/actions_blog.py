"""Blog ActionDefinitions (publish, list, get, update posts)."""

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
                    "description": "If true, post is published immediately. If false (default), saved as draft.",
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

    registry.register(ActionDefinition(
        name="platform_get_blog_post",
        description=(
            "Read the full content of a blog post by ID or slug. Returns the "
            "complete markdown content, metadata, and status. Use this to review "
            "a draft before editing or to read existing published content."
        ),
        category="blog",
        parameters={
            "type": "object",
            "properties": {
                "post_id": {
                    "type": "string",
                    "description": "UUID of the blog post.",
                },
                "slug": {
                    "type": "string",
                    "description": "URL slug of the blog post.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["blog", "read", "content"],
        examples=[
            "read blog post",
            "get blog post content",
            "show blog draft",
            "fetch article",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_create_blog_post",
        description=(
            "Create a complete blog post from a topic. Builds a standardized "
            "research-write-publish-cover mission and dispatches it to the "
            "coordinator. Same mission fires whether triggered by a UI button, "
            "a scheduled playbook, or an agent suggesting a topic. Returns "
            "mission_id for progress tracking. Use this whenever you have a "
            "topic and want a complete blog post produced end-to-end."
        ),
        category="blog",
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "Specific topic for the blog post (e.g. 'Multi-agent AI for "
                        "Shopify stores'). Be concrete — not just a category."
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Broad content category (e.g. 'AI & Automation', 'Engineering', "
                        "'Research'). Defaults to 'AI & Automation'."
                    ),
                },
            },
            "required": ["topic"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["blog", "write", "create", "mission", "content"],
        examples=[
            "create a blog post about multi-agent AI",
            "write a new blog on Shopify automation",
            "create blog post topic: AI agents for SaaS",
            "start a blog mission about LLM observability",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_generate_cover_image",
        description=(
            "Generate and attach a cover image to an existing blog post. "
            "Single tool call: builds an image using the configured "
            "BLOG_COVER_MODEL (default Gemini Nano Banana Pro, overridable "
            "per-deployment), saves it to the platform image store, and "
            "updates the post's cover_image_url. Use this after a draft has "
            "been created via platform_publish_blog_post — the resulting "
            "post_id is the input here."
        ),
        category="blog",
        parameters={
            "type": "object",
            "properties": {
                "post_id": {
                    "type": "string",
                    "description": "UUID of the blog post to generate a cover for.",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Image direction — describe the visual concept. Will be "
                        "wrapped with framing instructions (16:9, abstract, no "
                        "embedded text) before being sent to the image model."
                    ),
                },
            },
            "required": ["post_id", "prompt"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["blog", "image", "cover", "design", "content"],
        examples=[
            "generate a cover image for the latest draft",
            "create cover art for post abc123",
            "make a blog cover image",
            "add a cover image to my blog post",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_blog_post",
        description=(
            "Update an existing blog post. Only updates the fields you provide — "
            "omitted fields are left unchanged. Use this to improve drafts, fix "
            "content, update tags/category, or set a cover image URL. "
            "Content should be in markdown format."
        ),
        category="blog",
        parameters={
            "type": "object",
            "properties": {
                "post_id": {
                    "type": "string",
                    "description": "UUID of the blog post to update.",
                },
                "title": {
                    "type": "string",
                    "description": "New title for the post.",
                },
                "content": {
                    "type": "string",
                    "description": "Updated blog post content in markdown format.",
                },
                "excerpt": {
                    "type": "string",
                    "description": "Updated excerpt/summary (max 300 chars).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Updated tags.",
                },
                "category": {
                    "type": "string",
                    "description": "Updated category.",
                },
                "cover_image_url": {
                    "type": "string",
                    "description": "URL to a cover image for the post.",
                },
                "seo_title": {
                    "type": "string",
                    "description": "SEO-optimised title for search engines (max 60 chars).",
                },
                "seo_description": {
                    "type": "string",
                    "description": "SEO meta description for search results (max 160 chars).",
                },
            },
            "required": ["post_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["blog", "write", "update", "edit", "content"],
        examples=[
            "update blog post",
            "edit blog post",
            "revise article",
            "improve draft",
            "set cover image on blog post",
        ],
    ))
