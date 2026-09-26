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
        promoted=True,
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
        promoted=True,
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
        promoted=True,
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
        promoted=True,
        tags=["blog", "write", "create", "mission", "content"],
        examples=[
            "create a blog post about multi-agent AI",
            "write a new blog on Shopify automation",
            "create blog post topic: AI agents for SaaS",
            "start a blog mission about LLM observability",
        ],
        accepts=("config",),
    ))

    # PRD-251 US-117: the one image tool. Without a post_id it makes a still in
    # the aspect ratio asked for and registers it as an image Deliverable.
    # The aspect enum repeats handlers_blog.IMAGE_ASPECT_RATIOS as literals (this
    # registry stays stdlib-light for the utterance linter; a test pins them).
    registry.register(ActionDefinition(
        name="platform_generate_cover_image",
        description=(
            "Generate an image with the configured image model (BLOG_COVER_MODEL, "
            "default Gemini Nano Banana Pro, overridable per-deployment) and save it "
            "to the platform image store. With a post_id: a 16:9 cover for that blog "
            "post, set as its cover_image_url — use this after a draft has been "
            "created via platform_publish_blog_post. Without a post_id: a still for a "
            "social post, a carousel or a slide, in the aspect ratio you ask for, "
            "saved to Deliverables; its deliverable_id can go into a social post's "
            "media under that ratio. The image carries no words: text belongs to "
            "the template or the post."
        ),
        category="blog",
        parameters={
            "type": "object",
            "properties": {
                "post_id": {
                    "type": "string",
                    "description": (
                        "UUID of the blog post to generate a cover for. Leave it out "
                        "for an image that is not a blog cover."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Image direction — describe the visual concept. Will be "
                        "wrapped with framing instructions (the aspect ratio, no "
                        "embedded text) before being sent to the image model."
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["16:9", "1:1", "4:5", "9:16"],
                    "description": (
                        "The image's shape without a post_id (default 16:9): 9:16 "
                        "for stories and reels, 4:5 or 1:1 for feed posts, 16:9 for "
                        "links and slides. A blog cover is always 16:9."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "What the image is called in Deliverables when there is no "
                        "post_id (default: the start of the prompt)."
                    ),
                },
            },
            "required": ["prompt"],
        },
        permission_level="write",
        requires_confirmation=False,
        promoted=True,
        tags=["blog", "image", "cover", "design", "content", "social", "still"],
        examples=[
            "generate a cover image for the latest draft",
            "create cover art for post abc123",
            "make a blog cover image",
            "add a cover image to my blog post",
            "generate a 4:5 image for an instagram post",
            "make a 9:16 still for our story",
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
        promoted=True,
        tags=["blog", "write", "update", "edit", "content"],
        examples=[
            "update blog post",
            "edit blog post",
            "revise article",
            "improve draft",
            "set cover image on blog post",
        ],
    ))
