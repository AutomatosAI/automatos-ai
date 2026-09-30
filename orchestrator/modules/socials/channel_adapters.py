"""PRD-251 D8, S3.2 (US-203): the seeded channel adapters, the channel half of the
registry's data. ``modules/socials/capabilities.py`` reads and checks it; the Wave 3
publishers run the sequences written here.

DATA ONLY. The channel action slugs live in this file and nowhere else in
``modules/socials`` or ``core/composio`` (a test scans for them). The slugs and their
parameters are Composio's (docs.composio.dev/toolkits/{linkedin,twitter,instagram,
tiktok,youtube}, checked 2026-09-25). A seeded adapter needs only its slugs in
``composio_actions_cache``: the bulk sync often leaves an action's ``parameters``
empty, so the documented parameters are carried here.

``CHANNEL_ADAPTERS``, per Composio channel toolkit (its app name, lower case):

- ``label``: the channel's name in the Socials tab.
- ``setup_note``: what connecting it takes beyond the Composio connect flow, or None.
- ``kinds``: per post kind (``social_post_targets.post_kind``), the ordered action
  sequence. Each step has:
  - ``id``: the name later steps refer to it by;
  - ``action``: the Composio action slug;
  - ``class``: ``read`` (an account lookup), ``upload`` (media the platform takes
    before the post: a file or a container), ``status`` (polls an upload or a
    publish) or ``publish`` (makes the post live). With Socials on, the post gate
    refuses an agent's direct call to a connected channel's ``publish`` action (D14);
  - ``params``: the documented parameters, each mapped from a source (below);
  - ``files``: the params that take a FILE. Publishing is file-first: the publisher
    passes the file through the executor's file resolver with this list as the
    adapter's own upload spec, never by widening the executor's global list;
  - ``urls``: the params that take nothing but a link the platform fetches (D9:
    needs public storage);
  - ``optional``: skipped at publish when it cannot run (refused by the deny list,
    missing from the action cache, a link with no public storage, or refused by the
    platform), and the target's receipt notes why; it never makes its post kind
    unavailable;
  - ``returns``: what the publisher reads back from the call, ``{name: path}``
    (``modules/socials/step_results.py``). ``$steps.<id>`` reads the step's ``id``;
    a ``publish`` step's ``id`` is the target's remote id, and a ``permalink`` any
    step returns is the receipt's link. Composio's docs could not be reached from the
    build (2026-09-30), so a path lists the documented field and the likely
    alternatives, ``a|b``; the owner's live publish confirms them;
  - ``permalink``: a link with ``{id}`` in it, built from the returned id when no
    step returns a link;
  - ``until`` (a ``status`` step): polled until the state at ``path`` is ``done``,
    or ``failed`` (the target fails with the platform's ``error``).
- ``never_offered``: publish actions the registry never offers (stale, deprecated,
  or pulling media from a URL on a domain the app owner verified, and Composio owns
  the app). The post gate refuses them all the same.

A source is ``$copy`` (the channel's copy), ``$title`` (the post's title),
``$media`` (the post's media file), ``$media[]`` (all its media files),
``$media.content_type`` and ``$media.bytes`` (facts of that file), ``$thumbnail``
(the post's still), ``$option.<name>`` (the target's option, chosen in the composer
or at publish), ``$steps.<id>`` (the ``id`` an earlier step returned: the
account, upload, container or publish) or ``$idempotency_key`` (the target's own
key, for an action that takes one; no seeded action does). ``a|b`` takes the first that resolves, a
list is a list parameter, and any other value is passed as it is.

``GENERIC_ADAPTER``: D8's "Generic (text + media)". A connected toolkit outside
``CHANNEL_ADAPTERS`` is offered when one of its cached actions has a non-empty
schema with a text field and a media field, and a name that creates a post: one
of ``post_words`` and none of ``skip_words`` among the words of its slug. A media
field is a name in ``media_fields`` (mapped to the post kinds it carries), or a
schema property flagged ``file_marker`` (a file of any kind). A name ending in one
of ``url_suffixes`` takes a link; any other takes a file. Offered channels are
labelled "unverified channel" until one of their targets has published. ``returns``
is what the post action answers (its id, and a link when it gives one).
"""

CHANNEL_ADAPTERS = {
    "linkedin": {
        "label": "LinkedIn",
        "setup_note": None,
        "kinds": {
            "text": [
                {"id": "me", "action": "LINKEDIN_GET_MY_INFO", "class": "read", "returns": {"id": "author_id|response_dict.author_id"}},
                {
                    "id": "post",
                    "action": "LINKEDIN_CREATE_LINKED_IN_POST",
                    "class": "publish",
                    "params": {"author": "$option.author|$steps.me", "commentary": "$copy"},
                    "returns": {"id": "id|post_id|share_id|x_restli_id|urn"},
                    "permalink": "https://www.linkedin.com/feed/update/{id}/",
                },
            ],
            # Composio's own LinkedIn image upload is broken (its issues #3094, #3113,
            # #3231): the executor hands this call to the workspace-scoped image
            # workaround (core/composio/linkedin_image_workaround.py).
            "image": [
                {"id": "me", "action": "LINKEDIN_GET_MY_INFO", "class": "read", "returns": {"id": "author_id|response_dict.author_id"}},
                {
                    "id": "post",
                    "action": "LINKEDIN_CREATE_LINKED_IN_POST",
                    "class": "publish",
                    "params": {
                        "author": "$option.author|$steps.me",
                        "commentary": "$copy",
                        "images": "$media[]",
                    },
                    "files": ["images"],
                    "returns": {"id": "id|post_id|share_id|x_restli_id|urn"},
                    "permalink": "https://www.linkedin.com/feed/update/{id}/",
                },
            ],
            "video": [
                {
                    "id": "upload",
                    "action": "LINKEDIN_UPLOAD_VIDEO",
                    "class": "upload",
                    "params": {"file": "$media"},
                    "files": ["file"],
                    "returns": {"id": "video_urn|video|asset|value.video|id"},
                },
                {
                    "id": "post",
                    "action": "LINKEDIN_CREATE_VIDEO_POST",
                    "class": "publish",
                    "params": {"video_urn": "$steps.upload", "commentary": "$copy"},
                    "returns": {"id": "id|post_id|share_id|x_restli_id|urn"},
                    "permalink": "https://www.linkedin.com/feed/update/{id}/",
                },
            ],
        },
    },
    "twitter": {
        "label": "X",
        "setup_note": (
            "Composio removed its managed X credentials in February 2026 — connect X with "
            "your own X API app in Composio"
        ),
        "kinds": {
            "text": [
                {
                    "id": "post",
                    "action": "TWITTER_CREATION_OF_A_POST",
                    "class": "publish",
                    "params": {"text": "$copy"},
                    "returns": {"id": "id|data.id|tweet_id"},
                    "permalink": "https://x.com/i/web/status/{id}",
                },
            ],
            "image": [
                {
                    "id": "media",
                    "action": "TWITTER_UPLOAD_MEDIA",
                    "class": "upload",
                    "params": {"media": "$media", "media_type": "$media.content_type"},
                    "files": ["media"],
                    "returns": {"id": "media_id_string|media_id|id"},
                },
                {
                    "id": "post",
                    "action": "TWITTER_CREATION_OF_A_POST",
                    "class": "publish",
                    "params": {"text": "$copy", "media_media_ids": ["$steps.media"]},
                    "returns": {"id": "id|data.id|tweet_id"},
                    "permalink": "https://x.com/i/web/status/{id}",
                },
            ],
            # Video takes X's chunked upload, which Composio runs as one action.
            "video": [
                {
                    "id": "media",
                    "action": "TWITTER_UPLOAD_LARGE_MEDIA",
                    "class": "upload",
                    "params": {
                        "media": "$media",
                        "media_type": "$media.content_type",
                        "total_bytes": "$media.bytes",
                    },
                    "files": ["media"],
                    "returns": {"id": "media_id_string|media_id|id"},
                },
                {
                    "id": "processed",
                    "action": "TWITTER_GET_MEDIA_UPLOAD_STATUS",
                    "class": "status",
                    "params": {"media_id": "$steps.media"},
                    # No processing_info: X has nothing left to process.
                    "until": {
                        "path": "processing_info.state",
                        "done": ["succeeded"],
                        "failed": ["failed"],
                        "error": "processing_info.error.message|processing_info.error.name",
                        "absent": "done",
                    },
                },
                {
                    "id": "post",
                    "action": "TWITTER_CREATION_OF_A_POST",
                    "class": "publish",
                    "params": {"text": "$copy", "media_media_ids": ["$steps.media"]},
                    "returns": {"id": "id|data.id|tweet_id"},
                    "permalink": "https://x.com/i/web/status/{id}",
                },
            ],
        },
        "never_offered": ["TWITTER_CREATE_TWEET"],  # stale: never offered, and refused by the post gate
    },
    "instagram": {
        "label": "Instagram",
        "setup_note": None,
        "kinds": {
            "image": [
                {"id": "account", "action": "INSTAGRAM_GET_USER_INFO", "class": "read", "returns": {"id": "id|user_id"}},
                {
                    "id": "container",
                    "action": "INSTAGRAM_POST_IG_USER_MEDIA",
                    "class": "upload",
                    "params": {"ig_user_id": "$steps.account", "image_file": "$media", "caption": "$copy"},
                    "files": ["image_file"],
                    "returns": {"id": "id|creation_id|media_id"},
                },
                {
                    "id": "publish",
                    "action": "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH",
                    "class": "publish",
                    "params": {"ig_user_id": "$steps.account", "creation_id": "$steps.container"},
                    "returns": {"id": "id|creation_id|media_id"},
                },
            ],
            "reel": [
                {"id": "account", "action": "INSTAGRAM_GET_USER_INFO", "class": "read", "returns": {"id": "id|user_id"}},
                {
                    "id": "container",
                    "action": "INSTAGRAM_POST_IG_USER_MEDIA",
                    "class": "upload",
                    "params": {
                        "ig_user_id": "$steps.account",
                        "video_file": "$media",
                        "media_type": "REELS",
                        "caption": "$copy",
                    },
                    "files": ["video_file"],
                    "returns": {"id": "id|creation_id|media_id"},
                },
                {
                    "id": "publish",
                    "action": "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH",
                    "class": "publish",
                    "params": {"ig_user_id": "$steps.account", "creation_id": "$steps.container"},
                    "returns": {"id": "id|creation_id|media_id"},
                },
            ],
            # Two to ten images.
            "carousel": [
                {"id": "account", "action": "INSTAGRAM_GET_USER_INFO", "class": "read", "returns": {"id": "id|user_id"}},
                {
                    "id": "container",
                    "action": "INSTAGRAM_CREATE_CAROUSEL_CONTAINER",
                    "class": "upload",
                    "params": {
                        "ig_user_id": "$steps.account",
                        "caption": "$copy",
                        "child_image_files": "$media[]",
                    },
                    "files": ["child_image_files"],
                    "returns": {"id": "id|creation_id|media_id"},
                },
                {
                    "id": "publish",
                    "action": "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH",
                    "class": "publish",
                    "params": {"ig_user_id": "$steps.account", "creation_id": "$steps.container"},
                    "returns": {"id": "id|creation_id|media_id"},
                },
            ],
        },
        "never_offered": ["INSTAGRAM_CREATE_POST"],  # deprecated: never offered, and refused by the post gate
    },
    "tiktok": {
        "label": "TikTok",
        "setup_note": None,
        "kinds": {
            # The privacy level is one the creator info allows, chosen at publish;
            # is_aigc labels generated footage.
            "video": [
                {"id": "creator", "action": "TIKTOK_QUERY_CREATOR_INFO", "class": "read"},
                {
                    "id": "upload",
                    "action": "TIKTOK_UPLOAD_VIDEO",
                    "class": "publish",
                    "params": {
                        "file_to_upload": "$media",
                        "caption": "$copy",
                        "privacy_level": "$option.privacy_level",
                        "is_aigc": "$option.is_aigc",
                        "publish": True,
                    },
                    "files": ["file_to_upload"],
                    "returns": {"id": "publish_id"},
                },
                {
                    "id": "published",
                    "action": "TIKTOK_FETCH_PUBLISH_STATUS",
                    "class": "status",
                    "params": {"publish_id": "$steps.upload"},
                    "until": {
                        "path": "status",
                        "done": ["PUBLISH_COMPLETE"],
                        "failed": ["FAILED"],
                        "error": "fail_reason",
                    },
                },
            ],
        },
        # They pull media from a URL on a domain the app owner verified; Composio owns the app.
        "never_offered": ["TIKTOK_PUBLISH_VIDEO", "TIKTOK_POST_PHOTO"],  # never offered: URL pull
    },
    "youtube": {
        "label": "YouTube",
        "setup_note": None,
        "kinds": {
            "video": [
                {
                    "id": "upload",
                    "action": "YOUTUBE_UPLOAD_VIDEO",
                    "class": "publish",
                    "params": {
                        "title": "$title",
                        "description": "$copy",
                        "categoryId": "$option.category_id",
                        "privacyStatus": "$option.privacy_status",
                        "tags": "$option.tags",
                        "videoFilePath": "$media",
                    },
                    "files": ["videoFilePath"],
                    "returns": {"id": "id|videoId|video_id"},
                    "permalink": "https://www.youtube.com/watch?v={id}",
                },
                # The custom thumbnail takes only a public link (D9).
                {
                    "id": "thumbnail",
                    "action": "YOUTUBE_UPDATE_THUMBNAIL",
                    "class": "upload",
                    "params": {"videoId": "$steps.upload", "thumbnailUrl": "$thumbnail"},
                    "urls": ["thumbnailUrl"],
                    "optional": True,
                },
            ],
        },
    },
}

GENERIC_ADAPTER = {
    "post_words": ["POST", "PUBLISH"],
    "skip_words": [
        "GET", "LIST", "SEARCH", "FETCH", "FIND", "RETRIEVE", "DELETE", "REMOVE", "UPDATE",
        "EDIT", "COMMENT", "COMMENTS", "REPLY", "REPLIES", "LIKE", "UNLIKE", "REACTION",
        "REACTIONS", "INSIGHTS", "ANALYTICS", "STATS",
    ],
    "text_fields": ["text", "caption", "message", "content", "commentary", "body", "status", "description"],
    "media_fields": {
        "image": ["image"],
        "images": ["image"],
        "image_file": ["image"],
        "image_files": ["image"],
        "image_url": ["image"],
        "image_urls": ["image"],
        "photo": ["image"],
        "photos": ["image"],
        "photo_url": ["image"],
        "photo_urls": ["image"],
        "video": ["video"],
        "video_file": ["video"],
        "video_url": ["video"],
        "video_urls": ["video"],
        "media": ["image", "video"],
        "media_file": ["image", "video"],
        "media_files": ["image", "video"],
        "media_url": ["image", "video"],
        "media_urls": ["image", "video"],
    },
    "url_suffixes": ["_url", "_urls"],
    "file_marker": "file_uploadable",
    # What a generic post action returns: its id, and a link when it gives one.
    "returns": {"id": "id|post_id|data.id", "permalink": "permalink|url|link"},
}
