"""PRD-251 S0.3a: the publish seam.

``publish_post`` is the ONE way a post leaves Automatos. It calls
``assert_publishable`` before anything else, so a post whose approval no longer
matches its content (D6) never reaches a channel, whatever the policy plane
mode.

Wave 0 has no channel publishers: after the guard it raises
:class:`PublishingUnavailable`. S3.3 fills ``_publish_targets`` with the
per-channel Composio publishers; nothing here calls Composio yet.
"""
from __future__ import annotations

from typing import Any

from core.models.socials import SocialPost
from modules.socials.service import SocialsError, assert_publishable

PUBLISHING_UNAVAILABLE_MESSAGE = "Channel publishing arrives in Wave 3"


class PublishingUnavailable(SocialsError):
    """No channel publisher exists yet (Wave 0)."""

    def __init__(self, message: str = PUBLISHING_UNAVAILABLE_MESSAGE):
        super().__init__(message)


def _publish_targets(db: Any, post: SocialPost) -> Any:
    """The seam S3.3 fills: resolve the post's targets and publish each one
    through the workspace's own Composio connections."""
    raise PublishingUnavailable()


def publish_post(db: Any, post: SocialPost) -> Any:
    """Publish an approved post. The guard runs first; nothing runs if it refuses."""
    assert_publishable(post)
    return _publish_targets(db, post)
