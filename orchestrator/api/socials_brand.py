"""The brand kit's part in what Socials generates (PRD-251B B9, B10; US-B303..US-B305).

``modules/socials`` never imports ``modules/documents`` (``.importlinter``), so the api layer
hands a render and the AI options what they take from the brand kit:

* the style profile, as one paragraph after every prompt (``brand_style_text``);
* the workspace's default toolkit per slot kind, tried first (``media_tools.prefer_for``);
* links to the kit's liked style references (:func:`liked_reference_links`), which a toolkit
  gets only when its generate action takes a reference image (the registry's
  ``reference_image`` flag). None unless the workspace sends them (``send_liked``) and a
  platform can reach our storage (D9: AWS S3, or the public media bucket, into which each is
  first copied). A reference whose object is not in storage is left out, never fatal.

Everything here reads the database or storage synchronously: code on the event loop runs it
in a worker thread.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.orm import Session

from api.socials_compose import brand_style_text
from config import config
from core.storage import ensure_bucket, get_public_s3_client, get_s3_client
from modules.documents import brand_references as refs
from modules.documents.brand_logo import brand_files_bucket, s3_brand_file_key
from modules.socials import media_tools
from modules.socials.media_urls import media_public_url_available, presigned_inline_url

logger = logging.getLogger(__name__)

MAX_REFERENCE_LINKS = 1  # a recipe sends one reference image
PUBLIC_REFERENCE_PREFIX = "brand-references"


def _public_key(ref: Mapping[str, Any]) -> Tuple[str, str]:
    """Where the reference's link points: the documents bucket, or a copy in the public media bucket."""
    bucket, key = brand_files_bucket(), s3_brand_file_key(ref["path"])
    get_s3_client().head_object(Bucket=bucket, Key=key)  # ClientError when the mirror never landed
    if not config.SOCIALS_PUBLIC_MEDIA_BUCKET:
        return bucket, key
    public = f"{PUBLIC_REFERENCE_PREFIX}/{ref['path']}"
    ensure_bucket(config.SOCIALS_PUBLIC_MEDIA_BUCKET)
    get_s3_client().copy_object(Bucket=config.SOCIALS_PUBLIC_MEDIA_BUCKET, Key=public, CopySource={"Bucket": bucket, "Key": key})
    return config.SOCIALS_PUBLIC_MEDIA_BUCKET, public


def _link(ref: Mapping[str, Any]) -> Optional[str]:
    try:
        bucket, key = _public_key(ref)
        return presigned_inline_url(get_public_s3_client(), bucket, key, ref.get("content_type") or "image/png")
    except (BotoCoreError, ClientError):
        logger.warning("[Socials] style reference %s has no link a toolkit can fetch; it is left out", ref.get("id"), exc_info=True)
        return None


def liked_reference_links(settings: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Links to the liked style references, newest first, at most MAX_REFERENCE_LINKS (the module docstring)."""
    style = refs.style_of(settings)
    liked, _avoided = refs.split_by_stance(style["references"])
    if not style["send_liked"] or not liked or not media_public_url_available():
        return ()
    links = []
    for ref in reversed(liked):
        link = _link(ref)
        if link:
            links.append(link)
        if len(links) >= MAX_REFERENCE_LINKS:
            break
    return tuple(links)


def generation_inputs(db: Session, workspace: Any) -> Dict[str, Any]:
    """What a render's footage plan takes from the brand kit and the AI tools section."""
    return {
        "style": brand_style_text(db, workspace.id),
        "prefer": media_tools.prefer_for(workspace.settings),
        "references": liked_reference_links(workspace.settings),
    }
