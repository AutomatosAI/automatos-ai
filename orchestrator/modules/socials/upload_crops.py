"""PRD-251C (US-C303; O6, the PRD's recommendation: crop): a still the person brings, cropped
for each channel.

A post whose visual is the person's own still (an upload or a Library picture as the whole
post: ``media["original"]``, no template, format ``image``) renders as a template's post
does: once per shape its channels need (``channel_sizes.render_sizes`` over
:data:`CROP_SIZES`), through media-render. The built-in composition (:data:`CROP_BLOCKS`)
fills the frame with the picture, centred (``object-fit: cover``), with nothing on it. The
crops join ``media`` by aspect beside the original, which stays for the next render (the
channels may change). Publishing then gives each channel its own crop
(``publish_sources.media_for``), and the preview shows those files. A video the person
brings is sent as it is.

Pure: what to render and how; ``api/socials_render_crops.py`` starts the render.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.media_render_bundle import build_bundle
from core.social_templates import SOCIAL_IMAGE
from modules.socials import channel_sizes
from modules.socials.render import EXECUTION_PREFIX

ORIGINAL = "original"
IMAGE = "image"
CROP_SLOT = "photo"
CROP_PATH = "assets/slots/photo.png"
# One size per shape a channel shows a still in best (channel_sizes.KIND_ASPECT): square,
# 4:5, 9:16, a link card and 16:9.
CROP_SIZES: Tuple[str, ...] = ("1080x1080", "1080x1350", "1080x1920", "1200x628", "1600x900")

CROP_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={{ size.width }}, height={{ size.height }}" />
    <title>Your picture, cropped</title>
    <script src="assets/vendor/gsap.min.js"></script>
    <style>
      /* PRD-251C (US-C303): the person's own picture fills the frame, centred: a crop, nothing on it. */
      * { margin: 0; padding: 0; box-sizing: border-box; }
      html, body { width: {{ size.width }}px; height: {{ size.height }}px; overflow: hidden; background: var(--brand-ink, #1a1714); }
      #root { position: relative; width: 100%; height: 100%; overflow: hidden; }
      .clip { position: absolute; inset: 0; }
      .photo { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center; }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="1" data-width="{{ size.width }}" data-height="{{ size.height }}">
      <section id="card" class="clip" data-start="0" data-duration="1" data-track-index="0">
        <img class="photo" data-slot="photo" data-layout-allow-overflow src="assets/slots/photo.png" alt="" />
      </section>
      <audio id="mix" src="assets/audio/mix.wav" data-start="0" data-duration="1" data-track-index="1" data-volume="1"></audio>
    </div>
    <script>
      window.__timelines = window.__timelines || {};
      // A still: nothing moves. The timeline only spans the card's one second.
      const tl = gsap.timeline({ paused: true });
      tl.set({}, {}, 1);
      window.__timelines["main"] = tl;
    </script>
  </body>
</html>
"""

CROP_BLOCKS: Mapping[str, Any] = {
    "html": CROP_HTML,
    "variables_schema": {},
    "sizes": list(CROP_SIZES),
    "slots": {CROP_SLOT: {"kind": "image", "path": CROP_PATH, "label": "Your picture", "generate": False}},
}


def own_still(post: Any) -> Optional[str]:
    """The id of the post's own still (its ``media["original"]``) when that is its visual: no
    template and an image post; ``None`` otherwise."""
    if getattr(post, "template_id", None) is not None or getattr(post, "format", None) != IMAGE:
        return None
    originals = (getattr(post, "media", None) or {}).get(ORIGINAL)
    first = originals[0] if isinstance(originals, list) and originals else None
    return str(first) if isinstance(first, str) and first else None


def crop_sizes(post: Any) -> List[str]:
    """The sizes the post's channels need, one per shape (the first, square, with no channel)."""
    return channel_sizes.render_sizes(CROP_SIZES, getattr(post, "targets", None) or ())


def crop_bundles(post: Any) -> List[Dict[str, Any]]:
    """Each size's render bundle: the composition's one slot shows the picture, whose file
    joins the bundle as the render starts (``render.with_slot_files``)."""
    return [
        build_bundle(
            workspace_id=post.workspace_id, reference=f"{EXECUTION_PREFIX}{post.id}", blocks=CROP_BLOCKS, values={},
            brand_kit={}, size=size, keep_slots=(CROP_SLOT,), fmt=SOCIAL_IMAGE,
        )
        for size in crop_sizes(post)
    ]
