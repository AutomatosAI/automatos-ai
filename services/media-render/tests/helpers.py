"""Builders shared by the suite: a minimal composition and bundle, the test token."""

from __future__ import annotations

from typing import Any, Dict

TOKEN = "t0ken-for-tests"
STORAGE = "https://media.example-storage.test/automatos/"

PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <script src="assets/vendor/gsap.min.js"></script>
    <style>
      html, body { width: 1080px; height: 1920px; margin: 0; background: var(--brand-bg, #14171c); }
      #title { position: absolute; left: 90px; right: 90px; top: 800px; color: var(--brand-text, #f3efe6); }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="__DURATION__" data-width="1080" data-height="1920">
      __BODY__
      <h1 id="title">{{ headline }}</h1>
      <audio id="mix" src="assets/audio/mix.wav" data-start="0" data-duration="__DURATION__" data-track-index="1"></audio>
    </div>
    <script>
      window.__timelines = window.__timelines || {};
      const tl = gsap.timeline({ paused: true });
      tl.fromTo("#title", { opacity: 0 }, { opacity: 1, duration: 0.5 }, 0.2);
      window.__timelines["main"] = tl;
    </script>
  </body>
</html>
"""


def page(body: str = "", *, duration: float = 3) -> str:
    """A minimal composition document that passes the bundle's static checks."""
    return PAGE.replace("__BODY__", body).replace("__DURATION__", f"{duration:g}")


def bundle(workspace: str = "ws-a", **overrides: Any) -> Dict[str, Any]:
    """A minimal valid render bundle; keyword arguments replace its top-level fields."""
    body: Dict[str, Any] = {
        "workspace_id": workspace,
        "composition": {"html": page()},
        "variables": {"headline": "On brand"},
    }
    body.update(overrides)
    return body


def data_uri(mime: str, payload: bytes) -> str:
    import base64

    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")
