"""Socials (PRD-251): on-brand social posts, approved one by one, scheduled on
the calendar and published through the workspace's own Composio connections.

- :mod:`modules.socials.settings`: the two switches (the platform master switch
  and the workspace switch) and the route gate ``require_socials_enabled``.
- :mod:`modules.socials.service`: the post lifecycle — the status machine, the
  content hash an approval binds to (D6), unsourced claims (D7) and the publish
  guard ``assert_publishable``.
- :mod:`modules.socials.sources`: a claim's source resolved in the caller's
  workspace (S1.4), and the source picker's search.
- :mod:`modules.socials.publisher`: ``publish_post``, the one way a post leaves
  Automatos (the guard first; channel publishers arrive in Wave 3).
"""
