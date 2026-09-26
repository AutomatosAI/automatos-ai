"""Socials (PRD-251): on-brand social posts, approved one by one, scheduled on
the calendar and published through the workspace's own Composio connections.

- :mod:`modules.socials.settings`: the two switches (the platform master switch
  and the workspace switch) and the route gate ``require_socials_enabled``.
- :mod:`modules.socials.service`: the post lifecycle — the status machine, the
  content hash an approval binds to (D6), unsourced claims (D7) and the publish
  guard ``assert_publishable``.
- :mod:`modules.socials.sources`: a claim's source resolved in the caller's
  workspace (S1.4), and the source picker's search.
- :mod:`modules.socials.report_charts`: a chart bound to a report (S1.7): the
  report's top rows filling a chart template (the infographic), and the check
  that a render shows the report's rows as the report has them now.
- :mod:`modules.socials.capabilities`: the media capability registry (D16) —
  which actions of the workspace's connected Composio toolkits Socials may call,
  and what for (an allowlist per toolkit, under the deny list).
- :mod:`modules.socials.recipes`: one small recipe per Composio media toolkit
  (D12), through the registry and the Composio executor: speech from Fish
  Audio or ElevenLabs (``recipes.voice``, S1.5); footage and stills from fal.ai,
  Kie.ai or Higgsfield MCP (``recipes.footage``, S1.8).
- :mod:`modules.socials.media_caps`: the post's media cap and the workspace's
  monthly media cap, checked before any spend (D13).
- :mod:`modules.socials.publisher`: ``publish_post``, the one way a post leaves
  Automatos (the guard first; channel publishers arrive in Wave 3).
"""
