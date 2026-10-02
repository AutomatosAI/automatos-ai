"""Socials (PRD-251): on-brand social posts, approved one by one, scheduled on
the calendar and published through the workspace's own Composio connections.

- :mod:`modules.socials.settings`: the two switches (the platform master switch
  and the workspace switch) and the route gate ``require_socials_enabled``.
- :mod:`modules.socials.service`: the post lifecycle — the status machine, the
  content hash an approval binds to (D6), unsourced claims (D7) and the publish
  guard ``assert_publishable``.
- :mod:`modules.socials.targets`: a post's channels (US-204), one target per
  channel and post kind: approved content, so the hash covers them; their shape,
  their ``sp:`` keys and how a new set replaces the rows.
- :mod:`modules.socials.campaigns`: campaigns, named sets of posts, and their
  series approval (D6, S2.4): each shown post approved as a single approval,
  hash-bound, when the workspace's series approval switch is on.
- :mod:`modules.socials.sources`: a claim's source resolved in the caller's
  workspace (S1.4), and the source picker's search.
- :mod:`modules.socials.text_search`: what a person types into a search,
  matched literally and case-insensitively with LIKE: the source picker's, and
  the posts list's ``q`` (US-205).
- :mod:`modules.socials.report_charts`: a chart bound to a report (S1.7): the
  report's top rows filling a chart template (the infographic), and the check
  that a render shows the report's rows as the report has them now.
- :mod:`modules.socials.capabilities`: the capability registry (D16) — which
  actions of the workspace's connected Composio toolkits Socials may call, and
  what for (an allowlist per toolkit, under the deny list); and (D8) which social
  channels the workspace can post to, each post kind's action sequence, and which
  actions publish (the post gate refuses those to agents, D14). The channels'
  data is :mod:`modules.socials.channel_adapters`.
- :mod:`modules.socials.recipes`: one small recipe per Composio media toolkit
  (D12), through the registry and the Composio executor: speech from Fish
  Audio or ElevenLabs (``recipes.voice``, S1.5); footage and stills from fal.ai,
  Kie.ai or Higgsfield MCP (``recipes.footage``, S1.8).
- :mod:`modules.socials.media_caps`: the post's media cap and the workspace's
  monthly media cap, checked before any spend (D13).
- :mod:`modules.socials.media_urls`: presigned inline links to a post's media
  (D9, S3.4): the approval view's, and a link a platform fetches, which needs
  public storage.
- :mod:`modules.socials.publisher`: ``begin_publish`` / ``begin_retry``, the one
  way a post leaves Automatos (the approval guard first, then a compare-and-set
  claim), and :mod:`modules.socials.publishing`, the one engine that publishes
  every channel's targets through Composio from the adapter data (Wave 3, US-301).
"""
