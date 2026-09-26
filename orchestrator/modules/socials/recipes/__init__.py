"""PRD-251 D12: one small recipe per Composio media toolkit.

Each recipe maps what Socials needs ("speak this script line", "a 5 s 9:16 shot
from this prompt") onto one toolkit's actions, as the media capability registry
offers them (``modules/socials/capabilities.py``: connected, allowlisted, in the
cached schemas, not denied). Every call goes through ``ComposioToolExecutor``,
the workspace's own connection: Automatos holds no provider key and writes no
provider client (D15). What a tool returns is copied into our storage the
moment it arrives (``files.py``).

- :mod:`modules.socials.recipes.voice`: Fish Audio and ElevenLabs speech (S1.5).
- :mod:`modules.socials.recipes.footage`: footage and stills for a template's
  slots (S1.8): the plan, the caps and the money (D13), submit and poll, the
  copy into our storage; the per-toolkit recipes are
  :mod:`modules.socials.recipes.footage_toolkits` (fal.ai, Kie.ai, Higgsfield MCP).
- :mod:`modules.socials.recipes.toolkit`: what every recipe shares: calling an
  offered action, reading its answer, a credit balance and the credit window.
"""
