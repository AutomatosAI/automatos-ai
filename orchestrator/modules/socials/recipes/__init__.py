"""PRD-251 D12: one small recipe per Composio media toolkit.

Each recipe maps what Socials needs ("speak this script line", "a 5 s 9:16 shot
from this prompt") onto one toolkit's actions, as the media capability registry
offers them (``modules/socials/capabilities.py``: connected, allowlisted, in the
cached schemas, not denied). Every call goes through ``ComposioToolExecutor``,
the workspace's own connection: Automatos holds no provider key and writes no
provider client (D15). What a tool returns is copied into our storage the
moment it arrives (``files.py``).

- :mod:`modules.socials.recipes.voice`: Fish Audio and ElevenLabs speech (S1.5).
"""
