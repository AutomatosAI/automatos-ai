# PRD-251 reference compositions

These are the compositions behind the four reference videos: the three in PRD-251A §1 and the Markets posh cut made after it. They are text only; the footage, voice and music files stay outside the repo. Wave 1 ports them into `document_templates` rows (D4). Each hardcoded colour, font, logo, line of copy, clip and data point becomes a variable fed from the brand kit, the post and its sources.

| File | Reference video | Seeded template |
|---|---|---|
| `v1-ui-story.html` | v1: a product story with animated UI, no footage | "UI story promo" |
| `v2-cinematic-product.html` | v2: the Shopify story with four cinematic slots | "cinematic product promo" |
| `academy-app-promo.html` | Academy: an app promo with a phone frame | "app promo" |
| `markets-posh.html` + `markets-posh-build.py` + `markets-posh-charts.py` | Markets posh cut: footage and stills with data cards generated from live data | "data story" (the data → SVG → HTML pattern for S1.7) |

The supporting scripts show how the audio and the paid footage were produced:
- `mix-reference.py`: the ffmpeg mix (ducking, SFX, −14 LUFS).
- `synth-vo-reference.py`: Kokoro, one WAV per line.
- `higgsfield-client-reference.py`: estimate → cap → submit → poll → download. In Wave 1 it is reached through Composio recipes (D12), never as a direct client.

Composition rules and the traps behind them are in PRD-251A §2, stage 7.
