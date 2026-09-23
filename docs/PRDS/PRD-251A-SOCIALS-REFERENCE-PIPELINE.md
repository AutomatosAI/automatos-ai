# PRD-251A: Socials reference pipeline — how the three promo videos and the image posts were made, what they cost, and where each step goes in Automatos

> **Status:** REFERENCE 2026-09-23. This is the companion to `PRD-251-SOCIALS.md`. It records a working spike, not code in this repo. The artefacts live in the workspace root, outside every repo: `brag-output-2026-09-22-225946/`, `brag-output-2026-09-23-v2/`, `brag-output-2026-09-23-academy/` and `brag-output-2026-09-23-images/`.
>
> **Why it exists:** the owner saw the output and said "If we can do this for client in automatos... gold... Document all the process so we can build this into Automatos." Everything below was run end to end on 22–23 Sep 2026, and every number was measured on that run.

---

## 1. What was produced

| Asset | Path (workspace root) | Spec | External spend | Render time |
|---|---|---|---|---|
| **v1 promo, template only** | `brag-output-2026-09-22-225946/brag.mp4` | 39.5 s · 1080×1920 · 30 fps · H.264 + AAC · −14.5 LUFS | **$0** | 2 m 42 s |
| **v2 Shopify promo, cinematic** | `brag-output-2026-09-23-v2/automatos-v2-cinematic.mp4` | 40 s · 4 Cinema Studio shots (18 s) · −14.6 LUFS | **$8.32** | 4 m 30 s |
| **Automatos posts** | `brag-output-2026-09-23-v2/posts/post1-3.png` | 1080×1350 · 3 Soul Cinema stills | **$0.018** | seconds |
| **Academy promo, cinematic** | `brag-output-2026-09-23-academy/automatos-academy-cinematic.mp4` | 38 s · 4 shots (17 s) · Academy brand · −15.0 LUFS | **$7.86** | 3 m 37 s |
| **Academy posts** | `brag-output-2026-09-23-academy/posts/post1-3.png` | 1080×1350 · 3 Soul Cinema stills | **$0.018** | seconds |
| **Image-model comparison** | `brag-output-2026-09-23-images/image-model-comparison.png` | same brief through 7 model variants | **≈ $0.47 + 2 token-metered calls** | — |
| **Product-photo edit test** | `brag-output-2026-09-23-images/out/e1_ms_edit.png` | Marketing Studio with `image_urls` | **$0.255** | — |
| **Client demo posts** | `brag-output-2026-09-23-images/posts/client1-3.png` | "Harbourline Coffee Roasters" brand kit | reuses the above | seconds |

- **Machine:** all renders ran on an 8-core Apple-silicon Mac.
- **Pricing source:** prices are Higgsfield list prices, taken from their estimate endpoint or their published token formula (§3). Reconcile against the account statement.
- **Demo business:** "Harbourline Coffee Roasters" is the simulated business from the customer nights. The client posts are demos.

---

## 2. The pipeline, stage by stage

Each stage lists what was done, the exact settings, the traps hit, and where it goes in Automatos. The Automatos home is the matching PRD-251 decision or story.

### Stage 0 — Brief and claims
- **Every claim is sourced or dropped.**
  - The Web Summit cost figures were left out. `docs/WEBSUMMIT-MESSAGE.md` is marked "PARKED … not as a claim anyone may quote yet", and the docs disagree ($3.42 vs $4.26).
  - The Academy figures (227 lessons, 85 labs, 1,691 practice questions) were counted from `automatos-academy/public/content/**`. The count used the same rules as the server's `/stats` endpoint (`automatos-academy/server/catalog.js:225-270`) and covered the 10 live tracks only.
- **Workspace copy rules are respected.** The Academy repo enforces its own copy rules (`automatos-academy-app/src/compliance/copy.ts:7-12`):
  - no "guarantee" and no pass promise; the only approved outcome wording is "prepares you for";
  - vendor names in plain text;
  - a non-affiliation note, which the end card carries.
- **Script copy is the product's own public copy** (landing page, app strings) plus a few lines written for the video.
- **Automatos home:** D7 and S1.4 (facts carry sources), and D5 (brand-kit `voice` with banned phrases). A per-workspace "required disclaimer" belongs in the brand kit too.

### Stage 1 — Brand extraction (from code, not screenshots)
- **Automatos (Studio Dark):**
  - Tokens from `frontend/app/globals.css` `.studio.dark`: ink `#1A1714`, card `#221F1C`, cream `#F0E8DB`, burnt orange `#E96235`, olive `#90AF5A`.
  - Fonts: Geist, Geist Mono and Newsreader, downloaded as woff2 from Google Fonts (OFL).
  - Logo: the sailboat from the Brand Assets SVGs (`logomark_white.svg` is one path with three sub-paths, split for a hull-then-sails animation).
- **Academy ("Academy Periwinkle" Night):**
  - Palette from `automatos-academy-app/src/theme/palette.ts`: `#141D30` / `#1C2740` / `#24314C`, text `#E6EDF8`, periwinkle `#A9BFDD`, coral `#FF6B42`, gold A+ `#DCC363`.
  - Funnel Display and Funnel Sans TTFs from the app's assets.
  - The app's real `hero-loop.mp4` brain animation and `icon.png`.
- **Client demo (Harbourline):** a hand-made kit of green `#1F3B2D`, cream `#F3EDE2`, copper `#C8742C`, with Newsreader and Geist. This is what the brand kit must produce for any customer.
- **Screens are rebuilt in HTML/CSS from the real components and strings.** Examples: the chat "activity trail", the board columns, the Questions card, the Academy question card and tutor. This keeps text sharp and true to the product.
- **Automatos home:** D5 and S1.3. The brand kit needs font **files**, because Hyperframes lint requires an `@font-face` pointing at a local file for every named font.

### Stage 2 — Script and voice-over
- **Script shape:** 9–10 lines of at most about 12 words each, which is about 25 s of speech in a 38–40 s video. Captions repeat the lines that aren't already on-screen headlines.
- **Voice:** Kokoro-82M through `kokoro-onnx`, voice `af_heart`, speed 0.95, `lang=en-us`. One WAV per line (24 kHz mono) so every line can be placed exactly. The reference is `synth_vo.py` in the Appendix.
- **Pronunciation check:** run lines through `phonemizer` before rendering.
  - "Automatos" → `ˌɔːɾəmˈɑːɾoʊz`.
  - "AI" → `ˌeɪˈaɪ` (good).
  - "A plus" → `ɐ plˈʌs`, which is wrong; writing "A-plus" gives `ˈeɪplˈʌs`.
- **Word timing** comes from energy-gap detection (10 ms hops, a threshold of 4% of the line's peak, gaps of 70–80 ms or more). That is how "Orders. Stock. Customers. The books." each move a board card on the word.
- **Traps:**
  - **espeak-ng truncates its data path at about 160 characters** (the error was `…/site-packages//phontab`). `phonemizer` resolves symlinks, so copy the data to a short real path.
  - `hyperframes tts` needs `HYPERFRAMES_PYTHON` pointing at a Python with `kokoro-onnx`.
  - `phonemizer` and espeak-ng are **GPL-3.0**. Keep them in their own process (`media-render`).
- **Automatos home:** D11 and S1.5.
  - Kokoro is the default.
  - Fish Audio and ElevenLabs come through the workspace's Composio connection (D15, Composio-first). Fish Audio's Composio tool, `FISH_AUDIO_SYNTHESIZE_SPEECH`, uses its free S2.1 model; the paid `s2.1-pro` is $15 per million UTF-8 bytes, about **$0.005** for a 40 s script.
  - No owner recordings.

### Stage 3 — Music
- **Royalty-free only.** Chart-trending songs are not licensed for business promos; Meta points brands to royalty-free music or its Sound Collection. So pick **CC BY or CC0 tracks in the trending style**.
- **Tracks used, all by Sascha Ende from ende.app (CC BY 4.0, attribution appreciated):**
  - "Where the Night Begins" (melodic techno), for v1.
  - "Da Da Da Da Da De De De De" (Afro-house), for v2.
  - "Spring of 2026" (tropical house), for the Academy promo.
- **Structure analysis:**
  - A 2 s map of loudness, energy below 150 Hz (bass/kick) and the 300–3,000 Hz share (vocals/chops) finds instrumental grooves (good under voice) and chant breakdowns (keep them in the gaps).
  - A second pass at 0.1 s resolution pins drops and cuts. Examples: the breakdown at 53.4 s and the drop at 62.2 s in the Afro-house track; the drop at 52.9 s in "Spring of 2026".
- **Window choice:** offset the track so a musical event lands on a script beat.
  - v1: the music cuts on "clock off".
  - v2: the bass drops out after "You have to" (3.35 s), and the drop hits on "The right agent takes it" (12.15 s).
  - Academy: the drop hits on "Meet Automatos Academy" (5.0 s).
- **Automatos home:** S1.6. Precompute the drop, breakdown and groove windows per track when seeding the library, and store licence and attribution fields.

### Stage 4 — Mix
**One ffmpeg filter graph** (reference `mix_v2.py` in the Appendix):
1. The music excerpt gets fades.
2. Each voice line is placed with `adelay`, and the lines are mixed together.
3. The music is ducked by `sidechaincompress` keyed on the voice: threshold 0.015, ratio 10, attack 10 ms, release 420 ms. The bed then sits at ×0.6.
4. SFX (Kenney, CC0) are placed with `adelay` at volumes between 0.3 and 0.7.
5. Everything goes through `amix` and then `loudnorm` (I = −14 LUFS, TP = −1.5 dB), out as 48 kHz stereo WAV.
6. The composition carries **one** `<audio>` element.

**Measured on the final MP4s:** −14.5 (v1), −14.6 (v2) and −15.0 (Academy) LUFS integrated; true peak −1.4 dBFS on all three.

**Automatos home:** the mix step inside `media-render` (D3).

### Stage 5 — Cinematic footage (optional, the customer's key)
- **API:**
  - Submit: `POST https://api.higgsfield.ai/higgsfield/cinema-studio/4.0`, header `Authorization: Key <id>:<secret>`.
  - Poll: `GET /requests/{id}/status`. States are `queued`, `in_progress`, then `completed` / `failed` / `nsfw` / `canceled`.
  - Output: the video URL is on a public CloudFront host and retained for at least 7 days. **Copy it into Automatos storage straight away.**
  - Failed and moderated requests are not charged.
- **Parameters used:**
  - `aspect_ratio: 9:16`
  - `resolution: 720p` (the maximum)
  - `duration: 4–5` (the minimum is 4)
  - `generate_audio: false` (the music is ours)
  - `camera_movement` (`slow-zoom-in`, `dolly-in`, `static-shot`, `aerial-pullback`, `crane-up`), `camera_lens` (`anamorphic`, `clean-sharp`), `light: practicals`
- **Pricing is token-metered:** tokens = ⌈seconds × w × h × 24 / 1024⌉, at **$0.0214 per 1,000 tokens** for 480p and 720p. At 720p vertical that is **$0.462 per second**: $1.85 for a 4 s clip, $2.31 for 5 s. For this model the estimate endpoint returns a *description* instead of a number, so compute the cost locally (the Appendix client does).
- **What came back:**
  - 720×1280 at 24 fps, video only.
  - 8 of 8 shots completed first time, and all 8 were used.
  - A batch of four submitted together came back in about 3–4 minutes (Academy run: job file written 01:03:49, last clip downloaded 01:07:28).
- **Prompt pattern:** subject and scene, then light, then camera, then "no readable text, no logos", then the brand's colour words ("cool blue tones with warm coral highlights").
- **Never generate product UI or text.** It warps. UI stays HTML-rendered.
- **Stretching a short clip:** extract the last frame (`ffmpeg -sseof -0.05 … -frames:v 1`) and hold it with a slow zoom. The v2 end card continues the aerial pull-back by scaling the video wrapper from 1.12 to 1.10, then the freeze frame from 1.10 to 1.00.
- **Automatos home:** D12 and S1.8.
  - The workspace connects a generation toolkit in Composio: fal.ai, Kie.ai or Higgsfield MCP. A per-toolkit recipe submits, polls and copies the output.
  - Cost is estimated and capped before spend, then booked per D13 and S4.4.
  - Composio's Higgsfield is the **account** kind (OAuth, credits), not the platform API used here. Whether it reaches Cinema Studio 4.0 is on PRD-251's "Verify at build" list.

### Stage 6 — Still images (optional, the customer's key): model comparison
**The brief:** "Premium lifestyle product photo: kraft-paper coffee bag, blank label, rustic café counter, latte art, morning light", in 3:4. The labelled sheet is `brag-output-2026-09-23-images/image-model-comparison.png`.

| Model (endpoint) | Price per image | Output | Finding |
|---|---|---|---|
| **Soul Cinema** (`higgsfield-ai/soul/cinema`) | **$0.006**; a batch of 4 is $0.022 ($0.0055 each) | 1536×2048 | The best value. Cinematic, and the label stayed blank. The default for post backgrounds. |
| **Soul 2** (`higgsfield-ai/soul/v2/standard`) | **$0.006** | 1536×2048 | Good, but it **invented garbled label text** (roughly "… COFFEE BRANS") despite "no text". |
| **Soul standard** (`higgsfield-ai/soul/standard`) | **$0.188** | 1536×2048 | 31× Soul 2's price for no visible gain, and it also made up text. Don't offer it. |
| **Marketing Studio** (`marketing-studio/image`) | **$0.245** at 2k/high (the response shows a 25% discount) | 1744×2336 | The cleanest commercial product look. **It edits real product photos.** |
| **Marketing Studio flare / sunburst** (`…/image/flare`, `…/image/sunburst`) | Token-metered: image output $30 per million tokens, text input $5 per million | 1744×2336 | Similar quality. There is no up-front price, so budget a ceiling. |

- **The client feature is product-photo editing.** `marketing-studio/image` with `image_urls: [<public URL of the customer's product photo>]`:
  - It accepts up to 16 references.
  - Presets come from `GET /marketing-studio/image/presets`; `enhance_prompt=true` needs a preset plus one or two images (product first).
  - In the test (**$0.255**) it kept the bag's shape, zip and label exactly and placed it in a new bright kitchen-shelf scene.
  - For Automatos this means **one uploaded product photo becomes a series of ad scenes.**
- **Recommended defaults:**
  - Soul Cinema, batch of 4 (the customer picks one), for post backgrounds.
  - Marketing Studio for product ads made from the customer's own photos.
  - All words are overlaid in HTML and never generated.

### Stage 7 — Composition (Hyperframes, Apache-2.0, pinned `0.8.62`)
- **Structure:** one HTML file per video.
  - The root is `data-composition-id="main"` at 1080×1920 with `data-duration`.
  - A persistent background layer: brand gradient, perspective grid "floor", glows, grain.
  - **Top-level** `<video>` clips. A video may never have a timed ancestor; wrap it only in untimed divs when it needs to scale.
  - Scene `<section class="clip">` elements, and caption clips.
  - One pre-mixed `<audio>`.
  - GSAP from a local file, with **one paused timeline registered at the end**.
- **9:16 safe zones:** keep text between y ≈ 240 and y ≈ 1560. Captions go in the y ≈ 1452–1560 band. The platform's UI covers the bottom 350 px or so and the right edge.
- **Lint and check failures we hit, with the fixes:**

| Problem | Fix |
|---|---|
| Layering flagged as overlap | `data-layout-allow-overlap` must go on the **text element itself**; it is not inherited |
| A tween on `left`/`top` fails lint (`gsap_non_transform_motion`) | Put the position in CSS and tween transforms or opacity |
| Rotating an outer `<svg>` spinner with `svgOrigin` | `svgOrigin` works only on SVG child elements; use `transformOrigin: "50% 50%"` |
| Two display lines flagged as overlapping | Line-height ≥ 1.1, or add a gap |
| Orange text on tinted pills below 4.5:1 contrast | Use `#F07A50` |
| The same `<img>` source twice flagged as a duplicate | Use distinct file names |
| A named font with no local file | Every named font needs a local `@font-face` |

- **Automatos home:** D4 and S1.2 (templates as data). The three compositions become the **seed templates**:
  - "UI story promo" (v1);
  - "cinematic product promo" (v2), with clip slots;
  - "app promo" (Academy), with a phone frame.

  The copy, palette, fonts, clip slots, voice lines and music cue become variables.

### Stage 8 — QA (automatic and visual)
- `hyperframes check` must pass: lint, runtime, layout, motion and WCAG contrast, 63/63 text checks on the Academy promo.
- `hyperframes snapshot --at t1,t2,…` produces contact sheets that were **looked at**. They caught three problems the checks missed:
  - the tagline over a bright-orange laptop screen (v2);
  - the brain icon landing on a person's head (Academy);
  - a label sitting on the bright part of the brain.
- The MP4 is verified with `ffprobe` (streams, duration, fps) and `ebur128` (loudness), plus frames pulled from the final file.
- **Automatos home:** S1.1. The render service runs `check` before `render` and returns its report. The composer shows snapshot frames before approval.

### Stage 9 — Render
- **Command:** `hyperframes render --quality delivery --fps 30 --output …` on Node ≥ 22. It uses Puppeteer with chrome-headless-shell, downloaded automatically into `~/.cache/hyperframes/chrome`, plus the system ffmpeg.
- **Times:** 2 m 42 s to 4 m 30 s for 38–40 s at 1080×1920, with hardware GPU and screenshot capture. Output is 19–32 MB of H.264 + AAC.
- **Environment:** always set `HYPERFRAMES_NO_TELEMETRY=1`, `DO_NOT_TRACK=1` and `HYPERFRAMES_SKIP_SKILLS=1`. The CLI sends PostHog telemetry by default, and `init` otherwise installs skills globally.
- **Trap:** on macOS the first launch of a freshly downloaded Chrome timed out while the OS scanned it. The retry worked. Linux containers should pre-warm it.
- **Automatos home:** D3 `media-render` as an async job, and S1.1.

### Stage 10 — Static posts
- HTML at 1080×1350 (4:5) over a still, using brand-kit CSS variables. **All text is HTML.**
- Screenshotted with chrome-headless-shell: `--headless --screenshot=out.png --window-size=1080,1350 --virtual-time-budget=4000 --allow-file-access-from-files`.
- Visual review caught one miss: the client brand name sat on busy foliage, fixed with a cream pill.
- **Automatos home:** S1.2 social image templates, rendered by `media-render`, or by the existing `workspace_html_to_png` worker path for simple cards.

### Stage 11 — Publish (not run in this spike)
- It goes through Composio per PRD-251 D8:
  - Instagram needs public HTTPS media URLs (D9 presigned inline URLs).
  - TikTok: upload, then publish or publish from a URL, then a status check.
  - YouTube: a file upload.
  - LinkedIn: only after S0.4 fixes the cross-tenant credential defect.

---

## 3. Cost model (measured unit prices)

| Unit | Cost | Notes |
|---|---|---|
| **Template-only video** (v1 style) | **$0** external | Render compute only: about 3 min of an 8-core machine |
| **Cinematic shot**, Cinema Studio 720p 9:16 | **$0.462/s**: $1.85 for 4 s, $2.31 for 5 s | Four shots per video came to about $7.86–$8.32 |
| **Post background**, Soul Cinema or Soul 2 | **$0.006**, or $0.0055 in a batch of 4 | A batch of 4 gives the customer a choice |
| **Product ad from the customer's photo**, Marketing Studio edit | **$0.255** | Keeps the product; new scene |
| **Voice**, Kokoro | **$0** | Local |
| **Voice**, fish.audio `s2.1-pro` | about **$0.005** per 40 s script | $15 per million UTF-8 bytes; a free tier exists |
| **Voice**, ElevenLabs | Depends on plan | The customer's key |

**One small business for a month**, using the owner's target of "a few cool ads a month, but daily posts":

| Plan | Contents | External cost |
|---|---|---|
| **Images only** | 30 daily posts (8 batches of 4, pick 1 each) ≈ $0.18 · 8 product ads from their photos ≈ $2.04 | **≈ $2.20/month** |
| **Plus template videos** | Add 3 template-only promo videos at $0 | **≈ $2.20/month** |
| **Plus light cinematic** | 3 videos with one 5 s AI shot each ≈ $6.93 | **≈ $9/month** |
| **Full cinematic** | 3 videos with four AI shots each ≈ $24 | **≈ $26/month** |

Rendering compute on `media-render` is extra and internal.

---

## 4. Where each step goes in Automatos (PRD-251)

| Pipeline stage | Automatos component | PRD-251 |
|---|---|---|
| Brand extraction | Workspace brand kit, extended with font files, mark, handles, voice, disclaimers | D5 · S1.3 |
| Claims and sources | Claim slots with a source picker; unsourced claims need an override | D7 · S1.4 |
| Script | Auto drafts it in the composer; the weekly Playbook drafts the week | S2.2 · S4.2 |
| Voice | Kokoro by default, or Fish Audio / ElevenLabs through the workspace's Composio connection. No recordings. | D11 · S1.5 · D15 |
| Music | Library with licence, attribution and precomputed drop/groove cues | S1.6 |
| Footage and stills | The workspace's Composio generation toolkit (fal.ai, Kie.ai, Higgsfield MCP) through a per-toolkit recipe. The Appendix client's shape (estimate → submit → poll → download, spend cap) becomes that recipe; it is no longer our own provider client | D12 · D13 · D16 · S1.8 · S4.4 |
| Composition | Templates as data; the three seed templates above | D4 · S1.2 |
| QA | `check` plus snapshots inside the render job; previews in the composer | S1.1 · S2.2 |
| Render | `media-render` service: Node 22, Chromium, ffmpeg, Hyperframes pinned, Kokoro | D3 · S1.1 |
| Posts | Social image templates | S1.2 |
| Approval | Per post, bound to the content hash | D6 · S2.3 |
| Publish | Composio adapters and the capability registry | D8 · S3.2 · S3.3 |

**Proposed additions to PRD-251, from this spike.** Number 1 was adopted on 2026-09-23 as D13's `socials.media_monthly_cap_usd`; the rest await the owner's decision.
1. A monthly cap on AI spend for Socials per workspace, on top of the workspace budget. *(Adopted: D13.)*
2. Format rotation in the weekly Playbook: mix cards, carousels, fact cards and one short video a week.
3. A reuse library for paid media. Every generated clip and still becomes a tagged Deliverable that later posts can reuse, so the $8 of v2 footage could feed a month of posts.
4. Marketing Studio product-photo editing as a first-class composer action: "Upload product photo → N ad scenes".

---

## 5. Traps (all hit in this spike)

**Voice and audio**
- espeak-ng truncates its data path at about 160 characters, and `phonemizer` resolves symlinks. Keep the data at a short real path.
- `phonemizer` and espeak-ng are GPL-3.0. Keep them in a separate process.
- Trending songs are not licensed for business promos. CC BY music needs its credit line.

**Higgsfield**
- Cinema Studio and the Marketing Studio token-metered variants return **pricing descriptions**, not numbers, from the estimate endpoint. Compute or cap the cost locally.
- Output URLs are public and live for at least 7 days. Copy them into our storage straight away.
- Cinema Studio tops out at 720p; scale up under crisp HTML graphics. Clips are 24 fps; the render is 30.
- Generated images invent text even when asked not to (Soul 2, Soul standard, some batch variants). Never rely on generated text; overlay it.

**Hyperframes**
- It needs Node ≥ 22. The machine default here was Node 20, so the Node 22 already installed under nvm was used per command.
- It sends telemetry by default, and `init` installs skills globally unless `HYPERFRAMES_SKIP_SKILLS=1` is set.
- A freshly downloaded Chrome can time out on first launch under macOS.
- The composition rules in Stage 7 each broke a check once.

**Content**
- Parked or unsourced numbers must never reach a render.
- Visual review catches what the automatic checks miss: busy backgrounds behind text, icons landing on faces, labels on bright areas.

**Keys**
- A customer's paid tools are connected in Composio (D15), never pasted into chat as keys. The demo key for this spike was pasted in chat. It lived in a mode-600 file outside the repos while in use and was deleted after each run; the owner revokes it.

---

## Appendix — reference implementation (outside the repo; paths are relative to the workspace root)

- **Compositions:**
  - `brag-output-2026-09-22-225946/composition/index.html` (v1)
  - `brag-output-2026-09-23-v2/composition/index.html` (v2)
  - `brag-output-2026-09-23-academy/composition/index.html` (Academy)
- **Higgsfield client** with estimate, token-metered fallback, spend cap, submit, poll and download: `brag-output-2026-09-23-images/scripts/hf.py`. Jobs are JSON lists of `{name, endpoint, args, out}`, run as `python hf.py estimate|run jobs.json --cap <usd> [--only a,b]`.
- **Mixes:**
  - `brag-output-2026-09-23-v2/composition/scripts/mix_v2.py`
  - `brag-output-2026-09-23-academy/composition/scripts/mix_academy.py`
  - `brag-output-2026-09-22-225946/composition/scripts/mix.py`
- **Voice:** `…/composition/scripts/synth_vo.py`, with the lines in `vo_lines.json`.
- **Posts:**
  - `brag-output-2026-09-23-v2/posts/post*.html`
  - `brag-output-2026-09-23-academy/posts/post*.html`
  - `brag-output-2026-09-23-images/posts/client*.html`
- **Comparison sheet:** `brag-output-2026-09-23-images/compare.html`, rendered to `image-model-comparison.png`.

When PRD-251 is built, these files move into `media-render` and the template seeds, and the workspace-root copies are deleted (automatos-ai `CLAUDE.md` §5: delete what's superseded).
