# Article 50: what Automatos provides and what the deployer does

Article 50 of the EU AI Act (Regulation (EU) 2024/1689) puts transparency duties on two parties. The **provider** of an AI system is Automatos. The **deployer** is the customer who uses Automatos in their own business, and also Automatos itself for its own sites and posts. This guide lists, surface by surface, what Automatos builds for Article 50 and what the deployer still has to do.

It describes what PRD-264 (`docs/PRDS/PRD-264-EU-AI-ACT-ARTICLE-50.md`) builds. It is not legal advice, and it claims no conformity or certification. Counsel confirms the legal reading (PRD-264 D9).

## Read this first: what is live

**On 19 Oct 2026 none of the PRD-264 code was on `main`.** Each row below names the story that builds it and the wave it ships in. A notice or a mark is live only once its story is merged **and** deployed. Until then, the deployer adds the disclosure in their own copy.

The public record of what is live in production is the posture page at automatos.app/eu-ai-act. It is updated only after each notice and mark has been checked in production (PRD-264 EUAI-15). Where this guide and the posture page differ, the posture page is the current state.

## The duties in short

The dates and Guidelines paragraphs are as cited in PRD-264 §1. The Guidelines are the Commission's Article 50 Guidelines, C(2026) 5054 final (20 Jul 2026). The AI Omnibus, Regulation (EU) 2026/1744, left Article 50 in place and added one grace period, for 50(2) only.

| Article | Duty | Whose duty | Applies from |
|---|---|---|---|
| 50(1) | A person is told they are dealing with an AI, unless that is obvious. An AI agent says it is an AI and on whose behalf it acts (¶31). A notice only in the Terms, a URL, documentation, a brand name or metadata is not enough (¶38). | Provider | 2 Aug 2026 |
| 50(2) | Generated audio, images, video and text are marked in a machine-readable format and are detectable as AI-generated. The people exposed to the content can detect it (¶75). Marks carry no personal data (¶94). Short strings such as UI labels, captions and alt text, and source code, are out of scope (¶68). | Provider | 2 Aug 2026 for systems placed on the market after that date; 2 Dec 2026 for systems already on the market |
| 50(3) | Emotion recognition and biometric categorisation: inform the people exposed | Deployer | 2 Aug 2026 |
| 50(4) ¶1 | Deep fakes (realistic AI images, audio or video of real people, places or things, and cloned voices) are disclosed | Deployer | 2 Aug 2026 |
| 50(4) ¶2 | AI text published to inform the public on matters of public interest is disclosed, unless a person reviewed it and holds editorial responsibility | Deployer | 2 Aug 2026 |
| 50(5) | The notice is clear and distinguishable, given at the latest at first exposure, and accessible | Provider and deployer | 2 Aug 2026 |

The open-source exemption (Art. 2(12)) does not cover Article 50, so the local edition (`AUTH_EDITION=local`) carries the same notices and marks as the hosted one (`saas`).

## Surface by surface

"Provides" is what PRD-264 builds into the product. "The deployer" is what the customer does. No workspace setting, widget setting, theme, persona or environment flag turns an Automatos notice off (PRD-264 FR-2).

### Article 50(1): telling people they are dealing with an AI

| Surface | Automatos provides | The deployer | Story · wave |
|---|---|---|---|
| Chat widget (server side) | The widget's public config always carries `ai_disclosure`: a fixed badge (`AI`; `IA` for Portuguese, Spanish, French and Italian; `KI` for German) and a first-turn line, "You are chatting with an AI assistant for {site}.", where `{site}` is the Site's public `display_name`. The first event of the widget chat stream is the notice, before any token. | May reword the first-turn line. The server refuses a line that does not contain the locale's AI term (D17). The badge can't be changed. | EUAI-02a · EU-A1 |
| Chat widget (SDK), proactive and cart-idle popups | A badge beside the title that a `title` override doesn't remove; the notice as the opening turn; "AI assistant" in both popups; aria labels that name the AI. An old server without `ai_disclosure` gets the English default, never an unlabelled widget. | **A custom widget client** (not the Automatos SDK) must render `ai_disclosure` itself: the badge persistently and the line before the first answer. Don't cover or hide the badge with site CSS. | EUAI-02b · EU-W (automatos-widget-sdk) |
| Website chats and the Shopify theme embed | They load the published widget, so they get EUAI-02b with no change of their own. | Nothing beyond the widget row. | EUAI-02b · EU-W |
| In-app chat (`/chat`, Mission Mode, the Play chat, a theme-owned `/chat`, Play's other chat surfaces) | "Auto is an AI" (or "{agent} is an AI"), rendered by the host outside the theme's slot. If a theme hides it, the chat input is paused. Every agent's system prompt ends with a rule to answer truthfully that it is an AI, appended after the persona. | **A theme author** can't remove, cover or restyle the label away; a theme that hides it pauses the chat. | EUAI-11a, EUAI-11b · EU-A2 |
| Auto Live voice | The call opens with "Hi, this is Auto, your AI assistant." before the caller speaks. A spoken reminder after a reconnect and at an interval (`VOICE_AI_REMINDER_MINUTES`, default 10). "AI voice assistant" on the call panel for the whole call. | Nothing. A self-hosted instance may change the greeting (`VOICE_AI_GREETING`); one that doesn't say it is an AI is replaced by the default. | EUAI-04 · EU-A1 |
| Telegram, Slack, Discord and WhatsApp bots | "AI assistant for {principal}." before the first reply to a person, again after an idle gap (`AI_DISCLOSURE_IDLE_HOURS`, default 24), and when someone asks if they are talking to a human. Both outbound paths (adapters and drivers) carry it. The generic webhook driver gets `ai_generated` and `ai_disclosure` payload fields instead. | **A webhook receiver** that shows the text to a person must show the `ai_disclosure` field with it. Set a business profile `company_name` so the notice names you (FR-15). | EUAI-03 · EU-A1 |
| Email and messages an agent sends | The first line of the body: "Written and sent by an AI agent on behalf of {principal}." It is added before the owner's send card is built, so the owner approves the final text. | Don't remove the line from approved sends. Sends a person writes directly are unchanged. | EUAI-07 · EU-C1 |
| Academy web and app tutors | "AI tutor" in the header and first turn; the tutor prompt says it is an AI. | Not a deployer surface (Automatos's own product). | EUAI-11c, EUAI-11d · automatos-academy, automatos-academy-app |

`{principal}` is the business profile's `company_name`, then the workspace name only when it is not the auto-created "{name or email}'s Workspace" form and contains no "@". Otherwise the notice names no one (PRD-264 FR-15), because a workspace name can carry the owner's name or email.

### Article 50(2): marking generated content and making it detectable

| Content | Automatos provides | The deployer | Story · wave |
|---|---|---|---|
| media-render videos, stills, preview frames, JPEGs, the audio mix and voice lines | IPTC `DigitalSourceType` in XMP after the last encode (`trainedAlgorithmicMedia`, or `compositeWithTrainedAlgorithmicMedia` when AI and non-AI elements are mixed); then a signed C2PA manifest. Renders made only from templates and the person's own uploads get no AI mark. | Don't strip the metadata before publishing. A file you edit or re-encode yourself is a new file: it no longer carries Automatos's marks and the public check has no record of it. | EUAI-06a · EU-B1; EUAI-06b · EU-B2 |
| Generated PDFs, DOCX and XLSX | PDF: IPTC `DigitalSourceType` in the XMP metadata. DOCX and XLSX: the custom document properties `AIGenerated` (the IPTC term) and `AIInstanceID`. A provenance row for the public check. A document a person fills in by hand is not marked. | Don't remove the properties. | EUAI-09 · EU-C2 |
| Generated images (the image tool, images in chat answers) | PNG and JPEG get the IPTC XMP mark and a provenance row. An image that already carries a C2PA manifest from its model is kept byte for byte. WebP and GIF are not marked (see the gap analysis). | Don't strip the metadata. | EUAI-09 · EU-C2 |
| Chat answers (in-app) | `"ai_generated": true` on the frames that open and close each assistant message, and `data-ai-generated="true"` on the rendered message. Chat text is not recorded for the public check. | Nothing. | EUAI-09 · EU-C2 |
| Widget chat answers | `"ai_generated": true` on every assistant message event. | A custom client keeps the flag where it passes messages on. | EUAI-02a · EU-A1 |
| Blog posts an agent wrote | The blog APIs return `ai_generated: true` for a post with an agent author. EUAI-08b makes it "agent-written and not marked reviewed". The blog widget shows "AI-generated" and emits JSON-LD (EUAI-08c). | See 50(4) ¶2 below. | EUAI-09 · EU-C2; EUAI-08b · EU-C1; EUAI-08c · EU-W |
| Detection | A public page, `/verify`, that hashes a file in the browser and asks `GET /api/public/provenance/{sha256}` whether Automatos made it. No account, no upload. See [DETECTION.md](./DETECTION.md). | Point people who ask to `/verify`. The local edition answers for the files it made itself. | EUAI-10 · EU-C2 |

How the marking stands against the Code of Practice, and where it falls short (no watermark, D11), is in [ARTICLE-50-GAP-ANALYSIS.md](./ARTICLE-50-GAP-ANALYSIS.md).

### Article 50(4): the deployer's duties, and the tools Automatos gives for them

These duties are the deployer's by law. Automatos provides tools and defaults; the deployer decides and stays responsible (PRD-264 §7).

| Duty | Automatos provides | The deployer | Story · wave |
|---|---|---|---|
| Deep fakes (¶1) | Socials: "Realistic AI people, places, products or voice: label as AI-generated", on by default when a post carries AI media. When on, a visible label on the video at the start and at an interval, "AI-generated." at the start of the caption, and a spoken line for audio-only posts. Each network's own AI flag is set at publish where its API has one (TikTok `is_aigc`, and YouTube if the publish action accepts it); the receipt says what was set for each channel. | Decide per post. Switching the label off makes disclosure yours ("you are responsible for disclosing deep fakes"). Disclose deep fakes you publish anywhere else. | EUAI-08a · EU-C1; EUAI-05 · EU-B1 |
| Public-interest text (¶2) | Agent-written blog posts are labelled "AI-generated" until a workspace member marks them reviewed; editing clears the review. | Mark a post reviewed only when a person has reviewed it and holds editorial responsibility. Disclose AI text you publish on public-interest matters outside the blog widget. | EUAI-08b · EU-C1; EUAI-08c · EU-W |
| Emotion recognition, biometric categorisation (50(3)) | Automatos has no such feature. | If you build one with Automatos, informing the people exposed is yours. | — |

## What every deployer does

- **Keep the notices.** Don't remove, cover or hide an AI notice or badge: not with CSS, a theme, a custom client or a template. You may reword the widget's first-turn line within the locale rule (D17).
- **Don't strip the marks.** Don't remove the XMP, C2PA or document-property marks from files Automatos made. If you edit or re-encode a file, the result is a new file that no longer carries them; how you disclose it is then your decision.
- **Disclose what the law puts on you:** deep fakes and public-interest text you publish (50(4)), and any emotion-recognition or biometric-categorisation use (50(3)).
- **Name yourself.** Set the business profile's company name, so channel bots and agent sends say on whose behalf they act.
- **Until a notice is live** (the posture page lists it), add the disclosure in your own copy.

## Not covered here

- **The Terms.** The Terms clauses (no removing or hiding notices and marks; 50(3) and 50(4) duties are the deployer's) ship in the product site's and the company site's Terms in wave EU-D (EUAI-12), with the legal sign-off of D9 recorded there. The GitBook page with the same content ships in the same wave.
- **An imperceptible watermark.** Not built (D11). See the gap analysis.
- **High-risk obligations** (Annex III from 2 Dec 2027, Annex I from 2 Aug 2028). PRD-132.

Related: [DETECTION.md](./DETECTION.md) · [ARTICLE-50-GAP-ANALYSIS.md](./ARTICLE-50-GAP-ANALYSIS.md) · [README.md](./README.md)
