# Article 50(2): Code of Practice gap analysis

Automatos has not signed the Code of Practice on transparency of AI-generated content (final, Jun 2026). Signing is voluntary (Art. 50(7)). A provider that doesn't sign still has to show by other means how it meets Article 50(2). This document is that account: for each commitment the Code asks of its signatories, it says what Automatos does, what it doesn't, and why the marking it does build is effective, interoperable, robust and reliable as far as technically feasible.

Gerard decided on 19 Oct 2026 not to sign before Web Summit and to write this analysis instead (PRD-264 D12), and not to build an imperceptible watermark now (D11). Signing is revisited after Web Summit (below).

It describes what PRD-264 (`docs/PRDS/PRD-264-EU-AI-ACT-ARTICLE-50.md`) builds. It claims no conformity or certification, and it is not legal advice. Counsel confirms the legal reading (D9).

## Status: what is live

**On 19 Oct 2026 none of the PRD-264 marking or detection code was on `main`.** Every measure below is live only once its story is merged and deployed. The stories ship in waves EU-B1 (EUAI-06a, target merge Fri 23 Oct), EU-C2 (EUAI-09, EUAI-10, target Thu 29 Oct) and EU-B2 (EUAI-06b, target Fri 30 Oct). The posture page at automatos.app/eu-ai-act lists what has been checked in production (EUAI-15) and links this analysis.

## The legal frame

As cited in PRD-264 §1:

- **Article 50(2):** generated audio, images, video and text are marked in a machine-readable format and are detectable as AI-generated. The technical solutions must be effective, interoperable, robust and reliable as far as technically feasible.
- **Detection** is available to the people exposed to the content (Guidelines C(2026) 5054 final, ¶75). **Marks carry no personal data** (¶94). Short strings such as UI labels, captions and alt text, and source code, are out of scope (¶68).
- **Dates.** 2 Aug 2026 for systems placed on the market after that date: media-render first shipped on 25–26 Sep 2026, so its marking is due now. **2 Dec 2026** for systems already on the market before 2 Aug (the AI Omnibus, Regulation (EU) 2026/1744): chat, the widget, images and documents. Automatos targets Fri 30 Oct 2026 for all of them (D1).
- **The Code** (Art. 50(7)): signatories commit to signed metadata plus an imperceptible watermark, and to interoperable detection, by **2 Feb 2027**. The Code's own text is the reference; this is PRD-264's summary of it.

## What Automatos builds, in layers

| Layer | What it is | Covers | Story |
|---|---|---|---|
| 1. Metadata mark | The IPTC Digital Source Type term in each format's standard metadata location: XMP in MP4, PNG, JPEG, WAV and PDF; custom document properties (`AIGenerated`, `AIInstanceID`) in DOCX and XLSX. `trainedAlgorithmicMedia` when every element is AI-made, `compositeWithTrainedAlgorithmicMedia` when AI and non-AI elements are mixed. Every mark carries a random per-file instance id (FR-16) and nothing else. | media-render outputs; generated documents; model images | EUAI-06a, EUAI-09 |
| 2. Signed manifest | A C2PA manifest with a `c2pa.actions` assertion (`c2pa.created`, the same digital source type) and the claim generator "Automatos AI media-render/{version}". An upstream manifest is kept as an ingredient. | media-render outputs | EUAI-06b |
| 3. Provenance record and public check | One `content_provenance` row per marked file (SHA-256, kind, date; the workspace for tenancy and erasure only). `/verify` hashes a file in the visitor's browser and asks `GET /api/public/provenance/{sha256}`, which answers yes or no, the kind and the date. See [DETECTION.md](./DETECTION.md). | Every file that gets a mark and a row | EUAI-10 |
| 4. Payload flags for text | `"ai_generated": true` on assistant messages in the in-app chat stream and the widget stream, `data-ai-generated="true"` in the page, `ai_generated` in the blog APIs and on the generic webhook driver. | Chat answers, widget answers, blog posts, webhook messages | EUAI-09, EUAI-02a, EUAI-03 |
| 5. Visible and platform labels (supporting) | Not 50(2) marks, but they carry the disclosure where metadata is stripped: each network's AI flag at publish, the Socials "AI-generated" label, the blog "AI-generated" label, and the 50(1) notices. | Social posts, blog posts, conversations | EUAI-05, EUAI-08a, EUAI-08b, EUAI-08c |

## Commitment by commitment

### Signed metadata

- **Does:** media-render's videos, stills, preview frames, JPEGs, the audio mix and voice lines carry the IPTC XMP mark (EUAI-06a) and a signed C2PA manifest (EUAI-06b, not yet built on 19 Oct). If signing fails, the file still ships with its XMP mark, and the failure is logged and counted: a render never ships unmarked. Generated PDFs, DOCX, XLSX, PNG and JPEG carry the IPTC mark (EUAI-09). An image that already carries a C2PA manifest from the model that made it is kept byte for byte, so its signed manifest survives.
- **Doesn't:** documents and model images from EUAI-09 are **not signed**. Their mark is unsigned metadata: anyone can read it, and anyone can also forge or remove it. Until the C2PA trust-list certificate arrives (D15; the application is due on Tue 20 Oct), manifests are signed with a self-signed certificate, so verifiers show "unknown signer". They are still machine-readable and detectable, and the certificate is swapped by configuration with no code change.
- **Gap:** signing for documents and model images. Closing it means a C2PA signer for those formats in the orchestrator, which is a new dependency and needs a maintainer issue first (AGENTS.md).

### Watermark

- **Does:** nothing. No imperceptible watermark is built (D11, taken 19 Oct 2026).
- **Doesn't:** a mark that survives re-encoding, cropping, screenshots or a social network's upload pipeline. Every mark Automatos writes is metadata, and metadata is lost when a file is re-encoded or its metadata stripped.
- **Why this is still the 50(2) reading Automatos takes:** Article 50(2) asks for marking that is effective, interoperable, robust and reliable as far as technically feasible. It doesn't name a watermark; the watermark is a Code commitment. What carries the disclosure when metadata is lost: the provenance record still answers "yes" for the original file; the network's own AI flag is set at publish (EUAI-05); and the visible labels (EUAI-08a, EUAI-08b) are part of the content itself.
- **Gap:** **this is the main gap against the Code.** Closing it is EUAI-13: an open-source audio watermark (AudioSeal-type) and image watermark (TrustMark-type) in media-render, chosen in a maintainer issue, with the detection route reading the watermark too. If Gerard signs the Code after Web Summit, EUAI-13 is scheduled for Tue 2 Feb 2027.

### Detection

- **Does:** a public page, `/verify`, in both editions, with no account. The browser computes the file's SHA-256 and sends only the hash; the file never leaves the visitor's machine. The answer is yes or no, the kind and the date, and never the workspace, person, prompt or file name (D16). It is rate-limited per IP. The metadata marks and C2PA manifests can also be read without Automatos, with any XMP reader (exiftool, for example) or any Content Credentials tool (contentcredentials.org/verify, linked from `/verify`).
- **Doesn't:** the hash check proves an exact match only. An edited or re-encoded copy has a new hash and reads "no", and "no" does not prove that AI wasn't involved. Chat answers are marked in the payload and not recorded, so a chat answer can't be looked up by hash. A local-edition instance answers only for the files it made itself. With no watermark, a stripped copy can't be detected at all.
- **Gap:** detection of altered copies, which needs the watermark (EUAI-13).

### Interoperability

- **Does:** the marks use published vocabularies and containers that other tools already read: the IPTC Digital Source Type vocabulary, XMP for media, images and PDF, and C2PA manifests for media-render files. The detection route is a documented, unauthenticated HTTP interface ([DETECTION.md](./DETECTION.md)), so anyone can call it from their own tools.
- **Doesn't:** DOCX and XLSX have no standard AI field that the pinned libraries write, so their mark is a custom document property with Automatos's own names (`AIGenerated`, `AIInstanceID`). Generic tools show it as a custom property, not as an AI mark. WebP and GIF images carry no mark and get no provenance row: no library the platform ships writes their XMP without re-encoding the picture. The hash route is Automatos's own interface; it is not part of a shared, cross-provider detection protocol.
- **Gap:** a shared detection interface with other providers, should the Code's signatories settle on one; a standard mark for office documents; marking for WebP and GIF.

### The 2 Feb 2027 date

- **Applies to:** Code signatories. Automatos is not one, so the date is not an Automatos obligation today.
- **Where Automatos stands against it:** the metadata, signing and detection layers are targeted for Fri 30 Oct 2026. That is ahead of the Code's date and of the 2 Dec 2026 legal date for systems already on the market; media-render's legal date (2 Aug 2026) has already passed, so its marking is overdue until EUAI-06a and EUAI-06b are live. The watermark is not built.
- **If Gerard signs after Web Summit:** EUAI-13 is scheduled for Tue 2 Feb 2027, and the other gaps in this analysis get dates of their own in the same decision.

## Text that is marked differently, or not at all

Article 50(2) covers text as well as media. How Automatos's text outputs stand:

| Output | Machine-readable mark | Visible notice | Note |
|---|---|---|---|
| In-app chat answers | `ai_generated` in the stream, `data-ai-generated` in the page (EUAI-09) | "Auto is an AI" (EUAI-11a) | Not recorded for the hash check |
| Widget chat answers | `ai_generated` on every assistant message event (EUAI-02a) | Badge and first-turn line (EUAI-02a, EUAI-02b) | — |
| Generated documents | IPTC term in the file's metadata (EUAI-09) | — | Hash check answers for the file |
| Blog posts by an agent | `ai_generated` in the blog APIs (EUAI-09, EUAI-08b); JSON-LD on the post page (EUAI-08c) | "AI-generated" until reviewed (EUAI-08b, EUAI-08c) | — |
| Channel bot messages | `ai_generated` and `ai_disclosure` on the generic webhook driver only (EUAI-03) | "AI assistant for {principal}." (EUAI-03) | **Gap:** no machine-readable mark inside a Telegram, Slack, Discord or WhatsApp message |
| Email and messages an agent sends | None | "Written and sent by an AI agent on behalf of {principal}." at the top (EUAI-07) | **Gap:** no machine-readable mark in the email itself (a header, for example) |
| Social captions | None | "AI-generated." at the start when the deep-fake label is on (EUAI-08a) | ¶68 puts short strings such as captions out of scope; whether a full social caption counts is a reading for counsel (D9) |
| Auto Live voice audio | None (streamed audio, never a file) | Spoken greeting and reminders (EUAI-04) | **Gap:** live voice audio carries no machine-readable mark |

## Effective, interoperable, robust and reliable

- **Effective.** The mark is in the place each format keeps its metadata, in the term detectors read, on every file Automatos makes with an AI element. A document a person fills in by hand, or a render made only from templates and the person's own uploads, gets no mark, so the mark means what it says.
- **Interoperable.** Published vocabularies and containers (IPTC, XMP, C2PA), readable by third-party tools, plus a documented public route. The custom office-document properties and the Automatos-only route are the limits.
- **Robust.** Marks are written after the last encode, so the platform's own processing doesn't drop them (media-render's JPEG step re-marks its output). An upstream C2PA manifest is kept, never stripped (FR-9). A copy edited or re-encoded outside Automatos loses the metadata. That is the limit of metadata, and the reason the watermark gap is named above.
- **Reliable.** CI reads each mark back from a real output file for every format: media-render's formats with exiftool (EUAI-06a) and the C2PA reader (EUAI-06b); PDF, DOCX, XLSX, PNG, JPEG and one SSE message for EUAI-09. A document that can't be marked is deleted and the error raised, so it is never delivered unmarked; an image that can't be read as its type is not kept. Two files made from identical input never share a hash (FR-16), so the public check can't confirm a predictable file by guessing.
- **No personal data.** The marks carry the IPTC term and a random instance id only (¶94). Manifests carry no user or workspace identifier. The public answer never reads the workspace, and the route itself logs no hash. PRD-264 FR-17 also asks that the answers carry `Cache-Control: no-store` and that the request-metrics path collapses the hash; on 19 Oct 2026 neither was built (gap 9 below).

## The gaps in one list

1. **No imperceptible watermark** (D11). EUAI-13, scheduled for Tue 2 Feb 2027 if Gerard signs the Code.
2. Documents and model images carry **unsigned** metadata; only media-render files are C2PA-signed.
3. **Self-signed** C2PA certificate until the trust-list certificate arrives (D15).
4. DOCX and XLSX marks are **custom properties**, not a standard AI field.
5. **WebP and GIF** images carry no mark.
6. No machine-readable mark in **channel bot messages** (other than the webhook driver), **agent-sent email**, or **live voice audio**.
7. Detection matches **exact files only**; altered copies read "no".
8. The detection route is **Automatos's own interface**, not a shared cross-provider protocol.
9. The public route's answers don't yet carry `Cache-Control: no-store`, and request metrics don't yet collapse the hash in the path (PRD-264 FR-17). The server's access log records the path, which holds the hash, as it does for every route ([DETECTION.md](./DETECTION.md)).

## Revisit

Signing the Code is revisited on **Mon 16 Nov 2026**, the first working day after Web Summit (Lisbon, 10–12 Nov 2026). Gerard decides then whether to sign. If he signs, EUAI-13 is scheduled for Tue 2 Feb 2027 and each gap above gets a date. If he doesn't, this analysis is reviewed again and updated whenever a gap closes or a PRD-264 story's state changes.

Related: [ARTICLE-50.md](./ARTICLE-50.md) (what Automatos provides and what the deployer does) · [DETECTION.md](./DETECTION.md) · [README.md](./README.md)
