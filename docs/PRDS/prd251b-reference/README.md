# PRD-251B visual spec

The mockup behind PRD-251B, as built on 2026-10-02. Live copy: <https://claude.ai/artifact/T4W1hez7DD8u5gyshyaXTQ> (a Design canvas; each screen has a Play button).

| File | Screen | Stories |
|---|---|---|
| `Main.dc.html` | Socials calendar, the tab's home: month grid, channel filter, Today rail, plan card, status key | US-B107, US-B108, US-B208 |
| `Editor.dc.html` | Post editor: brief, format, length, channels and sizes, look, claims, when, live preview per channel | US-B109, US-B110, US-B305 |
| `Queue.dc.html` | Approval queue: today's posts with deadlines, exact media, copy, sources, actions | US-B111 |
| `Plan.dc.html` | Plan: goal and dates, cadence, research sources, making and approving, content bank | US-B207 |
| `Brand.dc.html` | Brand kit: basics, style references with like/avoid, style profile, AI tools and caps | US-B301–US-B304 |
| `Nav.dc.html` | Shared header and Socials sub-navigation | US-B107 |
| `studio.css` | Shared styles of the mockup: intent only (spacing, colour roles), not to be copied | all |
| `TOKENS.md` | Each `studio.css` role mapped to the app token or `components/ui` component to build it with (binding, B1) | all |

Each `.dc.html` is a self-contained HTML page with its data and handlers in the `<script type="text/x-dc">` block at the end, so the copy, the states and the sample data are all readable as text.

The sample content is illustrative and not product data:
- the Web Summit countdown plan;
- "today" as Wed 14 Oct;
- the topic titles and the placeholder reference images.

Build with the app's components and tokens (PRD-251B B1): `TOKENS.md` names the token for every role; two owner choices (the accent shade, orange primaries) are recorded there with their defaults.
