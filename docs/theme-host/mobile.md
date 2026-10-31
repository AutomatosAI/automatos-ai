# The phone's theme: the `mobile` block and its answer

PRD-261 M-01..M-04 (host contract 1.1.0) and PRD-271 W1 (host contract 1.2.0: a theme may pick one of the
phone's layouts). A phone app draws a member's theme from data alone
(Automatos Go reads this answer, and `GET /api/themes/active`, when it opens and each time it returns to the
foreground; the discovery document lists it as the `mobile_theme` capability). It never loads a theme's
`dist/entry.js` or `dist/theme.css`: an app may not download and run code (App Store guideline
2.5.2). A theme says how it looks on a phone in an optional `mobile` block in
`theme.json`, and the phone reads the result from one route.

## The block in `theme.json`

```jsonc
"mobile": {
  "tokens": { "light": { "--background": "0 0% 100%", "--foreground": "#14181c" }, "dark": { "…": "…" } },
  // or, instead of "tokens":
  "fromLooks": true,
  "fonts": { "sans": "assets/fonts/mobile/inter-regular.ttf", "serif": "…", "mono": "…" },
  "mascot": "clips",
  // PRD-271 (host 1.2.0), optional: one of the phone's layouts and its contract's slots
  "layout": "candy@1",
  "slots": { "page": "#FFF4F9", "accent": { "look": "--bubble-accent" }, "roundness": { "look": "roundness" } }
}
```

| Field | Rule (checked on import by `services/themes/mobile_manifest.py` and its TypeScript twin) |
|---|---|
| `tokens` or `fromLooks` | Exactly one. `fromLooks` is only ever `true`. |
| `tokens.light`, `tokens.dark` | `light` is required, `dark` optional. Keys are Studio's 24 colour tokens (`MOBILE_TOKEN_NAMES` in `services/themes/contract.py`), spelled with the leading `--`, and must include `--background` and `--foreground`. Each value is `H S% L%` (such as `0 0% 100%`) or `#rrggbb`. |
| `fromLooks` | The member's look is painted through `boot/prepaint.json`, which the package must have, and the table's `defaults`, painted as the web paints them (below), must give `--background` and `--foreground` as `H S% L%` triplets. A Studio token written `#rrggbb` in the table paints nothing, on the web or the phone. Defaults that don't fit the table would paint nothing either, so the reader refuses that table whatever the block (`PREPAINT_DEFAULTS_UNFIT`, [Theme host](_index.md) §1). |
| `fonts` | Any of `sans`, `serif`, `mono`; each a different `assets/fonts/mobile/<name>.ttf` or `.otf` in the package (one sfnt face, at most 400 KB). |
| `mascot` | `clips`, `stills` or `none` (absent means `none`). `clips` needs each state of the top-level `mascot.states` to have `assets/mascot/<state>.mp4` and `.png`; `stills` needs the `.png`. |
| `layout`, `slots` | Optional, and together (PRD-271). `layout` names a phone layout and its contract major, `"<id>@<major>"`: one of [the contracts](#the-phone-layouts-and-their-contracts). `slots` fills that contract: no slot it lacks, every slot it requires, each value of its slot's kind or `{"look": …}` (below). The Studio skin (`tokens` or `fromLooks`) stays required: an app without the layout draws it. |
| `engines.automatos` | Must not admit a host older than 1.1.0, or 1.2.0 when the block has `layout` (an older host refuses the field as `UNKNOWN_FIELD`). |

The review measures AA contrast for the block (4.5:1 for text on its fill, 3:1 for `--primary` and
`--ring` on `--background`) on exactly what the answer below can serve: per scheme for `tokens`; for
`fromLooks`, the table's `defaults` (what every member with no saved look gets; listed first with id
`boot/prepaint.json#defaults` and kind `defaults`, which no look id can be, so a preset named
`defaults` keeps its own entry, kind `preset`) and then each preset, each painted as the web paints
it. A pair under its minimum blocks the listing, and so does a scheme or painting that gives no
`--background` or `--foreground` (a phone can't draw it).

A block with a layout is measured too: the contract's pairs (a text or mark slot on a ground slot,
or on a colour the layout fixes, such as HUD's text) over its slots, once when every slot is a value,
else for the prepaint table's `defaults` and then each preset, each resolved as the answer below
resolves it. The Studio skin is still measured beside it, and a pair under its minimum in either
blocks the listing (`services/themes/layout_contrast.py`).

## The phone layouts and their contracts

Automatos Go draws every screen with a layout built into the app (App Store guideline 2.5.2 allows no
downloaded code), and Automatos adds layouts in app releases. Each layout publishes a **contract**: the
slots it paints from. The contracts are `orchestrator/services/themes/data/phone-layouts.json`
(`services/themes/layout_contracts.py` reads it), the host's own data, copied byte for byte by the
browser (`frontend/lib/theme-host/phone-layouts.json`), the design kit and Automatos Go.

| Contract | Home theme | Required slots | Optional slots (default) | Fixed by the layout | Tone | Auto |
|---|---|---|---|---|---|---|
| `play@1` | `automatos/play` | colours `bg`, `surface`, `text`, `text2`, `accent`, `accentInk`, `autoLight`; tints `line`, `chip`, `panel` | `glow` 0-40 (24), `shell` bronze, gold, silver or cobalt (bronze), `error` (#FF6B6B) | | `bg` against `text` | bundled: the theme's clips or stills when the block brings them, else its own Auto in `shell` and `autoLight` |
| `hud@1` | `automatos/hud` | colours `bg`, `accent`, `accentInk` | `grid` 0-30 (14), `scan` on or off (on), `glow` 0-40 (22), `auto` on or off (on) | `text`, `text2`, `line`, `track`, `error` | always dark | bundled (`auto`) |
| `arcade@1` | `automatos/arcade` | colours `page`, `win`, `tx`, `tx2`, `bd`, `bar`, `barTx`, `ac`, `aci`, `hl`, `hlt`, `hp`, `xp`, `mp`, `qs`; tints `sh`, `bevl`, `bevd` | the 13 map colours (Overworld's), `art` bronze or shades, `pixelScale` 2 or 3 (3), `scanlines` off, soft or strong (soft), `windowStyle` classic or thin (classic) | | `page` against `tx` | drawn: the pixel sprite in `art`'s colours |
| `candy@1` | `automatos/candy` | colours `page`, `bandA`, `bandB`, `band1`-`band3`, `win`, `chip`, `text`, `quiet`, `border`, `bandQuiet`, `accent`, `accentInk`, `accentText`, `userA`, `userB`, `userInk`, `stageInk`, `cardInk`, `g1a`-`g3c`; tints `line`, `tint` | tints `blob1`-`blob3` (Blush's), `gradient` soft, rich or vivid (rich), `roundness` 16-40 (28), `glow` 0-10 (6) | danger, info and agent colours, derived in the app per tone | `page` against `text` | drawn: the glossy shell, fixed colours |

Each contract also lists its pairs (Play 5, HUD 4, Arcade 15, Candy 34, the layout's own pairs of the
home theme's `design/contrast.json` re-keyed to slots), which text each font role draws on it, and an
`example`: the home theme's default look as slots, which passes its own contract. A slot with a
default is optional, one without is required. Within a major the only change is a new optional slot
(the `revision` goes up), which an older app ignores, drawing its own default, which is the
contract's. Anything else (a new required slot, a slot removed, renamed or retyped, a narrower range,
a changed pair) is the next major, `candy@2`, kept beside `candy@1` while any listed theme names it.

**Slot kinds** (`services/themes/layout_slots.py`, `frontend/lib/theme-host/mobile-layout.ts`):

| Kind | A value | From the look |
|---|---|---|
| `colour` | `#rrggbb` or `H S% L%` | `{"look": "--name"}`: a variable `boot/prepaint.json` writes |
| `tint` | a colour, `rgba(r,g,b,a)` (no spaces; each channel 0 to 255, the alpha 0 to 1 with up to four decimals) or `transparent` | `{"look": "--name"}` |
| `number` | a whole number in the slot's range | `{"look": "knob"}`: the knob's number, or a choice key that writes one (Candy's `roundness` keys `"16"` to `"40"`) |
| `choice` | one of the slot's choices | `{"look": "knob"}`: the knob's choice |

A variable and a knob are named as the browser's first paint keeps them: `--` then 1 to 64 lower-case
letters, digits or dashes; a letter then up to 31 letters, digits or `_`. On import
(`services/themes/mobile_layout.py`), a slot from the look needs `boot/prepaint.json`; its variable
must be one the table's knobs or pairs write (a numeric rule's variable is painted from a number, so
it is none), its knob one of the table's looks; and the table's `defaults` must fill it. Each refusal
is `MOBILE_CANNOT_PAINT` naming the slot.

## The answer: `GET /api/themes/mobile/{install_id}`

`orchestrator/api/themes_mobile.py`, built by `orchestrator/services/themes/mobile.py`. The frontend
type is `MobileTheme` in `frontend/lib/api-client.ts` (`apiClient.getMobileTheme(installId)`); the
phone app copies it.

```jsonc
{
  "schema": 1, "installId": "<uuid>", "themeId": "acme/harbour", "version": "1.1.0",
  "mobile": null | {
    "source": "tokens" | "fromLooks",
    "tone": "light" | "dark" | null,
    "tokens": { "light": { "--background": "0 0% 100%", "…": "…" } | null, "dark": { … } | null },
    "fonts": { "sans": "/api/themes/files/<digest>/assets/fonts/mobile/inter-regular.ttf" },
    "mascot": { "mode": "clips" | "stills" | "none",
                "states": [ { "name": "idle",
                              "clip": "/api/themes/files/<digest>/assets/mascot/idle.mp4" | null,
                              "still": "/api/themes/files/<digest>/assets/mascot/idle.png" | null } ] },
    "layout": null | { "contract": "candy@1", "tone": "light",
                       "slots": { "page": "#FFF4F9", "line": "rgba(42,24,48,.09)", "roundness": 28, "gradient": "vivid" } }
  }
}
```

- **`tokens` source.** The block's own schemes, exactly as `theme.json` writes them. `tone` is null:
  the phone follows the device's light or dark setting. A scheme the block leaves out is null.
- **`fromLooks` source.** The member's active look for that theme in the install's workspace (the
  looks the look editor saves on the server, carried to the theme's current look schema), or the
  prepaint table's `defaults` when the member has saved none. The look is painted exactly as the
  web's first paint paints it (next section), keeping Studio's colour tokens only, and placed under
  its tone (the table below). The other scheme is null. A saved look whose painting gives no
  `--background` or `--foreground` is not served: the phone gets the defaults (logged on the
  server), the same rule the import check and the review hold.
- **Token values** are `H S% L%` or `#rrggbb` for `tokens`, as `theme.json` writes them, and `H S% L%`
  only for `fromLooks`, as the table writes them; the phone turns them into colours with look-kit's
  `hsl.ts`.
- **`mobile` is null** when the theme has no `mobile` block, when the member may not run it
  (a workspace's own or unreviewed theme the member has not opted in to), when it is revoked, or
  when a `fromLooks` theme's prepaint table can't be read (logged on the server).
- **`layout`** (PRD-271). Every slot of the contract the block names: values as written,
  `{"look": …}` resolved for the look the skin paints (under `fromLooks` the one its tokens come from,
  else the member's saved look; none saved paints the defaults), slots the block leaves out at the
  contract's default. A colour or tint keeps the form it was written or painted in (`#rrggbb`,
  `H S% L%`, `rgba()`, `transparent`), a number is whole, a choice is one of the slot's. `tone` is the
  contract's rule over the slots (HUD always `dark`; the others `dark` when the ground slot is darker
  than the text slot), so core, the kit and the phone agree; Light / Dark / System doesn't apply on a
  layout. A look that can't fill a slot paints the defaults whole; defaults that can't either, or a
  table that can't be read, give `layout` null (logged), and the phone draws the skin. A block with
  no layout gives `layout` null too: the field's presence tells a phone the server has layouts, and an
  older server simply lacks it. `schema` stays 1, and there is no new route or discovery capability.
- **Which layout a phone draws** for a theme that names `candy@1`: an app that implements `candy@1`
  draws Candy from the slots; an app without Candy, or with only another major, and every app shipped
  before PRD-271, draws the theme's Studio skin; an older server refuses the package on import
  (`layout` is `UNKNOWN_FIELD` there), as any field newer than the host is. The layout's own home theme
  (the table's `home`) keeps its native path on the phone, painted from the phone's own looks.
- **Headers.** `Cache-Control: private, no-store`: the answer follows the member's look.

## Painting a look: the web's first paint

The phone shows exactly what the web paints before React runs (`frontend/lib/theme-host/boot.ts`,
`THEME_BOOT_SCRIPT`, over the table `boot-sanitize.ts` `sanitizePrepaint` lets through). The server
side is `services/themes/mobile_tokens.py` `paint_look`, shared by the import check, the review and
the answer:

1. **The table the web gets.** Knobs whose name is a look field (a letter, then letters, digits or
   `_`, at most 32), option keys of 1 to 64 characters, and in each option the variables whose name
   and value pass the boot's own gate (`css_value.is_boot_value`: no `color-mix()` or gradients; the
   reader refuses a cell's value it drops, `first_paint.py`); pair tables whose two knobs are look fields; numeric rules with every field
   (`template` `rgba-alpha`, a true/false `integer`, `min` <= `max`, `scale` > 0, `capTable` null or
   two knobs); `defaults` with text or number values. If the defaults don't fit what is left, the
   web drops the whole table and paints nothing from it.
2. **Fits.** A look fits when it names an option of every knob table and a cell of every pair table,
   and every numeric rule's knob is a number in its range (whole when `integer`). A look that doesn't
   fit is painted as the table's `defaults`, whole: never the look's own values mixed with the
   defaults.
3. **The walk.** Every knob table's entry, then every pair table's, a later one overwriting. A
   numeric rule paints its variable last with `rgba()`, which is no phone colour, so the phone gets
   no value for it. Only Studio's colour tokens whose final value is an `H S% L%` triplet are kept:
   the web reads every one of them as `hsl(var(--x))` (`frontend/tailwind.config.ts`, `globals.css`),
   so a `#rrggbb` there paints nothing, on the web or the phone. A web-only theme may still write one
   (the import check doesn't refuse it); a `fromLooks` theme whose defaults give `--background` or
   `--foreground` only that way can't paint (`MOBILE_CANNOT_PAINT`), and the review fails a preset
   that does.

The tone the tokens go under, by `theme.json` `tone`:

| `tone` | On the web | On the phone |
|---|---|---|
| `light`, `dark` | Fixed. | That tone, whatever the look. |
| `from-look` | The look's tone (the look runtime's cookie, for a theme it runs). | The table's tone rule at the painted look's value; with no rule (or one the web drops), the painted ground's tone. |
| `user` (or absent) | The member's own light or dark setting. | The painted ground's tone: the phone has no member tone cookie. |

The painted ground's tone is look-kit's `isDarkGround`: `dark` when `--background` is darker (by
WCAG relative luminance) than `--foreground`, else `light`.

## What the phone may load

Only the URLs in the answer. Each is `module_url(digest, path)` (`services/themes/serve.py`), a path
relative to the API base, served by `GET /api/themes/files/{digest}/{path}` under the same rules as
the answer: membership of a workspace that has the theme installed, plus the member's opt-in for
private code. A URL is given only for a file the package lists, and never for a JS or CSS file. A
mascot clip is the state's `.mp4` (the `.webm` is the web's); `stills` gives no clip; `none` gives
no states. A font is the `.ttf` or `.otf` the block names; a role the block leaves out is absent.

## Who may call

The route is authorised as `GET /api/themes/boot/{install_id}`: the install's workspace comes from
the install row, and the caller must belong to it (`services/themes/membership.py`). It never reads
the request's workspace, so an `X-Workspace-ID` naming another workspace changes nothing. An
unknown, malformed or foreign install id gets one and the same 404.

- **A person signed in with Clerk** (`auth_type` `clerk`): any workspace where they are an active
  member or the owner. A phone signed in this way works today; its token must also pass core's
  `azp` rule.
- **The local edition's operator** (`AUTH_EDITION=local`): every workspace on the instance.
- **A phone paired to the local edition** (`auth_type` `device`, `Authorization: Bearer amd_…`):
  exactly the one workspace its pairing is bound to, never every workspace, the operator's own phone
  included (`PERSON_AUTH_TYPES = {"clerk", "device"}` in `services/themes/membership.py`). The file
  URLs load through `/api/themes/files/...` on the same bearer, under the same rule.
- **Anyone else gets the 404**, an API key included.

The phone sees the look the member last saved on the server, not one still being edited in a
browser tab.

## Routes a phone may call

The server lists `mobile_theme` among the `capabilities` of its discovery document
(`GET /.well-known/automatos.json`, `services/mobile/discovery.py`). The capability stands for
`GET /api/themes/mobile/{install_id}` (`CAPABILITY_ROUTES`). A phone reads the member's theme and
looks through the same routes the web app uses, under the membership rule above:

| Route | What it answers |
|---|---|
| `GET /api/themes/active` | The member's theme in the workspace: their choice, the workspace default or Studio |
| `PUT /api/themes/choice` | Saves the member's choice in the workspace |
| `GET` / `PUT /api/themes/looks/{theme_id}` | The member's looks for one theme in the workspace |
| `GET /api/themes/mobile/{install_id}` | The installed theme's phone answer, above |
| `GET /api/themes/files/{digest}/{path}` | The font and mascot files the answer names |

`orchestrator/tests/test_prd266_device_theme_routes.py` runs a paired phone through the mobile,
active, choice and looks routes end to end.
