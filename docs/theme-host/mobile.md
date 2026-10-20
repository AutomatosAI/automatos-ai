# The phone's theme: the `mobile` block and its answer

PRD-261 M-01..M-04 (host contract 1.1.0). The phone app draws a member's theme from data alone. It
never loads a theme's `dist/entry.js` or `dist/theme.css`: an app may not download and run code
(App Store guideline 2.5.2). A theme says how it looks on a phone in an optional `mobile` block in
`theme.json`, and the phone reads the result from one route.

## The block in `theme.json`

```jsonc
"mobile": {
  "tokens": { "light": { "--background": "0 0% 100%", "--foreground": "#14181c" }, "dark": { "…": "…" } },
  // or, instead of "tokens":
  "fromLooks": true,
  "fonts": { "sans": "assets/fonts/mobile/inter-regular.ttf", "serif": "…", "mono": "…" },
  "mascot": "clips"
}
```

| Field | Rule (checked on import by `services/themes/mobile_manifest.py` and its TypeScript twin) |
|---|---|
| `tokens` or `fromLooks` | Exactly one. `fromLooks` is only ever `true`. |
| `tokens.light`, `tokens.dark` | `light` is required, `dark` optional. Keys are Studio's 24 colour tokens (`MOBILE_TOKEN_NAMES` in `services/themes/contract.py`), spelled with the leading `--`, and must include `--background` and `--foreground`. Each value is `H S% L%` (such as `0 0% 100%`) or `#rrggbb`. |
| `fromLooks` | The member's look is painted through `boot/prepaint.json`, which the package must have, and the table's `defaults`, painted as the web paints them (below), must give `--background` and `--foreground` as `H S% L%` triplets. Defaults that don't fit the table paint nothing, on the web or the phone, and neither does a Studio token written `#rrggbb` in the table. |
| `fonts` | Any of `sans`, `serif`, `mono`; each a different `assets/fonts/mobile/<name>.ttf` or `.otf` in the package (one sfnt face, at most 400 KB). |
| `mascot` | `clips`, `stills` or `none` (absent means `none`). `clips` needs each state of the top-level `mascot.states` to have `assets/mascot/<state>.mp4` and `.png`; `stills` needs the `.png`. |
| `engines.automatos` | Must not admit a host older than 1.1.0. |

The review measures AA contrast for the block (4.5:1 for text on its fill, 3:1 for `--primary` and
`--ring` on `--background`) on exactly what the answer below can serve: per scheme for `tokens`; for
`fromLooks`, the table's `defaults` (what every member with no saved look gets; listed first with id
`boot/prepaint.json#defaults` and kind `defaults`, which no look id can be, so a preset named
`defaults` keeps its own entry, kind `preset`) and then each preset, each painted as the web paints
it. A pair under its minimum blocks the listing, and so does a scheme or painting that gives no
`--background` or `--foreground` (a phone can't draw it).

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
                              "still": "/api/themes/files/<digest>/assets/mascot/idle.png" | null } ] }
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
- **Headers.** `Cache-Control: private, no-store`: the answer follows the member's look.

## Painting a look: the web's first paint

The phone shows exactly what the web paints before React runs (`frontend/lib/theme-host/boot.ts`,
`THEME_BOOT_SCRIPT`, over the table `boot-sanitize.ts` `sanitizePrepaint` lets through). The server
side is `services/themes/mobile_tokens.py` `paint_look`, shared by the import check, the review and
the answer:

1. **The table the web gets.** Knobs whose name is a look field (a letter, then letters, digits or
   `_`, at most 32), option keys of 1 to 64 characters, and in each option the variables whose name
   and value pass the boot's own gate (smaller than the import check's: no `color-mix()` or
   gradients); pair tables whose two knobs are look fields; numeric rules with every field
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
- **Anyone else gets the 404**, an API key included.

**Paired phones do not work yet.** A phone paired to an account calls as a `device` caller, and
theme membership counts only `clerk` callers (`PERSON_AUTH_TYPES = {"clerk"}` in
`services/themes/membership.py`). Until the mobile PRD adds the `device` auth type there, a paired
phone gets 404 from this route and from `/api/themes/files/...`. That change belongs to the mobile
PRD, not to PRD-261.

The phone sees the look the member last saved on the server, not one still being edited in a
browser tab.
