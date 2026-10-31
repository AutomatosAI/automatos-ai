# PRD-271: Phone layout contracts — a creator's theme picks one of Automatos Go's built-in layouts

**Status:** Draft for Gerard, 31 Oct 2026. Every §8 recommendation stands unless Gerard changes it (his 29 Oct rule).
W1 built (draft PR, 31 Oct 2026); where the build found the code saying otherwise, the PRD follows the code: a tint
also takes `transparent` and Candy's blobs are tints (§4.1, §4.3); a `{"look": …}` variable is one the table's knobs
or pairs write, never a numeric rule's (§4.1); the defaults must fill every slot taken from the look, not only the
required ones (§4.4, C02); the skin's look under `fromLooks` is the one its tokens are painted from (§4.4); the review
corpus gains the `contrast` check (C04); and the core files (§7).
**Owner:** Gerard · **Grounded on:** `automatos-ai` `main @ 1a8956c87`, `automatos-go` `main @ 39b638c` and
`feat/a9-candy @ 99ed5aa` (Candy, A9, Ralph running), `automatos-design-kit` `main @ eb516bd`.
**Gerard's decision (31 Oct 2026, 01:40):**
- Creators may put their theme on one of Automatos Go's phone layouts (Studio, Play, HUD, Arcade, and Candy once A9
  merges) and paint it in their own colours, fonts and Auto.
- Automatos keeps adding layouts in app releases. App Store guideline 2.5.2 allows no downloaded code, so a layout is
  native code in the app.
- Each layout publishes a **contract**: the slots it paints from. A theme names a layout in its `mobile` block and
  fills those slots.
- Core validates the block, Go draws the layout from the theme's slots, the kit checks the block and previews it.

**Revises:** PRD-001 D58 ("themes never bring code or layouts in v1"): a theme still brings no code, but it may now
pick a layout. Also the kit's `guides/mobile-skins.md` ("A layout is matched by theme id … A package can't name a
layout").
**Scope:** automatos-ai (contracts, validator, phone answer, review), automatos-design-kit (check, skin, harness,
guides) and automatos-go (painting). The kit and Go stories are written here so the design is reviewed once. On
approval they move into the kit's PRD and into Go's PRD-001 as Wave A10, as PRD-268's did.
**Depends on:** PRD-261 M-01..M-04 (the `mobile` block and its answer, shipped); Go A9 (Candy) for `candy@1` only.

Labels: **[T]** read in code at the commits above; **[I]** inferred; **[G]** Gerard's decision above.

## 1. What exists today

| Piece | Today [T] | Gap |
|---|---|---|
| The block | `services/themes/mobile_manifest.py` `MOBILE_FIELDS = ("tokens", "fromLooks", "fonts", "mascot")` (:43); lists in `contract.py` :45-:63; TS twin `frontend/lib/theme-host/mobile-manifest.ts` | No `layout`, no slots |
| The answer | `GET /api/themes/mobile/{install_id}` (`api/themes_mobile.py`), built by `services/themes/mobile.py` `mobile_theme` / `_mobile_block` (:174); the look painted by `mobile_tokens.paint_look`, which keeps **Studio tokens as triplets only** (`_phone_tokens` :351) | Can't carry a layout's own variables (`#rrggbb`, `rgba()`, numbers) |
| Review | `review_stage.contrast_check` over `mobile_contrast.measure_mobile` (:186): the 10 Studio pairs; a fail is `MOBILE_CONTRAST_FAIL_LEVEL = error`, which blocks the listing | No per-layout pairs |
| Which theme gets which layout | By theme id, hard-coded twice: Go `src/styles/themes/follow.ts` `BUILT_IN_THEME_STYLES` (:26) and the kit's `packages/theme-harness/src/phone/skin.ts` (:26). Core has no copy | Not data; can't name a layout |
| Go's theme painting | `registry/style-id.ts` `layoutOf` returns `'studio'` for every theme (:27); `provider/resolve.ts` `theme()` (:181) paints the Studio layout only | Layouts other than Studio can't take a theme |
| Go's answer parser | `themes/schemas.ts` `MobileThemeSchema`: `schema: z.literal(1)` (:163), `source: z.enum(['tokens','fromLooks'])` (:153); unknown fields ignored (:4) | An additive field is safe for every shipped build |

How each layout gets its paint today [T]:

| Layout | Paint inputs | Vars its components read |
|---|---|---|
| Studio | a skin's 24 Studio tokens, or Studio's own | the Studio-keyed `Tokens` |
| Play | a Play look = keys into Play's catalogue (`core/frozen/lib/looks/types.ts`: "never raw colours"): ground `{bg, surface, text, text2, line, chip, panel}`, accent `{hex, ink}`, shell, light, glow — core's generic `lib/look-kit` `LookCatalogue` already models these entries | `tokens/play.ts` `PLAY_VAR_NAMES` (23), via look-kit `lookToCssVars` |
| HUD | ground `bg`, accent `{hex, ink}`, grid, glow; text, text2, line, track and error **fixed** (`tokens/hud-catalogue.ts` `HUD_FIXED`) so text keeps AA on every ground | `tokens/hud.ts` `HUD_VAR_NAMES` (17) |
| Arcade | a palette: 15 colours + shadow and two bevels (`tokens/arcade-catalogue.ts` `PaletteColours`), 13 map colours, an art set; scanlines, window style, pixel scale | `tokens/arcade.ts` `ARCADE_VAR_NAMES` (19) |
| Candy (A9) | a palette: 21 colours (line and tint among them) and three blobs; three card gradients (9 stops) and card ink; gradient, roundness, glow. Danger and the info and agent colours are the layout's own (`DANGER`, `CARD_SETS` blue and violet made readable on the palette's grounds) | `tokens/candy.ts` `CANDY_VAR_NAMES` |

Every derived colour (sheets, dusk, accent text, error text, line and fill alphas, the glow cap, the Studio-keyed set
through `HUD_ROLES`, `ARCADE_ROLES`, `CANDY_ROLES`) is a function of those inputs. **So a layout's inputs are its
contract.** Each first-party theme also lists its AA pairs in the kit's `themes/<id>/design/contrast.json` (Play 11
layout pairs, HUD 13, Arcade 15, Candy 34, plus Studio's): those are the contract's pairs, re-keyed to slot names.

About 15 Go call sites read a style's raw look object (catalogue keys) instead of its variables, which a theme
without a catalogue can't supply: `arcade/PauseButton.tsx:25`, `arcade/parts/use-arcade-paint.ts:21`,
`arcade/sprite/{AgentAvatar.tsx:22, ItemIcon.tsx:15, PixelSprite.tsx:89}`, `dress/dress.ts:249`,
`ground/ground.ts:55,65`, `homes/ShelfCards.tsx:38-41`, `hud/LooksDrawer.tsx:54`, `mascot/ring-palette.ts:56-59`,
`voice/screen/dress.ts:52`, `theme.tsx:62`. They are the main Go cost (US-024c).

## 2. Goals

- A creator names a layout and its contract version in `theme.json` and fills its slots, with fixed values or from
  their own looks; `automatos-theme check` and the Marketplace review refuse exactly the same mistakes.
- A member whose app has that layout sees the theme drawn on it, in the theme's colours, fonts and Auto; a member
  whose app doesn't sees the theme's Studio skin. Nobody sees a broken screen.
- One table of contracts, owned by core, copied byte for byte by the kit and Go. No theme-id list in code.
- A layout can grow (an optional slot) without breaking a published theme; a breaking change is a new major.
- Nothing new reaches the phone but colours, numbers and named choices. No URL, no code.

## 3. Non-goals

- Creators adding layouts or code. A new layout is an Automatos app release; the kit's issue route for proposing one
  stays.
- Layouts on the web. The web theme is separate and unchanged.
- Changing how the first-party themes paint on the phone (D-4).

## 4. Design

### 4.1 The block (D-1)

```jsonc
"mobile": {
  "fromLooks": true,                         // the Studio skin: "tokens" or "fromLooks", still required (D-3)
  "fonts": { "serif": "assets/fonts/mobile/bubble-title.ttf", "sans": "assets/fonts/mobile/bubble-body.ttf" },
  "mascot": "stills",
  "layout": "candy@1",                       // <layout id>@<contract major>
  "slots": {
    "page": "#FFF4F9",                       // a fixed value: colour, tint, number or choice, as the slot's kind says
    "accent": { "look": "--bubble-accent" }, // from the member's look: a variable boot/prepaint.json writes
    "roundness": { "look": "roundness" },    // from the member's look: a knob (number and choice slots)
    "gradient": "vivid"
  }
}
```

- `layout` and `slots` come together. Each slot value is a literal of the slot's kind, or `{"look": …}`: a
  `--variable` the theme's `boot/prepaint.json` writes (colour and tint slots) or a look knob (number and choice
  slots). One theme can mix both. `{"look": …}` needs `boot/prepaint.json`.
- Slot kinds: `colour` (`#rrggbb` or `H S% L%`), `tint` (a colour, `rgba(r,g,b,a)` with no spaces, or `transparent`:
  Candy's prepaint paints its aura blobs `transparent` off Vivid), `number` (an integer in the slot's range),
  `choice` (one of the slot's named options).
- A `{"look": "--name"}` names a variable the table's knobs or pairs write, as the browser's first paint keeps it; a
  numeric rule's variable is painted last from the look's number, so it is never a slot's source (take the knob
  itself into a number slot). A `{"look": "knob"}` names a knob of the table's looks (`lookSpecOf`).
- Fonts and the mascot stay the block's existing fields; the contract says which text each font role draws on that
  layout and which mascot modes it honours (D-6).
- A block with `layout` needs host contract 1.2.0: `engines.automatos` must admit no older host (as
  `_engines_errors` does for 1.1.0 today). An older core already refuses `layout` as `UNKNOWN_FIELD`.

### 4.2 The contracts table (D-2)

`orchestrator/services/themes/data/phone-layouts.json`, beside `signing-keys.json` and `core-pages.json` (both read
from that folder at runtime today: `signature_verify.py:51`, `contract.py:122`). It is the host's contract, like
`contract.py`, not seed content.

```jsonc
{ "schema": "automatos-phone-layouts/v1",
  "layouts": {
    "candy": { "name": "Candy", "home": "automatos/candy",
      "versions": { "1": {
        "revision": 1,
        "tone": { "rule": "ground", "ground": "page", "text": "text" },     // HUD: { "rule": "dark" }
        "slots": {
          "page":      { "kind": "colour", "group": "Page" },
          "line":      { "kind": "tint",   "group": "Page" },
          "roundness": { "kind": "number", "min": 16, "max": 40, "default": 28, "group": "Shape" },
          "gradient":  { "kind": "choice", "choices": ["soft", "rich", "vivid"], "default": "rich", "group": "Cards" }
        },
        "fixed": {},                                                        // HUD: its fixed text, line, error
        "pairs": [ ["text", "page", "text"], ["accent", "page", "ui"] ],      // text 4.5:1, ui 3:1
        "fonts": { "serif": "titles", "sans": "body" },
        "mascot": "drawn",                                                  // or "bundled" (D-6)
        "example": { "page": "#FBF0F4", "…": "…" } } } } } }               // the home theme's default look as slots
```

- A slot with a `default` is optional; one without is required. `home` is the first-party theme that owns the layout
  (it replaces `BUILT_IN_THEME_STYLES`, D-4). `group` labels the harness's schematic. `example` must pass its own
  contract, and it is what Go's parity test paints (US-024g).
- **Versions.** Within a major, the only change is a new optional slot with a default (the `revision` goes up). An
  app that predates it ignores it and uses its own default, which is the contract's. Anything else (a new required
  slot, a removed, renamed or retyped slot, a narrower range, a changed pair) is the next major, `candy@2`, added
  beside `1`. Core keeps a major while any listed theme names it.
- **Copies.** The kit copies the file byte for byte to `packages/theme-cli/src/core/data/` (as it copies
  `core-pages.json`). Go copies it through `scripts/core-manifest.json` `copy` (as the route manifest) to
  `src/core/phone-layouts.json`. Core's TS twin reads a byte-identical copy at `frontend/lib/theme-host/`, held equal
  by `validator-parity.test.ts`.
- Studio is not in the table: its contract is the existing skin (`tokens` / `fromLooks`, `contract.py` lists).

### 4.3 The four contracts at major 1

Slot lists are each layout's paint inputs (§1), so they are exactly what Go's code already reads; derived colours are
never slots. Pairs are the layout's pairs from the kit's `design/contrast.json`, among slots and the layout's fixed
colours. Defaults are the home theme's default preset's values [T].

| | play@1 | hud@1 | arcade@1 | candy@1 |
|---|---|---|---|---|
| Required colours | `bg`, `surface`, `text`, `text2`, `accent`, `accentInk`, `autoLight` | `bg`, `accent`, `accentInk` | `page`, `win`, `bar`, `barTx`, `tx`, `tx2`, `bd`, `ac`, `aci`, `hl`, `hlt`, `hp`, `xp`, `mp`, `qs` | `page`, `bandA`, `bandB`, `band1`-`band3`, `win`, `chip`, `text`, `quiet`, `border`, `bandQuiet`, `accent`, `accentInk`, `accentText`, `userA`, `userB`, `userInk`, `stageInk`, `cardInk`, `g1a`-`g3c` |
| Required tints | `line`, `chip`, `panel` | — | `sh`, `bevl`, `bevd` | `line`, `tint` |
| Optional | `error` (#FF6B6B), `glow` 0-40, `shell` (bronze, gold, silver, cobalt) | `grid` 0-30, `glow` 0-40, `scan` on/off, `auto` on/off | the 13 map colours `g1`-`wall` (Overworld's), `art` (bronze, shades), `pixelScale` ("2", "3"), `scanlines` (off, soft, strong), `windowStyle` (classic, thin) | tints `blob1`-`blob3`, `gradient`, `roundness` 16-40, `glow` 0-10 |
| Fixed by the layout | coral error unless set | text, text2, line, track, error (`HUD_FIXED`, the table's `fixed`) | — | danger fill and ink per tone; info blue and agent violet, made readable on the theme's grounds (derived in the app per tone; no pair names them, so they are not in the table's `fixed`) |
| Tone | ground: `bg` vs `text` | always dark | ground: `page` vs `tx` | ground: `page` vs `text` |
| Pairs | 5 (text and text2 on bg and surface, ink on accent) | 4 (fixed text and text2 on bg, ink on accent, accent on bg at 3:1) | 15 | 34 |
| Fonts | serif display, sans body, mono labels | serif display, mono labels and transcript, sans body | serif titles and labels, sans running text | serif titles, sans body |
| Auto | bundled (`shell`, `autoLight`) or the theme's files | bundled (`auto`) or the theme's files | drawn: the pixel sprite in `art`'s colours | drawn: the glossy shell, fixed colours |

### 4.4 The phone answer

`mobile_theme` keeps `schema: 1` and every existing field (the Studio skin), and adds one:

```jsonc
"mobile": { "source": "fromLooks", "tone": "light", "tokens": { … }, "fonts": { … }, "mascot": { … },
  "layout": null | { "contract": "candy@1", "tone": "light",
                     "slots": { "page": "#FFF4F9", "line": "rgba(42,24,48,.09)", "roundness": 28, "gradient": "vivid" } } }
```

- Every slot of the contract is present: literals as written, `{"look": …}` resolved for the **same look** the skin
  paints (under `fromLooks` the look its tokens are painted from, `_phone_look`; else `_stored_active_look`; none
  paints the defaults), optional slots left out filled with the contract's default.
- Resolution is a sibling of `_phone_tokens`: `mobile_tokens.paint_vars(table, look, names)` walks the same table
  in the same order and keeps the named variables whatever their form (`#rrggbb`, triplet, `rgba()`), each checked
  against its slot's kind. A look that doesn't fit paints the defaults whole, as today. Defaults that can't fill a
  slot give `layout: null` (logged): the phone draws the Studio skin. A block with no layout gives `layout: null` too.
- `tone` is the contract's tone rule over the resolved slots, so core, kit and Go agree.
- No new route and no new discovery capability: the field's presence is the signal, and an older server simply lacks
  it (D-9).

### 4.5 Fallback (D-3)

For a theme that names `candy@1`:

| App build | Core | Drawn |
|---|---|---|
| implements candy@1 | has PRD-271 | Candy, from the slots |
| lacks Candy, or implements only another major | has PRD-271 | the theme's Studio skin |
| any build shipped today | has PRD-271 | the Studio skin (`schemas.ts` ignores `layout`) |
| any | older | nothing: the package is refused on import (`layout` is `UNKNOWN_FIELD`), as any field newer than the host is |

### 4.6 Go: follow, rows, picker (D-8)

- **Follow.** `followStudio`: a home theme (table `home`) → that built-in style, painted from the phone's own synced
  looks, as today. Any other theme with a painting → `themeChoice(installId)`, drawn on its `layout` when this build
  implements that contract (`IMPLEMENTED_CONTRACTS`), else on Studio.
- **Rows.** `installed.ts` `mayOffer` excludes home themes by the table, not by `BUILT_IN_THEME_STYLES`; an
  `OfferedTheme` carries its painted layout.
- **Picker.** A theme row sits under its layout's section ("Candy · Bubblegum · 1.0.0"), its preview card painted
  with its slots; a theme on Studio stays under Studio. "Same as Studio" selects it when Studio's active theme is it.
- **Look row.** For a creator theme on any layout: read-only, "Change this look in Studio" (D-5). Built-in look pages
  (Make it yours, Looks, Options, Make it sweet) stay for the built-in styles only.

### 4.7 Security and privacy

Unchanged in kind: data only. Slots are colours, tints, integers and named choices; every value passes the boot
value gate and its slot's kind, so no `url(`, `var(` or text reaches the phone. The answer adds no URL; font and
mascot URLs keep `_file_url`'s rule (listed in the package, never JS or CSS) and the files route's membership.
Membership, opt-in and revocation keep `mobile` null exactly as today. The table is public data.

## 5. User stories

### Wave W1: core (automatos-ai) · size M · tracking issue on approval (host contract change)

#### US-271-C01: The contracts table
- [ ] `services/themes/data/phone-layouts.json` with play@1, hud@1, arcade@1, candy@1 (§4.3); loader
  `services/themes/layout_contracts.py` (stdlib only, like `contract.py`): `contract_of(layout)`, `home_layouts()`
- [ ] Tests: the table's own shape; every `example` passes its contract (kinds, ranges, pairs); every pair names a
  slot or a fixed colour; the frontend copy is byte-identical
#### US-271-C02: Checking the block
- [ ] `services/themes/mobile_layout.py` (keeps `mobile_manifest.py` small), called from `mobile_errors`:
  `MOBILE_FIELDS` gains `layout` and `slots`; `BAD_MOBILE` at `mobile.layout` for an unknown layout or major, and at
  `mobile.slots.<name>` for an unknown slot, a missing required one, a wrong kind, an out-of-range number, an unknown
  choice; `layout` without `slots` and the reverse refused
- [ ] `THEME_HOST_VERSION` and `shared-modules.json` `hostVersion` → 1.2.0; `LAYOUT_MIN_HOST = "1.2.0"`
- [ ] At package check: each `{"look": …}` names a variable or knob the prepaint table has, and the defaults fill
  every slot the block takes from the look (`MOBILE_CANNOT_PAINT` naming the slot). A number slot from a knob takes its number, or a
  choice key that names one (Candy's `roundness` is a string key since kit 1.0.1)
- [ ] TS twin `frontend/lib/theme-host/mobile-layout.ts`; new `manifests` cases in `tests/fixtures/themes/cases.json`
  hold both languages to the same verdicts
#### US-271-C03: The phone answer
- [ ] `mobile.py` `_mobile_block` adds `layout` (§4.4); `mobile_tokens.paint_vars`; `frontend/lib/api-client.ts`
  `MobileThemeBlock.layout`
- [ ] Tests in `test_prd261_mobile_endpoint.py`'s style: fixed slots; look slots follow the member's saved look;
  a look that doesn't fit paints the defaults; defaults that can't fill → `layout` null and the skin still served;
  revoked or not opted in → `mobile` null; `schema` stays 1; a paired phone gets the same
#### US-271-C04: Review contrast per layout
- [ ] `measure_mobile` measures the contract's pairs over the slots: once for fixed slots; for look slots, the
  defaults then each preset (as `_look_tokens`); the Studio skin is still measured. A failing pair blocks the
  listing (`MOBILE_CONTRAST_FAIL_LEVEL`); new `review-cases.json` cases, the corpus gaining the `contrast` check
  (the kit's K02 follows it)
#### US-271-C05: Docs
- [ ] `docs/theme-host/mobile.md`: the `layout` and `slots` fields, the table, the answer's `layout`, the fallback

### Wave K: the kit (automatos-design-kit) · size M · after W1 merges
#### US-271-K01: Vendor and check
- [ ] Re-vendor core at W1's merge (`src/core/README.md` pin; the table, `mobile-layout.ts`, both corpora);
  `check`, `pack` and `skin --from skin.json` take a block with `layout`; `skin` still refuses the `automatos`
  publisher
#### US-271-K02: Answer and contrast ports
- [ ] `src/mobile-answer.ts` builds `layout` as core does; `src/check/mobile-contrast.ts` measures the contract's
  pairs; the review corpus agrees
#### US-271-K03: The harness
- [ ] `src/phone/skin.ts`: `BUILT_IN_THEME_STYLES` deleted, home themes read from the table; `PhoneMode` gains
  `layout`
- [ ] The phone view for a layout theme can't draw Go's native layout, so it shows: the contract's slots by group,
  the contrast panel per pair with the nearest passing colour (`nearest-passing.ts`), a labelled schematic (each
  group a panel in its colours with sample text in each pair), and a toggle to the Studio skin as an app without the
  layout draws it; under looks, one look at a time
- [ ] The skin designer gains a layout mode: pick a contract, start from its `example`, edit slots, export
  `skin.json`
#### US-271-K04: Guides and fixtures
- [ ] `guides/mobile-skins.md` "The phone's layouts" rewritten (a theme may name a layout; the contract table; the
  fallback); `guides/contract.md` "The `mobile` block"
- [ ] One fixture theme per layout under `packages/theme-harness/fixtures/`, and a Candy fixture whose slots all map
  to Candy's own `--candy-*` prepaint variables (the parity case Go uses)

### Wave A10: Go (automatos-go PRD-001, US-024) · size L · after W1; candy@1 after A9 merges
#### US-024a: Contracts in the app
- [ ] The table vendored by `scripts/core-manifest.json` `copy`; `src/styles/registry/layouts.ts`:
  `IMPLEMENTED_CONTRACTS` and `homeStyleOf(themeId)` from `home`; `BUILT_IN_THEME_STYLES` deleted
  (`follow.ts`, `installed.ts`); test: every implemented contract is in the table
#### US-024b: Reading the layout
- [ ] `MobileBlockSchema.layout` optional, `.catch(null)`; `themes/layout-paint.ts` turns it into a
  `LayoutPaint` (defaults from the table, kinds checked, a contract this build lacks → null); `ThemePaint.layout`
#### US-024c: Painting from slots
- [ ] `playVars`/`hudVars`/`arcadeVars`/`candyVars` and their Studio-keyed sets take an entry or palette, from the
  catalogue (home theme) or from slots; the ~15 look-object readers in §1 move to variables
- [ ] `layoutOf` and `resolveStyle`'s motion use the painted layout; `resolve.ts` `theme()` draws the theme's layout
#### US-024d: Follow, rows, picker, look row
- [ ] §4.6 in `follow.ts`, `installed.ts`, `plan.ts` and `registry/picker.ts`; tests: a creator theme on a layout
  this build has is followed onto it; on one it lacks, onto Studio with its skin; a home theme still opens its
  built-in style and look page; a theme gone while on screen is back on Studio with the toast, as today
#### US-024e: Mascot and type
- [ ] D-6 per contract (`resolve.ts` `mascot`: `'theme'` when the block brings files on a bundled-Auto layout); the
  theme's fonts by role on each layout through `themedStyleType`; tests per layout
#### US-024f: Parity and a11y
- [ ] Each contract's `example` painted through slots equals the home theme's default look painted natively (every
  variable and Studio-keyed token), and so does the kit's Candy parity fixture; every Studio-keyed pair a role table
  derives from passing slots passes; the a11y audit includes a slot-painted theme on each layout
#### US-024g: Docs: `docs/styles-and-themes.md`; PRD-001 §4.6 and D58 revised
JS only: no native module and no bundled font, so it can ship over the air to builds on the same runtime [I].

## 6. Waves

| Wave | Repo | Stories | Size | Needs |
|---|---|---|---|---|
| W1 | automatos-ai | C01-C05 | M: one Ralph wave, one PR | approval (host 1.2.0) |
| K | automatos-design-kit | K01-K04 | M: one wave | W1 merged |
| A10 | automatos-go | US-024a-g | L: one wave | W1 merged; A9 merged for candy@1 |

**WebSummit (10-12 Nov).** Achievable before it: W1 and K. Creators can then write, check, preview and list a theme
on any of the four contracts, and every shipped app draws its Studio skin. A10 after WebSummit: it touches the style
system across ~15 files while A9 is still running, and Go's build freezes before the event (PRD-001 §9). The cut is
Gerard's (Q1).

## 7. Where the code goes

| Repo | Files |
|---|---|
| core | new `services/themes/{data/phone-layouts.json, layout_contracts.py, layout_slots.py (the kinds), layout_paint.py (the resolution the reader, the answer and the review share), mobile_layout.py, layout_contrast.py}`, `frontend/lib/theme-host/{phone-layouts.json, mobile-layout.ts, mobile-colour.ts}`; changed `mobile_manifest.py`, `mobile.py`, `mobile_tokens.py`, `mobile_contrast.py`, `review_stage.py`, `package.py`, `contract.py`, `shared-modules.json`, `mobile-manifest.ts`, `api-client.ts`, `test.yml` (`FRONTEND_PATHS`), `docs/theme-host/mobile.md`, both corpora |
| kit | `packages/theme-cli/src/core/*` (re-vendor), `mobile-answer.ts`, `check/mobile-contrast.ts`, `skin-check.ts`; `packages/theme-harness/src/phone/skin.ts`, `app/phone/` (layout view, designer); `guides/` |
| Go | `scripts/core-manifest.json`, `src/styles/registry/{layouts.ts, style-id.ts, picker.ts}`, `themes/{schemas,layout-paint,types,follow,installed}.ts`, `provider/resolve.ts`, `tokens/{play,hud,arcade,candy}.ts`, the §1 readers |

## 8. Decisions (each recommendation stands unless Gerard changes it)

| | Decision | Recommendation |
|---|---|---|
| D-1 | Contract shape | `"layout": "<id>@<major>"` + `slots`; each slot a literal or `{"look": …}` (a prepaint variable or knob), so one theme can mix fixed and look-driven slots. Not per-preset values: a member's look need not be a preset |
| D-2 | Source of truth and versions | Core's `services/themes/data/phone-layouts.json`; the kit and Go copy it byte for byte. Optional slot with a default = same major; anything else = next major, both kept |
| D-3 | Fallback | The Studio skin (`tokens` or `fromLooks`) stays required. Go's parser demands `source` and `schema: 1`, so it is required by construction; an app without the layout or major draws it |
| D-4 | First-party themes | Keep their native path (look pages, offline painting, look sync). Only the theme-id map moves into the table (`home`); `BUILT_IN_THEME_STYLES` is deleted in Go and the kit. Their blocks don't change, so they don't need host 1.2.0 |
| D-5 | Looks on the phone | A creator theme shows the member's look as saved in Studio; its look row is read-only. Built-in look pages stay built-in |
| D-6 | Mascot | Bundled-Auto layouts (Play, HUD) show the theme's clips or stills when the block brings them, else their own Auto in the theme's `shell`/`autoLight`/`auto`. Drawn-Auto layouts (Arcade, Candy) always draw their own; Arcade's art set is a slot, Candy's colours are fixed at @1 |
| D-7 | Tone | Each contract fixes it (HUD dark; the others from ground and text); Light / Dark / System doesn't apply on a layout, as for the built-ins |
| D-8 | Go: follow, rows, picker | "Same as Studio" and a phone pick draw a creator theme on its layout when the build implements the contract, else on Studio; home themes come from the table; a theme row sits under its layout's section, previewed in its slots |
| D-9 | Discovery | No new capability; the answer's `layout` field is the signal |
| D-10 | Validation and review | Core refuses unknown, missing, mistyped and out-of-range slots and unresolvable look slots; contract pairs measured like Studio's; a failing pair blocks the listing; the Studio skin is measured too |

## 9. Open questions for Gerard

1. **WebSummit cut.** W1 and K before, A10 after (rec); or A10 for play, hud and arcade before Go's freeze.
2. **Phone look page for a creator theme.** Read-only (rec); or a list of the theme's presets that saves the
   member's active look through `PUT /api/themes/looks/{theme_id}` (needs the presets served to the phone).
3. **Drawn Auto.** Fixed per layout (rec); or allow a theme's stills to replace Arcade's sprite and Candy's shell.
4. **HUD's text.** Fixed, as HUD's design keeps AA on every ground (rec); or slots, with pairs.
5. **Host 1.2.0.** One bump shared with PRD-268 US-T01's status tokens, whichever merges first (rec).
6. **Storefront.** Show "On Automatos Go: Candy layout" on a listing (rec: yes, a small W1 addition to
   `services/themes/listings.py` and the storefront card).
