# On your phone

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/api/themes_mobile.py](../../orchestrator/api/themes_mobile.py)
- [orchestrator/services/themes/mobile.py](../../orchestrator/services/themes/mobile.py)
- [orchestrator/services/themes/mobile_manifest.py](../../orchestrator/services/themes/mobile_manifest.py)
- [orchestrator/services/themes/mobile_tokens.py](../../orchestrator/services/themes/mobile_tokens.py)
- [orchestrator/services/themes/membership.py](../../orchestrator/services/themes/membership.py)
- [orchestrator/services/themes/serve.py](../../orchestrator/services/themes/serve.py)
- [orchestrator/services/themes/defaults.py](../../orchestrator/services/themes/defaults.py)
- [orchestrator/services/themes/looks.py](../../orchestrator/services/themes/looks.py)
- [orchestrator/services/mobile/discovery.py](../../orchestrator/services/mobile/discovery.py)
- [orchestrator/api/themes_install.py](../../orchestrator/api/themes_install.py)
- [orchestrator/api/themes_looks.py](../../orchestrator/api/themes_looks.py)
- [orchestrator/tests/test_prd266_device_theme_routes.py](../../orchestrator/tests/test_prd266_device_theme_routes.py)
- [docs/theme-host/mobile.md](../theme-host/mobile.md)

</details>

Automatos Go, the phone app, talks to the same server as the web app. The server keeps your theme choice, your looks
and each theme's phone skin, so the phone can show the theme you chose on the web. A theme never sends code to the
phone: only colours, font files and mascot images.

---

## How the phone follows your theme

| Your theme on the web | On the phone |
|---|---|
| **Studio** | The phone's own Studio layout. |
| **Play** or **HUD** | Automatos Go has a Play layout and a HUD layout built in, and switches to the one you chose. Your looks for Play and HUD are the same records on the server as on the web, so a look you change on one shows on the other. |
| **Any other installed theme** | The phone keeps its Studio layout and repaints it from the theme's phone skin: its colours, its fonts and its mascot. |
| **A theme with no phone skin** | Nothing of the theme reaches the phone. |

A theme brings no layout and no code to the phone. The Play and HUD layouts are the app's own, and every other theme
can only recolour the Studio layout, change its fonts and give it a mascot.

### Where this comes from

- **Your choice.** The server keeps one theme choice per member per workspace. The phone reads the same answer the
  browser reads: your choice, else the workspace default, else Studio.
- **Your looks.** Looks are kept on the server per member, per workspace and per theme, for a theme installed in the
  workspace. See [Looks](looks.md#where-your-looks-are-kept).
- **The phone skin.** A theme can carry a phone block (`mobile` in its `theme.json`). The server turns it into data for
  the phone: colours, the URLs of font files and the URLs of the mascot's clips or still images. It never gives a phone
  the URL of a JavaScript or CSS file.

---

## The phone skin

| Part | What it is |
|---|---|
| **Colours** | Either the theme's own light and (optionally) dark colours, or your active look for that theme painted into colours (the theme's default look when you have saved none). |
| **Light or dark** | With the theme's own colours, the phone follows its own light or dark setting. With colours from your look, the tone comes from the theme and your look: the theme's fixed tone, the tone your look sets, or else light or dark by how dark your look's ground is. |
| **Fonts** | Up to three: a sans, a serif and a mono face, as `.ttf` or `.otf` files. |
| **Mascot** | Video clips with still images, still images only, or none, one per mascot state. Play's and HUD's mascot is Auto. |

When the colours come from your look, the phone shows the look you last saved on the server, not one still being edited
in a browser tab.

The scan report measures the contrast of every theme with a phone block: text on its ground at 4.5:1, the primary and
focus-ring colours at 3:1. A pair under its minimum is an **Error**, so that version can't be listed in the
Marketplace; a private upload still installs. See the **Contrast (AA)** row of the
[scan report](installing-and-choosing.md#the-checks).

---

## What the phone needs

| Need | Why |
|---|---|
| A server that offers phone themes | The server lists `mobile_theme` among the capabilities in its discovery document (`/.well-known/automatos.json`). This release does. The app offers installed themes only when it is listed. |
| The theme installed in the workspace the phone uses | A phone paired to a server belongs to exactly the workspace it was paired in. A phone signed in as you can use any workspace you belong to. |
| Your opt-in, for a private theme | The phone gets no skin for a private version you haven't opted in to. Opt in to that version in **Settings › Appearance › Themes**. |
| A phone block in the theme (any theme but Play and HUD) | A theme without one gives the phone nothing. |
| A version that isn't withdrawn | A withdrawn version gives the phone nothing. |

To pair a phone, use **Settings › Phones & devices**.

For the exact answer the phone reads, see [The phone's theme: the `mobile` block](../theme-host/mobile.md). To add a
phone block to your own theme, see the Design Kit's
[contract guide](https://github.com/AutomatosAI/automatos-design-kit/blob/main/guides/contract.md).

### The routes behind it

The server routes a phone can use. The web app reads your theme, your looks and the theme's files through the same
routes.

| Route | What it answers |
|---|---|
| `GET /api/themes/active` | Your theme in the workspace: your choice, the workspace default or Studio. |
| `PUT /api/themes/choice` | Saves your choice in the workspace. |
| `GET` / `PUT /api/themes/looks/{themeId}` | Your looks for one theme in the workspace. |
| `GET /api/themes/mobile/{installId}` | An installed theme's phone skin, or `mobile: null` when there is none for you. |
| `GET /api/themes/files/{digest}/{path}` | The font and mascot files the skin names. |
