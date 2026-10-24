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
- [orchestrator/services/themes/contract.py](../../orchestrator/services/themes/contract.py)
- [frontend/components/settings/SettingsPanel.tsx](../../frontend/components/settings/SettingsPanel.tsx)
- [orchestrator/api/themes_install.py](../../orchestrator/api/themes_install.py)
- [orchestrator/api/themes_looks.py](../../orchestrator/api/themes_looks.py)
- [orchestrator/tests/test_prd266_device_theme_routes.py](../../orchestrator/tests/test_prd266_device_theme_routes.py)
- [docs/theme-host/mobile.md](../theme-host/mobile.md)

</details>

**Automatos Go, the phone app, doesn't show the themes installed in your workspace yet. That is planned.** This page
says what the server already keeps and serves for a phone, so the theme you choose and the looks you save on the web
are ready for it.

A theme never sends code to a phone: only colours, font files and mascot images.

---

## What the server keeps for a phone

- **Your choice.** The server keeps one theme choice per member per workspace. A phone gets the same answer a browser
  gets: your choice, else the workspace default, else Studio.
- **Your looks.** Looks are kept on the server per member, per workspace and per theme, for a theme installed in the
  workspace. A phone reads the same records as the web. See [Looks](looks.md#where-your-looks-are-kept).
- **The phone skin.** A theme can carry a phone block (`mobile` in its `theme.json`). The server turns it into data for
  a phone: colours, the addresses of font files and the addresses of the mascot's clips or still images. It never gives
  a phone the address of a JavaScript or CSS file.

---

## The phone skin

| Part | What it is |
|---|---|
| **Colours** | Either the theme's own light and (optionally) dark colours, or your active look for that theme painted into colours (the theme's default look when you have saved none). |
| **Light or dark** | With the theme's own colours, the server leaves light or dark to the phone's own setting. With colours from your look, the tone comes from the theme and your look: the theme's fixed tone, the tone your look sets, or else light or dark by how dark your look's ground is. |
| **Fonts** | Up to three: a sans, a serif and a mono face, as `.ttf` or `.otf` files. |
| **Mascot** | Video clips with still images, still images only, or none, one per mascot state. |

When the colours come from your look, a phone gets the look you last saved on the server, not one still being edited
in a browser tab.

The scan report measures the contrast of every theme with a phone block: each text colour on its own fill (eight pairs)
at 4.5:1, and the primary and focus-ring colours on the ground at 3:1. A pair under its minimum is an **Error**, so that
version can't be listed in the Marketplace; a private upload still installs. See the **Contrast (AA)** row of the
[scan report](installing-and-choosing.md#the-checks).

---

## When a phone gets no skin

| Case | Why |
|---|---|
| The theme isn't installed in the workspace the phone uses | A phone paired to a server belongs to exactly the workspace it was paired in. A phone signed in as you can use any workspace you belong to. |
| A private theme you haven't opted in to | A phone gets no skin for a private version until you opt in to that version in **Settings › Appearance › Themes**. |
| The theme has no phone block | A theme without one gives a phone nothing. |
| The version is withdrawn | A withdrawn version gives a phone nothing. |

To pair a phone, use **Settings › Phones & devices**.

For the exact answer a phone reads, and the routes it may call, see
[The phone's theme: the `mobile` block](../theme-host/mobile.md) (technical). To add a phone block to your own theme,
see the Design Kit's [contract guide](https://github.com/AutomatosAI/automatos-design-kit/blob/main/guides/contract.md).
