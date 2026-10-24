# Looks

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [frontend/app/looks/page.tsx](../../frontend/app/looks/page.tsx)
- [frontend/app/looks/style/page.tsx](../../frontend/app/looks/style/page.tsx)
- [frontend/components/theme-host/no-look-editor-note.tsx](../../frontend/components/theme-host/no-look-editor-note.tsx)
- [frontend/components/theme-host/theme-look-runtime.tsx](../../frontend/components/theme-host/theme-look-runtime.tsx)
- [frontend/components/ui/theme-toggle.tsx](../../frontend/components/ui/theme-toggle.tsx)
- [frontend/lib/theme-sdk/looks.ts](../../frontend/lib/theme-sdk/looks.ts)
- [frontend/lib/theme-host/looks/provider.tsx](../../frontend/lib/theme-host/looks/provider.tsx)
- [frontend/lib/theme-host/looks/store.ts](../../frontend/lib/theme-host/looks/store.ts)
- [frontend/lib/theme-host/looks/state.ts](../../frontend/lib/theme-host/looks/state.ts)
- [frontend/lib/theme-host/looks/storage.ts](../../frontend/lib/theme-host/looks/storage.ts)
- [frontend/lib/theme-host/looks/server.ts](../../frontend/lib/theme-host/looks/server.ts)
- [frontend/lib/theme-host/looks/transfer.ts](../../frontend/lib/theme-host/looks/transfer.ts)
- [frontend/lib/theme-host/looks/tone-cookie.ts](../../frontend/lib/theme-host/looks/tone-cookie.ts)
- [frontend/lib/theme-host/looks/motion.ts](../../frontend/lib/theme-host/looks/motion.ts)
- [frontend/lib/theme-host/installed-entry.ts](../../frontend/lib/theme-host/installed-entry.ts)
- [orchestrator/api/themes_looks.py](../../orchestrator/api/themes_looks.py)
- [orchestrator/services/themes/looks.py](../../orchestrator/services/themes/looks.py)
- [orchestrator/services/themes/look_schema.py](../../orchestrator/services/themes/look_schema.py)
- [orchestrator/services/themes/updates.py](../../orchestrator/services/themes/updates.py)

</details>

A look is one way a theme can look: a set of its settings, saved under a name. Change the look and the whole app
follows at once. Looks belong to a theme. Classic and Studio have none.

---

## Presets, the default look and your own looks

| Kind | What it is |
|---|---|
| **Settings** ("knobs") | What a look is made of: the theme decides which ones it has, such as a ground, an accent, a typeface or a glow. |
| **Presets** | Looks that ship with the theme. You can pick one and tune it, but you can't change, rename or delete the preset itself: "A preset can't be changed. Save it as a new look instead." |
| **The default look** | The preset the theme names as its default. You see it until you pick or save a look of your own, and again after you delete the look you were on. |
| **Your looks** | Looks you saved. You can keep up to 100 per theme. |

---

## Where you edit a look

- **Appearance › Customise look…** opens **Make it yours** (`/looks`). The Appearance menu is the sun, moon or
  screen button at the top of the app, next to the notifications bell. The item shows only while a theme is your style
  and its layout is showing (a window 1024 pixels wide or more).
- The theme draws this page itself. Under Classic, Studio or a theme without a look page, `/looks` shows a note: "This
  style has no look to edit. Choose a theme with looks in the Appearance menu to make it yours."
- In a workspace with no theme to choose, the note says how to get one: "No theme is installed in this workspace yet. A
  workspace owner or admin can add one from Marketplace › Themes; then choose it in the Appearance menu to make it
  yours."
- In a narrow window: "It opens on a laptop or desktop screen: this theme's pages need a wider window."
- `/looks/style` is the theme's **Style sheet**: its own page that shows the theme's styles on your active look.

### What a look page can do

Automatos gives every theme the same set of look actions. The theme decides which buttons it shows and what they are
called.

| Action | What happens |
|---|---|
| Change a setting | The active look changes at once, everywhere in the app. Each change is checked against the theme's rules first. |
| Save | Keeps the look among your looks. |
| Rename | Renames one of your looks. |
| Delete | Removes one of your looks. Deleting the look you are on takes you back to the theme's default look. |
| Export | Gives the look as a `.look.json` file. |
| Import | Takes a `.look.json` file or pasted text. It must pass the theme's checks, and it gets a fresh id. A file of 1 MB or more is refused before it is read. |

What Play and HUD show (their look pages are part of each theme, in the
[Design Kit](https://github.com/AutomatosAI/automatos-design-kit)):

| Theme | Its look page |
|---|---|
| **Play** | **Make it yours** (`/looks`): an editor with **Save as new look** and **Reset** (back to the picked look's saved values); **This look**, with **Save**, **Rename**, **Duplicate** and **Delete** for one of your looks (a preset can only be duplicated), **Export** and **Copy JSON**; and **Import a look** (paste a look's JSON, or choose a `.json` file). |
| **HUD** | **Look**, with the look drawer beside a live preview: the looks row (its presets and your looks), the settings, **Save as new look**, **Reset** and **Delete** (for one of your own looks; it asks first); and **Share**, with **Export** and **Import** (paste a look's JSON). |

---

## Where your looks are kept

- **On the server, for you.** One record per member, per workspace and per theme. Other members never see your looks,
  and your looks in another workspace are separate.
- **By theme, not by version.** An update, a roll back, an uninstall and a later install of the same theme all keep
  your looks.
- **With a copy in your browser.** The page paints at once from the copy, and the theme keeps working on it when the
  server can't be reached. A change goes to the copy at once and to the server once you pause. A change the server
  never got is sent on the next load. Other tabs follow through the copy.
- **Merged once.** The first time a browser meets the server's copy, any looks only that browser kept join the
  server's.
- **When looks can't be kept** (the browser blocks site storage, or the server refused them), a theme's look page can
  say so. Play and HUD do.

Looks are kept only for a theme installed in the workspace. For any other theme the server answers "This theme isn't
installed in this workspace."

### After a theme update

A new version may change the theme's settings. Your looks are carried over each time they are read:

- a setting the new version dropped is left out;
- a setting it added, or a value it no longer accepts, takes that setting's default;
- a look that still doesn't fit is left out, and if it was your active look you get the theme's default look.

Reading never rewrites what is stored, so rolling back finds your looks as you saved them.

---

## Light, dark and motion

- **Tone.** A theme can let its look decide light or dark. The Appearance menu's tone items then rest, with "Set by
  your look". A theme that always uses one tone shows "Set by your theme". Otherwise the tone is yours.
- **Reduced motion.** When your device or browser asks for reduced motion, Automatos tells the theme, so its styles can
  stop loops and transitions. What a theme does with that is up to the theme.

Your looks are kept on the server, so a phone can read the same records: see [On your phone](on-your-phone.md).

To design a theme's settings and presets, see the Design Kit's
[looks guide](https://github.com/AutomatosAI/automatos-design-kit/blob/main/guides/looks.md).
