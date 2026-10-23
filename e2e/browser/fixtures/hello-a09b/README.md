# hello-a09b — one look, two browsers (PRD-261 A-09b)

The fixture behind `specs/looks-server.spec.ts`. It is not a theme: the story is Play's looks
moving from this browser's storage to the server (`GET/PUT /api/themes/looks/automatos/play`), so
the spec imports this look on Play's Looks page in one browser context and expects it in a second,
fresh context after a reload. Since W-A8 that is installed Play, the theme kit's package, whose
look runtime keeps the looks under the same theme id. Each fixture is its story's own (PA-17).

| File | What it is |
|---|---|
| `look.json` | A Play look (`automatos-interface-look/v1`, interface `play`), catalogue keys only. The spec gives it a name unique to the run before it pastes it, so a look left by an earlier run never satisfies the check. |

## The look data of a theme with looks (PRD-261 A-19c)

A-19c puts a theme's presets and its tone rule into the boot (plan §2.2.3), so a generic look
runtime (A-19d1, A-19d2) runs a theme's looks from data alone. `theme/` is that data for a small
theme, `automatos-e2e/hello-a09b`, not Play: three knobs, three presets and a prepaint table with
a tone rule. `boot.json` is what the boot route answers for it and what the browser keeps.

| File | What it is |
|---|---|
| `theme/looks/schema.json` | The look schema: `ground` (paper, ink, dusk), `accent` (blue, green), `glow` (0 to 40). Interface `hello-a09b`. |
| `theme/looks/presets/{paper,dusk,ink}.look.json` | The presets; `paper` is the default look (`boot.json` `themeJson.look`). |
| `theme/boot/prepaint.json` | The prepaint table, with `tone: {knob: "ground", values}`: paper is light, ink and dusk are dark. |
| `boot.json` | `prepaint`, `lookGuard` and `looks` (`{defaultLook, presets}`: the default first, the rest in file-name order) as `orchestrator/services/themes/serve.py` answers them for `theme/` (`orchestrator/tests/test_prd261_boot_looks.py`), and as `frontend/lib/theme-host/boot-sanitize.ts` keeps them (`boot-server.test.ts`: nothing dropped). |
