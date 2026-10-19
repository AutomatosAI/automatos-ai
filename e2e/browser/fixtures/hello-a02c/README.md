# hello-a02c — the installed package behind the theme choice (PRD-261 A-02c)

The fixture behind `specs/theme-choice.spec.ts`. The package is uploaded to the local backend and
opted in to, so the root layout's server-side boot (`GET /api/themes/boot/{installId}`, which no
browser route can intercept) paints it after the tab reloads into it.

| File | What it is |
|---|---|
| `build.py` | Builds A-06a's `hello` package under the id `automatos-e2e/hello-a02c` and the name `Hello choice A-02c`, so it never meets another spec's upload and the theme menu lists it by its own name. The spec runs it with `python3 build.py --out <file>`. |

The spec never changes the local operator's stored choice, which every other spec's tab now reads
once (A-02c): the menu's `PUT /api/themes/choice` and the browser's `GET /api/themes/active` are
answered in the spec, from what the first browser stored.
