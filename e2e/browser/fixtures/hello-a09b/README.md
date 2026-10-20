# hello-a09b — one look, two browsers (PRD-261 A-09b)

The fixture behind `specs/looks-server.spec.ts`. It is not a theme: the story is builtin Play's
looks moving from this browser's storage to the server (`GET/PUT /api/themes/looks/automatos/play`),
so the spec imports this look on Play's Looks page in one browser context and expects it in a
second, fresh context after a reload. Each fixture is its story's own (PA-17).

| File | What it is |
|---|---|
| `look.json` | A builtin Play look (`automatos-interface-look/v1`, interface `play`), catalogue keys only. The spec gives it a name unique to the run before it pastes it, so a look left by an earlier run never satisfies the check. |
