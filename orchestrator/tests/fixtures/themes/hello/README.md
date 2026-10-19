# Hello theme

The PRD-261 test theme. `tests/fixtures/themes/build.py` zips this folder into
`automatos-e2e.hello-1.0.0.autotheme`; its variants add a mascot clip (`hello-clip`) or swap
in an entry that throws while it loads (`hello-throws`).

`orchestrator/.gitignore` ignores every `dist/` folder, so `dist/entry.js` and
`dist/theme.css` here were added with `git add -f`. A new file under `dist/` needs the same.
