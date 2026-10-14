# Play's display type (PRD-260 K6)

The display faces of the Play theme's type styles, self-hosted and loaded with
`next/font/local` in [`app/play-fonts.ts`](../../play-fonts.ts) (`preload: false`,
so Classic and Studio download none of them).

| File | Face | Type style | Licence |
|---|---|---|---|
| `instrument-serif-latin-400-normal.woff2` | Instrument Serif 400 | Editorial | [`LICENSE-instrument-serif.txt`](LICENSE-instrument-serif.txt) |
| `instrument-serif-latin-400-italic.woff2` | Instrument Serif 400 italic | Editorial | [`LICENSE-instrument-serif.txt`](LICENSE-instrument-serif.txt) |
| `bricolage-grotesque-latin-700-normal.woff2` | Bricolage Grotesque 700 | Characterful | [`LICENSE-bricolage-grotesque.txt`](LICENSE-bricolage-grotesque.txt) |
| `fredoka-latin-600-normal.woff2` | Fredoka 600 | Playful | [`LICENSE-fredoka.txt`](LICENSE-fredoka.txt) |

- **Source:** the latin subsets of the Fontsource packages Automatos Play self-hosts
  (`@fontsource/instrument-serif`, `@fontsource/bricolage-grotesque`, `@fontsource/fredoka`),
  copied byte for byte. No npm package is added for them.
- **Licence:** all three families are under the SIL Open Font License 1.1; each licence file
  sits beside its fonts and travels with them.
- **Geist and Geist Mono** (the Clean type style and Play's body and mono text) are not here:
  Play reuses the core's `next/font` variables, `--font-geist-sans` and `--font-geist-mono`.
- Never load these through `next/font/google` (the Turbopack trap) or an `@fontsource` package.
