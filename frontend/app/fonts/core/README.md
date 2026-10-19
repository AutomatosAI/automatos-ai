# The core's type

Geist (sans), Geist Mono (mono) and Newsreader (serif), self-hosted and loaded with
`next/font/local` in [`app/core-fonts.ts`](../../core-fonts.ts) under the variables
`--font-geist-sans`, `--font-geist-mono` and `--font-newsreader`. `app/layout.tsx` puts their
classes on `<html>` for every page; the `.studio` scope and Play's Clean type style read them.

| File | Face | Licence |
|---|---|---|
| `geist-latin-wght-normal.woff2` | Geist, variable weight 100–900 | [`LICENSE-geist.txt`](LICENSE-geist.txt) |
| `geist-mono-latin-wght-normal.woff2` | Geist Mono, variable weight 100–900 | [`LICENSE-geist-mono.txt`](LICENSE-geist-mono.txt) |
| `newsreader-latin-400-normal.woff2` | Newsreader 400 | [`LICENSE-newsreader.txt`](LICENSE-newsreader.txt) |
| `newsreader-latin-400-italic.woff2` | Newsreader 400 italic | [`LICENSE-newsreader.txt`](LICENSE-newsreader.txt) |
| `newsreader-latin-500-normal.woff2` | Newsreader 500 | [`LICENSE-newsreader.txt`](LICENSE-newsreader.txt) |
| `newsreader-latin-500-italic.woff2` | Newsreader 500 italic | [`LICENSE-newsreader.txt`](LICENSE-newsreader.txt) |
| `newsreader-latin-600-normal.woff2` | Newsreader 600 | [`LICENSE-newsreader.txt`](LICENSE-newsreader.txt) |
| `newsreader-latin-600-italic.woff2` | Newsreader 600 italic | [`LICENSE-newsreader.txt`](LICENSE-newsreader.txt) |

- **Source:** the latin subsets of the Fontsource packages, copied byte for byte from
  jsDelivr's npm CDN: `@fontsource-variable/geist@5.3.0`, `@fontsource-variable/geist-mono@5.3.0`
  and `@fontsource/newsreader@5.3.0`. No npm package is added for them.
- **Licence:** all three families are under the SIL Open Font License 1.1; each package's
  licence file sits beside its fonts and travels with them.
- Never load these through `next/font/google`: it fetches from Google at build time, and
  `next build` failed whenever that fetch did (`An error occurred in next/font`). Never an
  `@fontsource` package either.
