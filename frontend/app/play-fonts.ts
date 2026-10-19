import localFont from 'next/font/local'

/**
 * PRD-260 K6 — Play's display type, self-hosted (app/fonts/play, OFL) and
 * loaded with next/font/local. Each face is bound to a --pl-font-* variable
 * that lib/looks/catalogue.ts's type styles read; Geist comes from the core's
 * own next/font variables. `preload: false`: the @font-face rules ship with
 * every page, but a file downloads only when a Play type style uses it, so
 * Classic and Studio pay nothing.
 *
 * next/font requires each call's options as literals at module scope (its
 * compiler reads them at build time), so the values are written in place.
 */

const instrumentSerif = localFont({
  src: [
    { path: './fonts/play/instrument-serif-latin-400-normal.woff2', weight: '400', style: 'normal' },
    { path: './fonts/play/instrument-serif-latin-400-italic.woff2', weight: '400', style: 'italic' },
  ],
  variable: '--pl-font-instrument-serif',
  display: 'swap',
  preload: false,
})

const bricolage = localFont({
  src: [{ path: './fonts/play/bricolage-grotesque-latin-700-normal.woff2', weight: '700', style: 'normal' }],
  variable: '--pl-font-bricolage',
  display: 'swap',
  preload: false,
})

const fredoka = localFont({
  src: [{ path: './fonts/play/fredoka-latin-600-normal.woff2', weight: '600', style: 'normal' }],
  variable: '--pl-font-fredoka',
  display: 'swap',
  preload: false,
})

/** The classes that define the three --pl-font-* variables, for <html>. */
export const PLAY_FONT_VARIABLE_CLASSES: readonly string[] = [
  instrumentSerif.variable,
  bricolage.variable,
  fredoka.variable,
]
