import localFont from 'next/font/local'

/**
 * The core's typography — Geist sans, Geist Mono mono, Newsreader serif — self-hosted
 * (app/fonts/core, OFL) and loaded with next/font/local, so `next build` fetches nothing
 * from Google (next/font/google failed the build whenever that fetch did). Tiempos is
 * unlicensed; Newsreader is the closest free warm display serif. The .studio scope and
 * Play's Clean type style read the --font-* variables these bind.
 *
 * next/font requires each call's options as literals at module scope (its compiler reads
 * them at build time), so the values are written in place.
 */

const geistSans = localFont({
  src: [{ path: './fonts/core/geist-latin-wght-normal.woff2', weight: '100 900', style: 'normal' }],
  variable: '--font-geist-sans',
  display: 'swap',
})

const geistMono = localFont({
  src: [{ path: './fonts/core/geist-mono-latin-wght-normal.woff2', weight: '100 900', style: 'normal' }],
  variable: '--font-geist-mono',
  display: 'swap',
})

const newsreader = localFont({
  src: [
    { path: './fonts/core/newsreader-latin-400-normal.woff2', weight: '400', style: 'normal' },
    { path: './fonts/core/newsreader-latin-400-italic.woff2', weight: '400', style: 'italic' },
    { path: './fonts/core/newsreader-latin-500-normal.woff2', weight: '500', style: 'normal' },
    { path: './fonts/core/newsreader-latin-500-italic.woff2', weight: '500', style: 'italic' },
    { path: './fonts/core/newsreader-latin-600-normal.woff2', weight: '600', style: 'normal' },
    { path: './fonts/core/newsreader-latin-600-italic.woff2', weight: '600', style: 'italic' },
  ],
  variable: '--font-newsreader',
  display: 'swap',
})

/** The classes that define --font-geist-sans, --font-geist-mono and --font-newsreader, for <html>. */
export const CORE_FONT_VARIABLE_CLASSES: readonly string[] = [
  geistSans.variable,
  geistMono.variable,
  newsreader.variable,
]
