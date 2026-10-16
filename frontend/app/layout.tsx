
import type { Metadata } from 'next'
import { Geist, Geist_Mono, Newsreader } from 'next/font/google'
import { cookies } from 'next/headers'
import { Providers } from '../components/providers'
import { LOOK_PREPAINT_SCRIPT } from '@/lib/looks/prepaint'
import { PLAY_TONE_COOKIE, playToneFor } from '@/lib/looks/tone'
import { parseUiStyle, STYLE_CLASS_SCRIPT, UI_STYLE_COOKIE } from '@/lib/ui-style'
import { isThemeHostSpikeOn } from '@/lib/theme-host/spike-flag'
import { ThemeImportMap } from '@/components/theme-host/theme-import-map'
import { PLAY_FONT_VARIABLE_CLASSES } from './play-fonts'
import './globals.css'
import './play-theme.css'

// Force dynamic rendering to prevent build-time Clerk errors
export const dynamic = 'force-dynamic'

// Studio rebrand typography — Geist sans, Geist Mono mono, Newsreader serif.
// Tiempos is unlicensed; Newsreader is the closest free warm display serif
// on Google Fonts. The .studio scope picks these up via CSS variables.
const geistSans = Geist({
  subsets: ['latin'],
  variable: '--font-geist-sans',
  display: 'swap',
})
const geistMono = Geist_Mono({
  subsets: ['latin'],
  variable: '--font-geist-mono',
  display: 'swap',
})
const newsreader = Newsreader({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  style: ['normal', 'italic'],
  variable: '--font-newsreader',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Automatos AI Platform',
  description: 'Enterprise AI automation and agent management platform',
}

// PRD-260 M07: <html className> is the font variables only, the same string on every render, so
// a server re-render (router.refresh after a client-side style switch) never rewrites the class
// attribute and drops next-themes' tone class. The style's classes come from STYLE_CLASS_SCRIPT.
const HTML_FONT_CLASSES = [
  geistSans.variable,
  geistMono.variable,
  newsreader.variable,
  ...PLAY_FONT_VARIABLE_CLASSES,
].join(' ')

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // PRD-244 D1: the Style axis (Classic | Studio | Play) is read from its cookie
  // here, so the chosen chrome renders on first paint with no classic flash —
  // Play as "studio play" (PRD-260 D1). The Tone axis (Light | Dark | System)
  // stays with next-themes in the browser; under Play the look's ground forces it,
  // and its own cookie lets ThemeProvider force it from the first paint (PRD-260 K4).
  const store = await cookies()
  const uiStyle = parseUiStyle(store.get(UI_STYLE_COOKIE)?.value)
  const playTone = playToneFor(uiStyle, store.get(PLAY_TONE_COOKIE)?.value)

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={HTML_FONT_CLASSES}
    >
      <head>
        {/* PRD-261 A-01: the theme import map, rendered only by a build with the spike on (off in
            every shipped build). It must precede every module script, so it is the first child. */}
        {isThemeHostSpikeOn() && <ThemeImportMap />}
        {/* PRD-260 M07: the style's classes go on <html> before first paint, ahead of the look
            script below, which needs 'play'. A constant per style from lib/ui-style.ts; the
            parsed style only picks it, no cookie text goes into it. React doesn't own these
            classes, so a server re-render changes only this script's text, which never runs again. */}
        <script dangerouslySetInnerHTML={{ __html: STYLE_CLASS_SCRIPT[uiStyle] }} />
        {/* PRD-260 D4, K3: under Play only, a constant built from lib/looks/catalogue.ts puts
            the stored look on <html> before first paint. It reads only the stored look, checked
            against the catalogue's keys; no user text ever goes into it. */}
        {uiStyle === 'play' && <script dangerouslySetInnerHTML={{ __html: LOOK_PREPAINT_SCRIPT }} />}
      </head>
      <body>
        <Providers initialUiStyle={uiStyle} initialPlayTone={playTone}>
          {children}
        </Providers>
      </body>
    </html>
  )
}
