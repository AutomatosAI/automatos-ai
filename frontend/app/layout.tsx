
import type { Metadata } from 'next'
import { Geist, Geist_Mono, Newsreader } from 'next/font/google'
import { cookies } from 'next/headers'
import { Providers } from '../components/providers'
import { SKIN_PREPAINT_SCRIPT } from '@/lib/skins/prepaint'
import {
  htmlAttrsFor,
  lockedToneFor,
  STUDIO_HTML_CLASS,
  UI_SKIN_BASE_COOKIE,
  UI_SKIN_TONE_COOKIE,
  UI_STYLE_COOKIE,
} from '@/lib/ui-style'
import './globals.css'

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

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // PRD-244 D1: the Style axis (Classic | Studio) is read from its cookie here,
  // so the Studio chrome renders on first paint with no classic flash. The Tone
  // axis (Light | Dark | System) stays with next-themes in the browser.
  // PRD-259 §4: a skin (`skin:<id>`) carries its base in a second cookie
  // (decision K1), so the studio class follows the base with no database call,
  // and <html data-skin> tells the pre-paint script which cached skin to apply.
  // A single-tone skin's lock rides in a third cookie: ThemeProvider renders with
  // forcedTheme, so next-themes' script forces that tone before paint.
  const store = await cookies()
  const styleCookie = store.get(UI_STYLE_COOKIE)?.value
  const attrs = htmlAttrsFor(styleCookie, store.get(UI_SKIN_BASE_COOKIE)?.value)
  const lockedTone = lockedToneFor(styleCookie, store.get(UI_SKIN_TONE_COOKIE)?.value)
  const uiStyle = attrs.style
  const htmlClass = [
    geistSans.variable,
    geistMono.variable,
    newsreader.variable,
    attrs.studioClass ? STUDIO_HTML_CLASS : '',
  ].filter(Boolean).join(' ')

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={htmlClass}
      data-skin={attrs.skinId ?? undefined}
    >
      <head>
        {/* A constant built from constants only (lib/skins/prepaint.ts): nothing from the request. */}
        <script dangerouslySetInnerHTML={{ __html: SKIN_PREPAINT_SCRIPT }} />
      </head>
      <body>
        <Providers initialUiStyle={uiStyle} initialSkinBase={attrs.base} initialLockedTone={lockedTone}>
          {children}
        </Providers>
      </body>
    </html>
  )
}
