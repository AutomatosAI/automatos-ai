
import type { Metadata } from 'next'
import { cookies } from 'next/headers'
import { Providers } from '../components/providers'
import { parseUiStyle, UI_STYLE_COOKIE } from '@/lib/ui-style'
import { createThemeBootResolver, type CookieReader } from '@/lib/theme-host/boot'
import { CORE_FONT_VARIABLE_CLASSES } from '@/app/core-fonts'
import { installedThemeBootResolver } from '@/lib/theme-host/boot-server'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { ThemeBootScripts } from '@/components/theme-host/theme-boot'
import { ThemeImportMap } from '@/components/theme-host/theme-import-map'
import './globals.css'

// Force dynamic rendering to prevent build-time Clerk errors
export const dynamic = 'force-dynamic'

// Studio rebrand typography — Geist sans, Geist Mono mono, Newsreader serif — self-hosted
// and loaded with next/font/local in app/core-fonts.ts (no build-time Google fetch). The
// .studio scope picks these up via CSS variables.

export const metadata: Metadata = {
  title: 'Automatos AI Platform',
  description: 'Automatos Studio: agents, documents, socials and a marketplace on the open-source Automatos OS',
}

// PRD-260 M07: <html className> is the font variables only, the same string on every render, so
// a server re-render (router.refresh after a client-side style switch) never rewrites the class
// attribute and drops next-themes' tone class. The style's classes come from the theme boot
// (PRD-261 A-03). Only the core's faces are defined here: no theme's font is compiled into the
// app (A-19f1); an installed theme brings its own with its stylesheet.
const HTML_FONT_CLASSES = CORE_FONT_VARIABLE_CLASSES.join(' ')

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // PRD-244 D1: the Style axis (Classic | Studio | a theme) is read from its cookie here, so the
  // chosen chrome renders on first paint with no classic flash. PRD-261 A-03: the style's boot
  // (lib/theme-host/boot.ts) carries its <html> classes, the look it paints before first paint
  // and its tone; the Tone axis stays with next-themes in the browser, and a theme whose tone
  // follows its look forces it from the first paint through the boot's tone (PRD-260 K4).
  // PRD-261 A-08c: an installed theme's boot (its classes, prepaint and module integrity) comes
  // from the backend with the user's session, within 300 ms, else the base boot (boot-server.ts).
  const store = await cookies()
  const readCookie: CookieReader = (name) => store.get(name)?.value
  const uiStyle = parseUiStyle(readCookie(UI_STYLE_COOKIE))
  const boot = await createThemeBootResolver(readCookie, installedThemeBootResolver)(uiStyle)
  // PRD-261 W-A6b: an installed theme that runs is a registry entry from its boot, the same on the
  // server render and the client's first render (null for every other style, and a base boot).
  const installedTheme = installedThemeEntry(uiStyle, boot.entry)

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={HTML_FONT_CLASSES}
    >
      <head>
        {/* PRD-261 A-03 (PA-13): the theme import map is on every page. It must precede every
            module script, so it is the first child; then the boot's JSON element and the one
            constant script that applies it before first paint. */}
        <ThemeImportMap integrity={boot.integrity} />
        <ThemeBootScripts boot={boot} />
      </head>
      <body>
        <Providers initialUiStyle={uiStyle} initialTone={boot.tone.value ?? undefined} initialTheme={installedTheme}>
          {children}
        </Providers>
      </body>
    </html>
  )
}
