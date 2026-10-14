
import type { Metadata } from 'next'
import { Geist, Geist_Mono, Newsreader } from 'next/font/google'
import { cookies } from 'next/headers'
import { Providers } from '../components/providers'
import { parseUiStyle, styleHtmlClasses, UI_STYLE_COOKIE } from '@/lib/ui-style'
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
  // PRD-244 D1: the Style axis (Classic | Studio | Play) is read from its cookie
  // here, so the chosen chrome renders on first paint with no classic flash —
  // Play as "studio play" (PRD-260 D1). The Tone axis (Light | Dark | System)
  // stays with next-themes in the browser.
  const uiStyle = parseUiStyle((await cookies()).get(UI_STYLE_COOKIE)?.value)
  const htmlClass = [
    geistSans.variable,
    geistMono.variable,
    newsreader.variable,
    ...styleHtmlClasses(uiStyle),
  ].join(' ')

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={htmlClass}
    >
      <body>
        <Providers initialUiStyle={uiStyle}>
          {children}
        </Providers>
      </body>
    </html>
  )
}
