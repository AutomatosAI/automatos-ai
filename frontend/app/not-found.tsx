import type { Metadata } from 'next'
import Link from 'next/link'

import { ThemeSlot } from '@/components/theme-host/theme-slot'

/**
 * PRD-260 §4, PRD-261 A-04b — the 404, for any address the app doesn't have. A server
 * component, so it can keep the document title Next's default 404 sets (P260-FX-21, M68): every
 * style's tab and screen reader say it, never the app's own title. The active theme's notFound@1
 * slot draws at its width (installed Play's: "This page wandered off"; an installed theme's code
 * loads only under its style, so Classic and Studio never download it); otherwise the plain page
 * below. Under a theme that draws one, nothing until the width is known (the server, hydration),
 * so a phone never mounts the theme's screen and a desktop never shows the plain page first
 * (P260-FX-25, M69).
 */

/** Next's own 404 title. */
export const NOT_FOUND_TITLE = '404: This page could not be found.'

export const metadata: Metadata = { title: NOT_FOUND_TITLE }

export default function NotFound() {
  return (
    <ThemeSlot slot="notFound@1">
      <PlainNotFound />
    </ThemeSlot>
  )
}

/** Next's default 404 in the style's own tokens, plus the way home. */
function PlainNotFound() {
  return (
    <main className="flex min-h-[100dvh] flex-col items-center justify-center gap-6 p-6 text-center text-foreground">
      <div className="flex items-center gap-5">
        <h1 className="border-r border-border pr-5 text-2xl font-medium">404</h1>
        <h2 className="text-sm font-normal">This page could not be found.</h2>
      </div>
      <Link href="/" className="text-sm text-muted-foreground underline underline-offset-4 hover:text-foreground">
        Go home
      </Link>
    </main>
  )
}
