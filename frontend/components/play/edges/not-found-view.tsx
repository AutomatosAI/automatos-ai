'use client'

import Link from 'next/link'

import { LostScreen } from '@/components/play/edges/lost-screen-lazy'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { useIsPlay } from '@/hooks/use-studio-theme'

/**
 * PRD-260 §4 — the 404, for any address the app doesn't have. Under Play at
 * >= 1024 px (D7) it is Play's "This page wandered off" (Ember mock 09); under
 * Classic, Studio, or below 1024 px, a plain page with the words of Next's own
 * 404 and a link home, so those users see no new design. A client component:
 * the width is known only in the browser, as for every Play page. app/not-found.tsx
 * renders it and sets the page's title (P260-FX-21, M68). Every route carries
 * the 404, so Play's lost screen comes through its lazy slot: Classic and
 * Studio never download it (P260-FX-23, M17).
 */
export function NotFoundView() {
  const isPlay = useIsPlay()
  const narrow = useIsTabletOrBelow()
  return isPlay && !narrow ? <LostScreen /> : <PlainNotFound />
}

/** Next's default 404 in the theme's own tokens, plus the way home. */
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
