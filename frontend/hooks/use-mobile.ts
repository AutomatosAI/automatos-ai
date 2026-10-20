import { useState, useEffect } from 'react'

import { useIsBelowWidth } from '@/hooks/use-below-width'

const MOBILE_BREAKPOINT = 768
const TABLET_BREAKPOINT = 1024
/** The server has no window: a wide one, as its HTML has always been drawn (hydration reads it too). */
const TABLET_OR_BELOW_ON_SERVER = false

export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
    const handler = (e: MediaQueryListEvent | MediaQueryList) => setIsMobile(e.matches)
    handler(mql)
    mql.addEventListener('change', handler)
    return () => mql.removeEventListener('change', handler)
  }, [])

  return isMobile
}

/**
 * Below 1024 px, known on a browser mount's first render (hooks/use-below-width.ts, the reader
 * useThemeLayout shares). It used to start false and flip in an effect, so on a phone every client
 * navigation drew MainLayout's desktop branch for one render: under an installed theme, the
 * theme's frame mounted (its /health poll, Recent) and was taken back (P260-FX-25, M69). The server
 * and hydration still read false, so the server's HTML is unchanged.
 */
export function useIsTabletOrBelow(): boolean {
  return useIsBelowWidth(TABLET_BREAKPOINT, TABLET_OR_BELOW_ON_SERVER)
}
