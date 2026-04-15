'use client'

import { useEffect, useRef } from 'react'
import { usePathname } from 'next/navigation'
import { useUser } from '@clerk/nextjs'

/**
 * Auto-starts the Shepherd tour for the current page on first visit.
 * Tours only fire once per page per user — completing or skipping
 * persists to localStorage so it won't trigger again.
 *
 * Drop this hook into MainLayout — it handles everything.
 */
export function useAutoTour() {
  const pathname = usePathname()
  const { user, isLoaded } = useUser()
  const startedRef = useRef<string | null>(null)

  useEffect(() => {
    if (!isLoaded || !user?.id || !pathname) return

    // Don't re-trigger if we already started a tour for this exact path this render
    if (startedRef.current === pathname) return

    let cancelled = false

    const maybeStart = async () => {
      const { getTourForRoute } = await import('@/lib/shepherd/tour-registry')
      const { hasSeenTour } = await import('@/lib/shepherd/tour-storage')

      // Don't auto-start page tours until the welcome modal has been dismissed
      if (!hasSeenTour('welcome', user.id)) return

      const entry = getTourForRoute(pathname)
      if (!entry) return
      if (hasSeenTour(entry.id, user.id)) return
      if (cancelled) return

      startedRef.current = pathname
      const tour = await entry.factory(user.id)
      if (!cancelled) tour.start()
    }

    // Short delay lets the page render and data-tour elements mount
    const timer = setTimeout(maybeStart, 800)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [pathname, user?.id, isLoaded])
}
