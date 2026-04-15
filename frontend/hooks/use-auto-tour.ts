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
      const { hasSeenTour, hasCompletedOnboarding } = await import('@/lib/shepherd/tour-storage')

      // If the welcome modal is still pending (user hasn't dismissed it yet),
      // don't fire page tours on top of it.  But for existing users who never
      // got the welcome modal at all, don't block them — check whether there's
      // a DOM element for the modal currently open.
      const welcomeSeen = hasSeenTour('welcome', user.id) || hasCompletedOnboarding(user.id)
      if (!welcomeSeen) {
        // The welcome modal might be about to appear — bail and let the
        // modal's own handlers mark it seen, then next navigation triggers tours.
        return
      }

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
