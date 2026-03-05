'use client'

import { useEffect, useRef } from 'react'
import { usePathname } from 'next/navigation'
import { useUser } from '@clerk/nextjs'
import { hasSeenTour } from '@/lib/shepherd/tour-storage'
import { getTourForRoute } from '@/lib/shepherd/tour-registry'

/**
 * Auto-launches the page-specific tour on first visit (after welcome tour is done).
 * 1.5s delay to let page data load before showing tour tooltips.
 * Only fires once per page per user.
 */
export function usePageTour() {
  const pathname = usePathname()
  const { user } = useUser()
  const activeTourRef = useRef<any>(null)

  useEffect(() => {
    if (!user?.id || !pathname) return

    // Don't auto-trigger page tours until welcome tour is complete
    const welcomeDone = hasSeenTour('welcome', user.id)
    if (!welcomeDone) return

    const entry = getTourForRoute(pathname)
    if (!entry) return

    // Already seen this page's tour
    if (hasSeenTour(entry.id, user.id)) return

    const timerId = setTimeout(async () => {
      try {
        const tour = await entry.factory(user.id)
        activeTourRef.current = tour
        tour.start()
      } catch (err) {
        console.warn(`[page-tour] Failed to start ${entry.id} tour:`, err)
      }
    }, 1500)

    return () => {
      clearTimeout(timerId)
      // Cancel active tour if user navigates away mid-tour
      if (activeTourRef.current?.isActive?.()) {
        activeTourRef.current.cancel()
      }
      activeTourRef.current = null
    }
  }, [pathname, user?.id])
}
