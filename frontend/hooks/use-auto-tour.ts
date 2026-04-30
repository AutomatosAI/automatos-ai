'use client'

import { useEffect, useRef } from 'react'
import { usePathname } from 'next/navigation'
import { useUser } from '@clerk/nextjs'
import { useWorkspace } from '@/components/workspace-provider'

/**
 * Auto-starts the Shepherd tour for the current page on first visit.
 *
 * Gating:
 *   - Only fires for NEW workspaces (workspace.isNewWorkspace === true).
 *     Existing users never see auto-tours; they can still launch any tour
 *     manually via the Guide button.
 *   - Once a tour completes/skips, that page's tour never fires again
 *     (per-user localStorage).
 *
 * Drop this hook into MainLayout — it handles everything.
 */
export function useAutoTour() {
  const pathname = usePathname()
  const { user, isLoaded } = useUser()
  const { workspace, isLoading: wsLoading } = useWorkspace()
  const startedRef = useRef<string | null>(null)

  useEffect(() => {
    if (!isLoaded || !user?.id || !pathname) return
    if (wsLoading || !workspace) return

    // Auto-tours are for new workspaces only. Existing users get nothing
    // unless they manually open a tour from the Guide button.
    if (!workspace.isNewWorkspace) return

    // No Shepherd tours on mobile — welcome modal only
    if (typeof window !== 'undefined' && window.innerWidth <= 640) return

    // Don't re-trigger if we already started a tour for this exact path this render
    if (startedRef.current === pathname) return

    let cancelled = false

    const maybeStart = async () => {
      const { getTourForRoute } = await import('@/lib/shepherd/tour-registry')
      const { hasSeenTour, hasCompletedOnboarding } = await import('@/lib/shepherd/tour-storage')

      // If the welcome modal is still pending (user hasn't dismissed it yet),
      // don't fire page tours on top of it.
      const welcomeSeen = hasSeenTour('welcome', user.id) || hasCompletedOnboarding(user.id)
      if (!welcomeSeen) return

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
  }, [pathname, user?.id, isLoaded, workspace?.isNewWorkspace, wsLoading])
}
