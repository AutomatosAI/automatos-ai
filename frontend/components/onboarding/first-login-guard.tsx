'use client'

import { useEffect, useState } from 'react'
import { useUser } from '@clerk/nextjs'
import { WelcomeModal } from './welcome-modal'
import { useWorkspace } from '@/components/workspace-provider'
import { hasSeenTour, migrateFromLegacy, hasCompletedOnboarding } from '@/lib/shepherd/tour-storage'

export function FirstLoginGuard() {
  const { user, isLoaded } = useUser()
  const { workspace, isLoading: wsLoading } = useWorkspace()
  const [showWelcome, setShowWelcome] = useState(false)

  useEffect(() => {
    if (!isLoaded || !user || wsLoading || !workspace) return

    // Migrate legacy single-tour keys to per-tour format
    migrateFromLegacy(user.id)

    // Show welcome when backend confirms this is a fresh workspace (no agents yet)
    // AND the user hasn't already completed/skipped the welcome tour
    const welcomeSeen = hasSeenTour('welcome', user.id) || hasCompletedOnboarding(user.id)

    if (!welcomeSeen && workspace.isNewWorkspace) {
      const timerId = setTimeout(() => setShowWelcome(true), 1000)
      return () => clearTimeout(timerId)
    }
  }, [isLoaded, user, wsLoading, workspace])

  return (
    <WelcomeModal
      open={showWelcome}
      onOpenChange={setShowWelcome}
      userId={user?.id ?? ''}
    />
  )
}
