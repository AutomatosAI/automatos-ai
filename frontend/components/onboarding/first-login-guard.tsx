'use client'

import { useEffect, useState } from 'react'
import { useUser } from '@clerk/nextjs'
import { WelcomeModal } from './welcome-modal'
import { useWorkspace } from '@/components/workspace-provider'
import { hasCompletedOnboarding } from '@/lib/shepherd/tour-storage'

export function FirstLoginGuard() {
  const { user, isLoaded } = useUser()
  const { workspace, isLoading: wsLoading } = useWorkspace()
  const [showWelcome, setShowWelcome] = useState(false)

  useEffect(() => {
    if (!isLoaded || !user || wsLoading || !workspace) return

    // Show welcome when backend confirms this is a fresh workspace (no agents yet)
    // AND the user hasn't already completed/skipped onboarding (per-user check)
    const onboardingComplete = hasCompletedOnboarding(user.id)

    if (!onboardingComplete && workspace.isNewWorkspace) {
      // Small delay to let the app render first
      setTimeout(() => setShowWelcome(true), 1000)
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
