'use client'

import { useEffect, useState } from 'react'
import { useUser } from '@clerk/nextjs'
import { WelcomeModal } from './welcome-modal'
import { hasCompletedOnboarding } from '@/lib/shepherd/tour-storage'

export function FirstLoginGuard() {
  const { user, isLoaded } = useUser()
  const [showWelcome, setShowWelcome] = useState(false)

  useEffect(() => {
    if (!isLoaded || !user) return

    // Check if this is truly first login
    const onboardingComplete = hasCompletedOnboarding()
    const userCreatedRecently = user.createdAt &&
      (Date.now() - new Date(user.createdAt).getTime()) < 5 * 60 * 1000 // 5 mins

    if (!onboardingComplete && userCreatedRecently) {
      // Small delay to let the app render first
      setTimeout(() => setShowWelcome(true), 1000)
    }
  }, [isLoaded, user])

  return (
    <WelcomeModal
      open={showWelcome}
      onOpenChange={setShowWelcome}
    />
  )
}
