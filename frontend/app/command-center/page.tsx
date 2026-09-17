'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { ActivityPage as ActivityCommandCentre } from '@/components/activity/activity-page'
import { CommandCenterShell } from '@/components/command-center/command-center-shell'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function CommandCenterPage() {
  usePageAPI('activity')
  const isStudio = useIsStudio()
  const isMobileLayout = useIsTabletOrBelow()

  // PRD-244 (two styles, two tones): the Studio style renders the shell on
  // desktop; the Classic style renders the ActivityPage, which carries the same
  // tabs in its own style. Below 1024 px both styles use the ActivityPage
  // until the mobile pass (PRD-245).
  if (isStudio && !isMobileLayout) {
    return (
      <MainLayout fullBleed>
        <CommandCenterShell />
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <ActivityCommandCentre />
    </MainLayout>
  )
}
