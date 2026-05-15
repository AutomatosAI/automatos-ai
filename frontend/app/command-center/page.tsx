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

  // Studio desktop wraps the same working ActivityPage internals in a Studio
  // shell — every widget, drag-and-drop, and clickable event is preserved.
  // Mobile + classic theme keep the original page verbatim.
  if (isStudio && !isMobileLayout) {
    return (
      <MainLayout>
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
