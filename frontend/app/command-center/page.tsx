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

  // Studio desktop renders CD's round-4 Command Centre. Mobile and classic
  // theme still use the existing ActivityPage until CD ships a mobile pass.
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
