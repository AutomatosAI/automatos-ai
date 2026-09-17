'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { ActivityPage as ActivityCommandCentre } from '@/components/activity/activity-page'
import { CommandCenterShell } from '@/components/command-center/command-center-shell'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function CommandCenterPage() {
  usePageAPI('activity')
  const isMobileLayout = useIsTabletOrBelow()

  // PRD-244 W1 (D2): the shell IS the Command Centre at every desktop width,
  // whatever the theme — its rules are scoped to `.cc-page`, not `.studio`.
  // Below 1024 px the classic ActivityPage remains until the mobile pass
  // (PRD-245, D6); it is deleted there, not here.
  if (!isMobileLayout) {
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
