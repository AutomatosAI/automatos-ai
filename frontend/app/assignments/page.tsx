'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { AssignmentsPage } from '@/components/assignments/assignments-page'
import { StudioAssignmentsHub } from '@/components/assignments/studio/assignments-hub'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function AssignmentsRoute() {
  usePageAPI('assignments')
  const isStudio = useIsStudio()
  const isMobileLayout = useIsTabletOrBelow()

  // Studio desktop renders CD's HUB-A. Mobile + classic theme keep the
  // existing AssignmentsPage until CD ships a mobile pass.
  if (isStudio && !isMobileLayout) {
    return (
      <MainLayout fullBleed>
        <StudioAssignmentsHub />
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <AssignmentsPage />
    </MainLayout>
  )
}
