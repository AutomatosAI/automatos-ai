'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { AssignmentsPage } from '@/components/assignments/assignments-page'
import { StudioAssignmentsHub } from '@/components/assignments/studio/assignments-hub'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function AssignmentsRoute() {
  usePageAPI('assignments')
  const isMobileLayout = useIsTabletOrBelow()

  // PRD-244 W2 (D3): the hub is the Assignments page at every desktop width,
  // whatever the theme (its root is `.cc-page`, its rules hang off it). Below
  // 1024 px the classic AssignmentsPage remains until the mobile pass (PRD-245).
  if (!isMobileLayout) {
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
