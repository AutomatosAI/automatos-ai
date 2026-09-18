'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { AssignmentsPage } from '@/components/assignments/assignments-page'
import { StudioAssignmentsHub } from '@/components/assignments/studio/assignments-hub'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsStudio } from '@/hooks/use-studio-theme'

export default function AssignmentsRoute() {
  usePageAPI('assignments')
  const isStudio = useIsStudio()

  // PRD-244 (two styles, two tones): the Studio style renders the hub on
  // desktop; the Classic style keeps the AssignmentsPage. Below 1024 px both
  // styles use the AssignmentsPage until the mobile pass (PRD-245).
  if (isStudio) {
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
