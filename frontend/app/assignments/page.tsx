'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { AssignmentsPage } from '@/components/assignments/assignments-page'
import { usePageAPI } from '@/hooks/use-page-api'

export default function AssignmentsRoute() {
  usePageAPI('assignments')

  return (
    <MainLayout>
      <AssignmentsPage />
    </MainLayout>
  )
}
