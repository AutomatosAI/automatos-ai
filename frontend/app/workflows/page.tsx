'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { WorkflowManagement } from '@/components/workflows/workflow-management-simple'
import { usePageAPI } from '@/hooks/use-page-api'

export default function WorkflowsPage() {
  usePageAPI('workflows')
  
  return (
    <MainLayout>
      <WorkflowManagement />
    </MainLayout>
  )
}
