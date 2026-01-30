'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { MyToolsDashboard } from '@/components/tools/my-tools-dashboard'
import { usePageAPI } from '@/hooks/use-page-api'

export default function ToolsPage() {
  usePageAPI('tools')

  return (
    <MainLayout>
      <MyToolsDashboard />
    </MainLayout>
  )
}