'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { ToolsDashboard } from '@/components/tools/tools-dashboard'
import { usePageAPI } from '@/hooks/use-page-api'

export default function ToolsPage() {
  usePageAPI('tools')
  
  return (
    <MainLayout>
      <ToolsDashboard />
    </MainLayout>
  )
}