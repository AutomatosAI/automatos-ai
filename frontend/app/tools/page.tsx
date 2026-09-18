'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { ToolsDashboard } from '@/components/tools/tools-dashboard'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsStudio } from '@/hooks/use-studio-theme'

export default function ToolsPage() {
  usePageAPI('tools')
  const isStudio = useIsStudio()

  // PRD-244 W5c (two styles): the Studio style renders the dashboard in its
  // Studio frame on desktop; the Classic style keeps the classic frame. Below
  // 1024 px both styles use Classic until the mobile pass (PRD-245).
  if (isStudio) {
    return (
      <MainLayout fullBleed>
        <ToolsDashboard variant="studio" />
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <ToolsDashboard />
    </MainLayout>
  )
}
