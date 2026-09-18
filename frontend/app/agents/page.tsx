'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { AgentManagement } from '@/components/agents/agent-management'
import { AgentManagementStudio } from '@/components/agents/studio/agent-management-studio'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsStudio } from '@/hooks/use-studio-theme'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function AgentsPage() {
  // Set page name for API mock configuration
  usePageAPI('agents')
  const isStudio = useIsStudio()

  // PRD-244 W5a (two styles): the Studio style renders the Studio page on
  // desktop; the Classic style keeps AgentManagement. Below 1024 px both
  // styles use the Classic page until the mobile pass (PRD-245).
  if (isStudio) {
    return (
      <MainLayout fullBleed>
        <AgentManagementStudio />
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <AgentManagement />
    </MainLayout>
  )
}
