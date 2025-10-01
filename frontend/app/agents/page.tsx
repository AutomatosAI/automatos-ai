'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { AgentManagement } from '@/components/agents/agent-management'
import { usePageAPI } from '@/hooks/use-page-api'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function AgentsPage() {
  // Set page name for API mock configuration
  usePageAPI('agents')

  return (
    <MainLayout>
      <AgentManagement />
    </MainLayout>
  )
}
