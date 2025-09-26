'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { AgentManagement } from '@/components/agents/agent-management'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function AgentsPage() {
  return (
    <MainLayout>
      <AgentManagement />
    </MainLayout>
  )
}
