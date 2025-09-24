'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { EnhancedAgentManagement } from '@/components/agents/enhanced-agent-management'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function AgentsPage() {
  return (
    <MainLayout>
      <EnhancedAgentManagement />
    </MainLayout>
  )
}
