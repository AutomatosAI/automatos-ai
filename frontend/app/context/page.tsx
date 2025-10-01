'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { ContextEngineering } from '@/components/context/context-engineering'
import { usePageAPI } from '@/hooks/use-page-api'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function ContextPage() {
  usePageAPI('context')
  
  return (
    <MainLayout>
      <ContextEngineering />
    </MainLayout>
  )
}
