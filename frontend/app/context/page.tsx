
'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { ContextEngineering } from '@/components/context/context-engineering'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function ContextPage() {
  return (
    <MainLayout>
      <ContextEngineering />
    </MainLayout>
  )
}
