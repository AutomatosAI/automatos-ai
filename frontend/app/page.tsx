'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { Dashboard } from '@/components/dashboard/dashboard'
import { usePageAPI } from '@/hooks/use-page-api'

export default function Home() {
  usePageAPI('dashboard')
  
  return (
    <MainLayout>
      <Dashboard />
    </MainLayout>
  )
}
