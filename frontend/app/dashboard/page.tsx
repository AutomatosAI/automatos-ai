'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { Dashboard } from '@/components/dashboard/dashboard'
import { usePageAPI } from '@/hooks/use-page-api'

export default function DashboardPage() {
  usePageAPI('dashboard')

  return (
    <MainLayout>
      <Dashboard />
    </MainLayout>
  )
}


