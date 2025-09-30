'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { PerformanceAnalytics } from '@/components/analytics/performance-analytics'
import { usePageAPI } from '@/hooks/use-page-api'

export default function AnalyticsPage() {
  usePageAPI('analytics')
  
  return (
    <MainLayout>
      <PerformanceAnalytics />
    </MainLayout>
  )
}

