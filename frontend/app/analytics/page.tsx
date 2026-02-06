'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { AnalyticsPage as UnifiedAnalytics } from '@/components/analytics/analytics-page'
import { usePageAPI } from '@/hooks/use-page-api'

export default function AnalyticsRoute() {
  usePageAPI('analytics')

  return (
    <MainLayout>
      <UnifiedAnalytics />
    </MainLayout>
  )
}

