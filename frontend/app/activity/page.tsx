'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { ActivityPage as ActivityCommandCentre } from '@/components/activity/activity-page'
import { usePageAPI } from '@/hooks/use-page-api'

export default function ActivityRoute() {
  usePageAPI('activity')

  return (
    <MainLayout>
      <ActivityCommandCentre />
    </MainLayout>
  )
}
