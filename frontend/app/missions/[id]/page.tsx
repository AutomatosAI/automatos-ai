'use client'

import { use } from 'react'
import { MainLayout } from '@/components/layout/main-layout'
import { MissionDetailPage } from '@/components/missions/mission-detail-page'

export default function MissionDetailRoute({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = use(params)

  return (
    <MainLayout>
      <MissionDetailPage missionId={id} />
    </MainLayout>
  )
}
