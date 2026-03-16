import { MainLayout } from '@/components/layout/main-layout'
import { MissionDetailPage } from '@/components/missions/mission-detail-page'

export default async function MissionDetailRoute({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params

  return (
    <MainLayout>
      <MissionDetailPage missionId={id} />
    </MainLayout>
  )
}
