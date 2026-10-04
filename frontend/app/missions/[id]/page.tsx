import { MainLayout } from '@/components/layout/main-layout'
import { MissionDetailPage } from '@/components/missions/mission-detail-page'
import { MissionStepCheckToggle } from '@/components/missions/mission-step-check-toggle'

export default async function MissionDetailRoute({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params

  return (
    <MainLayout>
      <div className="flex flex-col h-full min-h-0">
        <MissionStepCheckToggle missionId={id} />
        <div className="flex-1 min-h-0">
          <MissionDetailPage missionId={id} />
        </div>
      </div>
    </MainLayout>
  )
}
