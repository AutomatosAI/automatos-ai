'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { PlaybooksPanel } from '@/components/playbooks/PlaybooksPanel'
import { StudioPlaybooksPage } from '@/components/assignments/studio/studio-playbooks-page'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function PlaybooksPage() {
  const isStudio = useIsStudio()
  const isMobileLayout = useIsTabletOrBelow()

  // Studio desktop: CD's standalone Playbooks library. Mobile + classic
  // theme keep the existing PlaybooksPanel.
  if (isStudio && !isMobileLayout) {
    return (
      <MainLayout fullBleed>
        <StudioPlaybooksPage />
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <PlaybooksPanel />
    </MainLayout>
  )
}


