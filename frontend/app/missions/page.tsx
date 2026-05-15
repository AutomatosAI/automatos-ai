'use client'

import { useRouter } from 'next/navigation'
import { useEffect } from 'react'

import { MainLayout } from '@/components/layout/main-layout'
import { StudioMissionsPage } from '@/components/assignments/studio/studio-missions-page'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function MissionsRoute() {
  const router = useRouter()
  const isStudio = useIsStudio()
  const isMobileLayout = useIsTabletOrBelow()

  // Classic theme + mobile fall back to /assignments?tab=missions —
  // we don't ship a non-Studio /missions page.
  useEffect(() => {
    if (!isStudio || isMobileLayout) {
      router.replace('/assignments?tab=missions' as any)
    }
  }, [isStudio, isMobileLayout, router])

  if (!isStudio || isMobileLayout) return null

  return (
    <MainLayout fullBleed>
      <StudioMissionsPage />
    </MainLayout>
  )
}
