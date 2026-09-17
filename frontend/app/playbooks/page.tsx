'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { MainLayout } from '@/components/layout/main-layout'
import { PlaybooksPanel } from '@/components/playbooks/PlaybooksPanel'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function PlaybooksPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const isStudio = useIsStudio()
  const isMobileLayout = useIsTabletOrBelow()

  // Studio style, desktop: the Assignments hub is the canonical view — forward
  // with the Playbooks tab pre-selected and other params preserved (e.g.
  // ?id=<recipe> for the detail panel). PRD-244 (two styles).
  useEffect(() => {
    if (!isStudio || isMobileLayout) return
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.set('tab', 'playbooks')
    router.replace(`/assignments?${params.toString()}` as any)
  }, [router, searchParams, isStudio, isMobileLayout])

  // Classic style (and every style below 1024 px) keeps the standalone panel.
  if (isStudio && !isMobileLayout) {
    return <MainLayout fullBleed>{null}</MainLayout>
  }

  return (
    <MainLayout>
      <PlaybooksPanel />
    </MainLayout>
  )
}
