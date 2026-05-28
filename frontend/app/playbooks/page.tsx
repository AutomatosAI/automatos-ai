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

  // Studio: the Assignments hub is the canonical view. Forward to it
  // with the Playbooks tab pre-selected and any other params preserved
  // (e.g. ?id=<recipe> for the detail panel).
  useEffect(() => {
    if (!isStudio || isMobileLayout) return
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.set('tab', 'playbooks')
    router.replace(`/assignments?${params.toString()}` as any)
  }, [router, searchParams, isStudio, isMobileLayout])

  // Classic theme + mobile keep the existing standalone Playbooks panel.
  if (isStudio && !isMobileLayout) {
    return <MainLayout fullBleed>{null}</MainLayout>
  }

  return (
    <MainLayout>
      <PlaybooksPanel />
    </MainLayout>
  )
}
