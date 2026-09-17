'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { MainLayout } from '@/components/layout/main-layout'
import { PlaybooksPanel } from '@/components/playbooks/PlaybooksPanel'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function PlaybooksPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const isMobileLayout = useIsTabletOrBelow()

  // Desktop, every theme (PRD-244 W2): the Assignments hub is the canonical
  // view — forward with the Playbooks tab pre-selected and other params
  // preserved (e.g. ?id=<recipe> for the detail panel). Below 1024 px the
  // standalone panel remains until the mobile pass (PRD-245).
  useEffect(() => {
    if (isMobileLayout) return
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.set('tab', 'playbooks')
    router.replace(`/assignments?${params.toString()}` as any)
  }, [router, searchParams, isMobileLayout])

  if (!isMobileLayout) {
    return <MainLayout fullBleed>{null}</MainLayout>
  }

  return (
    <MainLayout>
      <PlaybooksPanel />
    </MainLayout>
  )
}
