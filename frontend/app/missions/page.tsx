'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { MainLayout } from '@/components/layout/main-layout'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'

export default function MissionsRoute() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const isStudio = useIsStudio()
  const isMobileLayout = useIsTabletOrBelow()

  // Single source of truth: the Assignments hub renders both flip tabs.
  // /missions just forwards there with the Missions tab pre-selected,
  // preserving any other query params (e.g. ?state=awaiting_approval).
  useEffect(() => {
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.set('tab', 'missions')
    const target = isStudio && !isMobileLayout
      ? `/assignments?${params.toString()}`
      : `/assignments?tab=missions`
    router.replace(target as any)
  }, [router, searchParams, isStudio, isMobileLayout])

  return <MainLayout fullBleed>{null}</MainLayout>
}
