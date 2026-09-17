'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { MainLayout } from '@/components/layout/main-layout'

export default function MissionsRoute() {
  const router = useRouter()
  const searchParams = useSearchParams()

  // Single source of truth: the Assignments page renders the Missions tab.
  // /missions forwards there with the tab pre-selected and every other query
  // param preserved (e.g. ?state=awaiting_approval) — PRD-244 W2: in every
  // theme and at every width (the classic branch used to drop them).
  useEffect(() => {
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.set('tab', 'missions')
    router.replace(`/assignments?${params.toString()}` as any)
  }, [router, searchParams])

  return <MainLayout fullBleed>{null}</MainLayout>
}
