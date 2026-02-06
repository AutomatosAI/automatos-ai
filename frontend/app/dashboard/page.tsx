'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

// Dashboard removed — redirects to unified analytics page
export default function DashboardPage() {
  const router = useRouter()

  useEffect(() => {
    router.replace('/analytics')
  }, [router])

  return null
}
