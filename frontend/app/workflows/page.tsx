'use client'

import { useSearchParams } from 'next/navigation'
import { redirect } from 'next/navigation'
import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

/**
 * /workflows is deprecated. Redirect to /activity, or to
 * /activity/execution if openExecution params are present.
 */
export default function WorkflowsPage() {
  const searchParams = useSearchParams()
  const router = useRouter()

  const openExecution = searchParams.get('openExecution')
  const recipeId = searchParams.get('recipeId')

  useEffect(() => {
    if (openExecution) {
      router.replace(`/activity/execution?id=${openExecution}${recipeId ? `&recipeId=${recipeId}` : ''}`)
    } else {
      router.replace('/activity')
    }
  }, [openExecution, recipeId, router])

  return null
}
