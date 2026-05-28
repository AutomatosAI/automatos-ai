'use client'

import { useSearchParams, useRouter } from 'next/navigation'
import { MainLayout } from '@/components/layout/main-layout'
import { ExecutionKitchen } from '@/components/workflows/execution-kitchen'
import { PageHeader } from '@/components/shared'

export default function ExecutionPage() {
  const searchParams = useSearchParams()
  const router = useRouter()

  const executionId = searchParams.get('id')
  const recipeId = searchParams.get('recipeId')

  if (!executionId) {
    return (
      <MainLayout>
        <div className="flex items-center justify-center h-[60vh] text-muted-foreground">
          No execution ID provided.
        </div>
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <div className="space-y-4">
        <PageHeader
          title="Playbook"
          titleAccent="Execution"
          eyebrow="Activity · live run"
          lede="Every step, tool call, handoff, and decision streamed as it happens. Pause, replay, or branch the run from any row."
        />
        <ExecutionKitchen
          workflowId={0}
          onBack={() => router.push('/command-center')}
          executionType="recipe"
          recipeExecutionId={executionId}
          recipeId={recipeId || undefined}
          recipeSteps={[]}
        />
      </div>
    </MainLayout>
  )
}
