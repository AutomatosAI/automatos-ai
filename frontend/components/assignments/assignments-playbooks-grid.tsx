'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { BookOpen } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PlaybooksTab } from '@/components/workflows/playbooks-tab'

// ── Filter chips ────────────────────────────────────────────────────

type PlaybookFilter = 'all' | 'mine' | 'workspace' | 'imported'

const FILTER_CHIPS: ReadonlyArray<{ value: PlaybookFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'mine', label: 'Mine' },
  { value: 'workspace', label: 'Workspace' },
  { value: 'imported', label: 'Imported' },
]

// ── Component ───────────────────────────────────────────────────────

export function AssignmentsPlaybooksGrid() {
  const router = useRouter()
  const [activeFilter, setActiveFilter] = useState<PlaybookFilter>('all')

  const emptyState = (
    <div className="text-center py-12">
      <BookOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
      <p className="text-muted-foreground mb-4">No playbooks yet.</p>
      <div className="flex items-center justify-center gap-3">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => router.push('/chat?mode=plan&from=assignments')}
        >
          + Plan one with Auto
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => router.push('/marketplace?tab=playbooks')}
        >
          Browse Marketplace
        </Button>
      </div>
    </div>
  )

  return (
    <div className="space-y-4">
      {/* Filter chips — matches marketplace-playbooks-tab styling */}
      <div className="flex gap-2 overflow-x-auto pb-1">
        {FILTER_CHIPS.map((chip) => (
          <Button
            key={chip.value}
            variant={activeFilter === chip.value ? 'default' : 'outline'}
            size="sm"
            onClick={() => setActiveFilter(chip.value)}
            className={`whitespace-nowrap flex-shrink-0 ${
              activeFilter === chip.value
                ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                : 'border-secondary text-muted-foreground hover:bg-secondary'
            }`}
          >
            {chip.label}
          </Button>
        ))}
      </div>

      {/* Reuse existing PlaybooksTab for the card grid */}
      <PlaybooksTab
        viewMode="grid"
        onUseRecipe={() => {}}
        onExecuteRecipe={(_wfId, info) => {
          if (info?.recipeExecutionId) {
            router.push(`/activity/execution?id=${info.recipeExecutionId}&recipeId=${info.recipeId || ''}`)
          }
        }}
        emptyState={emptyState}
      />
    </div>
  )
}
