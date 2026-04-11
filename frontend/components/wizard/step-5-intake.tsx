'use client'

/**
 * PRD-130: Step 5 — Live Intake Progress
 * ========================================
 *
 * Presentation-only wrapper around <WizardProgressFeed /> which renders
 * the SSE-backed terminal feed. All state comes from the parent shell,
 * which owns the `useWizardProgress` hook.
 */

import { Card, CardContent } from '@/components/ui/card'
import { WizardProgressFeed } from './wizard-progress-feed'
import type {
  WizardProgressEvent,
  WizardProgressState,
} from '@/hooks/use-wizard-progress'

interface Step5Props {
  pageCount: number
  events: WizardProgressEvent[]
  state: WizardProgressState
}

export function Step5Intake({ pageCount, events, state }: Step5Props) {
  return (
    <Card className="bg-secondary/30 border-border/30">
      <CardContent className="py-6 space-y-4">
        <div>
          <div className="text-lg font-medium">
            Reading {pageCount} pages…
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Scraping content, embedding to RAG, and building your knowledge
            graph. Live feed below — this runs in the background, so you can
            leave the tab and come back.
          </p>
        </div>

        <WizardProgressFeed
          events={events}
          state={state}
          pageCount={pageCount}
        />
      </CardContent>
    </Card>
  )
}
