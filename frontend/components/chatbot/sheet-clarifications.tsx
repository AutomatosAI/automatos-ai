'use client'

/**
 * The questions a database query asks before it can run (a sheet artifact whose
 * metadata.status is 'needs_clarification'): each one a button that copies it, to
 * paste into the chat and answer.
 *
 * P260-FX-05 (M41): moved out of SheetArtifact, which is over the component limit,
 * and rendered by ArtifactViewer above it. The option buttons carry token classes
 * only: a bare dark:bg- or dark:border- class beside their hover: classes wins in
 * every dark tone and cancels the hover.
 */

import { toast } from 'sonner'

const NEEDS_CLARIFICATION = 'needs_clarification'

export interface SheetClarificationsProps {
  metadata?: Record<string, unknown>
}

/** The metadata's questions, kept only when they are strings (it is agent output). */
function questionsOf(metadata: Record<string, unknown> | undefined): string[] {
  if (metadata?.status !== NEEDS_CLARIFICATION) return []
  const raw = metadata.clarifications
  return Array.isArray(raw) ? raw.filter((q): q is string => typeof q === 'string') : []
}

async function copyQuestion(question: string): Promise<void> {
  if (!navigator.clipboard) {
    toast.error('Clipboard API is not available')
    return
  }
  try {
    await navigator.clipboard.writeText(question)
    toast.success('Question copied (paste into chat to answer)')
  } catch (error) {
    console.error('Failed to copy the clarification question:', error)
    toast.error('Failed to copy')
  }
}

export function SheetClarifications({ metadata }: SheetClarificationsProps) {
  const questions = questionsOf(metadata)
  if (questions.length === 0) return null
  const infoMessage = typeof metadata?.message === 'string' ? metadata.message : undefined

  return (
    <div className="mb-6 rounded-xl border border-warning/30 bg-warning/10 p-4">
      <div className="text-sm font-semibold text-warning">Clarification needed</div>
      {infoMessage && <div className="mt-1 text-sm text-warning/80">{infoMessage}</div>}
      <div className="mt-3 space-y-2">
        {questions.map((question, idx) => (
          <button
            key={idx}
            className="w-full rounded-xl border border-border/60 bg-card/50 px-3 py-2 text-left text-sm text-foreground hover:border-warning/40 hover:bg-warning/5"
            onClick={() => void copyQuestion(question)}
            type="button"
          >
            {question}
          </button>
        ))}
      </div>
      <div className="mt-3 text-xs text-warning/70">
        Tip: answer these in chat (you can paste), then re-run the query.
      </div>
    </div>
  )
}
