/**
 * AddToKnowledgeButton (F354, 5 Oct; PRE-11, 7 Oct)
 * =================================================
 *
 * The owner puts something into the knowledge base (a Deliverable's document, a ticket's
 * approved answer, a report), so an agent can find it later ("every file for customer
 * X"). Once added it says so and offers to take it back out; the source stays. Nothing
 * is ever added without this click (F305). It takes its source's add and remove
 * mutations (hooks/use-knowledge-copy.ts) and the id of the owner's copy, or null.
 */

'use client'

import { BookMinus, BookPlus, Check, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { ADDED_TEXT, type KnowledgeMutations } from '@/hooks/use-knowledge-copy'

export const ADD_TEXT = 'Add to Knowledge'
export const REMOVE_TEXT = 'Remove'

export interface AddToKnowledgeButtonProps extends KnowledgeMutations {
  /** The owner's document it was added as; null or absent when it was not added. */
  documentId?: number | null
  /** What stays when the copy is removed, for the Remove button's title: "the report". */
  keeps: string
}

export function AddToKnowledgeButton({ documentId, add, remove, keeps }: AddToKnowledgeButtonProps) {
  const busy = add.isLoading || remove.isLoading

  if (documentId == null) {
    return (
      <Button variant="outline" size="sm" onClick={() => add.mutate()} disabled={busy}>
        {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BookPlus className="mr-2 h-4 w-4" />}
        {ADD_TEXT}
      </Button>
    )
  }

  return (
    <span className="inline-flex items-center gap-1">
      <span
        className="inline-flex h-9 items-center rounded-xl border border-primary/40 bg-primary/10 px-3 text-sm text-primary"
        title={`In your Documents (document ${documentId})`}
      >
        <Check className="mr-2 h-4 w-4" />
        {ADDED_TEXT}
      </span>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => remove.mutate()}
        disabled={busy}
        aria-label="Remove from Knowledge"
        title={`Remove from Knowledge (${keeps} stays)`}
      >
        {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BookMinus className="mr-2 h-4 w-4" />}
        {REMOVE_TEXT}
      </Button>
    </span>
  )
}
