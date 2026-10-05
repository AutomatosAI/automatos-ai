/**
 * AddToKnowledgeButton (F354, 5 Oct)
 * ==================================
 *
 * In a Deliverable's panel: the owner puts this document (an invoice, a letter) into
 * the knowledge base, so an agent can find it later ("every file for customer X").
 * Once added it says so and offers to take it back out. Only documents show it
 * (`canAddToKnowledge`); nothing is ever added without this click (F305).
 */

'use client'

import { BookMinus, BookPlus, Check, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { Deliverable } from '@/hooks/use-deliverables-api'
import { ADDED_TEXT, useDeliverableKnowledge } from '@/hooks/use-deliverable-knowledge'

export const ADD_TEXT = 'Add to Knowledge'
export const REMOVE_TEXT = 'Remove'

export function AddToKnowledgeButton({ deliverable }: { deliverable: Deliverable }) {
  const { add, remove } = useDeliverableKnowledge(deliverable.id)
  const added = deliverable.knowledge_document_id != null
  const busy = add.isLoading || remove.isLoading

  if (!added) {
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
        title={`In your Documents (document ${deliverable.knowledge_document_id})`}
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
        title="Remove from Knowledge (the Deliverable stays)"
      >
        {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BookMinus className="mr-2 h-4 w-4" />}
        {REMOVE_TEXT}
      </Button>
    </span>
  )
}
