'use client'

/**
 * 3 Oct 2026 — a Delete that asks first, since a delete cannot be undone: the question, then
 * Delete <noun> or Keep it. A post's (delete-post-button.tsx) and a plan's (its page).
 */
import { useState } from 'react'
import { Trash2 } from 'lucide-react'

import { Button } from '@/components/ui/button'

const DESTRUCTIVE_OUTLINE = 'text-destructive hover:bg-destructive/10 hover:text-destructive'

interface DeleteAskedFirstProps {
  /** What is deleted, "post" or "plan": the question is "Delete post", its button "Delete post". */
  noun: string
  question: string
  busy: boolean
  onConfirm: () => void
}

export function DeleteAskedFirst({ noun, question, busy, onConfirm }: DeleteAskedFirstProps) {
  const [asking, setAsking] = useState(false)
  if (!asking) {
    return (
      <Button type="button" variant="outline" size="sm" className={DESTRUCTIVE_OUTLINE} onClick={() => setAsking(true)}>
        <Trash2 className="mr-1.5 h-4 w-4" aria-hidden />
        Delete
      </Button>
    )
  }
  return (
    <div role="alertdialog" aria-label={`Delete ${noun}`} className="flex flex-wrap items-center gap-2 rounded-lg border border-destructive/40 px-3 py-2">
      <span className="text-sm text-foreground">{question}</span>
      <Button type="button" size="sm" variant="destructive" disabled={busy} onClick={onConfirm}>
        {`Delete ${noun}`}
      </Button>
      <Button type="button" size="sm" variant="ghost" onClick={() => setAsking(false)}>Keep it</Button>
    </div>
  )
}
