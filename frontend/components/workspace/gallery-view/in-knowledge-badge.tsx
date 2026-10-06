/**
 * InKnowledgeBadge (F354)
 * =======================
 *
 * On a Deliverable's card and row: this document is in the owner's knowledge base
 * (added from its panel with Add to Knowledge), so the grid shows it without opening
 * each one. Nothing renders for a Deliverable that is not in knowledge.
 */

import { Check } from 'lucide-react'

import { cn } from '@/lib/utils'
import type { Deliverable } from '@/hooks/use-deliverables-api'

export const IN_KNOWLEDGE_TEXT = 'In Knowledge'

export function InKnowledgeBadge({ deliverable, className }: { deliverable: Deliverable; className?: string }) {
  if (deliverable.knowledge_document_id == null) return null
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary',
        className,
      )}
      title="In your Documents: agents can find it"
    >
      <Check className="h-3 w-3" aria-hidden />
      {IN_KNOWLEDGE_TEXT}
    </span>
  )
}
