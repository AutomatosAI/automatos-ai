/**
 * F354 (5 Oct): "Add to Knowledge" on a Deliverable, and removing it again.
 *
 * The owner picks the document (an invoice, a letter) that should be in the knowledge
 * base; nothing is added on its own (F305). The mutations are the shared pair
 * (use-knowledge-copy.ts); after either call the Deliverables queries refetch.
 *
 *   POST   /api/deliverables/{id}/knowledge   file the document as the owner's document
 *   DELETE /api/deliverables/{id}/knowledge   remove that copy (the Deliverable stays)
 */

import type { Deliverable } from '@/hooks/use-deliverables-api'
import { useKnowledgeCopy, type KnowledgeMutations } from '@/hooks/use-knowledge-copy'

/** The kinds of file the upload path reads the text of (services/owner_knowledge.py). */
export const KNOWLEDGE_EXTENSIONS = ['pdf', 'docx', 'xlsx', 'csv', 'md', 'txt'] as const
/** Agent reports and blog posts live elsewhere; only a Deliverable's own file is added here. */
const KNOWLEDGE_ARTIFACT_TYPES = ['document', 'spreadsheet']

/** The root of every Deliverables query (deliverableQueryKeys). */
const DELIVERABLES_QUERY = ['deliverables'] as const

function extensionOf(d: Pick<Deliverable, 'file_name' | 'file_path'>): string {
  const name = d.file_name || d.file_path || ''
  const dot = name.lastIndexOf('.')
  return dot >= 0 ? name.slice(dot + 1).toLowerCase() : ''
}

/** Whether a Deliverable is a document the owner can add to knowledge. */
export function canAddToKnowledge(
  d: Pick<Deliverable, 'artifact_type' | 'file_name' | 'file_path'>,
): boolean {
  if (!KNOWLEDGE_ARTIFACT_TYPES.includes(d.artifact_type)) return false
  return (KNOWLEDGE_EXTENSIONS as readonly string[]).includes(extensionOf(d))
}

/** Add and remove mutations for one Deliverable's knowledge copy. */
export function useDeliverableKnowledge(deliverableId: string): KnowledgeMutations {
  return useKnowledgeCopy(`/api/deliverables/${deliverableId}/knowledge`, DELIVERABLES_QUERY)
}
