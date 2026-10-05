/**
 * F354 (5 Oct): "Add to Knowledge" on a Deliverable, and removing it again.
 *
 * The owner picks the document (an invoice, a letter) that should be in the knowledge
 * base; nothing is added on its own (F305). Whether one was added rides the
 * Deliverable itself (`knowledge_document_id` on the list and detail answers), so
 * after either call the Deliverables queries refetch, and the Documents queries too.
 *
 *   POST   /api/deliverables/{id}/knowledge   file the document as the owner's document
 *   DELETE /api/deliverables/{id}/knowledge   remove that copy (the Deliverable stays)
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import type { Deliverable } from '@/hooks/use-deliverables-api'

/** The kinds of file the upload path reads the text of (services/owner_knowledge.py). */
export const KNOWLEDGE_EXTENSIONS = ['pdf', 'docx', 'xlsx', 'csv', 'md', 'txt'] as const
/** Agent reports and blog posts live elsewhere; only a Deliverable's own file is added here. */
const KNOWLEDGE_ARTIFACT_TYPES = ['document', 'spreadsheet']

/** The roots of every Deliverables query (deliverableQueryKeys) and every Documents query. */
const REFRESHED_QUERIES = [['deliverables'], ['documents']] as const

export const ADDED_TEXT = 'Added to Knowledge'
export const REMOVED_TEXT = 'Removed from Knowledge'

export interface KnowledgeAnswer {
  success: boolean
  deliverable_id: string
  document_id?: number
  already_added?: boolean
  removed?: number
}

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
export function useDeliverableKnowledge(deliverableId: string) {
  const queryClient = useQueryClient()
  const path = `/api/deliverables/${deliverableId}/knowledge`

  const refresh = () => {
    REFRESHED_QUERIES.forEach((queryKey) => queryClient.invalidateQueries({ queryKey: [...queryKey] }))
  }
  const failed = (fallback: string) => (error: Error) => toast.error(error.message || fallback)

  const add = useMutation<KnowledgeAnswer, Error, void>({
    mutationFn: () => apiClient.request<KnowledgeAnswer>(path, { method: 'POST' }),
    onSuccess: () => {
      refresh()
      toast.success(ADDED_TEXT)
    },
    onError: failed('Could not add it to Knowledge'),
  })

  const remove = useMutation<KnowledgeAnswer, Error, void>({
    mutationFn: () => apiClient.request<KnowledgeAnswer>(path, { method: 'DELETE' }),
    onSuccess: () => {
      refresh()
      toast.success(REMOVED_TEXT)
    },
    onError: failed('Could not remove it from Knowledge'),
  })

  return { add, remove }
}
