/**
 * "Add to Knowledge" and Remove, for whatever the owner adds (F354, 5 Oct; PRE-11, 7 Oct).
 *
 * The owner picks what goes into the knowledge base; nothing an agent makes is added on
 * its own (F305). One pair of mutations on one path: POST files the owner's copy (adding
 * again answers with that copy), DELETE removes it and the source stays. Whether it was
 * added rides the source's own answers (`knowledge_document_id`), so after either call the
 * source's queries refetch, and the Documents queries too.
 *
 *   a Deliverable   /api/deliverables/{id}/knowledge       (use-deliverable-knowledge.ts)
 *   a ticket card   /api/v1/tasks/{id}/add-to-knowledge
 *   a report        /api/reports/{id}/add-to-knowledge
 *
 * In SaaS only a workspace owner or admin may (the server answers 403 otherwise, and the
 * toast says so); in the local edition, the operator.
 */

import { useMutation, useQueryClient, type QueryKey, type UseMutationResult } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import { boardQueryKeys } from '@/hooks/use-board-tasks'
import { reportQueryKeys } from '@/hooks/use-reports-api'

export const ADDED_TEXT = 'Added to Knowledge'
export const REMOVED_TEXT = 'Removed from Knowledge'

/** The root of every Documents query. */
const DOCUMENTS_QUERY: QueryKey = ['documents']

export interface KnowledgeAnswer {
  success: boolean
  document_id?: number
  already_added?: boolean
  removed?: number
  deliverable_id?: string
  task_id?: number
  report_id?: string
}

export type KnowledgeMutation = UseMutationResult<KnowledgeAnswer, Error, void>

export interface KnowledgeMutations {
  add: KnowledgeMutation
  remove: KnowledgeMutation
}

/** Add and remove mutations for one source's knowledge copy at `path`. */
export function useKnowledgeCopy(path: string, sourceQuery: QueryKey): KnowledgeMutations {
  const queryClient = useQueryClient()

  const refresh = () => {
    for (const queryKey of [sourceQuery, DOCUMENTS_QUERY]) queryClient.invalidateQueries({ queryKey })
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

/** A ticket card's answer: the board's queries refetch. */
export function useCardKnowledge(taskId: string): KnowledgeMutations {
  return useKnowledgeCopy(`/api/v1/tasks/${taskId}/add-to-knowledge`, boardQueryKeys.all)
}

/** A report: the Reports queries refetch. */
export function useReportKnowledge(reportId: string): KnowledgeMutations {
  return useKnowledgeCopy(`/api/reports/${reportId}/add-to-knowledge`, reportQueryKeys.all)
}
