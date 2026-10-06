'use client'

/**
 * PRE-11 (Gerard, 7 Oct): in an approved ticket's view, the owner adds its answer to
 * Knowledge, sees that it was added, and can take it back out; the ticket stays.
 * Only an approved ticket with an answer offers it (the server refuses any other);
 * one already added always shows "Added to Knowledge", so it can be removed. Only a
 * workspace owner or admin sees it (everyone in the local edition): the server refuses
 * anyone else.
 */

import { AddToKnowledgeButton } from '@/components/knowledge/add-to-knowledge-button'
import { useCardKnowledge } from '@/hooks/use-knowledge-copy'
import { useMayManageKnowledge } from '@/hooks/use-may-manage-knowledge'
import type { BoardTask } from '@/types/board'

/** Whether the ticket's view offers Add to Knowledge (or shows it was added). */
export function offersKnowledge(task: BoardTask): boolean {
  if (task.knowledge_document_id != null) return true
  return task.status === 'done' && typeof task.result === 'string' && task.result.trim() !== ''
}

export function CardKnowledgeButton({ task }: { task: BoardTask }) {
  const may = useMayManageKnowledge()
  const mutations = useCardKnowledge(task.id)
  if (!may) return null
  return <AddToKnowledgeButton documentId={task.knowledge_document_id} keeps="the ticket" {...mutations} />
}
