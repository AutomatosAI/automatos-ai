/**
 * F354 (5 Oct): Add to Knowledge in a document Deliverable's panel (an invoice, a
 * letter), then "Added to Knowledge" with Remove; the Deliverable stays.
 */

'use client'

import { AddToKnowledgeButton } from '@/components/knowledge/add-to-knowledge-button'
import type { Deliverable } from '@/hooks/use-deliverables-api'
import { useDeliverableKnowledge } from '@/hooks/use-deliverable-knowledge'

export function DeliverableKnowledgeButton({ deliverable }: { deliverable: Deliverable }) {
  const mutations = useDeliverableKnowledge(deliverable.id)
  return <AddToKnowledgeButton documentId={deliverable.knowledge_document_id} keeps="the Deliverable" {...mutations} />
}
