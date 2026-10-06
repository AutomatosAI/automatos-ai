'use client'

/**
 * PRE-11 (Gerard, 7 Oct): in a report's view, the owner adds it to Knowledge (its text,
 * without its Execution Metrics), sees that it was added, and can take it back out; the
 * report stays. A report for a ticket waits for the ticket's approval: the server says so.
 * Only a workspace owner or admin sees it (everyone in the local edition).
 */

import { AddToKnowledgeButton } from '@/components/knowledge/add-to-knowledge-button'
import { useReportKnowledge } from '@/hooks/use-knowledge-copy'
import { useMayManageKnowledge } from '@/hooks/use-may-manage-knowledge'
import type { AgentReport } from '@/hooks/use-reports-api'

export function ReportKnowledgeButton({ report }: { report: Pick<AgentReport, 'id' | 'knowledge_document_id'> }) {
  const may = useMayManageKnowledge()
  const mutations = useReportKnowledge(report.id)
  if (!may) return null
  return <AddToKnowledgeButton documentId={report.knowledge_document_id} keeps="the report" {...mutations} />
}
