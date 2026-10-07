/**
 * PRE-11 (Gerard, 7 Oct): a report's view offers Add to Knowledge once the report has
 * loaded, then "Added to Knowledge" with Remove (the report stays). Only a workspace
 * owner or admin sees it in SaaS.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const request = vi.hoisted(() => vi.fn())
const caller = vi.hoisted(() => ({ role: 'owner' as string | null }))

vi.mock('@/lib/api-client', () => ({ apiClient: { request: (...args: unknown[]) => request(...args) } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => (caller.role ? { workspace: { role: caller.role } } : null),
}))

import type { AgentReport } from '@/hooks/use-reports-api'
import { ReportKnowledgeButton } from '../report-knowledge-button'
import { ReportViewerHeader } from '../report-viewer-parts'

const REPORT = {
  id: 'r-7f3a',
  title: 'Supplier terms',
  knowledge_document_id: null,
} as unknown as AgentReport

function withClient(node: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>)
}

beforeEach(() => {
  caller.role = 'owner'
  request.mockReset()
  request.mockResolvedValue({ success: true, report_id: REPORT.id, document_id: 9, already_added: false })
})
afterEach(() => cleanup())

describe('ReportKnowledgeButton', () => {
  it('adds the report', async () => {
    withClient(<ReportKnowledgeButton report={REPORT} />)

    fireEvent.click(screen.getByRole('button', { name: /Add to Knowledge/ }))

    await waitFor(() => expect(request).toHaveBeenCalledWith('/api/reports/r-7f3a/add-to-knowledge', { method: 'POST' }))
  })

  it('says it was added and removes the copy', async () => {
    withClient(<ReportKnowledgeButton report={{ ...REPORT, knowledge_document_id: 9 }} />)

    expect(screen.getByText('Added to Knowledge')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Remove from Knowledge' }))

    await waitFor(() => expect(request).toHaveBeenCalledWith('/api/reports/r-7f3a/add-to-knowledge', { method: 'DELETE' }))
  })
})

describe("the report viewer's header", () => {
  it('offers Add to Knowledge beside Download once the report has loaded', () => {
    withClient(<ReportViewerHeader report={REPORT} onClose={() => {}} onDownload={() => {}} />)
    expect(screen.getByRole('button', { name: /Add to Knowledge/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Download/ })).toBeInTheDocument()
    cleanup()

    withClient(<ReportViewerHeader onClose={() => {}} onDownload={() => {}} />)
    expect(screen.queryByRole('button', { name: /Add to Knowledge/ })).toBeNull()
  })
})

describe('who sees it', () => {
  it('an admin does; an editor, a viewer or a member does not', () => {
    caller.role = 'admin'
    withClient(<ReportKnowledgeButton report={REPORT} />)
    expect(screen.getByRole('button', { name: /Add to Knowledge/ })).toBeInTheDocument()

    for (const role of ['editor', 'viewer', 'member']) {
      cleanup()
      caller.role = role
      withClient(<ReportViewerHeader report={REPORT} onClose={() => {}} onDownload={() => {}} />)
      expect(screen.queryByRole('button', { name: /Add to Knowledge/ })).toBeNull()
      expect(screen.getByRole('button', { name: /Download/ })).toBeInTheDocument()
    }
  })
})
