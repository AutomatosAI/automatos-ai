/**
 * F354 (5 Oct): the owner adds a document Deliverable (an invoice, a letter) to
 * Knowledge from its panel, sees that it was added, and can take it back out.
 * Only documents offer it; a picture or an agent report does not.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const request = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({ apiClient: { request: (...args: unknown[]) => request(...args) } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { canAddToKnowledge } from '@/hooks/use-deliverable-knowledge'
import { DeliverableKnowledgeButton } from '../deliverable-knowledge-button'
import { DeliverablePreviewActions } from '../deliverable-preview-actions'

const INVOICE = {
  id: 'd-invoice-1',
  workspace_id: 'w1',
  source_type: 'task',
  source_id: '17',
  agent_id: 3,
  agent_name: 'Ledger',
  artifact_type: 'document',
  title: 'Invoice INV-0042',
  summary: null,
  storage_type: 'generated',
  file_path: 'generated/20261005_120000_Invoice_INV-0042.pdf',
  file_name: '20261005_120000_Invoice_INV-0042.pdf',
  file_type: 'pdf',
  file_size_bytes: 52_000,
  preview_url: '/api/documents/generated/20261005_120000_Invoice_INV-0042.pdf',
  preview_type: 'file',
  extra: { parties: { client_name: 'Northwind Traders' } },
  status: 'ready',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  knowledge_document_id: null as number | null,
}

function renderWithClient(node: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>)
}

function actions(deliverable: typeof INVOICE) {
  return (
    <DeliverablePreviewActions
      deliverable={deliverable}
      downloadUrl={deliverable.preview_url}
      downloading={false}
      deleting={false}
      onDownload={() => {}}
      onOpenInExplorer={() => {}}
      onDelete={() => {}}
    />
  )
}

beforeEach(() => {
  request.mockReset()
  request.mockResolvedValue({ success: true, deliverable_id: INVOICE.id, document_id: 42 })
})
afterEach(() => cleanup())

describe('which Deliverables can be added to Knowledge', () => {
  it('documents and spreadsheets the upload path reads, nothing else', () => {
    expect(canAddToKnowledge(INVOICE)).toBe(true)
    expect(canAddToKnowledge({ ...INVOICE, file_name: 'letter.docx', file_path: 'generated/letter.docx' })).toBe(true)
    expect(canAddToKnowledge({ ...INVOICE, artifact_type: 'spreadsheet', file_name: 'q3.xlsx' })).toBe(true)
    expect(canAddToKnowledge({ ...INVOICE, artifact_type: 'image', file_name: 'logo.png' })).toBe(false)
    expect(canAddToKnowledge({ ...INVOICE, artifact_type: 'report', file_name: 'standup.md' })).toBe(false)
    expect(canAddToKnowledge({ ...INVOICE, artifact_type: 'slide', file_name: 'deck.pptx' })).toBe(false)
  })
})

describe('DeliverableKnowledgeButton', () => {
  it('adds a document the owner picks', async () => {
    renderWithClient(<DeliverableKnowledgeButton deliverable={INVOICE} />)

    fireEvent.click(screen.getByRole('button', { name: /Add to Knowledge/ }))

    await waitFor(() => expect(request).toHaveBeenCalledWith('/api/deliverables/d-invoice-1/knowledge', { method: 'POST' }))
  })

  it('says it was added and offers to remove it', async () => {
    renderWithClient(<DeliverableKnowledgeButton deliverable={{ ...INVOICE, knowledge_document_id: 42 }} />)

    expect(screen.getByText('Added to Knowledge')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Add to Knowledge/ })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Remove from Knowledge' }))

    await waitFor(() => expect(request).toHaveBeenCalledWith('/api/deliverables/d-invoice-1/knowledge', { method: 'DELETE' }))
  })
})

describe("the Deliverable panel's actions", () => {
  it('offer Add to Knowledge for a document', () => {
    renderWithClient(actions(INVOICE))
    expect(screen.getByRole('button', { name: /Add to Knowledge/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Delete/ })).toBeInTheDocument()
  })

  it('do not offer it for a picture', () => {
    renderWithClient(actions({ ...INVOICE, artifact_type: 'image', file_name: 'logo.png', file_path: 'out/logo.png' }))
    expect(screen.queryByRole('button', { name: /Add to Knowledge/ })).toBeNull()
    expect(screen.getByRole('button', { name: /Open in Explorer/ })).toBeInTheDocument()
  })
})
