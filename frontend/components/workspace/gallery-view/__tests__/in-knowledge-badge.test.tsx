/**
 * F354: a Deliverable the owner added to knowledge says so on its card (grid) and its row
 * (list), so the owner sees it without opening each one; one not in knowledge shows nothing.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  useAuthenticatedBlobUrl: () => ({ src: null, error: null }),
}))
vi.mock('@/components/deliverables/deliverable-artwork', () => ({
  DeliverableArtwork: () => <div data-testid="artwork" />,
}))

import type { Deliverable } from '@/hooks/use-deliverables-api'
import { DeliverableCard } from '@/components/workspace/gallery-view/deliverable-card'
import { DeliverableRow } from '@/components/workspace/gallery-view/deliverable-row'
import { IN_KNOWLEDGE_TEXT } from '@/components/workspace/gallery-view/in-knowledge-badge'

afterEach(cleanup)

const INVOICE = {
  id: '5f0c2a8e-2b7d-4c55-9a51-0d6f1e2b2980',
  artifact_type: 'document',
  source_type: 'task',
  title: 'Invoice INV-0042',
  agent_name: 'Ledger',
  created_at: '2026-10-05T09:00:00Z',
  file_size_bytes: 20480,
  preview_url: '/api/documents/generated/inv-0042.pdf',
  thumbnail_url: null,
} as unknown as Deliverable

describe.each([
  ['card', DeliverableCard],
  ['row', DeliverableRow],
])('the Deliverable %s', (_name, View) => {
  it('says "In Knowledge" once the owner added it', () => {
    render(<View deliverable={{ ...INVOICE, knowledge_document_id: 12 }} />)
    expect(screen.getByText(IN_KNOWLEDGE_TEXT)).toBeTruthy()
  })

  it('shows nothing when it is not in knowledge', () => {
    render(<View deliverable={{ ...INVOICE, knowledge_document_id: null }} />)
    expect(screen.queryByText(IN_KNOWLEDGE_TEXT)).toBeNull()
  })

  it('shows nothing when the answer has no knowledge field', () => {
    render(<View deliverable={INVOICE} />)
    expect(screen.queryByText(IN_KNOWLEDGE_TEXT)).toBeNull()
  })
})
