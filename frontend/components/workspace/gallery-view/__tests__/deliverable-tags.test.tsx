/**
 * 7 Oct — a Deliverable's tags show as chips on its panel (the tags it was given, and a
 * board card's tags when the card produced it); a Deliverable with none shows no chips.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, within } from '@testing-library/react'

const detail = vi.hoisted(() => ({ current: {} as Record<string, unknown> }))

vi.mock('@/lib/api-client', () => ({
  apiClient: { getBaseUrl: () => 'https://api.test', getAuthHeaders: async () => ({}) },
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ open, children }: { open: boolean; children: React.ReactNode }) => (open ? <div>{children}</div> : null),
  SheetContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
}))
vi.mock('@/hooks/use-deliverables-api', () => ({
  useDeliverable: (id: string | null) => ({
    data: id ? { success: true, deliverable: detail.current } : undefined,
    isLoading: false,
    isError: false,
  }),
  useDeleteDeliverable: () => ({ mutate: vi.fn(), isLoading: false }),
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: () => <div data-testid="file-preview" />,
  inferPreviewType: () => 'markdown',
}))
vi.mock('../add-to-knowledge-button', () => ({ AddToKnowledgeButton: () => null }))

import { DeliverablePreview } from '../deliverable-preview'

function invoice(tags?: string[]) {
  return {
    id: 'd-1',
    title: 'Invoice HL-2026-0145',
    agent_name: 'Ops Manager',
    artifact_type: 'report',
    file_path: 'reports/invoice.md',
    file_name: 'invoice.md',
    file_type: 'md',
    summary: null,
    preview_url: null,
    content: '# Invoice',
    content_url: null,
    content_error: null,
    created_at: '2026-10-07T09:00:00Z',
    ...(tags ? { tags } : {}),
  }
}

afterEach(cleanup)

describe('a Deliverable panel shows its tags', () => {
  it('shows each tag as a chip', () => {
    detail.current = invoice(['invoice', 'harbourline', 'session'])
    render(<DeliverablePreview deliverableId="d-1" open onOpenChange={() => {}} />)

    const chips = within(screen.getByRole('list', { name: 'Tags' })).getAllByRole('listitem')
    expect(chips.map((chip) => chip.textContent)).toEqual(['invoice', 'harbourline', 'session'])
  })

  it('shows no chips for a Deliverable without tags', () => {
    detail.current = invoice([])
    render(<DeliverablePreview deliverableId="d-1" open onOpenChange={() => {}} />)
    expect(screen.queryByRole('list', { name: 'Tags' })).toBeNull()

    cleanup()
    detail.current = invoice()
    render(<DeliverablePreview deliverableId="d-1" open onOpenChange={() => {}} />)
    expect(screen.queryByRole('list', { name: 'Tags' })).toBeNull()
  })
})
