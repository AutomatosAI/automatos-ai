/**
 * F373 — a PDF Deliverable fills the side panel instead of opening as a thin strip.
 *
 * The PDF frames in an iframe, which has no height of its own: its `h-full` sat in a body
 * box with no height, so it fell back to the browser's 150px default and only the header
 * band showed. The panel is now a flex column and a PDF's (or an HTML page's) body box
 * grows to fill it; an image keeps its natural size. jsdom does no layout, so the test
 * checks the classes that make the boxes fill.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const api = vi.hoisted(() => ({
  getBaseUrl: () => 'https://api.test',
  getAuthHeaders: async () => ({ Authorization: 'Bearer test' }),
}))
const detail = vi.hoisted(() => ({
  current: {} as Record<string, unknown>,
}))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ open, children }: { open: boolean; children: React.ReactNode }) => (open ? <div>{children}</div> : null),
  SheetContent: ({ children, className }: { children: React.ReactNode; className?: string }) => (
    <div data-testid="sheet-content" className={className}>{children}</div>
  ),
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
vi.mock('../add-to-knowledge-button', () => ({ AddToKnowledgeButton: () => null }))

import { DeliverablePreview } from '../deliverable-preview'

function deliverable(fileName: string, fileType: string) {
  return {
    id: 'd-1',
    title: 'R4final Brand Board',
    agent_name: 'Designer',
    artifact_type: 'document',
    file_path: `documents/${fileName}`,
    file_name: fileName,
    file_type: fileType,
    summary: null,
    preview_url: `/api/documents/generated/${fileName}`,
    content: null,
    content_url: `/api/documents/generated/${fileName}`,
    content_error: null,
    created_at: '2026-10-05T09:00:00Z',
  }
}

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, blob: async () => new Blob(['%PDF'], { type: 'application/pdf' }) })),
  )
  Object.defineProperty(URL, 'createObjectURL', { value: vi.fn(() => 'blob:board-1'), configurable: true, writable: true })
  Object.defineProperty(URL, 'revokeObjectURL', { value: vi.fn(), configurable: true, writable: true })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('F373 a PDF Deliverable fills the side panel', () => {
  it('grows the PDF frame to the panel height', async () => {
    detail.current = deliverable('r4final-brand-board.pdf', 'pdf')
    render(<DeliverablePreview deliverableId="d-1" open onOpenChange={() => {}} />)

    const frame = await screen.findByTitle('r4final-brand-board.pdf')
    expect(frame.tagName).toBe('IFRAME')
    expect(frame.className).toContain('flex-1')

    const body = screen.getByTestId('deliverable-preview-body')
    expect(body.className.split(' ')).toEqual(expect.arrayContaining(['flex', 'flex-col', 'flex-1', 'min-h-[50vh]']))
    expect(body.contains(frame)).toBe(true)
    expect(body.parentElement?.className).toContain('flex-1')
    expect(screen.getByTestId('sheet-content').className.split(' ')).toEqual(
      expect.arrayContaining(['flex', 'flex-col']),
    )
  })

  it('leaves an image at its natural size', async () => {
    detail.current = deliverable('moodboard.png', 'png')
    render(<DeliverablePreview deliverableId="d-1" open onOpenChange={() => {}} />)

    await screen.findByRole('img', { name: 'moodboard.png' })
    expect(screen.getByTestId('deliverable-preview-body').className).toBe('')
  })
})
