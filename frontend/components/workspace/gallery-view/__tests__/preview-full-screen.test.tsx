/**
 * Gerard, 7 Oct: a PDF (or web page) Deliverable's preview switches to full screen and back.
 *
 * The panel's action row offers "Full screen" for a preview that fills the panel. It asks
 * the browser to make the preview box full screen; inside, "Exit full screen" (or Esc)
 * leaves it. Where the browser refuses or has no Fullscreen API, the box covers the window
 * instead. An image has no button: it already opens at full size.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react'

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
  // The jsdom document has no Fullscreen API: each test installs what it needs.
  delete (HTMLElement.prototype as { requestFullscreen?: unknown }).requestFullscreen
  delete (document as { exitFullscreen?: unknown }).exitFullscreen
  Object.defineProperty(document, 'fullscreenElement', { value: null, configurable: true, writable: true })
})

/** The browser's Fullscreen API, faked: entering marks the element and fires the change event. */
function installFullscreen(refuse = false) {
  const exit = vi.fn(async () => {
    Object.defineProperty(document, 'fullscreenElement', { value: null, configurable: true, writable: true })
    document.dispatchEvent(new Event('fullscreenchange'))
  })
  const target: { element: HTMLElement | null } = { element: null }
  const request = vi.fn(async function (this: HTMLElement) {
    if (refuse) throw new Error('Permissions check failed')
    target.element = this
    Object.defineProperty(document, 'fullscreenElement', { value: this, configurable: true, writable: true })
    document.dispatchEvent(new Event('fullscreenchange'))
  })
  Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', { value: request, configurable: true, writable: true })
  Object.defineProperty(document, 'exitFullscreen', { value: exit, configurable: true, writable: true })
  return { request, exit, target }
}

async function openPdf() {
  detail.current = deliverable('r4final-brand-board.pdf', 'pdf')
  render(<DeliverablePreview deliverableId="d-1" open onOpenChange={() => {}} />)
  await screen.findByTitle('r4final-brand-board.pdf')
  return screen.getByTestId('deliverable-preview-body')
}


describe('a PDF preview goes full screen and back', () => {
  it('asks the browser to make the preview box full screen, and leaves it', async () => {
    const fs = installFullscreen()
    const body = await openPdf()

    await act(async () => fireEvent.click(screen.getByRole('button', { name: /full screen/i })))
    expect(fs.request).toHaveBeenCalledTimes(1)
    expect(fs.target.element).toBe(body)
    expect(body.className).toContain('bg-background')

    await act(async () => fireEvent.click(screen.getByRole('button', { name: /exit full screen/i })))
    expect(fs.exit).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('button', { name: /exit full screen/i })).toBeNull()
    expect(body.className).toContain('min-h-[50vh]')
  })

  it('covers the window when the browser refuses, and Esc leaves it', async () => {
    installFullscreen(true)
    const body = await openPdf()

    await act(async () => fireEvent.click(screen.getByRole('button', { name: /full screen/i })))
    expect(body.className.split(' ')).toEqual(expect.arrayContaining(['fixed', 'inset-0']))
    expect(screen.getByRole('button', { name: /exit full screen/i })).toBeTruthy()

    await act(async () => fireEvent.keyDown(window, { key: 'Escape' }))
    expect(body.className).not.toContain('fixed')
  })

  it('covers the window when there is no Fullscreen API at all', async () => {
    const body = await openPdf()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: /full screen/i })))
    expect(body.className.split(' ')).toEqual(expect.arrayContaining(['fixed', 'inset-0']))
  })

  it('offers no full screen for an image', async () => {
    detail.current = deliverable('moodboard.png', 'png')
    render(<DeliverablePreview deliverableId="d-1" open onOpenChange={() => {}} />)
    await screen.findByRole('img', { name: 'moodboard.png' })
    expect(screen.queryByRole('button', { name: /full screen/i })).toBeNull()
  })
})
