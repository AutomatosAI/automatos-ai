/**
 * PRD-251 S0.4b — video is a Deliverable. An MP4 an agent wrote shows in
 * Deliverables Outputs under its own Videos row, with the video icon in the
 * list, and the preview plays it: a <video controls preload="metadata"> fed by
 * the Deliverable's own file URL (the /files/raw route every binary
 * Deliverable already uses — no second media route).
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'

const RAW_URL = '/api/workspaces/w1/files/raw?path=renders/launch.mp4'

const state = vi.hoisted(() => ({ deliverables: [] as any[], detail: null as any }))
const api = vi.hoisted(() => ({
  getBaseUrl: () => 'https://api.test',
  getAuthHeaders: async () => ({ Authorization: 'Bearer test' }),
}))

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ open, children }: { open: boolean; children: React.ReactNode }) => (open ? <div data-testid="sheet">{children}</div> : null),
  SheetContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
}))
vi.mock('@/hooks/use-deliverables-api', () => {
  const DEFAULT_FILTERS = {
    artifact_type: null, source_type: null, source_type_exclude: null, source_id: null,
    agent_id: null, date_range: 'all', search: '',
  }
  return {
    DEFAULT_FILTERS,
    FEED_DEFAULT_FILTERS: { ...DEFAULT_FILTERS, source_type_exclude: ['heartbeat'] },
    useDeliverables: (filters: { artifact_type: string | null; source_type: string | null }) => {
      const list = state.deliverables.filter(
        (d) => (!filters.artifact_type || d.artifact_type === filters.artifact_type)
          && (filters.source_type ? d.source_type === filters.source_type : d.source_type !== 'heartbeat'),
      )
      return { data: { pages: [{ deliverables: list, total: list.length, limit: 24, offset: 0 }] }, isLoading: false }
    },
    useDeliverable: (id: string | null) => ({
      data: id ? { success: true, deliverable: state.detail } : undefined,
      isLoading: false,
      isError: false,
    }),
    useDeleteDeliverable: () => ({ mutate: vi.fn(), isLoading: false }),
  }
})

import {
  DELIVERABLE_TYPES,
  DeliverableIcon,
  deliverableLabel,
  isDeliverableType,
  type DeliverableSize,
  type DeliverableType,
} from '@/components/icons/deliverable-icon'
import { DeliverableRow } from '@/components/workspace/gallery-view/deliverable-row'
import { DeliverablePreview } from '@/components/workspace/gallery-view/deliverable-preview'
import { OutputsFeed } from '@/components/deliverables/outputs-feed'

const VIDEO = {
  id: 'd-video-1',
  workspace_id: 'w1',
  source_type: 'chat',
  source_id: null,
  agent_id: 7,
  agent_name: 'Studio',
  artifact_type: 'video',
  title: 'Launch teaser',
  summary: null,
  storage_type: 'workspace',
  file_path: 'renders/launch.mp4',
  file_name: 'launch.mp4',
  file_type: 'mp4',
  file_size_bytes: 4_200_000,
  preview_url: RAW_URL,
  preview_type: 'file',
  extra: {},
  status: 'ready',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
}

const fetchMock = vi.fn(async () => ({ ok: true, blob: async () => new Blob(['mp4-bytes'], { type: 'video/mp4' }) }))

beforeEach(() => {
  state.deliverables = []
  state.detail = { ...VIDEO, content: null, content_url: RAW_URL }
  fetchMock.mockClear()
  vi.stubGlobal('fetch', fetchMock)
  // jsdom implements no object URLs.
  Object.defineProperty(URL, 'createObjectURL', { value: vi.fn(() => 'blob:video-1'), configurable: true, writable: true })
  Object.defineProperty(URL, 'revokeObjectURL', { value: vi.fn(), configurable: true, writable: true })
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

function glyphMarkup(type: DeliverableType, size: DeliverableSize): string {
  const { container, unmount } = render(<DeliverableIcon type={type} size={size} />)
  const markup = container.querySelector('svg')!.outerHTML
  unmount()
  return markup
}

async function expectThePlayer() {
  await waitFor(() => expect(document.querySelector('video')).not.toBeNull())
  const video = document.querySelector('video')!
  expect(video).toHaveAttribute('controls')
  expect(video).toHaveAttribute('preload', 'metadata')
  expect(video).toHaveAttribute('src', 'blob:video-1')
  expect(fetchMock).toHaveBeenCalledWith(`https://api.test${RAW_URL}`, { headers: { Authorization: 'Bearer test' } })
}

describe('video is a Deliverable type', () => {
  it('is canonical, labelled Videos, with its own hero, row and badge icons', () => {
    expect(DELIVERABLE_TYPES).toContain('video')
    expect(isDeliverableType('video')).toBe(true)
    expect(deliverableLabel('video')).toBe('Videos')
    const [hero, row, badge] = [glyphMarkup('video', 'hero'), glyphMarkup('video', 'row'), glyphMarkup('video', 'badge')]
    expect(new Set([hero, row, badge]).size).toBe(3)
    expect(row).not.toBe(glyphMarkup('slide', 'row'))
  })

  it('the Deliverables list row shows the video icon in the video accent', () => {
    const videoGlyph = glyphMarkup('video', 'row')
    const { container } = render(<DeliverableRow deliverable={VIDEO as any} />)
    const svgs = Array.from(container.querySelectorAll('svg')).map((svg) => svg.outerHTML)
    expect(svgs).toContain(videoGlyph)
    expect(container.querySelector('.text-red-400')).not.toBeNull()
  })
})

describe('an agent MP4 in Deliverables Outputs', () => {
  it('appears under its own Videos row with the video artwork, and today’s count names it', () => {
    state.deliverables = [VIDEO]
    render(<OutputsFeed />)
    const section = screen.getByRole('heading', { name: 'Videos' }).closest('section')!
    expect(within(section).getByText('1 item')).toBeInTheDocument()
    expect(within(section).getByText('Launch teaser')).toBeInTheDocument()
    expect(within(section).getByRole('img', { name: 'video preview' })).toBeInTheDocument()
    // Today's hero pill: "1 video" (a canonical type absent from the pill order would vanish).
    expect(screen.getByText('video')).toBeInTheDocument()
  })

  it('opens from the Videos row into a player', async () => {
    state.deliverables = [VIDEO]
    render(<OutputsFeed />)
    const section = screen.getByRole('heading', { name: 'Videos' }).closest('section')!
    fireEvent.click(within(section).getByText('Launch teaser'))
    await expectThePlayer()
  })
})

describe('the Deliverables preview', () => {
  it('renders <video controls preload="metadata"> fed by the Deliverable file URL', async () => {
    render(<DeliverablePreview deliverableId="d-video-1" open onOpenChange={() => {}} />)
    await expectThePlayer()
    expect(screen.queryByText(/Unable to load content/)).toBeNull()
  })

  it('plays a video Deliverable whatever its file name says', async () => {
    state.detail = { ...VIDEO, file_name: 'final-cut', file_path: 'renders/final-cut', file_type: null, content: null, content_url: RAW_URL }
    render(<DeliverablePreview deliverableId="d-video-1" open onOpenChange={() => {}} />)
    await expectThePlayer()
  })
})
