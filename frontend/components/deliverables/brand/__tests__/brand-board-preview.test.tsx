/**
 * PRD-255 Wave 2, US-010 — the brand board on the Brand kit page, against a mocked apiClient
 * holding one kit: page 1 of the board (GET /api/documents/brand-kit/board?format=png) is
 * fetched with the caller's auth and drawn again after each change the server stores (a save,
 * a logo variant uploaded); a refused save leaves it as it was; the PDF and the PNG download;
 * a board or a download that cannot be drawn says so.
 *
 * F372: the board is keyed on the kit's `updated_at` (the server stamps it on every save, by any
 * route), so a kit changed elsewhere (Auto, the designer, the API) draws it again when the page
 * reads the stamp: on focus, on its poll, or after the page's own change.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider, focusManager } from '@tanstack/react-query'
import { useState } from 'react'

import { designKit } from './brand-kit-design-kit'

const server = vi.hoisted(() => ({ kit: null as any, putError: null as Error | null, boardOk: true, saves: 0 }))
// What the server stamps on the kit at each save (save_brand_kit): F372.
const stamp = (n: number) => `2026-10-06T18:00:0${n}+00:00`

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    get: vi.fn(async (path: string) => {
      if (path === '/api/documents/brand-kit') return server.kit
      if (path === '/api/documents/brand-kit/suggestions') return { suggestions: {} }
      throw new Error(`unexpected GET ${path}`)
    }),
    put: vi.fn(async (path: string, body: any) => {
      if (path !== '/api/documents/brand-kit') throw new Error(`unexpected PUT ${path}`)
      if (server.putError) throw server.putError
      server.saves += 1
      server.kit = { ...server.kit, ...body, updated_at: stamp(server.saves) }
      return server.kit
    }),
    post: vi.fn(async (path: string) => {
      if (path !== '/api/documents/brand-kit/logo-dark') throw new Error(`unexpected POST ${path}`)
      server.saves += 1
      server.kit = { ...server.kit, logo_dark_path: 'w1/brand/logo_dark.png', updated_at: stamp(server.saves) }
      return server.kit
    }),
    delete: vi.fn(),
    getAuthHeaders: vi.fn(async () => ({ Authorization: 'Bearer t' })),
    getBaseUrl: vi.fn(() => ''),
  }
  return { apiClient, default: apiClient }
})

import { toast } from 'sonner'
import { BOARD_UNAVAILABLE, BrandBoardPreview } from '../brand-board-preview'
import { BrandKitDesign } from '../brand-kit-design'
import { useBrandKitForm } from '../use-brand-kit-form'

const BOARD_PNG = '/api/documents/brand-kit/board?format=png'
const BOARD_PDF = '/api/documents/brand-kit/board?format=pdf'

function Board() {
  const form = useBrandKitForm()
  return (
    <>
      <BrandBoardPreview changes={form.boardVersion} />
      <button type="button" onClick={() => void form.save()} disabled={!form.kit}>Save kit</button>
      <BrandKitDesign form={form} canEdit />
    </>
  )
}

function Harness() {
  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }))
  return <QueryClientProvider client={client}><Board /></QueryClientProvider>
}

// The fetch reaches the stub with jsdom's origin in front (as in brand-kit-logo-variants.test.tsx):
// what is checked is the path and the query.
const ORIGIN = /^https?:\/\/[^/]+/

function fetchedUrls(): string[] {
  return vi.mocked(fetch).mock.calls.map(([url]) => String(url).replace(ORIGIN, ''))
}

function endingWith(path: string) {
  const escaped = path.replace(/[.*+?^$()|[\]\\]/g, '\\$&')
  return expect.stringMatching(new RegExp(`${escaped}$`))
}

function boardDraws(): string[] {
  return fetchedUrls().filter((url) => url.startsWith(BOARD_PNG))
}

let clicked: { href: string; download: string }[] = []
let anchorClick: { mockRestore: () => void } | null = null

beforeEach(() => {
  server.kit = designKit({ updated_at: stamp(0) })
  server.saves = 0
  server.putError = null
  server.boardOk = true
  clicked = []
  vi.mocked(toast.error).mockClear()
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const ok = !String(url).includes('/brand-kit/board') || server.boardOk
    return { ok, status: ok ? 200 : 500, blob: async () => new Blob(['bytes']) }
  }))
  URL.createObjectURL = vi.fn(() => 'blob:board')
  URL.revokeObjectURL = vi.fn()
  anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    clicked.push({ href: this.href, download: this.download })
  })
})
afterEach(() => { cleanup(); vi.unstubAllGlobals(); anchorClick?.mockRestore() })

describe('the brand board on the Brand kit page', () => {
  it('shows page 1 of the board, fetched with the caller’s auth', async () => {
    render(<Harness />)
    const section = await screen.findByRole('region', { name: 'Brand board' })
    expect(await within(section).findByAltText('Brand board, page 1')).toHaveAttribute('src', 'blob:board')
    expect(fetch).toHaveBeenCalledWith(endingWith(`${BOARD_PNG}&v=${encodeURIComponent(stamp(0))}`), { headers: { Authorization: 'Bearer t' } })
  })

  it('draws the board again after each save', async () => {
    render(<Harness />)
    await screen.findByAltText('Brand board, page 1')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Save kit' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: 'Save kit' }))
    await waitFor(() => expect(boardDraws()).toEqual([`${BOARD_PNG}&v=${encodeURIComponent(stamp(0))}`, `${BOARD_PNG}&v=${encodeURIComponent(stamp(1))}`]))
    fireEvent.click(screen.getByRole('button', { name: 'Save kit' }))
    await waitFor(() => expect(boardDraws()).toContain(`${BOARD_PNG}&v=${encodeURIComponent(stamp(2))}`))
  })

  it('leaves the board as it was when the save is refused', async () => {
    server.putError = new Error('Invalid brand kit')
    render(<Harness />)
    await screen.findByAltText('Brand board, page 1')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Save kit' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: 'Save kit' }))
    await waitFor(() => expect(toast.error).toHaveBeenCalled())
    expect(boardDraws()).toEqual([`${BOARD_PNG}&v=${encodeURIComponent(stamp(0))}`])
  })

  it('draws the board again when a logo variant is uploaded', async () => {
    render(<Harness />)
    const variants = await screen.findByRole('region', { name: 'Logo variants' })
    const png = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'logo-dark.png', { type: 'image/png' })
    fireEvent.change(within(variants).getByLabelText('Logo for dark backgrounds file'), { target: { files: [png] } })
    await waitFor(() => expect(boardDraws()).toContain(`${BOARD_PNG}&v=${encodeURIComponent(stamp(1))}`))
  })

  it('draws the board again when the kit changed elsewhere, once the page reads the stamp (F372)', async () => {
    render(<Harness />)
    await screen.findByAltText('Brand board, page 1')
    // Auto, the designer or the API restored the kit: the server stamped it.
    server.kit = { ...server.kit, palette: { ...server.kit.palette, accent: '#c34a1a' }, updated_at: stamp(7) }
    focusManager.setFocused(false)
    focusManager.setFocused(true)
    await waitFor(() => expect(boardDraws()).toEqual([
      `${BOARD_PNG}&v=${encodeURIComponent(stamp(0))}`, `${BOARD_PNG}&v=${encodeURIComponent(stamp(7))}`,
    ]))
    focusManager.setFocused(undefined)
  })

  it('downloads the board as a PDF and as a PNG', async () => {
    render(<Harness />)
    const section = await screen.findByRole('region', { name: 'Brand board' })
    fireEvent.click(within(section).getByRole('button', { name: 'Download PDF' }))
    await waitFor(() => expect(clicked.map((c) => c.download)).toEqual(['brand-board.pdf']))
    fireEvent.click(within(section).getByRole('button', { name: 'Download PNG' }))
    await waitFor(() => expect(clicked.map((c) => c.download)).toEqual(['brand-board.pdf', 'brand-board.png']))
    expect(fetchedUrls()).toEqual(expect.arrayContaining([BOARD_PDF, BOARD_PNG]))
  })

  it('says so when the board cannot be drawn, and when a download fails', async () => {
    server.boardOk = false
    render(<Harness />)
    const section = await screen.findByRole('region', { name: 'Brand board' })
    expect(await within(section).findByText(BOARD_UNAVAILABLE)).toBeInTheDocument()
    fireEvent.click(within(section).getByRole('button', { name: 'Download PDF' }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(BOARD_UNAVAILABLE))
    expect(clicked).toEqual([])
  })
})
