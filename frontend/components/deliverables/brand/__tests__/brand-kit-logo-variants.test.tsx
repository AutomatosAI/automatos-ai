/**
 * PRD-255 Wave 1, US-007 — the logo's variants on the Brand kit page (FR-9), against a mocked
 * apiClient: the logo for dark backgrounds and the one-colour logo are uploaded to their own
 * routes (uploads only: no URL field), shown once stored, and removed.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'

import { designKit } from './brand-kit-design-kit'

const server = vi.hoisted(() => ({ kit: null as any }))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const FIELDS: Record<string, string> = {
    '/api/documents/brand-kit/logo-dark': 'logo_dark_path',
    '/api/documents/brand-kit/logo-mono': 'logo_mono_path',
  }
  const apiClient = {
    get: vi.fn(async (path: string) => {
      if (path === '/api/documents/brand-kit') return server.kit
      if (path === '/api/documents/brand-kit/suggestions') return { suggestions: {} }
      throw new Error(`unexpected GET ${path}`)
    }),
    put: vi.fn(),
    post: vi.fn(async (path: string) => {
      if (!FIELDS[path]) throw new Error(`unexpected POST ${path}`)
      server.kit = { ...server.kit, [FIELDS[path]]: `w1/brand/${FIELDS[path]}.png` }
      return server.kit
    }),
    delete: vi.fn(async (path: string) => {
      if (!FIELDS[path]) throw new Error(`unexpected DELETE ${path}`)
      server.kit = { ...server.kit, [FIELDS[path]]: '' }
      return server.kit
    }),
    getAuthHeaders: vi.fn(async () => ({})),
    getBaseUrl: vi.fn(() => ''),
  }
  return { apiClient, default: apiClient }
})

import { apiClient } from '@/lib/api-client'
import { BrandKitDesign } from '../brand-kit-design'
import { useBrandKitForm } from '../use-brand-kit-form'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

function Harness() {
  const form = useBrandKitForm()
  return <BrandKitDesign form={form} canEdit />
}

beforeEach(() => {
  server.kit = designKit()
  Object.values(api).forEach((fn) => fn.mockClear())
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, blob: async () => new Blob(['png']) })))
  URL.createObjectURL = vi.fn(() => 'blob:variant')
  URL.revokeObjectURL = vi.fn()
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('the logo variants', () => {
  it('uploads the logo for dark backgrounds to its own route, shows it, and removes it', async () => {
    render(<Harness />)
    const section = await screen.findByRole('region', { name: 'Logo variants' })
    expect(within(section).queryByLabelText(/public image URL/)).toBeNull()
    const png = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'logo-dark.png', { type: 'image/png' })
    fireEvent.change(within(section).getByLabelText('Logo for dark backgrounds file'), { target: { files: [png] } })

    expect(await within(section).findByRole('button', { name: /Replace logo for dark backgrounds/ })).toBeInTheDocument()
    const [path, form] = api.post.mock.calls[0] as [string, FormData]
    expect(path).toBe('/api/documents/brand-kit/logo-dark')
    expect((form.get('file') as File).name).toBe('logo-dark.png')
    expect(await within(section).findByAltText('Logo for dark backgrounds')).toHaveAttribute('src', 'blob:variant')

    fireEvent.click(within(section).getByRole('button', { name: /Remove/ }))
    await waitFor(() => expect(api.delete).toHaveBeenCalledWith('/api/documents/brand-kit/logo-dark'))
    expect(await within(section).findByRole('button', { name: /Upload logo for dark backgrounds/ })).toBeInTheDocument()
  })

  it('uploads the one-colour logo to its own route', async () => {
    render(<Harness />)
    const section = await screen.findByRole('region', { name: 'Logo variants' })
    const png = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'logo-mono.png', { type: 'image/png' })
    fireEvent.change(within(section).getByLabelText('One-colour logo file'), { target: { files: [png] } })
    await waitFor(() => expect(api.post).toHaveBeenCalledTimes(1))
    expect(api.post.mock.calls[0][0]).toBe('/api/documents/brand-kit/logo-mono')
    expect(await within(section).findByRole('button', { name: /Replace one-colour logo/ })).toBeInTheDocument()
  })

  it('reads a stored variant back when the kit loads', async () => {
    server.kit = designKit({ logo_mono_path: 'w1/brand/logo_mono.png' })
    render(<Harness />)
    const section = await screen.findByRole('region', { name: 'Logo variants' })
    expect(await within(section).findByAltText('One-colour logo')).toHaveAttribute('src', 'blob:variant')
    expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/\/api\/documents\/brand-kit\/logo-mono$/), { headers: {} })
  })
})
