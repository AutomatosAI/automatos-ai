/**
 * PRD-251 S1.3 (US-108, D5) — the brand kit dialog, opened from the Socials tab,
 * against a mocked apiClient that holds one brand kit:
 *
 * * an owner or admin opens "Brand kit" from the Socials tab; an editor or a
 *   viewer, who cannot save it (workspace:manage), gets no button;
 * * the dialog shows the D5 fields: the body font and the heading font, the
 *   uploaded font files, the logo mark, the social handles and the voice;
 * * Save sends them in PUT /api/documents/brand-kit; a count of tone words other
 *   than three to five (or none) holds Save back and says why, and a refusal from
 *   the server names the field;
 * * a woff2 goes up with the face it provides (POST …/brand-kit/fonts, FormData)
 *   and is listed; removing it calls DELETE …/brand-kit/fonts/{id}.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const FONT_ID = 'a'.repeat(32)
const NEW_FONT_ID = 'b'.repeat(32)

const server = vi.hoisted(() => ({
  role: 'owner' as string,
  kit: null as any,
  putError: null as Error | null,
}))

function startingKit() {
  return {
    name: 'Acme', tagline: 'Build better', logo_url: '', logo_path: '',
    primary_color: '#1a1a2e', secondary_color: '#16213e', accent_color: '#0f3460', text_color: '#1a1a2e',
    font_family: 'Inter, sans-serif',
    company: { name: 'Acme Ltd', address: '', email: '', phone: '', website: 'acme.com' },
    heading_font: '"Brand Display", serif',
    font_files: [{
      id: FONT_ID, family: 'Brand Display', weight: 700, style: 'normal',
      path: `w1/brand/fonts/${FONT_ID}.woff2`, file_name: 'brand-display-700.woff2', bytes: 4712,
    }],
    logo_mark_url: '', logo_mark_path: '',
    social_handles: { linkedin: 'acme-inc', twitter: 'acme' },
    voice: { tone: ['warm', 'plain-spoken', 'bold'], banned_phrases: ['game-changer'] },
  }
}

vi.mock('next/navigation', () => ({ usePathname: () => '/deliverables' }))
vi.mock('@/lib/auth-hooks', () => {
  const organization = { id: 'org-1' }
  return {
    useAuth: () => ({ isSignedIn: true, getToken: async () => 'token' }),
    useOrganization: () => ({ organization }),
  }
})
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const MANAGED = ['logo_path', 'logo_mark_path', 'font_files']
  const apiClient = {
    listSocialPosts: vi.fn(async () => ({ posts: [], total: 0 })),
    getSocialsUsage: vi.fn(async () => ({ render_minutes: null })),
    get: vi.fn(async (path: string) => {
      if (path === '/api/documents/brand-kit') return server.kit
      if (path === '/api/documents/brand-kit/suggestions') return { suggestions: {} }
      throw new Error(`unexpected GET ${path}`)
    }),
    put: vi.fn(async (path: string, body: any) => {
      if (path !== '/api/documents/brand-kit') throw new Error(`unexpected PUT ${path}`)
      if (server.putError) throw server.putError
      const changes = Object.fromEntries(Object.entries(body).filter(([key]) => !MANAGED.includes(key)))
      server.kit = { ...server.kit, ...changes }
      return server.kit
    }),
    post: vi.fn(async (path: string, form: FormData) => {
      if (path !== '/api/documents/brand-kit/fonts') throw new Error(`unexpected POST ${path}`)
      const file = form.get('file') as File
      const entry = {
        id: 'b'.repeat(32), family: form.get('family'), weight: Number(form.get('weight')), style: form.get('style'),
        path: `w1/brand/fonts/${'b'.repeat(32)}.woff2`, file_name: file.name, bytes: file.size,
      }
      server.kit = { ...server.kit, font_files: [...server.kit.font_files, entry] }
      return server.kit
    }),
    delete: vi.fn(async (path: string) => {
      const id = path.split('/').pop()
      if (!path.startsWith('/api/documents/brand-kit/fonts/')) throw new Error(`unexpected DELETE ${path}`)
      server.kit = { ...server.kit, font_files: server.kit.font_files.filter((font: any) => font.id !== id) }
      return server.kit
    }),
    getAuthHeaders: vi.fn(async () => ({})),
    getBaseUrl: vi.fn(() => ''),
  }
  return { apiClient, default: apiClient }
})

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { WorkspaceProvider } from '@/components/workspace-provider'
import { SocialsTab } from '@/components/deliverables/socials/socials-tab'
import { BrandKitDialog } from '@/components/documents/blocks/BrandKitDialog'
import { parseList, toneWordsProblem } from '@/components/documents/blocks/BrandKitSocial'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

const currentWorkspace = vi.fn(async () => ({
  ok: true,
  status: 200,
  json: async () => ({
    id: 'w1', name: 'Acme', slug: 'acme', plan: 'pro', role: server.role,
    plan_limits: {}, socials: { available: true, enabled: true }, settings: {},
  }),
}))

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <WorkspaceProvider>
        <SocialsTab />
      </WorkspaceProvider>
    </QueryClientProvider>,
  )
}

function renderDialog() {
  const onSaved = vi.fn()
  const onOpenChange = vi.fn()
  render(<BrandKitDialog open onOpenChange={onOpenChange} onSaved={onSaved} />)
  return { onSaved, onOpenChange }
}

/** The open dialog, once the kit has loaded into it. */
async function loadedDialog() {
  const dialog = await screen.findByRole('dialog')
  await within(dialog).findByDisplayValue('"Brand Display", serif')
  return dialog
}

beforeEach(() => {
  server.role = 'owner'
  server.kit = startingKit()
  server.putError = null
  Object.values(api).forEach((fn) => fn.mockClear())
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
  currentWorkspace.mockClear()
  vi.stubGlobal('fetch', currentWorkspace)
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('the Socials tab opens the brand kit', () => {
  it.each(['owner', 'admin'])('for an %s, and the dialog shows the D5 fields', async (role) => {
    server.role = role
    renderTab()

    fireEvent.click(await screen.findByRole('button', { name: 'Brand kit' }))
    const dialog = await loadedDialog()

    expect(within(dialog).getByRole('heading', { name: /Brand Kit/ })).toBeInTheDocument()
    expect(within(dialog).getByLabelText(/^Body font/)).toHaveValue('Inter, sans-serif')
    expect(within(dialog).getByLabelText(/^Heading font/)).toHaveValue('"Brand Display", serif')
    // The uploaded font files, and the controls to add one.
    const fonts = within(dialog).getByTestId('brand-kit-fonts')
    expect(within(fonts).getByText('Brand Display Bold')).toBeInTheDocument()
    expect(within(fonts).getByText(/brand-display-700\.woff2/)).toBeInTheDocument()
    expect(within(fonts).getByLabelText('Font family name')).toBeInTheDocument()
    expect(within(fonts).getByLabelText('Font file (woff2)')).toHaveAttribute('accept', '.woff2,font/woff2')
    // The logo mark, separate from the wordmark.
    expect(within(dialog).getByText(/Logo mark \(square\)/)).toBeInTheDocument()
    expect(within(dialog).getByRole('button', { name: /Upload logo mark/ })).toBeInTheDocument()
    expect(within(dialog).getByLabelText('Logo mark file')).toHaveAttribute('accept', 'image/png,image/jpeg')
    // A handle per network, and the voice.
    expect(within(dialog).getByLabelText('LinkedIn')).toHaveValue('acme-inc')
    expect(within(dialog).getByLabelText('X')).toHaveValue('acme')
    for (const network of ['Instagram', 'TikTok', 'YouTube']) {
      expect(within(dialog).getByLabelText(network)).toHaveValue('')
    }
    expect(within(dialog).getByLabelText(/^Tone words/)).toHaveValue('warm, plain-spoken, bold')
    expect(within(dialog).getByLabelText(/^Phrases the brand never uses/)).toHaveValue('game-changer')
    expect(api.get).toHaveBeenCalledWith('/api/documents/brand-kit')
  })

  it.each(['editor', 'viewer'])('not for an %s, who cannot save it', async (role) => {
    server.role = role
    renderTab()
    expect(await screen.findByText('No posts yet')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Brand kit' })).toBeNull()
    expect(api.get).not.toHaveBeenCalledWith('/api/documents/brand-kit')
  })
})

describe('the brand kit dialog', () => {
  it('saves the heading font, the handles and the voice', async () => {
    const { onSaved } = renderDialog()
    const dialog = await loadedDialog()

    fireEvent.change(within(dialog).getByLabelText(/^Heading font/), { target: { value: '"Brand Serif", Georgia, serif' } })
    fireEvent.change(within(dialog).getByLabelText('Instagram'), { target: { value: 'acme.studio' } })
    fireEvent.change(within(dialog).getByLabelText(/^Tone words/), { target: { value: 'warm, precise, Warm, curious, ' } })
    fireEvent.change(within(dialog).getByLabelText(/^Phrases the brand never uses/), { target: { value: 'game-changer\n\nsynergy' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(api.put).toHaveBeenCalledTimes(1))
    const [path, body] = api.put.mock.calls[0] as [string, any]
    expect(path).toBe('/api/documents/brand-kit')
    expect(body.heading_font).toBe('"Brand Serif", Georgia, serif')
    expect(body.font_family).toBe('Inter, sans-serif')
    expect(body.social_handles).toEqual({ linkedin: 'acme-inc', twitter: 'acme', instagram: 'acme.studio' })
    expect(body.voice).toEqual({ tone: ['warm', 'precise', 'curious'], banned_phrases: ['game-changer', 'synergy'] })
    await waitFor(() => expect(onSaved).toHaveBeenCalledWith(expect.objectContaining({ heading_font: '"Brand Serif", Georgia, serif' })))
    expect(toast.success).toHaveBeenCalledWith('Brand kit saved')
  })

  it('holds Save back while the tone words are not three to five, and says why', async () => {
    renderDialog()
    const dialog = await loadedDialog()

    fireEvent.change(within(dialog).getByLabelText(/^Tone words/), { target: { value: 'warm, bold' } })
    expect(within(dialog).getByRole('alert')).toHaveTextContent('Give 3 to 5 tone words, or none (2 now).')
    const save = within(dialog).getByRole('button', { name: 'Save' })
    expect(save).toBeDisabled()
    fireEvent.click(save)
    expect(api.put).not.toHaveBeenCalled()

    fireEvent.change(within(dialog).getByLabelText(/^Tone words/), { target: { value: '' } })
    expect(within(dialog).queryByRole('alert')).toBeNull()
    expect(save).not.toBeDisabled()
  })

  it('names the field when the server refuses the kit', async () => {
    server.putError = new Error(JSON.stringify({
      message: 'Invalid brand kit',
      errors: [{ loc: ['social_handles'], msg: "Value error, the twitter handle 'acme-hq' does not fit: 1 to 15 letters, digits or underscores" }],
    }))
    renderDialog()
    const dialog = await loadedDialog()

    fireEvent.change(within(dialog).getByLabelText('X'), { target: { value: 'acme-hq' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(
      "social_handles: the twitter handle 'acme-hq' does not fit: 1 to 15 letters, digits or underscores",
    ))
  })

  it('uploads a woff2 with the face it provides, lists it, and removes it', async () => {
    renderDialog()
    const dialog = await loadedDialog()
    const fonts = within(dialog).getByTestId('brand-kit-fonts')

    fireEvent.change(within(fonts).getByLabelText('Font family name'), { target: { value: 'Brand Serif' } })
    fireEvent.change(within(fonts).getByLabelText('Font weight'), { target: { value: '400' } })
    fireEvent.change(within(fonts).getByLabelText('Font style'), { target: { value: 'italic' } })
    const woff2 = new File([new Uint8Array([0x77, 0x4f, 0x46, 0x32])], 'brand-serif-italic.woff2', { type: 'font/woff2' })
    fireEvent.change(within(fonts).getByLabelText('Font file (woff2)'), { target: { files: [woff2] } })

    expect(await within(fonts).findByText('Brand Serif Regular italic')).toBeInTheDocument()
    const [path, form] = api.post.mock.calls[0] as [string, FormData]
    expect(path).toBe('/api/documents/brand-kit/fonts')
    expect([form.get('family'), form.get('weight'), form.get('style')]).toEqual(['Brand Serif', '400', 'italic'])
    expect((form.get('file') as File).name).toBe('brand-serif-italic.woff2')

    fireEvent.click(within(fonts).getByRole('button', { name: 'Remove Brand Serif Regular italic' }))
    await waitFor(() => expect(within(fonts).queryByText('Brand Serif Regular italic')).toBeNull())
    expect(api.delete).toHaveBeenCalledWith(`/api/documents/brand-kit/fonts/${NEW_FONT_ID}`)
    expect(within(fonts).getByText('Brand Display Bold')).toBeInTheDocument()
  })
})

describe('the voice helpers', () => {
  it('parse a list once per entry and count three to five tone words, or none', () => {
    expect(parseList(' warm, Warm ,, bold ', /,/)).toEqual(['warm', 'bold'])
    expect(parseList('a\n\nb\n', /\n/)).toEqual(['a', 'b'])
    expect(toneWordsProblem([])).toBeNull()
    expect(toneWordsProblem(['a', 'b', 'c'])).toBeNull()
    expect(toneWordsProblem(['a', 'b', 'c', 'd', 'e'])).toBeNull()
    expect(toneWordsProblem(['a'])).toBe('Give 3 to 5 tone words, or none (1 now).')
    expect(toneWordsProblem(['a', 'b', 'c', 'd', 'e', 'f'])).toBe('Give 3 to 5 tone words, or none (6 now).')
  })
})
