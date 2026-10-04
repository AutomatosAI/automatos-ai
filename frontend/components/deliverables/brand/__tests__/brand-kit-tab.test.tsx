/**
 * PRD-251B Wave 3 — the Brand kit tab (US-B301 to US-B304), against a mocked apiClient that
 * holds one brand kit, its style references and the AI tools:
 *
 * * the basics the BrandKitDialog held (the D5 fields, PRD-251 S1.3): the body and heading
 *   fonts, the uploaded font files, the logo mark, the social handles and the voice; Save
 *   sends them in PUT /api/documents/brand-kit; tone words other than three to five hold
 *   Save back, and a refusal from the server names the field; a woff2 goes up with its face;
 * * an editor or a viewer reads the kit and changes nothing;
 * * the style references: added (liked), turned to avoid, noted, removed; the kit full;
 * * what Auto takes from them, Read the references again, and whether liked images go;
 * * the AI tools: the rows, a default changed (only the changed one is sent) and the caps;
 *   a dropdown with one choice says what to connect (F252); with Socials off, a note instead;
 * * PRD-251C US-C406: the voice examples, Auto's draft beside the copy approved, removed by an
 *   owner; a viewer reads them only.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const FONT_ID = 'a'.repeat(32)
const NEW_FONT_ID = 'b'.repeat(32)

const server = vi.hoisted(() => ({
  role: 'owner' as string,
  socials: true,
  kit: null as any,
  putError: null as Error | null,
  style: null as any,
  tools: null as any,
  examples: [] as any[],
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

function reference(id: string, stance = 'like', note = '') {
  return { id, note, stance, content_type: 'image/png', bytes: 1200, created_at: '2026-10-03T09:00:00Z', url: `/api/documents/brand-kit/references/${id}/image` }
}

function startingStyle() {
  return {
    references: [reference('r1', 'like', 'Warm light'), reference('r2', 'avoid', 'Too busy')],
    profile: { palette: ['#1F2A44', '#F4B400'], mood: ['calm', 'confident'], composition: 'Wide shots, one subject.', avoid: 'Busy collages.', read_at: '2026-10-03T09:01:00Z', reference_ids: ['r1', 'r2'] },
    send_liked: true,
    limits: { references: 24, bytes: 10485760 },
  }
}

function startingTools() {
  return {
    toolkits: [
      { toolkit: 'fal_ai', label: 'fal.ai', kind: 'Images and footage', status: 'available', makes: ['video', 'image'] },
      { toolkit: 'kieai', label: 'Kie.ai', kind: 'Images and footage', status: 'connect' },
      { toolkit: 'templates', label: 'Templates', kind: 'Images', status: 'builtin' },
      { toolkit: 'kokoro', label: 'Kokoro', kind: 'Voice', status: 'builtin' },
    ],
    offered: {
      images: [{ value: 'templates', label: 'Templates (free)' }, { value: 'fal_ai', label: 'fal.ai' }],
      ai_images: [{ value: 'fal_ai', label: 'fal.ai' }, { value: 'ask', label: 'Ask each time' }],
      footage: [{ value: 'fal_ai', label: 'fal.ai' }, { value: 'off', label: 'Off' }],
      voice: [{ value: 'kokoro', label: 'Kokoro (free)' }],
    },
    defaults: { images: 'templates', ai_images: 'ask', footage: 'off', voice: 'kokoro' },
    caps: { monthly_usd: 30, per_post_usd: 10, problem: null },
    spend: { month_usd: 1.25, period_end: '2026-11-01T00:00:00Z' },
  }
}

vi.mock('next/navigation', () => ({
  usePathname: () => '/deliverables',
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams('tab=brand'),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: server.role, socials: { available: true, enabled: server.socials } } }),
}))
vi.mock('@/hooks/use-authed-image', () => ({ useAuthedImage: () => null }))
vi.mock('@/hooks/use-composio-api', () => ({ useInitiateConnection: () => ({ mutateAsync: vi.fn(), isLoading: false }) }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const MANAGED = ['logo_path', 'logo_mark_path', 'font_files']
  const apiClient = {
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
    getBrandStyle: vi.fn(async () => server.style),
    uploadBrandReference: vi.fn(async (_file: File, note: string, stance: string) => {
      server.style = { ...server.style, references: [...server.style.references, reference('r3', stance, note)] }
      return server.style
    }),
    updateBrandReference: vi.fn(async (id: string, changes: any) => {
      server.style = { ...server.style, references: server.style.references.map((r: any) => (r.id === id ? { ...r, ...changes } : r)) }
      return server.style
    }),
    deleteBrandReference: vi.fn(async (id: string) => {
      server.style = { ...server.style, references: server.style.references.filter((r: any) => r.id !== id) }
      return server.style
    }),
    readBrandStyle: vi.fn(async () => server.style),
    setBrandStyleSendLiked: vi.fn(async (on: boolean) => {
      server.style = { ...server.style, send_liked: on }
      return server.style
    }),
    getSocialMediaTools: vi.fn(async () => server.tools),
    updateSocialMediaTools: vi.fn(async (input: any) => {
      server.tools = { ...server.tools, defaults: { ...server.tools.defaults, ...(input.defaults ?? {}) } }
      return server.tools
    }),
    listSocialVoiceExamples: vi.fn(async () => ({ examples: server.examples })),
    deleteSocialVoiceExample: vi.fn(async (id: string) => {
      server.examples = server.examples.filter((example: any) => example.id !== id)
    }),
    getAuthHeaders: vi.fn(async () => ({})),
    getBaseUrl: vi.fn(() => ''),
  }
  return { apiClient, default: apiClient }
})

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { BrandKitTab, READ_ONLY_NOTE } from '@/components/deliverables/brand/brand-kit-tab'
import { SOCIALS_OFF_NOTE } from '@/components/deliverables/brand/brand-ai-tools'
import { parseList, toneWordsProblem } from '@/components/documents/blocks/BrandKitSocial'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <BrandKitTab />
    </QueryClientProvider>,
  )
}

/** The basics card, once the kit has loaded into it. */
async function basics() {
  const card = await screen.findByRole('region', { name: 'The basics' })
  await within(card).findByDisplayValue('"Brand Display", serif')
  return card
}

beforeEach(() => {
  server.role = 'owner'
  server.socials = true
  server.kit = startingKit()
  server.putError = null
  server.style = startingStyle()
  server.tools = startingTools()
  server.examples = [
    { id: 'v2', post_id: 'p2', draft: 'We are at Web Summit!!!', approved: 'Web Summit, stand B12.', created_at: '2026-10-20T09:00:00Z' },
    { id: 'v1', post_id: 'p1', draft: 'Big news coming.', approved: 'Monday: the local edition.', created_at: '2026-10-19T09:00:00Z' },
  ]
  Object.values(api).forEach((fn) => fn.mockClear())
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404, json: async () => ({}) })))
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('the basics', () => {
  it('shows the D5 fields', async () => {
    renderTab()
    const card = await basics()
    expect(within(card).getByLabelText(/^Body font/)).toHaveValue('Inter, sans-serif')
    const fonts = within(card).getByTestId('brand-kit-fonts')
    expect(within(fonts).getByText('Brand Display Bold')).toBeInTheDocument()
    expect(within(fonts).getByText(/brand-display-700\.woff2/)).toBeInTheDocument()
    expect(within(card).getByText(/Logo mark \(square\)/)).toBeInTheDocument()
    expect(within(card).getByRole('button', { name: /Upload logo mark/ })).toBeInTheDocument()
    expect(within(card).getByLabelText('Logo mark file')).toHaveAttribute('accept', 'image/png,image/jpeg')
    expect(within(card).getByLabelText('LinkedIn')).toHaveValue('acme-inc')
    expect(within(card).getByLabelText('X')).toHaveValue('acme')
    expect(within(card).getByLabelText(/^Tone words/)).toHaveValue('warm, plain-spoken, bold')
    expect(within(card).getByLabelText(/^Phrases the brand never uses/)).toHaveValue('game-changer')
    expect(api.get).toHaveBeenCalledWith('/api/documents/brand-kit')
  })

  it('saves the heading font, the handles and the voice', async () => {
    renderTab()
    const card = await basics()
    fireEvent.change(within(card).getByLabelText(/^Heading font/), { target: { value: '"Brand Serif", Georgia, serif' } })
    fireEvent.change(within(card).getByLabelText('Instagram'), { target: { value: 'acme.studio' } })
    fireEvent.change(within(card).getByLabelText(/^Tone words/), { target: { value: 'warm, precise, Warm, curious, ' } })
    fireEvent.change(within(card).getByLabelText(/^Phrases the brand never uses/), { target: { value: 'game-changer\n\nsynergy' } })
    fireEvent.change(within(card).getByLabelText(/^Sign-off/), { target: { value: 'Sam, Acme Coffee' } })
    fireEvent.click(within(card).getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(api.put).toHaveBeenCalledTimes(1))
    const [path, body] = api.put.mock.calls[0] as [string, any]
    expect(path).toBe('/api/documents/brand-kit')
    expect(body.heading_font).toBe('"Brand Serif", Georgia, serif')
    expect(body.social_handles).toEqual({ linkedin: 'acme-inc', twitter: 'acme', instagram: 'acme.studio' })
    expect(body.voice).toEqual({
      tone: ['warm', 'precise', 'curious'],
      banned_phrases: ['game-changer', 'synergy'],
      sign_off: 'Sam, Acme Coffee',
    })
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Brand kit saved'))
  })

  it('holds Save back while the tone words are not three to five, and says why', async () => {
    renderTab()
    const card = await basics()
    fireEvent.change(within(card).getByLabelText(/^Tone words/), { target: { value: 'warm, bold' } })
    expect(within(card).getByRole('alert')).toHaveTextContent('Give 3 to 5 tone words, or none (2 now).')
    expect(within(card).getByRole('button', { name: 'Save' })).toBeDisabled()
  })

  it('names the field when the server refuses the kit', async () => {
    server.putError = new Error(JSON.stringify({
      message: 'Invalid brand kit',
      errors: [{ loc: ['social_handles'], msg: "Value error, the twitter handle 'acme-hq' does not fit: 1 to 15 letters, digits or underscores" }],
    }))
    renderTab()
    const card = await basics()
    fireEvent.change(within(card).getByLabelText('X'), { target: { value: 'acme-hq' } })
    fireEvent.click(within(card).getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(
      "social_handles: the twitter handle 'acme-hq' does not fit: 1 to 15 letters, digits or underscores",
    ))
  })

  it('uploads a woff2 with the face it provides, lists it, and removes it', async () => {
    renderTab()
    const fonts = within(await basics()).getByTestId('brand-kit-fonts')
    fireEvent.change(within(fonts).getByLabelText('Font family name'), { target: { value: 'Brand Serif' } })
    fireEvent.change(within(fonts).getByLabelText('Font weight'), { target: { value: '400' } })
    fireEvent.change(within(fonts).getByLabelText('Font style'), { target: { value: 'italic' } })
    const woff2 = new File([new Uint8Array([0x77, 0x4f, 0x46, 0x32])], 'brand-serif-italic.woff2', { type: 'font/woff2' })
    fireEvent.change(within(fonts).getByLabelText('Font file (woff2)'), { target: { files: [woff2] } })

    expect(await within(fonts).findByText('Brand Serif Regular italic')).toBeInTheDocument()
    const [path, form] = api.post.mock.calls[0] as [string, FormData]
    expect(path).toBe('/api/documents/brand-kit/fonts')
    expect([form.get('family'), form.get('weight'), form.get('style')]).toEqual(['Brand Serif', '400', 'italic'])
    fireEvent.click(within(fonts).getByRole('button', { name: 'Remove Brand Serif Regular italic' }))
    await waitFor(() => expect(within(fonts).queryByText('Brand Serif Regular italic')).toBeNull())
    expect(api.delete).toHaveBeenCalledWith(`/api/documents/brand-kit/fonts/${NEW_FONT_ID}`)
  })

  it.each(['editor', 'viewer'])('an %s reads the kit and changes nothing', async (role) => {
    server.role = role
    renderTab()
    const card = await basics()
    expect(screen.getByText(READ_ONLY_NOTE)).toBeInTheDocument()
    expect(within(card).queryByRole('button', { name: 'Save' })).toBeNull()
    expect(within(card).getByLabelText(/^Body font/)).toBeDisabled()
    const references = await screen.findByRole('region', { name: 'Style references' })
    expect(within(references).queryByRole('button', { name: 'Add an image' })).toBeNull()
    expect(within(references).queryByRole('button', { name: /Remove Reference/ })).toBeNull()
  })
})

describe('the style references and what Auto takes from them', () => {
  it('adds an image as liked, turns one to avoid, notes it and removes it', async () => {
    renderTab()
    const section = await screen.findByRole('region', { name: 'Style references' })
    await within(section).findByRole('listitem', { name: 'Reference 1' })
    expect(within(section).getByText(/2 of 24/)).toBeInTheDocument()

    const png = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'light.png', { type: 'image/png' })
    fireEvent.change(within(section).getByLabelText('Style reference file'), { target: { files: [png] } })
    await waitFor(() => expect(api.uploadBrandReference).toHaveBeenCalledWith(png, '', 'like'))
    expect(await within(section).findByRole('listitem', { name: 'Reference 3' })).toBeInTheDocument()

    const first = within(section).getByRole('listitem', { name: 'Reference 1' })
    fireEvent.click(within(first).getByRole('button', { name: 'Avoid' }))
    await waitFor(() => expect(api.updateBrandReference).toHaveBeenCalledWith('r1', { stance: 'avoid' }))
    const note = within(first).getByLabelText('Reference 1 note')
    fireEvent.change(note, { target: { value: 'Soft morning light ' } })
    fireEvent.blur(note)
    await waitFor(() => expect(api.updateBrandReference).toHaveBeenCalledWith('r1', { note: 'Soft morning light' }))

    fireEvent.click(within(section).getByRole('button', { name: 'Remove Reference 2' }))
    await waitFor(() => expect(api.deleteBrandReference).toHaveBeenCalledWith('r2'))
  })

  it('a full kit takes no more', async () => {
    server.style = { ...startingStyle(), limits: { references: 2, bytes: 10485760 } }
    renderTab()
    const section = await screen.findByRole('region', { name: 'Style references' })
    await waitFor(() => expect(within(section).getByRole('button', { name: 'Add an image' })).toBeDisabled())
    expect(within(section).getByText(/remove one to add another/)).toBeInTheDocument()
  })

  it('shows the profile, reads the references again, and turns sending liked images off', async () => {
    renderTab()
    const aside = await screen.findByRole('complementary', { name: 'What Auto takes from these' })
    expect(await within(aside).findByText('#1F2A44')).toBeInTheDocument()
    expect(within(within(aside).getByRole('list', { name: 'Mood' })).getByText('calm')).toBeInTheDocument()
    expect(within(aside).getByText(/Wide shots, one subject/)).toBeInTheDocument()

    fireEvent.click(within(aside).getByRole('button', { name: /Read the references again/ }))
    await waitFor(() => expect(api.readBrandStyle).toHaveBeenCalledTimes(1))
    fireEvent.click(within(aside).getByRole('checkbox', { name: /Send liked images/ }))
    await waitFor(() => expect(api.setBrandStyleSendLiked).toHaveBeenCalledWith(false))
  })
})

describe('the AI tools', () => {
  it('lists the tools and saves a changed default with both caps', async () => {
    renderTab()
    const section = await screen.findByRole('region', { name: 'AI tools' })
    const rows = await within(section).findByRole('list', { name: 'Media toolkits' })
    expect(within(rows).getByText('Connected')).toBeInTheDocument()
    expect(within(rows).getByRole('button', { name: 'Connect Kie.ai' })).toBeInTheDocument()
    expect(within(rows).getAllByText('Built in · free')).toHaveLength(2)
    expect(within(section).getByText(/Spent this month: \$1\.25 of \$30\.00/)).toBeInTheDocument()
    // F252: a dropdown with one choice says why, and what to connect for more; one with a choice says nothing.
    expect(within(section).getByLabelText('Voice')).toHaveAccessibleDescription(
      'Only Kokoro (free) for now: no AI tool for this is set up on this platform.',
    )
    expect(within(section).getByLabelText('Images')).not.toHaveAccessibleDescription()

    fireEvent.change(within(section).getByLabelText('AI images'), { target: { value: 'fal_ai' } })
    fireEvent.change(within(section).getByLabelText('Per-post media cap (USD)'), { target: { value: '4.5' } })
    fireEvent.click(within(section).getByRole('button', { name: 'Save AI tools' }))
    await waitFor(() => expect(api.updateSocialMediaTools).toHaveBeenCalledWith({
      defaults: { ai_images: 'fal_ai' }, monthly_cap_usd: 30, per_post_cap_usd: 4.5,
    }))
  })

  it('says why there are none while Socials is off', async () => {
    server.socials = false
    renderTab()
    const section = await screen.findByRole('region', { name: 'AI tools' })
    expect(within(section).getByText(SOCIALS_OFF_NOTE)).toBeInTheDocument()
    expect(api.getSocialMediaTools).not.toHaveBeenCalled()
  })
})

describe('the voice examples (PRD-251C US-C406)', () => {
  it("shows Auto's draft beside the copy approved, and an owner removes one", async () => {
    renderTab()
    const card = await screen.findByRole('region', { name: 'Voice examples' })
    const first = await within(card).findByRole('listitem', { name: 'Web Summit, stand B12.' })
    expect(first).toHaveTextContent('Auto wrote: We are at Web Summit!!!')
    fireEvent.click(within(first).getByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(api.deleteSocialVoiceExample).toHaveBeenCalledWith('v2'))
    await waitFor(() => expect(within(card).queryByRole('listitem', { name: 'Web Summit, stand B12.' })).toBeNull())
  })

  it('a viewer reads them and removes nothing', async () => {
    server.role = 'viewer'
    renderTab()
    const card = await screen.findByRole('region', { name: 'Voice examples' })
    await within(card).findByRole('listitem', { name: 'Monday: the local edition.' })
    expect(within(card).queryByRole('button', { name: 'Remove' })).toBeNull()
  })
})

describe('the voice helpers', () => {
  it('parse a list once per entry and count three to five tone words, or none', () => {
    expect(parseList(' warm, Warm ,, bold ', /,/)).toEqual(['warm', 'bold'])
    expect(parseList('a\n\nb\n', /\n/)).toEqual(['a', 'b'])
    expect(toneWordsProblem([])).toBeNull()
    expect(toneWordsProblem(['a', 'b', 'c', 'd', 'e'])).toBeNull()
    expect(toneWordsProblem(['a'])).toBe('Give 3 to 5 tone words, or none (1 now).')
  })
})
