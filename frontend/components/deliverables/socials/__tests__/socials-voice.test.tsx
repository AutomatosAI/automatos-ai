/**
 * PRD-251 S1.5 (D11, D15) — the voice picker, against a mocked apiClient:
 *
 * * without a voice toolkit connected, Kokoro is the only voice offered, and
 *   Fish Audio and ElevenLabs are Connect buttons that start the Composio
 *   connect flow (POST /api/composio/connect/{app}, then its hosted sign-in in
 *   a new window);
 * * a connected toolkit is offered beside Kokoro; choosing it lists its voices,
 *   and picking one saves it on the post (PATCH voice); choosing Kokoro again
 *   saves `null`;
 * * a connected toolkit the workspace cannot use says why; a toolkit that does
 *   not list its voices takes a typed voice id;
 * * a role that cannot edit sees the voice, and no choices.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const server = vi.hoisted(() => ({
  sources: [] as any[],
  voices: [] as any[],
}))

vi.mock('next/navigation', () => ({ usePathname: () => '/deliverables' }))
vi.mock('@/lib/auth-hooks', () => {
  const organization = { id: 'org-1' }
  return {
    useAuth: () => ({ isLoaded: true, isSignedIn: true, getToken: async () => 'token' }),
    useOrganization: () => ({ organization }),
  }
})
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    getSocialVoiceSources: vi.fn(async () => ({ sources: server.sources, problem: null })),
    listSocialToolkitVoices: vi.fn(async (toolkit: string) => ({ toolkit, voices: server.voices })),
    updateSocialPost: vi.fn(async (id: string, changes: Record<string, unknown>) => ({ id, ...changes })),
    listSocialPosts: vi.fn(async () => ({ posts: [], total: 0 })),
    post: vi.fn(async (path: string) => ({ redirect_url: `https://connect.composio.dev/link/${path.split('/').pop()}` })),
  }
  return { apiClient, default: apiClient }
})

import { apiClient } from '@/lib/api-client'
import { WorkspaceProvider } from '@/components/workspace-provider'
import { SocialsVoicePicker, voiceLabel } from '@/components/deliverables/socials/socials-voice-picker'
import { speaksAScript } from '@/components/deliverables/socials/socials-status'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

const currentWorkspace = vi.fn(async () => ({
  ok: true,
  status: 200,
  json: async () => ({
    id: 'w1', name: 'Harbourline', slug: 'harbourline', plan: 'pro', role: 'owner',
    plan_limits: {}, socials: { available: true, enabled: true }, settings: {},
  }),
}))

const KOKORO = { toolkit: 'kokoro', label: 'Kokoro (built in)', status: 'available', builtin: true, lists_voices: false }
const source = (toolkit: string, label: string, status: string, extra: Record<string, unknown> = {}) => ({
  toolkit, label, status, builtin: false, lists_voices: status === 'available', ...extra,
})

function post(overrides: Record<string, unknown> = {}) {
  return {
    id: 'post-1', workspace_id: 'w1', created_by: 'user-1', title: 'Launch teaser', brief: null,
    copy: { base: 'Something big lands Monday.' }, format: 'video', template_id: 'tpl-1',
    variables: {}, sources: {}, media: {}, voice: null, status: 'draft', content_hash: 'a'.repeat(64),
    approved_hash: null, approved_by: null, approved_at: null, override_unsourced: false,
    review_log: [], scheduled_for: null, timezone: null,
    created_at: '2026-09-25T09:00:00Z', updated_at: '2026-09-25T09:00:00Z', ...overrides,
  } as any
}

function wrap(children: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return (
    <QueryClientProvider client={client}>
      <WorkspaceProvider>{children}</WorkspaceProvider>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  server.sources = [KOKORO]
  server.voices = []
  Object.values(api).forEach((fn) => fn.mockClear())
  vi.stubGlobal('fetch', currentWorkspace)
  vi.stubGlobal('open', vi.fn())
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('without a voice toolkit', () => {
  it('offers Kokoro only, and links Fish Audio and ElevenLabs to the Composio connect flow', async () => {
    server.sources = [KOKORO, source('elevenlabs', 'ElevenLabs', 'connect'), source('fish_audio', 'Fish Audio', 'connect')]
    render(wrap(<SocialsVoicePicker post={post()} editable />))

    const kokoro = await screen.findByRole('radio', { name: 'Kokoro (built in)' })
    expect(kokoro).toBeChecked()
    expect(screen.getAllByRole('radio')).toHaveLength(1)
    expect(screen.queryByRole('radio', { name: 'Fish Audio' })).not.toBeInTheDocument()

    const connect = screen.getByRole('button', { name: 'Connect Fish Audio' })
    expect(screen.getByRole('button', { name: 'Connect ElevenLabs' })).toBeInTheDocument()
    fireEvent.click(connect)
    await waitFor(() => expect(api.post).toHaveBeenCalledTimes(1))
    const [path, body] = api.post.mock.calls[0]
    expect(path).toBe('/api/composio/connect/FISH_AUDIO')
    expect(body).toEqual({ callback_url: `${window.location.origin}/tools/callback?connected=FISH_AUDIO` })
    await waitFor(() =>
      expect(window.open).toHaveBeenCalledWith(
        'https://connect.composio.dev/link/FISH_AUDIO', 'Connect Fish Audio', expect.any(String),
      ),
    )
    expect(api.updateSocialPost).not.toHaveBeenCalled()
  })

  it('reads the choices again when the connect flow reports back', async () => {
    server.sources = [KOKORO, source('fish_audio', 'Fish Audio', 'connect')]
    render(wrap(<SocialsVoicePicker post={post()} editable />))
    await screen.findByRole('button', { name: 'Connect Fish Audio' })

    server.sources = [KOKORO, source('fish_audio', 'Fish Audio', 'available')]
    window.dispatchEvent(new MessageEvent('message', { origin: window.location.origin, data: { type: 'COMPOSIO_CONNECTED', app: 'FISH_AUDIO' } }))

    expect(await screen.findByRole('radio', { name: 'Fish Audio' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Connect Fish Audio' })).not.toBeInTheDocument()
  })
})

describe('with a voice toolkit connected', () => {
  it('lists its voices when chosen, and picking one saves it on the post', async () => {
    server.sources = [KOKORO, source('fish_audio', 'Fish Audio', 'available')]
    server.voices = [{ id: 'v1', name: 'Energetic narrator' }, { id: 'v2', name: 'Calm narrator' }]
    render(wrap(<SocialsVoicePicker post={post()} editable />))

    fireEvent.click(await screen.findByRole('radio', { name: 'Fish Audio' }))
    const select = await screen.findByRole('combobox', { name: 'Fish Audio voice' })
    await screen.findByRole('option', { name: 'Energetic narrator' })
    expect(api.listSocialToolkitVoices).toHaveBeenCalledWith('fish_audio', '')
    expect(api.updateSocialPost).not.toHaveBeenCalled()

    fireEvent.change(select, { target: { value: 'v1' } })
    await waitFor(() =>
      expect(api.updateSocialPost).toHaveBeenCalledWith('post-1', {
        voice: { toolkit: 'fish_audio', voice_id: 'v1', name: 'Energetic narrator' },
      }),
    )
  })

  it('a search asks the toolkit once typing pauses, not per keystroke', async () => {
    server.sources = [KOKORO, source('fish_audio', 'Fish Audio', 'available')]
    server.voices = [{ id: 'v1', name: 'Energetic narrator' }]
    render(wrap(<SocialsVoicePicker post={post()} editable />))
    fireEvent.click(await screen.findByRole('radio', { name: 'Fish Audio' }))
    const search = await screen.findByRole('textbox', { name: 'Search Fish Audio voices' })
    await waitFor(() => expect(api.listSocialToolkitVoices).toHaveBeenCalledWith('fish_audio', ''))

    for (const typed of ['n', 'na', 'nar']) fireEvent.change(search, { target: { value: typed } })

    await waitFor(() => expect(api.listSocialToolkitVoices).toHaveBeenCalledWith('fish_audio', 'nar'))
    const asked = api.listSocialToolkitVoices.mock.calls.map(([, q]) => q)
    expect(asked).not.toContain('n')
    expect(asked).not.toContain('na')
  })

  it('choosing Kokoro again saves no toolkit voice', async () => {
    server.sources = [KOKORO, source('fish_audio', 'Fish Audio', 'available')]
    const spoken = post({ voice: { toolkit: 'fish_audio', voice_id: 'v1', name: 'Energetic narrator' } })
    render(wrap(<SocialsVoicePicker post={spoken} editable />))

    expect(await screen.findByRole('radio', { name: 'Fish Audio' })).toBeChecked()
    fireEvent.click(screen.getByRole('radio', { name: 'Kokoro (built in)' }))
    await waitFor(() => expect(api.updateSocialPost).toHaveBeenCalledWith('post-1', { voice: null }))
  })

  it('a toolkit that does not list its voices takes a typed voice id', async () => {
    server.sources = [KOKORO, source('elevenlabs', 'ElevenLabs', 'available', { lists_voices: false })]
    render(wrap(<SocialsVoicePicker post={post()} editable />))

    fireEvent.click(await screen.findByRole('radio', { name: 'ElevenLabs' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'ElevenLabs voice id' }), { target: { value: ' 21m00Tcm4TlvDq8ikWAM ' } })
    fireEvent.click(screen.getByRole('button', { name: 'Use this voice' }))
    await waitFor(() =>
      expect(api.updateSocialPost).toHaveBeenCalledWith('post-1', { voice: { toolkit: 'elevenlabs', voice_id: '21m00Tcm4TlvDq8ikWAM' } }),
    )
    expect(api.listSocialToolkitVoices).not.toHaveBeenCalled()
  })

  it('a connected toolkit the workspace cannot use says why', async () => {
    const reason = 'Fish Audio bills credit, and its balance action is not available here.'
    server.sources = [KOKORO, source('fish_audio', 'Fish Audio', 'unavailable', { reason })]
    render(wrap(<SocialsVoicePicker post={post()} editable />))

    expect(await screen.findByText(`Fish Audio: ${reason}`)).toBeInTheDocument()
    expect(screen.getAllByRole('radio')).toHaveLength(1)
  })
})

describe('reading', () => {
  it('a role that cannot edit sees the voice and no choices', async () => {
    server.sources = [KOKORO, source('fish_audio', 'Fish Audio', 'available')]
    const spoken = post({ voice: { toolkit: 'fish_audio', voice_id: 'v1', name: 'Energetic narrator' } })
    render(wrap(<SocialsVoicePicker post={spoken} editable={false} />))

    expect(await screen.findByText('Voice: Fish Audio — Energetic narrator')).toBeInTheDocument()
    expect(screen.queryAllByRole('radio')).toHaveLength(0)
  })

  it('names the voice, Kokoro by default', () => {
    expect(voiceLabel(null, [])).toBe('Kokoro (built in)')
    expect(voiceLabel({ toolkit: 'fish_audio', voice_id: 'v9' }, [])).toBe('fish_audio — v9')
  })

  it('only a post that speaks a script has a voice to choose', () => {
    expect(speaksAScript({ template_id: 'tpl', format: 'video' })).toBe(true)
    expect(speaksAScript({ template_id: 'tpl', format: null })).toBe(true)
    expect(speaksAScript({ template_id: 'tpl', format: 'carousel' })).toBe(false)
    expect(speaksAScript({ template_id: null, format: 'video' })).toBe(false)
  })
})
