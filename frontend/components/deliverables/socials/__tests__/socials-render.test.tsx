/**
 * PRD-251 S1.1c — rendering in the Socials tab, against a mocked apiClient:
 *
 * * the list shows "Render minutes: used / quota this month" from
 *   GET /api/socials/usage (a plan with no quota shows the minutes used alone,
 *   and a used-up month says so);
 * * a post with a template, for a role that authors, shows Render; clicking it
 *   calls POST /render and the post shows Rendering; a refused render (429, no
 *   minutes left) shows the server's reason;
 * * a failed render shows its reason and offers Render again;
 * * the list polls only while a post renders, and the minutes are read again
 *   when the last render ends.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const server = vi.hoisted(() => ({
  role: 'owner' as string,
  posts: [] as any[],
  usage: null as any,
  renderError: null as null | { status: number; message: string },
}))

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
  const apiClient = {
    listSocialPosts: vi.fn(async () => ({ posts: server.posts, total: server.posts.length })),
    getSocialsUsage: vi.fn(async () => ({ render_minutes: server.usage })),
    // S1.5: a video post's detail shows its voice; Kokoro alone here.
    getSocialVoiceSources: vi.fn(async () => ({
      sources: [{ toolkit: 'kokoro', label: 'Kokoro (built in)', status: 'available', builtin: true, lists_voices: false }],
      problem: null,
    })),
    renderSocialPost: vi.fn(async (id: string) => {
      if (server.renderError) {
        throw Object.assign(new Error(server.renderError.message), { status: server.renderError.status })
      }
      server.posts = server.posts.map((p) =>
        p.id === id
          ? { ...p, status: 'rendering', review_log: [...p.review_log, { at: 'now', by: 'user-1', action: 'render', comment: null }] }
          : p,
      )
      return server.posts.find((p) => p.id === id)
    }),
  }
  return { apiClient, default: apiClient }
})

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { WorkspaceProvider } from '@/components/workspace-provider'
import { SocialsTab } from '@/components/deliverables/socials/socials-tab'
import { SocialsRenderMinutes } from '@/components/deliverables/socials/socials-render-minutes'
import { formatRenderMinutes } from '@/components/deliverables/socials/socials-status'
import { renderPollInterval, SOCIALS_RENDER_POLL_MS } from '@/hooks/use-socials-api'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

const currentWorkspace = vi.fn(async () => ({
  ok: true,
  status: 200,
  json: async () => ({
    id: 'w1', name: 'Acme', slug: 'acme', plan: 'basic', role: server.role,
    plan_limits: {}, socials: { available: true, enabled: true }, settings: {},
  }),
}))

function usage(overrides: Record<string, unknown> = {}) {
  return {
    used_minutes: 3.5, used_seconds: 210, quota_minutes: 10, remaining_minutes: 6.5, exhausted: false,
    period_start: '2026-09-01T00:00:00+00:00', period_end: '2026-10-01T00:00:00+00:00', ...overrides,
  }
}

function post(overrides: Record<string, unknown> = {}) {
  return {
    id: 'post-1', workspace_id: 'w1', created_by: 'user-1', title: 'Launch teaser', brief: null,
    copy: { base: 'Something big lands Monday.' }, format: 'video', template_id: 'tpl-1',
    variables: {}, sources: {}, media: {}, status: 'draft', content_hash: 'a'.repeat(64),
    approved_hash: null, approved_by: null, approved_at: null, override_unsourced: false,
    review_log: [], scheduled_for: null, timezone: null,
    created_at: '2026-09-25T09:00:00Z', updated_at: '2026-09-25T09:00:00Z', ...overrides,
  }
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
}

function wrap(children: React.ReactNode, client: QueryClient = newClient()) {
  return (
    <QueryClientProvider client={client}>
      <WorkspaceProvider>{children}</WorkspaceProvider>
    </QueryClientProvider>
  )
}

const detail = (title: string) => screen.getByRole('article', { name: `Post: ${title}` })

beforeEach(() => {
  server.role = 'owner'
  server.posts = []
  server.usage = usage()
  server.renderError = null
  Object.values(api).forEach((fn) => fn.mockClear())
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
  vi.stubGlobal('fetch', currentWorkspace)
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('render minutes', () => {
  it('shows the minutes used and the plan quota this month', async () => {
    render(wrap(<SocialsTab />))
    expect(await screen.findByText('Render minutes: 3.5 / 10 this month')).toBeInTheDocument()
    expect(api.getSocialsUsage).toHaveBeenCalled()
  })

  it('a plan with no quota shows the minutes used alone', async () => {
    server.usage = usage({ used_minutes: 12, quota_minutes: null, remaining_minutes: null })
    render(wrap(<SocialsTab />))
    expect(await screen.findByText('Render minutes: 12 this month (no monthly quota)')).toBeInTheDocument()
  })

  it('a used-up month says so', async () => {
    server.usage = usage({ used_minutes: 10, remaining_minutes: 0, exhausted: true })
    render(wrap(<SocialsTab />))
    expect(await screen.findByText(/Render minutes: 10 \/ 10 this month — used up until next month/)).toBeInTheDocument()
  })

  it('formats minutes to one decimal place', () => {
    expect(formatRenderMinutes(3.46, 10)).toBe('3.5 / 10')
    expect(formatRenderMinutes(0, 240)).toBe('0 / 240')
    expect(formatRenderMinutes(7.25, null)).toBe('7.3')
  })

  it('reads the minutes again when the last render ends', async () => {
    const client = newClient()
    const view = render(wrap(<SocialsRenderMinutes rendering />, client))
    await screen.findByText('Render minutes: 3.5 / 10 this month')
    const reads = api.getSocialsUsage.mock.calls.length

    // The same query cache: only the end of the render can ask again.
    view.rerender(wrap(<SocialsRenderMinutes rendering />, client))
    expect(api.getSocialsUsage.mock.calls.length).toBe(reads)

    server.usage = usage({ used_minutes: 4.2 })
    view.rerender(wrap(<SocialsRenderMinutes rendering={false} />, client))
    expect(await screen.findByText('Render minutes: 4.2 / 10 this month')).toBeInTheDocument()
    expect(api.getSocialsUsage.mock.calls.length).toBe(reads + 1)
  })
})

describe('rendering a post', () => {
  it('Render moves a draft with a template to Rendering', async () => {
    server.posts = [post()]
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))

    fireEvent.click(within(detail('Launch teaser')).getByRole('button', { name: 'Render' }))

    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1'))
    expect(await screen.findByRole('heading', { name: 'Rendering 1' })).toBeInTheDocument()
    expect(within(detail('Launch teaser')).getByRole('status')).toHaveTextContent('Rendering.')
    expect(toast.success).toHaveBeenCalledWith('Rendering started')
    // A rendering post is not rendered again, nor edited.
    expect(within(detail('Launch teaser')).queryByRole('button', { name: /^Render/ })).toBeNull()
    expect(within(detail('Launch teaser')).queryByRole('button', { name: 'Save copy' })).toBeNull()
  })

  it('a render refused for minutes (429) shows the server reason and changes nothing', async () => {
    server.posts = [post()]
    server.renderError = {
      status: 429,
      message: 'This workspace has used 10.0 of its 10 render minutes this month on the Basic plan.',
    }
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    fireEvent.click(within(detail('Launch teaser')).getByRole('button', { name: 'Render' }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(server.renderError!.message))
    expect(within(detail('Launch teaser')).getByTestId('socials-post-status')).toHaveTextContent('Draft')
    expect(toast.success).not.toHaveBeenCalled()
  })

  it('a post without a template, or a viewer, gets no Render button', async () => {
    server.posts = [post({ template_id: null })]
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    expect(within(detail('Launch teaser')).queryByRole('button', { name: 'Render' })).toBeNull()
    cleanup()

    server.posts = [post()]
    server.role = 'viewer'
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    expect(within(detail('Launch teaser')).queryByRole('button', { name: 'Render' })).toBeNull()
  })

  it('a failed render shows its reason and offers Render again', async () => {
    server.posts = [post({
      status: 'failed',
      review_log: [
        { at: '2026-09-25T09:01:00Z', by: 'user-1', action: 'render', comment: null },
        { at: '2026-09-25T09:02:00Z', by: 'user-1', action: 'render_failed', comment: 'The composition failed its check with 1 error(s).' },
      ],
    })]
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))

    const card = detail('Launch teaser')
    expect(within(card).getByRole('alert')).toHaveTextContent('The last render failed: The composition failed its check with 1 error(s).')
    expect(within(card).getByRole('button', { name: 'Render again' })).toBeInTheDocument()
    // Failed is editable: fix the post, then render again.
    expect(within(card).getByRole('button', { name: 'Save copy' })).toBeInTheDocument()
    expect(within(card).getByText('Render failed')).toBeInTheDocument()
  })
})

describe('polling', () => {
  it('polls only while a post renders', () => {
    expect(renderPollInterval(undefined)).toBe(false)
    expect(renderPollInterval({ posts: [post()] as any, total: 1 })).toBe(false)
    expect(renderPollInterval({ posts: [post(), post({ id: 'p2', status: 'rendering' })] as any, total: 2 }))
      .toBe(SOCIALS_RENDER_POLL_MS)
  })
})
