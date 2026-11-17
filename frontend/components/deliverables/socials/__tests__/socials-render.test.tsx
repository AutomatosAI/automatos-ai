/**
 * PRD-251 S1.1c — rendering in the Socials tab, against a mocked apiClient:
 *
 * * the list shows "Render minutes: used / quota this month" from
 *   GET /api/socials/usage (a plan with no quota shows the minutes used alone,
 *   and a used-up month says so);
 * * a draft with a template, for a role that authors, opens in its editor from the List, and
 *   Submit for approval gives it the full render (POST /render, PRD-251D US-D102); the post
 *   then shows Rendering in its read-only view, with nothing to render or edit; a refused
 *   render (429, no minutes left) shows the server's reason;
 * * a failed render shows its reason and offers a render again, in the editor's Activity card;
 * * the list polls only while a post renders, and the minutes are read again
 *   when the last render ends.
 * * P256-FIX-SOC-5 (F437): a rendering post says when it started and when it gives up (the
 *   server's `render`), keeping the "Rendering." start; a plan's draft that waits for you
 *   (SOC-2's `needs_input`) says what it waits for.
 *
 * PRD-251D US-D203: these lines moved from the older detail pane to the post's Activity card
 * (studio/post-notices.tsx), under the editor and in the read-only view.
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

vi.mock('next/navigation', () => ({
  usePathname: () => '/deliverables',
  // PRD-251B US-B107: the Studio holds its view in the URL (router.push).
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  // US-B108: the calendar's List is the post list these tests drive.
  useSearchParams: () => new URLSearchParams('tab=socials&cal=list'),
}))
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
    // PRD-251D: the editor saves a post (its fields, then its channels) before it renders it.
    updateSocialPost: vi.fn(async (id: string) => server.posts.find((p) => p.id === id)),
    setSocialPostTargets: vi.fn(async (id: string) => server.posts.find((p) => p.id === id)),
    // The brand kit's name and a post's AI-label lines: nothing to say here.
    get: vi.fn(async () => ({ targets: [] })),
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
import { RENDERING_NOTE, renderingLine, waitingLine } from '@/components/deliverables/socials/studio/post-notices'
import { RENDERING_FOR_APPROVAL } from '@/components/deliverables/socials/studio/editor-submit'
import { format } from 'date-fns'

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
    // PRD-251D: an author's draft now opens in the editor, which reads these as a saved post has them.
    planned_for: null, length_seconds: null, targets: [], preview: null,
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

/** A post's read-only view (studio/socials-post-view.tsx). */
const detail = (title: string) => screen.getByRole('article', { name: `Post: ${title}` })
/** The editor's status, beside its title, and its Activity card. */
const editorStatus = () => screen.getByTestId('socials-editor-status')
const findActivity = () => screen.findByRole('region', { name: 'Activity' })

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
  it("the editor's Submit for approval renders a draft with a template, and the post shows Rendering", async () => {
    server.posts = [post()]
    render(wrap(<SocialsTab />))
    // PRD-251D: the List opens the draft in its editor, whose Submit for approval makes the full
    // render (US-D102): the older pane's Render.
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'Submit for approval' }))

    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1', {}))
    // While it renders the post page shows the post's read-only view.
    const card = await screen.findByRole('article', { name: 'Post: Launch teaser' })
    expect(within(card).getByTestId('socials-post-status')).toHaveTextContent('Rendering')
    expect(within(card).getByRole('status')).toHaveTextContent('Rendering.')
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(RENDERING_FOR_APPROVAL))
    // A rendering post is not rendered again, nor edited.
    for (const name of [/^Render/, 'Save copy', 'Save draft', 'Submit for approval']) {
      expect(screen.queryByRole('button', { name })).toBeNull()
    }
    fireEvent.click(screen.getByRole('button', { name: 'Back to calendar' }))
    expect(await screen.findByRole('heading', { name: 'Rendering 1' })).toBeInTheDocument()
  })

  it('a render refused for minutes (429) shows the server reason and changes nothing', async () => {
    server.posts = [post()]
    server.renderError = {
      status: 429,
      message: 'This workspace has used 10.0 of its 10 render minutes this month on the Basic plan.',
    }
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'Submit for approval' }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(server.renderError!.message))
    expect(editorStatus()).toHaveTextContent('Draft')
    expect(server.posts[0].status).toBe('draft')
    expect(toast.success).not.toHaveBeenCalled()
  })

  it('a viewer gets no render; a post without a template is rendered with Let Auto pick', async () => {
    // PRD-251D (US-D103): the editor renders a post with no template, Auto picking one, so a
    // template-less draft is offered Render preview where the older pane offered nothing.
    server.posts = [post({ template_id: null })]
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    expect(await screen.findByRole('button', { name: 'Render preview' })).toBeEnabled()
    cleanup()

    server.posts = [post()]
    server.role = 'viewer'
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    expect(within(detail('Launch teaser')).queryByRole('button', { name: /Render/ })).toBeNull()
    expect(screen.queryByRole('button', { name: /Render/ })).toBeNull()
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
    // Failed is editable: the List opens it in its editor, whose Activity card says why.
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))

    const activity = await findActivity()
    expect(within(activity).getByRole('alert')).toHaveTextContent('The last render failed: The composition failed its check with 1 error(s).')
    expect(within(activity).getByText('Render failed')).toBeInTheDocument()
    // Fix the post in the editor (Save draft), then render it again: the full render, which the
    // editor's header does not offer a failed post (its Render preview of a video is the preview).
    expect(screen.getByRole('button', { name: 'Save draft' })).toBeEnabled()
    expect(editorStatus()).toHaveTextContent('Failed')
    fireEvent.click(within(activity).getByRole('button', { name: 'Render again' }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('post-1'))
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Rendering started'))
  })
})

describe('P256-FIX-SOC-5: a render says since when, and a waiting draft says why', () => {
  const STARTED = '2026-10-14T16:22:00+00:00'
  const GIVES_UP = '2026-10-14T16:47:00+00:00'
  const asked = { at: '2026-10-14T07:00:00Z', by: 'the plan', action: 'needs_input', comment: 'fill in item_1_name, item_1_value before rendering' }

  it('a rendering post shows when it started and when it gives up, after "Rendering."', async () => {
    const render_ = { state: 'rendering', started_at: STARTED, gives_up_at: GIVES_UP, finished_at: null, why: null }
    server.posts = [post({ status: 'rendering', render: render_, review_log: [{ at: STARTED, by: 'user-1', action: 'render', comment: null, gives_up_at: GIVES_UP }] })]
    render(wrap(<SocialsTab />))
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    const line = `Rendering. Started ${format(new Date(STARTED), 'HH:mm')}; it gives up at ${format(new Date(GIVES_UP), 'HH:mm')} if it has not finished.`
    expect(within(detail('Launch teaser')).getByRole('status')).toHaveTextContent(line)
    expect(renderingLine({ render: null })).toBe(RENDERING_NOTE)  // a render the server gives no start for
  })

  it('a draft whose newest entry asks for your input says so, and one rendered since does not', async () => {
    server.posts = [post({ review_log: [asked] })]
    render(wrap(<SocialsTab />))
    // A draft opens in its editor; its Activity card says what it waits for.
    fireEvent.click(await screen.findByRole('button', { name: /Launch teaser/ }))
    expect(within(await findActivity()).getByRole('alert')).toHaveTextContent('Waiting for you: fill in item_1_name, item_1_value before rendering')
    const rendered = { at: '2026-10-14T08:00:00Z', by: 'user-1', action: 'render', comment: null }
    expect(waitingLine({ status: 'draft', review_log: [asked, rendered] } as any)).toBeNull()
    expect(waitingLine({ status: 'draft', review_log: [{ ...asked, fields: ['Item 1 name'] }] } as any)).toBe('Waiting for you: Fill in: Item 1 name')
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
