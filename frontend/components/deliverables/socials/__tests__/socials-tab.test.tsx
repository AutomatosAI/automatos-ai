/**
 * PRD-251 S0.5 — the Socials tab body against a mocked apiClient that applies
 * the S0.3 rules: the status machine (TRANSITIONS), the content hash, and D6 —
 * a content edit to an approved post voids the approval and sends it back to
 * needs_approval, and approve takes the content_hash of the version on screen
 * (a post changed since answers 409; so does a save another writer's commit
 * overtook). The workspace comes through the REAL WorkspaceProvider (its
 * GET /api/workspaces/current is the only stubbed fetch), so turning Socials on
 * is proven to refetch the workspace and swap the card for the list in place.
 *
 * PRD-251D US-D202/D203: the calendar's List opens a post as a calendar chip does, so these
 * tests follow it there: a post waiting for approval into the Queue's pane, a draft into the
 * editor (which submits, approves with the Queue's controls, US-D101, and saves an edit), an
 * archived post, or any post for a viewer, into the read-only view.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const server = vi.hoisted(() => ({
  role: 'owner' as string,
  socials: { available: true, enabled: false },
  posts: [] as any[],
  clock: 0,
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
  // The S0.3 status machine: action → {from: to}.
  const TRANSITIONS: Record<string, Record<string, string>> = {
    submit: { draft: 'needs_approval', changes_requested: 'needs_approval' },
    approve: { needs_approval: 'approved' },
    request_changes: { needs_approval: 'changes_requested' },
    reject: { needs_approval: 'archived' },
    edit: { approved: 'needs_approval', scheduled: 'needs_approval' },
  }
  const tick = () => new Date(Date.UTC(2026, 8, 23, 9, 0, server.clock++)).toISOString()
  const hashOf = (p: any) =>
    JSON.stringify([p.copy ?? {}, p.variables ?? {}, p.sources ?? {}, p.format ?? null, p.template_id ?? null, p.media ?? {}])
  const find = (id: string) => {
    const post = server.posts.find((p) => p.id === id)
    if (!post) throw new Error('Post not found')
    return post
  }
  const store = (next: any) => {
    server.posts = server.posts.map((p) => (p.id === next.id ? next : p))
    return next
  }
  const logged = (post: any, action: string, comment: string | null = null) => [
    ...post.review_log,
    { at: tick(), by: 'user-1', action, comment },
  ]
  const move = (id: string, action: string, comment: string | null = null, extra: (p: any) => object = () => ({})) => {
    const post = find(id)
    const to = TRANSITIONS[action][post.status]
    if (!to) throw new Error(`cannot ${action.replace('_', ' ')} a post that is ${post.status.replace('_', ' ')}`)
    return store({ ...post, ...extra(post), status: to, review_log: logged(post, action, comment), updated_at: tick() })
  }
  const apiClient = {
    setWorkspaceSocialsEnabled: vi.fn(async (enabled: boolean) => {
      server.socials = { ...server.socials, enabled }
      return { status: 'saved', socials: server.socials }
    }),
    listSocialPosts: vi.fn(async () => {
      const posts = [...server.posts].sort((a, b) => b.created_at.localeCompare(a.created_at))
      return { posts, total: posts.length }
    }),
    createSocialPost: vi.fn(async (input: any) => {
      const at = tick()
      const post: any = {
        id: `post-${server.posts.length + 1}`, workspace_id: 'w1', created_by: 'user-1',
        title: input.title, brief: input.brief ?? null, copy: input.copy ?? {}, format: null,
        template_id: null, variables: {}, sources: {}, media: {}, status: 'draft',
        approved_hash: null, approved_by: null, approved_at: null, override_unsourced: false,
        review_log: [], scheduled_for: null, timezone: null, created_at: at, updated_at: at,
      }
      post.content_hash = hashOf(post)
      server.posts = [...server.posts, post]
      return post
    }),
    updateSocialPost: vi.fn(async (id: string, changes: any) => {
      const post = find(id)
      const next = { ...post, ...changes }
      next.content_hash = hashOf(next)
      const contentChanged = next.content_hash !== post.content_hash
      // D6: a content edit voids the approval of an approved or scheduled post.
      const voided = contentChanged && TRANSITIONS.edit[post.status]
      return store({
        ...next,
        status: voided || post.status,
        review_log: voided ? logged(post, 'approval_voided') : logged(post, 'edit'),
        updated_at: tick(),
      })
    }),
    submitSocialPost: vi.fn(async (id: string) => move(id, 'submit')),
    approveSocialPost: vi.fn(async (id: string, contentHash: string) => {
      // D6: only the version the reviewer saw is approved; a post changed since is a 409.
      if (find(id).content_hash !== contentHash) {
        throw Object.assign(new Error('the post changed since you opened it'), { status: 409 })
      }
      return move(id, 'approve', null, (p) => ({ approved_hash: p.content_hash, approved_by: 'user-1', approved_at: tick() }))
    }),
    requestSocialPostChanges: vi.fn(async (id: string, comment: string) => move(id, 'request_changes', comment)),
    rejectSocialPost: vi.fn(async (id: string) => move(id, 'reject')),
    // PRD-251D: the editor saves a post's channels after its fields; the Queue reads the plans
    // and, sending a post back to Auto, asks for another take (the post is left as it is here).
    setSocialPostTargets: vi.fn(async (id: string) => find(id)),
    listSocialCampaigns: vi.fn(async () => ({ campaigns: [] })),
    retakeSocialPost: vi.fn(async (id: string) => find(id)),
    // The brand kit's name and a post's AI-label lines: nothing to say here.
    get: vi.fn(async () => ({ targets: [] })),
  }
  return { apiClient, default: apiClient }
})

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { WorkspaceProvider, useWorkspace } from '@/components/workspace-provider'
import { SocialsTab } from '@/components/deliverables/socials/socials-tab'
import { APPROVE_CARD } from '@/components/deliverables/socials/studio/editor-approve-card'
import { READ_ONLY_ROLE } from '@/components/deliverables/socials/studio/socials-post-view'
import { SOCIAL_POST_CHANGED_MESSAGE, SOCIAL_POST_REVIEW_STALE_MESSAGE } from '@/hooks/use-socials-api'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>

const currentWorkspace = vi.fn(async () => ({
  ok: true,
  status: 200,
  json: async () => ({
    id: 'w1', name: 'Acme', slug: 'acme', plan: 'pro', role: server.role,
    plan_limits: {}, socials: server.socials, settings: {},
  }),
}))

const contentHashOf = (p: any) =>
  JSON.stringify([p.copy ?? {}, p.variables ?? {}, p.sources ?? {}, p.format ?? null, p.template_id ?? null, p.media ?? {}])

function seedPost(overrides: Record<string, unknown>) {
  const at = new Date(Date.UTC(2026, 8, 22, 9, 0, server.posts.length)).toISOString()
  const post: any = {
    id: `seed-${server.posts.length + 1}`, workspace_id: 'w1', created_by: 'user-1', title: 'Seeded',
    brief: null, copy: { base: 'v1' }, format: null, template_id: null, variables: {}, sources: {},
    media: {}, status: 'draft', approved_hash: null, approved_by: null, approved_at: null,
    override_unsourced: false, review_log: [], scheduled_for: null, timezone: null,
    created_at: at, updated_at: at, ...overrides,
  }
  post.content_hash = contentHashOf(post)
  server.posts = [...server.posts, post]
  return post
}

/** F256: a post with no channel cannot be approved; the posts these tests approve go to LinkedIn. */
const CHANNEL = { id: 't-linkedin', toolkit: 'linkedin', post_kind: 'text', options: {}, status: 'pending' }

/** Marks the moment the provider has the workspace, so "renders nothing" is not vacuous. */
function WorkspaceLoaded() {
  const { workspace } = useWorkspace()
  return workspace ? <span data-testid="workspace-loaded" /> : null
}

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <WorkspaceProvider>
        <WorkspaceLoaded />
        <SocialsTab />
      </WorkspaceProvider>
    </QueryClientProvider>,
  )
}

/** A post's read-only view (studio/socials-post-view.tsx) and the status it shows. */
const detail = (title: string) => screen.getByRole('article', { name: `Post: ${title}` })
const statusOf = (title: string) => within(detail(title)).getByTestId('socials-post-status')
/** The editor's status, beside its title. */
const editorStatus = () => screen.getByTestId('socials-editor-status')
/** The Queue's pane, where a post that waits for approval opens. */
const findPane = () => screen.findByRole('region', { name: 'Post to approve' })
const nav = () => screen.getByRole('navigation', { name: 'Socials' })
/** Back to the calendar, whose layout is still the List. */
const toTheList = () => fireEvent.click(within(nav()).getByRole('button', { name: 'Calendar' }))

beforeEach(() => {
  server.role = 'owner'
  server.socials = { available: true, enabled: false }
  server.posts = []
  server.clock = 0
  Object.values(api).forEach((fn) => fn.mockClear())
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
  currentWorkspace.mockClear()
  vi.stubGlobal('fetch', currentWorkspace)
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('Socials available, this workspace switched off', () => {
  it.each(['owner', 'admin'])('an %s sees the Turn-on card; turning it on shows the empty list without a reload', async (role) => {
    server.role = role
    renderTab()

    const turnOn = await screen.findByRole('button', { name: 'Turn on Socials for this workspace' })
    expect(screen.queryByText('Ask an admin to turn on Socials')).toBeNull()
    fireEvent.click(turnOn)

    expect(await screen.findByText('No posts yet')).toBeInTheDocument()
    expect(api.setWorkspaceSocialsEnabled).toHaveBeenCalledWith(true)
    expect(screen.queryByRole('button', { name: 'Turn on Socials for this workspace' })).toBeNull()
    // The provider refetched GET /api/workspaces/current in place: two fetches, one mount.
    expect(currentWorkspace).toHaveBeenCalledTimes(2)
    expect(String((currentWorkspace.mock.calls[1] as unknown[])[0])).toContain('/api/workspaces/current')
    expect(api.listSocialPosts).toHaveBeenCalled()
  })

  it.each(['viewer', 'editor'])('a %s sees the ask-an-admin card and cannot turn it on', async (role) => {
    server.role = role
    renderTab()

    expect(await screen.findByText('Ask an admin to turn on Socials')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Turn on Socials/ })).toBeNull()
    expect(api.listSocialPosts).not.toHaveBeenCalled()
  })

  it('renders nothing while the platform switch is off', async () => {
    server.socials = { available: false, enabled: true }
    const { container } = renderTab()
    await screen.findByTestId('workspace-loaded')
    expect(container.textContent).toBe('')
    expect(screen.queryByRole('button')).toBeNull()
    expect(api.listSocialPosts).not.toHaveBeenCalled()
  })
})

describe('Socials on', () => {
  beforeEach(() => { server.socials = { available: true, enabled: true } })

  it('shows the empty state when there are no posts', async () => {
    renderTab()
    expect(await screen.findByText('No posts yet')).toBeInTheDocument()
  })

  it('creating a draft in the editor lists it under Draft with count 1', async () => {
    server.role = 'editor'
    renderTab()

    // PRD-251B US-B109: New post opens the editor; Save draft creates the post.
    fireEvent.click(await screen.findByRole('button', { name: /New post/ }))
    fireEvent.change(await screen.findByRole('textbox', { name: 'Title' }), { target: { value: 'Launch week teaser' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Brief' }), { target: { value: 'Tease Monday’s launch' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Copy' }), { target: { value: 'Something big lands Monday.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))

    await waitFor(() => expect(api.createSocialPost).toHaveBeenCalled())
    expect(api.createSocialPost).toHaveBeenCalledWith(expect.objectContaining({
      title: 'Launch week teaser', brief: 'Tease Monday’s launch', copy: { base: 'Something big lands Monday.', channels: {} },
    }))
    fireEvent.click(await screen.findByRole('button', { name: 'Back to calendar' }))
    const heading = await screen.findByRole('heading', { name: 'Draft 1' })
    expect(within(heading.closest('section')!).getByText('Launch week teaser')).toBeInTheDocument()
    expect(screen.queryByText('No posts yet')).toBeNull()
  })

  it('groups posts by status with counts, newest first', async () => {
    seedPost({ title: 'Older draft' })
    seedPost({ title: 'Waiting', status: 'needs_approval' })
    seedPost({ title: 'Newer draft' })
    renderTab()

    const drafts = (await screen.findByRole('heading', { name: 'Draft 2' })).closest('section')!
    const titles = within(drafts).getAllByRole('button').map((b) => b.textContent)
    expect(titles[0]).toContain('Newer draft')
    expect(titles[1]).toContain('Older draft')
    expect(screen.getByRole('heading', { name: 'Needs approval 1' })).toBeInTheDocument()
  })

  it('approving moves it to Approved; editing its copy afterwards moves it to Needs approval', async () => {
    // A text post: Submit sends it as it is (a post with a visual renders first, PRD-251D US-D102).
    const post = seedPost({ title: 'Launch week teaser', format: 'text', copy: { base: 'v1' }, targets: [CHANNEL] })
    renderTab()

    // The List opens the draft in its editor, which submits it.
    fireEvent.click(await screen.findByRole('button', { name: /Launch week teaser/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'Submit for approval' }))
    await waitFor(() => expect(editorStatus()).toHaveTextContent('Needs approval'))
    expect(api.submitSocialPost).toHaveBeenCalledWith(post.id)

    // Waiting for approval, it is approved where it is (US-D101), by the hash of the version on screen.
    fireEvent.click(within(await screen.findByRole('region', { name: APPROVE_CARD })).getByRole('button', { name: 'Approve' }))
    await waitFor(() => expect(editorStatus()).toHaveTextContent('Approved'))
    expect(api.approveSocialPost).toHaveBeenCalledWith(post.id, server.posts[0].content_hash)
    expect(server.posts[0].approved_hash).toBe(server.posts[0].content_hash)

    // An edit saved afterwards voids the approval (D6).
    fireEvent.change(screen.getByRole('textbox', { name: 'Copy' }), { target: { value: 'v2, sharper' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))

    await waitFor(() => expect(editorStatus()).toHaveTextContent('Needs approval'))
    expect(api.updateSocialPost).toHaveBeenLastCalledWith(post.id, expect.objectContaining({ copy: { base: 'v2, sharper', channels: {} } }))
    expect(server.posts[0].approved_hash).not.toBe(server.posts[0].content_hash)
    expect(within(screen.getByRole('region', { name: 'History' })).getByText('Approval voided by an edit')).toBeInTheDocument()

    // The List says the same.
    fireEvent.click(screen.getByRole('button', { name: 'Back to calendar' }))
    await screen.findByRole('heading', { name: 'Needs approval 1' })
    expect(screen.queryByRole('heading', { name: /^Approved/ })).toBeNull()
  })

  it('Approve sends the hash of the version on screen; a 409 (changed while reviewed) says so and reloads the post', async () => {
    const post = seedPost({ title: 'Launch week teaser', status: 'needs_approval', copy: { base: 'v1' }, targets: [CHANNEL] })
    renderTab()
    // From the List, a post waiting for approval opens in the Queue's pane, as from the calendar.
    fireEvent.click(await screen.findByRole('button', { name: /Launch week teaser/ }))
    const pane = await findPane()
    expect(within(pane).getByText('v1')).toBeInTheDocument()

    // Another editor changes the copy on the server; this screen still shows v1 (the list is cached).
    const edited = { ...server.posts[0], copy: { base: 'v2 from another editor' } }
    server.posts = [{ ...edited, content_hash: contentHashOf(edited) }]
    const fetches = api.listSocialPosts.mock.calls.length

    fireEvent.click(within(pane).getByRole('button', { name: 'Approve' }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(SOCIAL_POST_REVIEW_STALE_MESSAGE))
    expect(within(pane).getByRole('alert')).toHaveTextContent(SOCIAL_POST_REVIEW_STALE_MESSAGE)
    expect(api.approveSocialPost).toHaveBeenCalledWith(post.id, post.content_hash)
    await waitFor(() => expect(api.listSocialPosts.mock.calls.length).toBeGreaterThan(fetches))
    expect(await within(pane).findByText('v2 from another editor')).toBeInTheDocument()
    expect(server.posts[0].status).toBe('needs_approval')
    expect(server.posts[0].approved_hash).toBeNull()
    expect(toast.success).not.toHaveBeenCalled()

    // Now shown the edit, the reviewer approves exactly it; it leaves the Queue and the List says Approved.
    fireEvent.click(within(pane).getByRole('button', { name: 'Approve' }))
    expect(await screen.findByText('Nothing is waiting for your approval.')).toBeInTheDocument()
    expect(api.approveSocialPost).toHaveBeenLastCalledWith(post.id, server.posts[0].content_hash)
    expect(server.posts[0].approved_hash).toBe(server.posts[0].content_hash)
    toTheList()
    await screen.findByRole('heading', { name: 'Approved 1' })
  })

  it("Save draft answering 409 (another writer committed first) says so, refetches the posts and shows the server's version", async () => {
    // PRD-251D: the List opens a draft in its editor, where the copy is saved with Save draft.
    const post = seedPost({ title: 'Launch week teaser', copy: { base: 'v1' } })
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /Launch week teaser/ }))
    fireEvent.change(await screen.findByRole('textbox', { name: 'Copy' }), { target: { value: 'v2 mine' } })

    // Another editor's change commits first, so the server refuses this save.
    const edited = { ...server.posts[0], copy: { base: 'v2 from another editor' }, updated_at: '2026-09-23T10:00:00.000Z' }
    server.posts = [{ ...edited, content_hash: contentHashOf(edited) }]
    api.updateSocialPost.mockRejectedValueOnce(
      Object.assign(new Error('the post changed since you opened it'), { status: 409 }),
    )
    const fetches = api.listSocialPosts.mock.calls.length

    fireEvent.click(screen.getByRole('button', { name: 'Save draft' }))

    // The editor says the post changed, and reads the posts again.
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(SOCIAL_POST_CHANGED_MESSAGE))
    expect(api.updateSocialPost).toHaveBeenCalledWith(post.id, expect.objectContaining({ copy: { base: 'v2 mine', channels: {} } }))
    await waitFor(() => expect(api.listSocialPosts.mock.calls.length).toBeGreaterThan(fetches))
    expect(await screen.findByDisplayValue('v2 from another editor')).toBeInTheDocument()
    expect(server.posts[0].copy).toEqual({ base: 'v2 from another editor' })
    expect(toast.success).not.toHaveBeenCalled()
  })

  it('request changes takes a comment and moves the post to Changes requested', async () => {
    const post = seedPost({ title: 'Needs work', status: 'needs_approval' })
    renderTab()

    // PRD-251D: in the Queue's pane, Request changes sends the post back to Auto: the comment goes
    // with the request, and Auto is asked for another take with it.
    fireEvent.click(await screen.findByRole('button', { name: /Needs work/ }))
    const pane = await findPane()
    fireEvent.click(within(pane).getByRole('button', { name: 'Request changes' }))
    const send = within(pane).getByRole('button', { name: 'Send back to Auto' })
    expect(send).toBeDisabled()
    fireEvent.change(within(pane).getByLabelText('What should change?'), { target: { value: 'Tighten the hook' } })
    fireEvent.click(send)

    await waitFor(() => expect(api.retakeSocialPost).toHaveBeenCalledWith(post.id, 'Tighten the hook'))
    expect(api.requestSocialPostChanges).toHaveBeenCalledWith(post.id, 'Tighten the hook')
    expect(server.posts[0].status).toBe('changes_requested')

    // The List shows it under Changes requested; it opens in its editor, which shows the comment
    // in its history and lets the author resubmit.
    toTheList()
    const heading = await screen.findByRole('heading', { name: 'Changes requested 1' })
    fireEvent.click(within(heading.closest('section')!).getByRole('button', { name: /Needs work/ }))
    expect(await screen.findByRole('button', { name: 'Submit for approval' })).toBeInTheDocument()
    expect(editorStatus()).toHaveTextContent('Changes requested')
    expect(within(screen.getByRole('region', { name: 'History' })).getByText('Tighten the hook')).toBeInTheDocument()
  })

  it('reject takes an optional reason and archives the post', async () => {
    const post = seedPost({ title: 'Off brand', status: 'needs_approval' })
    renderTab()

    fireEvent.click(await screen.findByRole('button', { name: /Off brand/ }))
    const pane = await findPane()
    fireEvent.click(within(pane).getByRole('button', { name: 'Reject' }))
    fireEvent.change(within(pane).getByLabelText('Reason (optional)'), { target: { value: 'Not our voice' } })
    fireEvent.click(within(pane).getByRole('button', { name: 'Reject post' }))
    await waitFor(() => expect(api.rejectSocialPost).toHaveBeenCalledWith(post.id, 'Not our voice'))

    // PRD-251D US-D202: the List shows it under Archived, and it opens in its read-only view.
    toTheList()
    const heading = await screen.findByRole('heading', { name: 'Archived 1' })
    fireEvent.click(within(heading.closest('section')!).getByRole('button', { name: /Off brand/ }))
    expect(statusOf('Off brand')).toHaveTextContent('Archived')
  })

  it("an agent's draft reads as Drafted in the history, naming the agent (US-116)", async () => {
    seedPost({
      title: 'Agent draft', status: 'needs_approval', created_by: 'agent:7',
      review_log: [
        { at: '2026-09-22T09:00:00Z', by: 'agent:7', action: 'draft', comment: 'Drafted by Social Media Director.' },
        { at: '2026-09-22T09:01:00Z', by: 'agent:7', action: 'submit', comment: null },
      ],
    })
    renderTab()

    // PRD-251D: it opens in the Queue's pane; Edit post opens its editor, whose Activity card holds the history.
    fireEvent.click(await screen.findByRole('button', { name: /Agent draft/ }))
    fireEvent.click(within(await findPane()).getByRole('button', { name: 'Edit post' }))
    const history = await screen.findByRole('region', { name: 'History' })
    expect(within(history).getByText('Drafted')).toBeInTheDocument()
    expect(within(history).getByText('Drafted by Social Media Director.')).toBeInTheDocument()
    expect(within(history).getByText('Sent for approval')).toBeInTheDocument()
    expect(within(history).queryByText('draft')).toBeNull()
  })

  it('an editor may review; a viewer may only read', async () => {
    seedPost({ title: 'For review', status: 'needs_approval' })
    server.role = 'editor'
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /For review/ }))
    // PRD-251D: the Queue's pane; its copy is changed in the editor, which Edit post opens.
    const pane = await findPane()
    for (const name of ['Approve', 'Request changes', 'Reject', 'Edit post']) {
      expect(within(pane).getByRole('button', { name })).toBeInTheDocument()
    }
    cleanup()

    server.role = 'viewer'
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /For review/ }))
    // PRD-251D US-D203: one who may not approve reads the post, in the Queue, with nothing to press.
    const card = detail('For review')
    for (const name of ['Approve', 'Request changes', 'Reject', 'Save copy', 'Submit for approval', 'Save draft', 'Edit post']) {
      expect(within(card).queryByRole('button', { name })).toBeNull()
    }
    expect(within(card).queryAllByRole('button')).toHaveLength(0)
    expect(within(card).getByText(READ_ONLY_ROLE)).toBeInTheDocument()
    expect(READ_ONLY_ROLE).toBe('Your role can read posts but not change them.')
    expect(screen.queryByRole('button', { name: /New post/ })).toBeNull()
    expect(screen.queryByRole('button', { name: /Blank draft/ })).toBeNull()
  })
})
