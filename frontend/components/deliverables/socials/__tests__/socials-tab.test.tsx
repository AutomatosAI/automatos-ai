/**
 * PRD-251 S0.5 — the Socials tab body against a mocked apiClient that applies
 * the S0.3 rules: the status machine (TRANSITIONS), the content hash, and D6 —
 * a content edit to an approved post voids the approval and sends it back to
 * needs_approval, and approve takes the content_hash of the version on screen
 * (a post changed since answers 409; so does a save another writer's commit
 * overtook). The workspace comes through the REAL WorkspaceProvider (its
 * GET /api/workspaces/current is the only stubbed fetch), so turning Socials on
 * is proven to refetch the workspace and swap the card for the list in place.
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
  }
  return { apiClient, default: apiClient }
})

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { WorkspaceProvider, useWorkspace } from '@/components/workspace-provider'
import { SocialsTab } from '@/components/deliverables/socials/socials-tab'
import { SOCIAL_POST_CHANGED_MESSAGE } from '@/hooks/use-socials-api'

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

const detail = (title: string) => screen.getByRole('article', { name: `Post: ${title}` })
const statusOf = (title: string) => within(detail(title)).getByTestId('socials-post-status')

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

  it('creating a draft lists it under Draft with count 1', async () => {
    server.role = 'editor'
    renderTab()

    fireEvent.click(await screen.findByRole('button', { name: /New draft/ }))
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'Launch week teaser' } })
    fireEvent.change(screen.getByLabelText('Brief'), { target: { value: 'Tease Monday’s launch' } })
    fireEvent.change(screen.getByLabelText('Copy'), { target: { value: 'Something big lands Monday.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create draft' }))

    const heading = await screen.findByRole('heading', { name: 'Draft 1' })
    expect(within(heading.closest('section')!).getByText('Launch week teaser')).toBeInTheDocument()
    expect(api.createSocialPost).toHaveBeenCalledWith({
      title: 'Launch week teaser', brief: 'Tease Monday’s launch', copy: { base: 'Something big lands Monday.' },
    })
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
    const post = seedPost({ title: 'Launch week teaser', copy: { base: 'v1' } })
    renderTab()

    fireEvent.click(await screen.findByRole('button', { name: /Launch week teaser/ }))
    fireEvent.click(within(detail('Launch week teaser')).getByRole('button', { name: 'Submit for approval' }))
    await screen.findByRole('heading', { name: 'Needs approval 1' })

    fireEvent.click(within(detail('Launch week teaser')).getByRole('button', { name: 'Approve' }))
    await screen.findByRole('heading', { name: 'Approved 1' })
    expect(statusOf('Launch week teaser')).toHaveTextContent('Approved')
    expect(api.approveSocialPost).toHaveBeenCalledWith(post.id, post.content_hash)
    expect(server.posts[0].approved_hash).toBe(server.posts[0].content_hash)

    fireEvent.change(within(detail('Launch week teaser')).getByLabelText('Copy'), { target: { value: 'v2, sharper' } })
    fireEvent.click(within(detail('Launch week teaser')).getByRole('button', { name: 'Save copy' }))

    await screen.findByRole('heading', { name: 'Needs approval 1' })
    expect(screen.queryByRole('heading', { name: /^Approved/ })).toBeNull()
    expect(statusOf('Launch week teaser')).toHaveTextContent('Needs approval')
    expect(api.updateSocialPost).toHaveBeenCalledWith(post.id, { copy: { base: 'v2, sharper' } })
    expect(server.posts[0].approved_hash).not.toBe(server.posts[0].content_hash)
    expect(within(detail('Launch week teaser')).getByText('Approval voided by an edit')).toBeInTheDocument()
  })

  it('Approve sends the hash of the version on screen; a 409 (changed since) toasts and refetches the posts', async () => {
    const post = seedPost({ title: 'Launch week teaser', status: 'needs_approval', copy: { base: 'v1' } })
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /Launch week teaser/ }))
    expect(within(detail('Launch week teaser')).getByLabelText('Copy')).toHaveValue('v1')

    // Another editor changes the copy on the server; this screen still shows v1 (the list is cached).
    const edited = { ...server.posts[0], copy: { base: 'v2 from another editor' } }
    server.posts = [{ ...edited, content_hash: contentHashOf(edited) }]
    const fetches = api.listSocialPosts.mock.calls.length

    fireEvent.click(within(detail('Launch week teaser')).getByRole('button', { name: 'Approve' }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(SOCIAL_POST_CHANGED_MESSAGE))
    expect(api.approveSocialPost).toHaveBeenCalledWith(post.id, post.content_hash)
    await waitFor(() => expect(api.listSocialPosts.mock.calls.length).toBeGreaterThan(fetches))
    expect(await within(detail('Launch week teaser')).findByDisplayValue('v2 from another editor')).toBeInTheDocument()
    expect(statusOf('Launch week teaser')).toHaveTextContent('Needs approval')
    expect(server.posts[0].approved_hash).toBeNull()
    expect(toast.success).not.toHaveBeenCalled()

    // Now shown the edit, the reviewer approves exactly it.
    fireEvent.click(within(detail('Launch week teaser')).getByRole('button', { name: 'Approve' }))
    await screen.findByRole('heading', { name: 'Approved 1' })
    expect(api.approveSocialPost).toHaveBeenLastCalledWith(post.id, server.posts[0].content_hash)
    expect(server.posts[0].approved_hash).toBe(server.posts[0].content_hash)
  })

  it('Save copy answering 409 (another writer committed first) toasts and refetches the posts, as Approve does', async () => {
    const post = seedPost({ title: 'Launch week teaser', status: 'needs_approval', copy: { base: 'v1' } })
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /Launch week teaser/ }))
    fireEvent.change(within(detail('Launch week teaser')).getByLabelText('Copy'), { target: { value: 'v2 mine' } })

    // Another editor's change commits first, so the server refuses this save.
    const edited = { ...server.posts[0], copy: { base: 'v2 from another editor' } }
    server.posts = [{ ...edited, content_hash: contentHashOf(edited) }]
    api.updateSocialPost.mockRejectedValueOnce(
      Object.assign(new Error('the post changed since you opened it'), { status: 409 }),
    )
    const fetches = api.listSocialPosts.mock.calls.length

    fireEvent.click(within(detail('Launch week teaser')).getByRole('button', { name: 'Save copy' }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(SOCIAL_POST_CHANGED_MESSAGE))
    expect(api.updateSocialPost).toHaveBeenCalledWith(post.id, { copy: { base: 'v2 mine' } })
    await waitFor(() => expect(api.listSocialPosts.mock.calls.length).toBeGreaterThan(fetches))
    expect(await within(detail('Launch week teaser')).findByDisplayValue('v2 from another editor')).toBeInTheDocument()
    expect(server.posts[0].copy).toEqual({ base: 'v2 from another editor' })
    expect(toast.success).not.toHaveBeenCalled()
  })

  it('request changes takes a comment and moves the post to Changes requested', async () => {
    const post = seedPost({ title: 'Needs work', status: 'needs_approval' })
    renderTab()

    fireEvent.click(await screen.findByRole('button', { name: /Needs work/ }))
    fireEvent.click(within(detail('Needs work')).getByRole('button', { name: 'Request changes' }))
    const send = within(detail('Needs work')).getByRole('button', { name: 'Send request' })
    expect(send).toBeDisabled()
    fireEvent.change(within(detail('Needs work')).getByLabelText('What needs to change?'), { target: { value: 'Tighten the hook' } })
    fireEvent.click(send)

    await screen.findByRole('heading', { name: 'Changes requested 1' })
    expect(api.requestSocialPostChanges).toHaveBeenCalledWith(post.id, 'Tighten the hook')
    expect(statusOf('Needs work')).toHaveTextContent('Changes requested')
    expect(within(detail('Needs work')).getByText('Tighten the hook')).toBeInTheDocument()
    // Changes requested → the author can resubmit.
    expect(within(detail('Needs work')).getByRole('button', { name: 'Submit for approval' })).toBeInTheDocument()
  })

  it('reject archives the post', async () => {
    seedPost({ title: 'Off brand', status: 'needs_approval' })
    renderTab()

    fireEvent.click(await screen.findByRole('button', { name: /Off brand/ }))
    fireEvent.click(within(detail('Off brand')).getByRole('button', { name: 'Reject' }))
    await screen.findByRole('heading', { name: 'Archived 1' })
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

    fireEvent.click(await screen.findByRole('button', { name: /Agent draft/ }))
    const history = within(detail('Agent draft')).getByRole('region', { name: 'History' })
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
    for (const name of ['Approve', 'Request changes', 'Reject', 'Save copy']) {
      expect(within(detail('For review')).getByRole('button', { name })).toBeInTheDocument()
    }
    cleanup()

    server.role = 'viewer'
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /For review/ }))
    const card = detail('For review')
    for (const name of ['Approve', 'Request changes', 'Reject', 'Save copy', 'Submit for approval']) {
      expect(within(card).queryByRole('button', { name })).toBeNull()
    }
    expect(within(card).getByText('Your role can read posts but not change them.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /New draft/ })).toBeNull()
  })
})
