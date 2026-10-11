/**
 * 3 Oct 2026 (Gerard: "I see Needs you ... I can't add channels or remove, delete") — the
 * Queue's pane opens the post in the editor (Edit post), and a post with no channel says so
 * with a link that does the same. An owner or admin deletes a post after one confirmation;
 * an editor sees no Delete. Approve & post now approves the version on screen and publishes
 * it at once, after a confirmation that names its channels; it waits for a channel, and a
 * post with unsourced claims goes through Approve.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const state = vi.hoisted(() => ({ role: 'owner' }))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({
    workspace: { id: 'w1', name: 'Acme', role: state.role, socials: { available: true, enabled: true, series_approval: false } },
  }),
}))
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    getSocialPostMedia: vi.fn(async () => []),
    listSocialCampaigns: vi.fn(async () => ({ campaigns: [] })),
    approveSocialPost: vi.fn(async (id: string) => ({ id, status: 'approved' })),
    publishSocialPostNow: vi.fn(async (id: string) => ({ id, status: 'publishing' })),
    deleteSocialPost: vi.fn(async () => undefined),
    requestSocialPostChanges: vi.fn(async (id: string) => ({ id })),
    retakeSocialPost: vi.fn(async (id: string) => ({ id })),
    rejectSocialPost: vi.fn(),
    listSocialPosts: vi.fn(async () => ({ posts: [], total: 0 })),
  }
  return { apiClient, default: apiClient }
})
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({ FilePreview: () => null, inferPreviewType: () => 'image' }))

import { apiClient } from '@/lib/api-client'
import { SocialsQueue } from '@/components/deliverables/socials/studio/socials-queue'
import { ADD_A_CHANNEL } from '@/components/deliverables/socials/socials-post-review'
import { DELETE_CONFIRM } from '@/components/deliverables/socials/studio/delete-post-button'
import { ApproveAndPostNow, POST_NOW_LABEL } from '@/components/deliverables/socials/socials-approve-post-now'
import { renderWith } from './socials-editor-harness'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>
const edit = vi.fn()

function waiting(id: string, extra: Record<string, unknown> = {}) {
  return {
    id, title: `Post ${id}`, status: 'needs_approval', format: 'image', length_seconds: null, timezone: 'UTC',
    planned_for: null, scheduled_for: null, content_hash: `hash-${id}`, approved_hash: null, media: {},
    copy: { base: `Copy of ${id}.` }, variables: {}, sources: {}, review_log: [], campaign_id: null,
    targets: [{ id: `t-${id}`, toolkit: 'twitter', post_kind: 'image', options: {}, status: 'pending' }],
    created_at: '2026-10-13T09:00:00Z', ...extra,
  } as any
}

function renderQueue(post: any) {
  return renderWith(<SocialsQueue role={state.role as any} posts={[post]} selectedId={null} onSelect={vi.fn()} onEdit={edit} />)
}

const pane = () => screen.getByRole('region', { name: 'Post to approve' })
const withClient = (client: QueryClient, ui: React.ReactElement) => (
  <QueryClientProvider client={client}>{ui}</QueryClientProvider>
)

beforeEach(() => {
  state.role = 'owner'
  edit.mockClear()
  Object.values(api).forEach((fn) => fn.mockClear())
})
afterEach(() => cleanup())

describe('the Queue opens a post in the editor', () => {
  it('Edit post opens it, and so does the no-channel hint', () => {
    renderQueue(waiting('a', { targets: [] }))
    fireEvent.click(within(pane()).getByRole('button', { name: 'Edit post' }))
    fireEvent.click(within(pane()).getByRole('button', { name: ADD_A_CHANNEL }))
    expect(edit.mock.calls).toEqual([['a'], ['a']])
  })
})

describe('deleting a post', () => {
  it('an owner deletes it after one confirmation', async () => {
    renderQueue(waiting('a'))
    fireEvent.click(within(pane()).getByRole('button', { name: 'Delete' }))
    const ask = within(pane()).getByRole('alertdialog', { name: 'Delete post' })
    expect(ask).toHaveTextContent(DELETE_CONFIRM)
    expect(api.deleteSocialPost).not.toHaveBeenCalled()
    fireEvent.click(within(ask).getByRole('button', { name: 'Delete post' }))
    await waitFor(() => expect(api.deleteSocialPost).toHaveBeenCalledWith('a'))
  })

  it('Keep it changes nothing, and an editor sees no Delete', () => {
    renderQueue(waiting('a'))
    fireEvent.click(within(pane()).getByRole('button', { name: 'Delete' }))
    fireEvent.click(within(pane()).getByRole('button', { name: 'Keep it' }))
    expect(within(pane()).queryByRole('alertdialog', { name: 'Delete post' })).toBeNull()
    cleanup()
    state.role = 'editor'
    renderQueue(waiting('b'))
    expect(within(pane()).queryByRole('button', { name: 'Delete' })).toBeNull()
    expect(api.deleteSocialPost).not.toHaveBeenCalled()
  })
})

describe('Approve & post now', () => {
  it('approves the version on screen, then publishes it, after naming its channels', async () => {
    renderQueue(waiting('a'))
    fireEvent.click(within(pane()).getByRole('button', { name: POST_NOW_LABEL }))
    const ask = within(pane()).getByRole('alertdialog', { name: POST_NOW_LABEL })
    expect(ask).toHaveTextContent('Approve and post now to X? It goes live at once.')
    fireEvent.click(within(ask).getByRole('button', { name: 'Post now' }))
    await waitFor(() => expect(api.publishSocialPostNow).toHaveBeenCalledWith('a'))
    expect(api.approveSocialPost).toHaveBeenCalledWith('a', 'hash-a')
    expect(api.approveSocialPost.mock.invocationCallOrder[0]).toBeLessThan(api.publishSocialPostNow.mock.invocationCallOrder[0])
  })

  // 11 Oct 2026: the confirmation was open on post A when A was approved; A left the queue,
  // the pane fell back to the next post (B, no channels) with the confirmation still open,
  // and Post now approved B and asked to publish it (409). A was never published.
  it('a confirmation left open never moves to the next post', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const queue = (posts: any[]) => withClient(client,
      <SocialsQueue role="owner" posts={posts} selectedId={null} onSelect={vi.fn()} onEdit={edit} />)
    const view = render(queue([waiting('a')]))
    fireEvent.click(within(pane()).getByRole('button', { name: POST_NOW_LABEL }))
    expect(within(pane()).getByRole('alertdialog', { name: POST_NOW_LABEL })).toHaveTextContent('to X?')
    view.rerender(queue([waiting('b', { targets: [] })]))
    expect(within(pane()).getByRole('heading', { name: 'Post b' })).toBeInTheDocument()
    expect(within(pane()).queryByRole('alertdialog', { name: POST_NOW_LABEL })).toBeNull()
    expect(within(pane()).getByRole('button', { name: POST_NOW_LABEL })).toBeDisabled()
    expect(api.approveSocialPost).not.toHaveBeenCalled()
    expect(api.publishSocialPostNow).not.toHaveBeenCalled()
  })

  it('Post now in an open confirmation waits whenever the button would', () => {
    const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
    const post = waiting('a')
    const view = render(withClient(client, <ApproveAndPostNow post={post} disabled={false} />))
    fireEvent.click(screen.getByRole('button', { name: POST_NOW_LABEL }))
    view.rerender(withClient(client, <ApproveAndPostNow post={post} disabled />))
    expect(screen.getByRole('button', { name: 'Post now' })).toBeDisabled()
  })

  it('waits for a channel, and a post with unsourced claims goes through Approve', () => {
    renderQueue(waiting('a', { targets: [] }))
    expect(within(pane()).getByRole('button', { name: POST_NOW_LABEL })).toBeDisabled()
    cleanup()
    renderQueue(waiting('b', { variables: { members: { value: 1200, claim: true } }, sources: {} }))
    expect(within(pane()).getByRole('button', { name: POST_NOW_LABEL })).toBeDisabled()
  })
})
