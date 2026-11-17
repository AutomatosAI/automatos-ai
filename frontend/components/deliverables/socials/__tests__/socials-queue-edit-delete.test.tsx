/**
 * 3 Oct 2026 (Gerard: "I see Needs you ... I can't add channels or remove, delete") — the
 * Queue's pane opens the post in the editor (Edit post), and a post with no channel says so
 * with a link that does the same. An owner or admin deletes a post after one confirmation;
 * an editor sees no Delete.
 *
 * One Approve (17 Nov 2026, Gerard: "are you sure, are you really sure?"): a post with no slot has
 * one Approve, "Approve & post now to X", which approves the version on screen on its one click;
 * the server publishes it (the approve route), so the screen never calls publish-now itself, and
 * no second question is asked. There is no separate Approve & post now. It waits for a channel, a
 * post with unsourced claims still asks to confirm them by name first (D7), and that confirmation
 * never carries over to the next post the pane shows (11 Oct 2026).
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
import { APPROVE, APPROVE_AND_POST_NOW } from '@/components/deliverables/socials/studio/queue-model'
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

const POST_NOW = `${APPROVE_AND_POST_NOW} X`
const CLAIMED = { variables: { members: { value: 1200, claim: true } }, sources: {} }

describe('one Approve for a post with no slot', () => {
  it('says it posts now to its channels, and approves the version on screen on one click', async () => {
    renderQueue(waiting('a'))
    expect(POST_NOW).toBe('Approve & post now to X')
    fireEvent.click(within(pane()).getByRole('button', { name: POST_NOW }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('a', 'hash-a', { postNow: true }))
    expect(api.approveSocialPost).toHaveBeenCalledTimes(1)
    // No second question, and the server publishes it (asked by post_now): the screen never calls publish-now.
    expect(within(pane()).queryByRole('alertdialog')).toBeNull()
    expect(api.publishSocialPostNow).not.toHaveBeenCalled()
  })

  it('is the only Approve: there is no separate Approve & post now', () => {
    renderQueue(waiting('a'))
    expect(within(pane()).getAllByRole('button', { name: /^Approve/ }).map((b) => b.textContent)).toEqual([POST_NOW])
    expect(within(pane()).queryByRole('button', { name: 'Approve & post now' })).toBeNull()
    expect(within(pane()).queryByRole('button', { name: 'Post now' })).toBeNull()
  })

  it('a claim with no source asks to confirm it by name first, and only the confirmation approves it', async () => {
    renderQueue(waiting('a', CLAIMED))
    fireEvent.click(within(pane()).getByRole('button', { name: POST_NOW }))
    const ask = within(pane()).getByRole('alertdialog', { name: 'Approve with unsourced claims' })
    expect(ask).toHaveTextContent('members')
    expect(api.approveSocialPost).not.toHaveBeenCalled()
    fireEvent.click(within(ask).getByRole('button', { name: 'Approve with unsourced claims' }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('a', 'hash-a', { overrideUnsourced: true, postNow: true }))
    expect(api.publishSocialPostNow).not.toHaveBeenCalled()
  })

  // 11 Oct 2026: a confirmation was open on post A when A was approved; A left the queue, the pane
  // fell back to the next post (B, no channels) with the confirmation still open, and its click
  // approved B. Keyed by the post, a confirmation left open never moves to the next one.
  it('a confirmation left open never moves to the next post', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const queue = (posts: any[]) => withClient(client,
      <SocialsQueue role="owner" posts={posts} selectedId={null} onSelect={vi.fn()} onEdit={edit} />)
    const view = render(queue([waiting('a', CLAIMED)]))
    fireEvent.click(within(pane()).getByRole('button', { name: POST_NOW }))
    expect(within(pane()).getByRole('alertdialog', { name: 'Approve with unsourced claims' })).toBeInTheDocument()
    view.rerender(queue([waiting('b', { targets: [] })]))
    expect(within(pane()).getByRole('heading', { name: 'Post b' })).toBeInTheDocument()
    expect(within(pane()).queryByRole('alertdialog')).toBeNull()
    expect(within(pane()).getByRole('button', { name: APPROVE })).toBeDisabled()
    expect(api.approveSocialPost).not.toHaveBeenCalled()
    expect(api.publishSocialPostNow).not.toHaveBeenCalled()
  })

  it('waits for a channel: with none it is a plain Approve, and off', () => {
    renderQueue(waiting('a', { targets: [] }))
    expect(within(pane()).getByRole('button', { name: APPROVE })).toBeDisabled()
    expect(within(pane()).queryByRole('button', { name: /post now/ })).toBeNull()
  })
})
