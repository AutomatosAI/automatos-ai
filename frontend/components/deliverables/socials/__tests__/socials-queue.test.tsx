/**
 * PRD-251B US-B111 — the Queue (Queue.dc.html).
 *
 * * The heading's three forms; the posts waiting for approval by their slot's day, today first.
 * * The pane shows the selected post exactly (PRD-251's approval view: the media, each
 *   channel's copy, the claims and sources) and approves the hash on screen; a 409 says it
 *   changed; unsourced claims still ask for the second confirmation.
 * * Send back to Auto requests changes with the comment, then asks for another take with it;
 *   Make another take asks for one without guidance.
 * * Approve all shown is offered only with series approval on; it approves each shown post by
 *   its hash (the series path for one series campaign) and reports one that changed.
 * * F256: a post with no channel publishes nothing: its Approve is disabled and says why, and
 *   Approve all shown leaves it out.
 * * PRD-251C US-C204: a plan's week is one section, in slot order, first; Approve the week
 *   approves its posts by their hashes (series switch off) and names one that changed; a
 *   viewer sees the week without the button.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'

const state = vi.hoisted(() => ({ seriesOn: false, role: 'owner' }))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({
    workspace: { id: 'w1', name: 'Acme', role: state.role, socials: { available: true, enabled: true, series_approval: state.seriesOn } },
  }),
}))
vi.mock('@/lib/api-client', () => {
  const apiClient = {
    getSocialPostMedia: vi.fn(async () => []),
    listSocialCampaigns: vi.fn(async () => ({ campaigns: [] })),
    approveSocialPost: vi.fn(),
    requestSocialPostChanges: vi.fn(async (id: string) => ({ id })),
    retakeSocialPost: vi.fn(async (id: string) => ({ id })),
    undoSocialPostRetake: vi.fn(async (id: string) => ({ id })),
    rejectSocialPost: vi.fn(),
    approveSocialCampaignSeries: vi.fn(),
    approveSocialPlanBatch: vi.fn(),
    listSocialPosts: vi.fn(async () => ({ posts: [], total: 0 })),
  }
  return { apiClient, default: apiClient }
})
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({ FilePreview: () => null, inferPreviewType: () => 'image' }))

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { SocialsQueue } from '@/components/deliverables/socials/studio/socials-queue'
import { queueHeading, timeLeft } from '@/components/deliverables/socials/studio/queue-model'
import { SOCIAL_POST_REVIEW_STALE_MESSAGE } from '@/hooks/use-socials-api'
import { NO_CHANNEL_HINT } from '@/components/deliverables/socials/socials-post-review'
import { UNDO_TAKE } from '@/components/deliverables/socials/studio/queue-pane'
import { RETAKE_STARTED } from '@/hooks/use-socials-queue'
import { renderWith } from './socials-editor-harness'

const api = apiClient as unknown as Record<string, ReturnType<typeof vi.fn>>
const select = vi.fn()

function waiting(id: string, slot: string | null, extra: Record<string, unknown> = {}) {
  return {
    id, title: `Post ${id}`, status: 'needs_approval', format: 'image', length_seconds: null, timezone: 'UTC',
    planned_for: slot, scheduled_for: null, content_hash: `hash-${id}`, approved_hash: null, media: {},
    copy: { base: `Copy of ${id}.` }, variables: {}, sources: {}, review_log: [], campaign_id: null,
    targets: [{ id: `t-${id}`, toolkit: 'twitter', post_kind: 'image', options: {}, status: 'pending' }],
    created_at: '2026-10-13T09:00:00Z', ...extra,
  } as any
}

const TODAY = [waiting('a', '2026-10-14T12:00:00Z'), waiting('b', '2026-10-14T18:00:00Z')]
const LATER = waiting('c', '2026-10-16T09:00:00Z')
const UNSLOTTED = waiting('d', null)

function renderQueue(posts: any[]) {
  return renderWith(<SocialsQueue role={state.role as any} posts={posts} selectedId={null} onSelect={select} />)
}

const pane = () => screen.getByRole('region', { name: 'Post to approve' })

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date('2026-10-14T07:20:00Z'))
  state.seriesOn = false
  state.role = 'owner'
  Object.values(api).forEach((fn) => fn.mockClear())
  api.approveSocialPost.mockImplementation(async (id: string) => ({ id, status: 'scheduled' }))
  vi.mocked(toast.success).mockClear()
  vi.mocked(toast.error).mockClear()
})
afterEach(() => { cleanup(); vi.useRealTimers() })

describe('the Queue', () => {
  it('says how many need you today, in three forms', () => {
    expect(queueHeading(2)).toBe('2 posts need you today')
    expect(queueHeading(1)).toBe('1 post needs you today')
    expect(queueHeading(0)).toBe('All caught up for today')
    expect(timeLeft('2026-10-14T12:00:00Z', new Date('2026-10-14T07:20:00Z'))).toBe('in 4h 40m')
  })

  it('lists the waiting posts by day, today first, then later, then no slot', () => {
    renderQueue([UNSLOTTED, LATER, ...TODAY])
    expect(screen.getByRole('heading', { name: '2 posts need you today' })).toBeInTheDocument()
    const list = screen.getByRole('complementary', { name: 'Waiting for approval' })
    expect(within(list).getAllByRole('heading').map((h) => h.textContent)).toEqual(['Wed 14 Oct', 'Fri 16 Oct', 'No slot'])
    expect(within(list).getAllByRole('button').map((b) => b.textContent?.match(/Post (\w)/)?.[1])).toEqual(['a', 'b', 'c', 'd'])
    expect(within(list).getAllByText('Needs you')).toHaveLength(4)
  })

  it('shows the selected post exactly and approves the hash on screen', async () => {
    renderQueue(TODAY)
    expect(within(pane()).getByText('Post a')).toBeInTheDocument()
    expect(within(pane()).getByText('12:00 · X · Image')).toBeInTheDocument()
    expect(within(pane()).getByText('Publishes 12:00 · in 4h 40m')).toBeInTheDocument()
    expect(within(pane()).getByText('Copy of a.')).toBeInTheDocument() // the channel copy (SocialsPostEvidence)
    expect(within(pane()).getByText(/You approve exactly this file and this copy/)).toBeInTheDocument()
    fireEvent.click(within(pane()).getByRole('button', { name: 'Approve · publishes 12:00' }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('a', 'hash-a'))
  })

  it('a post that changed while it was shown says so', async () => {
    api.approveSocialPost.mockRejectedValue(Object.assign(new Error('changed'), { status: 409 }))
    renderQueue(TODAY)
    fireEvent.click(within(pane()).getByRole('button', { name: 'Approve · publishes 12:00' }))
    expect(await within(pane()).findByText(SOCIAL_POST_REVIEW_STALE_MESSAGE)).toBeInTheDocument()
  })

  it('unsourced claims still ask for the second confirmation', () => {
    renderQueue([waiting('a', '2026-10-14T12:00:00Z', { variables: { members: { value: 1200, claim: true } } })])
    fireEvent.click(within(pane()).getByRole('button', { name: 'Approve · publishes 12:00' }))
    expect(within(pane()).getByRole('alertdialog', { name: 'Approve with unsourced claims' })).toBeInTheDocument()
    expect(api.approveSocialPost).not.toHaveBeenCalled()
  })

  it('Send back to Auto requests changes, then another take with the comment', async () => {
    renderQueue(TODAY)
    fireEvent.click(within(pane()).getByRole('button', { name: 'Request changes' }))
    fireEvent.change(within(pane()).getByLabelText('What should change?'), { target: { value: 'Use the Missions screenshot' } })
    expect(within(pane()).getByText('Auto redrafts it and it comes back here.')).toBeInTheDocument()
    fireEvent.click(within(pane()).getByRole('button', { name: 'Send back to Auto' }))
    await waitFor(() => expect(api.retakeSocialPost).toHaveBeenCalledWith('a', 'Use the Missions screenshot'))
    expect(api.requestSocialPostChanges).toHaveBeenCalledWith('a', 'Use the Missions screenshot')
  })

  it('Make another take asks for one without guidance', async () => {
    renderQueue(TODAY)
    fireEvent.click(within(pane()).getByRole('button', { name: /Make another take/ }))
    await waitFor(() => expect(api.retakeSocialPost).toHaveBeenCalledWith('a'))
  })

  it("F378: a retake's toast says what Auto still needs from the owner", async () => {
    api.retakeSocialPost.mockResolvedValueOnce({ id: 'a', take: { warnings: [], questions: ['Source line'] } })
    renderQueue(TODAY)
    fireEvent.click(within(pane()).getByRole('button', { name: /Make another take/ }))
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(`${RETAKE_STARTED} Auto needs: Source line`))
  })

  it('F378: a post Auto retook offers the earlier take back, until it is restored', async () => {
    const retook = { at: '2026-10-14T07:00:00Z', by: 'owner', action: 'retake', comment: null, previous: { copy: { base: 'Old.' } } }
    renderQueue([waiting('a', '2026-10-14T12:00:00Z', { review_log: [retook] })])
    fireEvent.click(within(pane()).getByRole('button', { name: UNDO_TAKE }))
    await waitFor(() => expect(api.undoSocialPostRetake).toHaveBeenCalledWith('a'))
    cleanup()
    const undone = { at: '2026-10-14T07:05:00Z', by: 'owner', action: 'retake_undone', comment: null, undid: retook.at }
    renderQueue([waiting('a', '2026-10-14T12:00:00Z', { review_log: [retook, undone] })])
    expect(within(pane()).queryByRole('button', { name: UNDO_TAKE })).toBeNull()
  })

  it('F256: a post with no channel cannot be approved, and says nothing would post', () => {
    renderQueue([waiting('a', '2026-10-14T12:00:00Z', { targets: [] })])
    expect(within(pane()).getByRole('button', { name: 'Approve' })).toBeDisabled()
    expect(within(pane()).queryByRole('button', { name: /publishes/ })).toBeNull()
    expect(within(pane()).getByText(NO_CHANNEL_HINT)).toBeInTheDocument()
    expect(within(pane()).getByText('No channels chosen yet.')).toBeInTheDocument()
  })

  it('F256: Approve all shown leaves out a post with no channel', async () => {
    state.seriesOn = true
    api.approveSocialPost.mockImplementation(async (id: string) => ({ id, status: 'scheduled' }))
    renderQueue([...TODAY, waiting('e', '2026-10-14T20:00:00Z', { targets: [] })])
    fireEvent.click(screen.getByRole('button', { name: 'Approve all shown' }))
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Approved 2 posts.'))
    expect(api.approveSocialPost.mock.calls.map(([id]) => id)).toEqual(['a', 'b'])
  })

  it('Approve all shown is absent while series approval is off', () => {
    renderQueue(TODAY)
    expect(screen.queryByRole('button', { name: 'Approve all shown' })).toBeNull()
  })

  it('with series approval on, approves each shown post by its hash and reports one that changed', async () => {
    state.seriesOn = true
    api.approveSocialPost.mockImplementation(async (id: string) => {
      if (id === 'b') throw Object.assign(new Error('changed'), { status: 409 })
      return { id, status: 'scheduled' }
    })
    renderQueue(TODAY)
    fireEvent.click(screen.getByRole('button', { name: 'Approve all shown' }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith('Not approved: Post b, because it changed since it was shown.'))
    expect(api.approveSocialPost).toHaveBeenCalledWith('a', 'hash-a')
    expect(api.approveSocialPost).toHaveBeenCalledWith('b', 'hash-b')
    expect(toast.success).toHaveBeenCalledWith('Approved 1 post.')
  })

  it('the shown posts of one series campaign go through the series path', async () => {
    state.seriesOn = true
    api.listSocialCampaigns.mockResolvedValue({ campaigns: [{ id: 'c-1', name: 'WebSummit countdown', approval_mode: 'series' }] })
    api.approveSocialCampaignSeries.mockResolvedValue({ campaign: {}, approved: [{}, {}], left: [] })
    renderQueue(TODAY.map((post) => ({ ...post, campaign_id: 'c-1' })))
    expect(await within(pane()).findByText('12:00 · X · Image · WebSummit countdown')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Approve all shown' }))
    await waitFor(() =>
      expect(api.approveSocialCampaignSeries).toHaveBeenCalledWith('c-1', [
        { post_id: 'a', content_hash: 'hash-a' }, { post_id: 'b', content_hash: 'hash-b' },
      ]),
    )
    expect(api.approveSocialPost).not.toHaveBeenCalled()
  })
})

describe("PRD-251C: the week's review", () => {
  const PLAN = { id: 'p1', name: 'Countdown', kind: 'plan', approval_mode: 'per_post', make: { rhythm: 'weekly', batch_day: 'sun' } }
  const week = (id: string, slot: string) => waiting(id, slot, { campaign_id: 'p1', batch_key: '2026-W43' })
  const WEEK = [week('w2', '2026-10-20T09:00:00Z'), week('w1', '2026-10-19T09:00:00Z')]
  const CHANGED = 'the post changed after you were shown it: review the current version'

  it("shows a plan's week first, in slot order, and approves it in one sitting", async () => {
    api.listSocialCampaigns.mockResolvedValueOnce({ campaigns: [PLAN] })
    api.approveSocialPlanBatch.mockResolvedValueOnce({ approved: [{ id: 'w1' }], left: [{ post_id: 'w2', title: 'Post w2', reason: 'changed', message: CHANGED }] })
    renderQueue([LATER, ...WEEK])
    const section = await screen.findByRole('region', { name: 'Week of 19 Oct · Countdown · 2 posts' })
    expect(within(section).getAllByText(/^Post w/).map((node) => node.textContent)).toEqual(['Post w1', 'Post w2'])
    const regions = screen.getAllByRole('region').map((region) => region.getAttribute('aria-label'))
    expect(regions.indexOf('Week of 19 Oct · Countdown · 2 posts')).toBeLessThan(regions.indexOf('Fri 16 Oct'))
    fireEvent.click(within(section).getByRole('button', { name: 'Approve the week' }))
    await waitFor(() => expect(api.approveSocialPlanBatch).toHaveBeenCalledWith('p1', '2026-W43', [
      { post_id: 'w1', content_hash: 'hash-w1' }, { post_id: 'w2', content_hash: 'hash-w2' },
    ]))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(`Not approved: Post w2, because ${CHANGED}.`))
    expect(toast.success).toHaveBeenCalledWith('Approved 1 post.')
  })

  it('a viewer sees the week without Approve the week', async () => {
    state.role = 'viewer'
    api.listSocialCampaigns.mockResolvedValueOnce({ campaigns: [PLAN] })
    renderQueue(WEEK)
    const section = await screen.findByRole('region', { name: 'Week of 19 Oct · Countdown · 2 posts' })
    expect(within(section).queryByRole('button', { name: 'Approve the week' })).toBeNull()
  })
})

