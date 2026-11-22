/**
 * PRD-251B US-B111 — the Queue (Queue.dc.html).
 *
 * * The heading's three forms; the posts waiting for approval by their slot's day, today first.
 * * The pane shows the selected post exactly (PRD-251's approval view: the media, each
 *   channel's copy, the claims and sources) and approves the hash on screen; a 409 says it
 *   changed; unsourced claims still ask for the second confirmation.
 * * One Approve (17 Nov 2026): its label says what the server does with the post, at the Queue's
 *   clock: "Approve · posts 12:00" for a planned slot still ahead, else a schedule kept from before an
 *   edit still ahead (scheduled there), "Approve & post now to X" for no slot and no schedule (published
 *   at once, on one click, nothing asked), plain "Approve" for a slot or a kept schedule that passed, or
 *   no channel; the pane's badge names the same time (`approvalSlot`). There is no
 *   separate Approve & post now, and the toast says what the approval did.
 * * Send back to Auto requests changes with the comment, then asks for another take with it;
 *   Make another take asks for one without guidance.
 * * Approve all shown is offered only with series approval on; it approves each shown post by
 *   its hash (the series path for one series campaign) and reports one that changed.
 * * F256: a post with no channel publishes nothing: its Approve is disabled and says why, and
 *   Approve all shown leaves it out.
 * * PRD-251C US-C204: a plan's week is one section, in slot order, first; Approve the week
 *   approves its posts by their hashes (series switch off) and names one that changed; a
 *   viewer sees the week without the button.
 * * P256-FIX-SOC-2: a plan's draft that waits for the owner's input is listed apart, "Waits for
 *   you": it opens its editor and offers no Approve; the Queue still lists only needs_approval.
 * * PRD-251D US-D108: a post whose file is older than its words (`file_current` false) shows the
 *   sentence the server refuses with in place of a working Approve, and Render again renders it.
 * * An approved post with no time (neither `planned_for` nor `scheduled_for`) posts nothing: it is
 *   listed apart, "Approved, no time yet", each row opening the post's page, and the nav's count
 *   adds it to what waits for approval.
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
    renderSocialPost: vi.fn(async (id: string) => ({ id, status: 'rendering' })),  // PRD-251D US-D108: Render again
  }
  return { apiClient, default: apiClient }
})
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({ FilePreview: () => null, inferPreviewType: () => 'image' }))

import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { SocialsQueue } from '@/components/deliverables/socials/studio/socials-queue'
import {
  APPROVE, APPROVE_AND_POST_NOW, APPROVE_POSTS_AT, approvalSlot, approveLabel, approvePostsNow,
  approvedWithNoTime, queueCount, queueHeading, queuedPosts, timeLeft, waitingForATime, waitingForInput,
} from '@/components/deliverables/socials/studio/queue-model'
import { APPROVED_NO_TIME, APPROVED_NO_TIME_HINT } from '@/components/deliverables/socials/studio/queue-waiting'
import { APPROVED_AND_PUBLISHING, SOCIAL_POST_REVIEW_STALE_MESSAGE } from '@/hooks/use-socials-api'
import { NO_CHANNEL_HINT, RENDER_AGAIN } from '@/components/deliverables/socials/socials-post-review'
import { FILE_OLDER_THAN_WORDS } from '@/components/deliverables/socials/socials-review'
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
/** Approve on a post slotted for 12:00 today, still ahead at the Queue's clock (07:20). */
const APPROVE_AT_NOON = `${APPROVE_POSTS_AT} 12:00`
const LATER = waiting('c', '2026-10-16T09:00:00Z')
const UNSLOTTED = waiting('d', null)

function renderQueue(posts: any[], onEdit?: (postId: string) => void) {
  return renderWith(<SocialsQueue role={state.role as any} posts={posts} selectedId={null} onSelect={select} onEdit={onEdit} />)
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

describe('one Approve says what it does', () => {
  const NOW = new Date('2026-10-14T07:20:00Z')
  const LINKEDIN = { id: 't-li', toolkit: 'linkedin', post_kind: 'image', options: {}, status: 'pending' }

  it('labels each case as the server decides it', () => {
    expect(APPROVE_AT_NOON).toBe('Approve · posts 12:00')
    expect(approveLabel(waiting('a', '2026-10-14T12:00:00Z'), NOW)).toBe(APPROVE_AT_NOON)
    expect(approveLabel(waiting('d', null), NOW)).toBe('Approve & post now to X')
    expect(approveLabel(waiting('d', null, { targets: [...UNSLOTTED.targets, LINKEDIN] }), NOW))
      .toBe(`${APPROVE_AND_POST_NOW} X, LinkedIn`)
    // A slot that passed is left approved for a person to reschedule: plain Approve.
    expect(approveLabel(waiting('e', '2026-10-14T06:00:00Z'), NOW)).toBe(APPROVE)
    expect(APPROVE).toBe('Approve')
    // A schedule kept from before an edit sent it back is approved into while still ahead, as a slot
    // is, and before a planned slot that passed; one that passed is plain Approve, as is no channel.
    const KEPT_AHEAD = waiting('f', null, { scheduled_for: '2026-10-15T09:00:00Z' })
    expect(approveLabel(KEPT_AHEAD, NOW)).toBe(`${APPROVE_POSTS_AT} 09:00`)
    expect(approvalSlot(KEPT_AHEAD, NOW)).toBe('2026-10-15T09:00:00Z')
    const PLANNED_PASSED_KEPT_AHEAD = waiting('h', '2026-10-14T06:00:00Z', { scheduled_for: '2026-10-14T15:30:00Z' })
    expect(approveLabel(PLANNED_PASSED_KEPT_AHEAD, NOW)).toBe(`${APPROVE_POSTS_AT} 15:30`)
    expect(approveLabel(waiting('a2', '2026-10-14T12:00:00Z', { scheduled_for: '2026-10-14T15:30:00Z' }), NOW))
      .toBe(APPROVE_AT_NOON)  // a planned slot still ahead comes first
    const KEPT_PASSED = waiting('k', null, { scheduled_for: '2026-10-14T06:00:00Z' })
    expect(approveLabel(KEPT_PASSED, NOW)).toBe(APPROVE)
    expect(approvalSlot(KEPT_PASSED, NOW)).toBeNull()
    expect(approveLabel(waiting('g', null, { targets: [] }), NOW)).toBe(APPROVE)
    // Only the post-now label sends post_now: a post with a slot or a kept schedule, or no channel, never does.
    expect(approvePostsNow(UNSLOTTED)).toBe(true)
    for (const other of [waiting('a', '2026-10-14T12:00:00Z'), waiting('e', '2026-10-14T06:00:00Z'), KEPT_AHEAD, KEPT_PASSED, waiting('g', null, { targets: [] })]) {
      expect(approvePostsNow(other), other.id).toBe(false)
    }
  })

  it('a post with no slot approves on one click, asks nothing more, and says it is publishing', async () => {
    api.approveSocialPost.mockImplementation(async (id: string) => ({ id, status: 'publishing' }))
    renderQueue([UNSLOTTED])
    fireEvent.click(within(pane()).getByRole('button', { name: 'Approve & post now to X' }))
    // The click says what its label said: post now. The server publishes only if it agrees.
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('d', 'hash-d', { postNow: true }))
    expect(within(pane()).queryByRole('alertdialog')).toBeNull()
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(APPROVED_AND_PUBLISHING))
  })

  it('a slot that passed shows a plain Approve, which approves it', async () => {
    renderQueue([waiting('e', '2026-10-14T06:00:00Z')])
    expect(within(pane()).queryByRole('button', { name: /· posts|post now/ })).toBeNull()
    fireEvent.click(within(pane()).getByRole('button', { name: APPROVE }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('e', 'hash-e'))
  })

  it('a schedule kept from before an edit, still ahead after its planned slot passed, is the time the button and the badge name', async () => {
    renderQueue([waiting('h', '2026-10-14T06:00:00Z', { scheduled_for: '2026-10-14T15:30:00Z' })])
    expect(within(pane()).getByText('Publishes 15:30 · in 8h 10m')).toBeInTheDocument()
    fireEvent.click(within(pane()).getByRole('button', { name: `${APPROVE_POSTS_AT} 15:30` }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('h', 'hash-h'))
  })

  it('there is no separate Approve & post now', () => {
    renderQueue(TODAY)
    expect(within(pane()).getAllByRole('button', { name: /^Approve/ }).map((b) => b.textContent)).toEqual([APPROVE_AT_NOON])
    expect(within(pane()).queryByRole('button', { name: 'Approve & post now' })).toBeNull()
  })
})

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
    fireEvent.click(within(pane()).getByRole('button', { name: APPROVE_AT_NOON }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('a', 'hash-a'))
  })

  it('a post that changed while it was shown says so', async () => {
    api.approveSocialPost.mockRejectedValue(Object.assign(new Error('changed'), { status: 409 }))
    renderQueue(TODAY)
    fireEvent.click(within(pane()).getByRole('button', { name: APPROVE_AT_NOON }))
    expect(await within(pane()).findByText(SOCIAL_POST_REVIEW_STALE_MESSAGE)).toBeInTheDocument()
  })

  it('unsourced claims still ask for the second confirmation', () => {
    renderQueue([waiting('a', '2026-10-14T12:00:00Z', { variables: { members: { value: 1200, claim: true } } })])
    fireEvent.click(within(pane()).getByRole('button', { name: APPROVE_AT_NOON }))
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
    expect(within(pane()).queryByRole('button', { name: /· posts|post now/ })).toBeNull()
    expect(within(pane()).getByText(NO_CHANNEL_HINT)).toBeInTheDocument()
    expect(within(pane()).getByText('No channels chosen yet.')).toBeInTheDocument()
  })

  it('US-D108: a post whose file is older than its words cannot be approved, says so, and Render again renders it', async () => {
    renderQueue([waiting('a', '2026-10-14T12:00:00Z', { media: { '1:1': ['d-square'] }, file_current: false })])
    expect(within(pane()).getByText(FILE_OLDER_THAN_WORDS)).toBeInTheDocument()
    expect(within(pane()).getByRole('button', { name: APPROVE_AT_NOON })).toBeDisabled()
    fireEvent.click(within(pane()).getByRole('button', { name: RENDER_AGAIN }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('a'))
    expect(api.approveSocialPost).not.toHaveBeenCalled()
  })

  it('US-D108: a post whose file still shows it keeps a working Approve and no sentence', () => {
    renderQueue([waiting('a', '2026-10-14T12:00:00Z', { media: { '1:1': ['d-square'] }, file_current: true })])
    expect(within(pane()).getByRole('button', { name: APPROVE_AT_NOON })).toBeEnabled()
    expect(within(pane()).queryByText(FILE_OLDER_THAN_WORDS)).toBeNull()
    expect(within(pane()).queryByRole('button', { name: RENDER_AGAIN })).toBeNull()
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

describe('P256-FIX-SOC-2: a planned post that waits for you', () => {
  const asked = {
    at: '2026-10-14T07:00:00Z', by: 'the plan', action: 'needs_input', comment: 'Fill in Figure 1 before rendering.',
    fields: ['Figure 1', 'Figure 1 meaning'],
  }
  const WAITS = waiting('w', '2026-10-15T09:00:00Z', { status: 'draft', review_log: [asked] })
  const PLAIN_DRAFT = waiting('p', null, { status: 'draft' })

  it('keeps the drafts whose last entry asks for input; the Queue still lists only needs_approval', () => {
    expect(waitingForInput([PLAIN_DRAFT, WAITS, ...TODAY]).map((post) => post.id)).toEqual(['w'])
    const rendered = { at: '2026-10-14T08:00:00Z', by: 'owner', action: 'render', comment: null }
    expect(waitingForInput([{ ...WAITS, review_log: [asked, rendered] }])).toEqual([])
    expect(queuedPosts([PLAIN_DRAFT, WAITS, ...TODAY]).map((post) => post.id)).toEqual(['a', 'b'])
  })

  it('a waiting row opens the editor and offers no Approve', () => {
    const edit = vi.fn()
    renderQueue([WAITS, ...TODAY], edit)
    const group = screen.getByRole('region', { name: 'Waits for you' })
    expect(within(group).getByText('Fill in: Figure 1, Figure 1 meaning')).toBeInTheDocument()
    expect(within(group).queryByRole('button', { name: /Approve/ })).toBeNull()
    const list = screen.getByRole('complementary', { name: 'Waiting for approval' })
    expect(within(list).queryByText('Post w')).toBeNull()
    expect(within(pane()).getByText('Post a')).toBeInTheDocument()
    fireEvent.click(within(group).getByRole('button', { name: /Post w/ }))
    expect(edit).toHaveBeenCalledWith('w')
  })
})

describe('an approved post with no time', () => {
  // Approved with no slot, the server leaves it approved "for a person to schedule": nothing posts it.
  const NO_TIME = waiting('n', null, { status: 'approved', approved_hash: 'hash-n' })
  const OLDER = waiting('o', null, { status: 'approved', created_at: '2026-10-01T09:00:00Z' })
  // Approved after its slot passed: it stays approved, but on the calendar's grid at that slot.
  const SLOT_PASSED = waiting('pl', '2026-10-13T09:00:00Z', { status: 'approved' })
  const SLOT_AHEAD = waiting('u', '2026-10-20T09:00:00Z', { status: 'approved' })
  const SCHEDULE_ONLY = waiting('so', null, { status: 'approved', scheduled_for: '2026-10-20T09:00:00Z' })
  const ON_SCHEDULE = waiting('s', null, { status: 'scheduled', scheduled_for: '2026-10-20T09:00:00Z' })

  it('counts a post approved with neither a planned slot nor a schedule, and nothing else', () => {
    expect(approvedWithNoTime(NO_TIME)).toBe(true)
    expect(approvedWithNoTime({ status: 'approved' } as any)).toBe(true)  // a post read without the fields
    for (const other of [SLOT_PASSED, SLOT_AHEAD, SCHEDULE_ONLY, ON_SCHEDULE, UNSLOTTED]) {
      expect(approvedWithNoTime(other), other.id).toBe(false)
    }
    expect(waitingForATime([NO_TIME, SLOT_PASSED, OLDER, ON_SCHEDULE, ...TODAY]).map((post) => post.id)).toEqual(['o', 'n'])
  })

  it("the nav's count adds them to the posts waiting for approval", () => {
    expect(queueCount([NO_TIME, OLDER, SLOT_PASSED, ON_SCHEDULE, UNSLOTTED, ...TODAY])).toBe(5)
    expect(queueCount([SLOT_AHEAD, ON_SCHEDULE])).toBe(0)
  })

  it('lists them apart, says they post nothing yet, and a row opens the post without approving anything', () => {
    const edit = vi.fn()
    renderQueue([NO_TIME, ...TODAY], edit)
    const group = screen.getByRole('region', { name: APPROVED_NO_TIME })
    expect(within(group).getByRole('heading', { name: APPROVED_NO_TIME })).toBeInTheDocument()
    expect(within(group).getByText(APPROVED_NO_TIME_HINT)).toBeInTheDocument()
    expect(within(group).queryByRole('button', { name: /Approve/ })).toBeNull()
    const list = screen.getByRole('complementary', { name: 'Waiting for approval' })
    expect(within(list).queryByText('Post n')).toBeNull()
    expect(within(pane()).getByText('Post a')).toBeInTheDocument()
    fireEvent.click(within(group).getByRole('button', { name: /Post n/ }))
    expect(edit).toHaveBeenCalledWith('n')
    expect(api.approveSocialPost).not.toHaveBeenCalled()
  })

  it('shows when nothing waits for approval, and is absent when no approved post lacks a time', () => {
    renderQueue([NO_TIME])
    expect(screen.getByRole('region', { name: APPROVED_NO_TIME })).toBeInTheDocument()
    expect(screen.getByText('Nothing is waiting for your approval.')).toBeInTheDocument()
    cleanup()
    renderQueue([SLOT_PASSED, SLOT_AHEAD, ...TODAY])
    expect(screen.queryByRole('region', { name: APPROVED_NO_TIME })).toBeNull()
  })
})
