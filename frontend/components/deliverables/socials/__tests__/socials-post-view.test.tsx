/**
 * PRD-251D US-D203 — one post page for every status and every role. The post page shows the
 * post's read-only view (studio/socials-post-view.tsx) to a role that cannot author, in every
 * status, and to an author for a post the editor does not hold (rendering, publishing,
 * published, partly published, archived); the Queue shows it to a person who may not approve.
 *
 * * It shows what the older detail pane showed: the status, the brief, the copy, each channel's
 *   copy, the receipts, the history, the render lines, an approval an edit reset, the voice.
 * * It offers nothing the role may not do: a viewer gets no button at all; an author gets Retry
 *   on a partly published post's failed channels, and nothing else in these statuses.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { format } from 'date-fns'

const api = vi.hoisted(() => ({
  getSocialPostMedia: vi.fn(),
  getSocialVoiceSources: vi.fn(),
  retrySocialPost: vi.fn(),
  renderSocialPost: vi.fn(),
  listSocialCampaigns: vi.fn(),
  listSocialPosts: vi.fn(),
  get: vi.fn(),
}))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api, default: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'viewer', socials: { available: true, enabled: true, series_approval: false } } }),
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  FilePreview: ({ url }: { url: string }) => <div data-testid="file-preview">{url}</div>,
  inferPreviewType: () => 'image',
}))

import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost, SocialPostTarget, SocialReviewEntry } from '@/lib/api-client'
import { SOCIAL_STATUS_LABELS, SOCIAL_STATUS_ORDER } from '@/components/deliverables/socials/socials-status'
import { RENDERING_NOTE, RENDERS_AS_SAVED } from '@/components/deliverables/socials/studio/post-notices'
import { SocialsEditorActivity } from '@/components/deliverables/socials/studio/socials-editor-activity'
import { SocialsPostPage } from '@/components/deliverables/socials/studio/socials-post-page'
import { READ_ONLY_ROLE } from '@/components/deliverables/socials/studio/socials-post-view'
import { SocialsQueue } from '@/components/deliverables/socials/studio/socials-queue'

const go = vi.fn()
const STARTED = '2026-10-14T16:22:00+00:00'
const GIVES_UP = '2026-10-14T16:47:00+00:00'

function target(toolkit: string, over: Partial<SocialPostTarget> = {}): SocialPostTarget {
  return {
    id: `t-${toolkit}`, toolkit, post_kind: 'text', options: {}, status: 'pending', attempts: 0,
    remote_id: null, permalink: null, error: null, published_at: null, notes: [], ...over,
  }
}

function post(over: Partial<SocialPost> = {}): SocialPost {
  return {
    id: 'p1', workspace_id: 'w1', created_by: 'u1', title: 'Harvest Club', brief: 'Open day on Friday',
    copy: { base: 'Harvest Club opens Friday.', channels: { twitter: 'Friday. Harvest Club.' } }, format: 'image',
    template_id: null, variables: {}, sources: {}, media: {}, status: 'draft', content_hash: 'h1',
    approved_hash: null, approved_by: null, approved_at: null, override_unsourced: false, review_log: [],
    scheduled_for: null, timezone: null, planned_for: null, length_seconds: null, targets: [target('linkedin')],
    created_at: '2026-10-14T07:00:00Z', updated_at: '2026-10-14T07:00:00Z', ...over,
  } as SocialPost
}

const entry = (action: string, comment: string | null = null, extra: Partial<SocialReviewEntry> = {}): SocialReviewEntry => ({
  at: '2026-10-14T08:00:00Z', by: 'u1', action, comment, ...extra,
})

function withClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

function renderPage(p: SocialPost, role: Workspace['role']) {
  return withClient(<SocialsPostPage role={role} posts={[p]} postId={p.id} loading={false} go={go} />)
}

const view = () => screen.getByRole('article', { name: 'Post: Harvest Club' })
const historyOf = (card: HTMLElement) => within(within(card).getByRole('region', { name: 'History' }))

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getSocialPostMedia.mockResolvedValue([])
  api.getSocialVoiceSources.mockResolvedValue({
    sources: [{ toolkit: 'kokoro', label: 'Kokoro (built in)', status: 'available', builtin: true, lists_voices: false }],
    problem: null,
  })
  api.retrySocialPost.mockImplementation(async (id: string) => post({ id, status: 'publishing' }))
  api.renderSocialPost.mockImplementation(async (id: string) => post({ id, status: 'rendering' }))
  api.listSocialCampaigns.mockResolvedValue({ campaigns: [] })
  api.listSocialPosts.mockResolvedValue({ posts: [], total: 0 })
  api.get.mockResolvedValue({ targets: [] })
  go.mockReset()
})
afterEach(cleanup)

describe('a role that cannot author', () => {
  it('gets the read-only view in every status, with the post as it is and no button at all', () => {
    for (const status of SOCIAL_STATUS_ORDER) {
      // A partly published post with a failed channel: Retry is an author's.
      const targets = status === 'partially_published'
        ? [target('linkedin', { status: 'published' }), target('twitter', { status: 'failed', error: '429 Too Many Requests' })]
        : [target('linkedin')]
      renderPage(post({ status, targets }), 'viewer')
      const card = view()
      expect(within(card).getByTestId('socials-post-status')).toHaveTextContent(SOCIAL_STATUS_LABELS[status])
      expect(within(card).getByText(READ_ONLY_ROLE)).toBeInTheDocument()
      expect(within(card).getByText('Open day on Friday')).toBeInTheDocument()
      expect(within(within(card).getByRole('region', { name: 'Copy' })).getByText('Harvest Club opens Friday.')).toBeInTheDocument()
      expect(within(within(card).getByRole('region', { name: 'Channels' })).getByText('LinkedIn')).toBeInTheDocument()
      expect(within(card).queryAllByRole('button'), status).toHaveLength(0)
      expect(screen.queryByRole('textbox')).toBeNull()
      cleanup()
    }
  })

  it('a member reads the same way', () => {
    renderPage(post({ status: 'approved' }), 'member')
    expect(within(view()).getByText(READ_ONLY_ROLE)).toBeInTheDocument()
    expect(within(view()).queryAllByRole('button')).toHaveLength(0)
  })

  it('sees why the last render failed, what a waiting draft waits for, and an approval an edit reset', () => {
    renderPage(post({ status: 'failed', review_log: [entry('render'), entry('render_failed', 'The composition failed its check.')] }), 'viewer')
    expect(within(view()).getByRole('alert')).toHaveTextContent('The last render failed: The composition failed its check.')
    cleanup()

    renderPage(post({ review_log: [entry('needs_input', 'Fill in Figure 1 before rendering.', { fields: ['Figure 1'] })] }), 'viewer')
    expect(within(view()).getByRole('alert')).toHaveTextContent('Waiting for you: Fill in: Figure 1')
    cleanup()

    renderPage(post({ status: 'needs_approval', review_log: [entry('approve'), entry('approval_voided', 'The content changed after approval.')] }), 'viewer')
    expect(within(view()).getByText('Approval reset: content changed')).toBeInTheDocument()
    expect(historyOf(view()).getByText('Approval voided by an edit')).toBeInTheDocument()
  })

  it('reads the voice of a post that speaks a script, with no choice to make', async () => {
    renderPage(post({ format: 'video', template_id: 'tpl-video' }), 'viewer')
    expect(await within(view()).findByText('Voice: Kokoro (built in)')).toBeInTheDocument()
    expect(screen.queryByRole('radio')).toBeNull()
  })

  it('in the Queue, one who may not approve reads the post there, with no approver action', () => {
    for (const role of ['viewer', 'member'] as const) {
      withClient(<SocialsQueue role={role} posts={[post({ status: 'needs_approval' })]} selectedId={null} onSelect={vi.fn()} onEdit={vi.fn()} />)
      const card = view()
      expect(screen.queryByRole('region', { name: 'Post to approve' })).toBeNull()
      expect(within(card).getByText(READ_ONLY_ROLE)).toBeInTheDocument()
      for (const name of ['Approve', 'Approve & post now', 'Request changes', 'Reject', 'Make another take', 'Edit post']) {
        expect(screen.queryByRole('button', { name })).toBeNull()
      }
      expect(within(card).queryAllByRole('button')).toHaveLength(0)
      cleanup()
    }
  })
})

describe('an author, for a post the editor does not hold', () => {
  it('a rendering post says when its render started and when it gives up, and offers nothing', () => {
    const render_: SocialPost['render'] = { state: 'rendering', started_at: STARTED, gives_up_at: GIVES_UP, finished_at: null, why: null }
    renderPage(post({ status: 'rendering', render: render_ }), 'owner')
    const card = view()
    expect(within(card).getByRole('status')).toHaveTextContent(
      `Rendering. Started ${format(new Date(STARTED), 'HH:mm')}; it gives up at ${format(new Date(GIVES_UP), 'HH:mm')} if it has not finished.`,
    )
    expect(within(card).queryAllByRole('button')).toHaveLength(0)
    expect(within(card).queryByText(READ_ONLY_ROLE)).toBeNull()
    expect(screen.queryByRole('textbox')).toBeNull()
    cleanup()

    // A render the server gives no start for says so in general words.
    renderPage(post({ status: 'rendering' }), 'owner')
    expect(within(view()).getByRole('status')).toHaveTextContent(RENDERING_NOTE)
  })

  it('a publishing post shows each channel publishing, and offers nothing', () => {
    renderPage(post({ status: 'publishing', targets: [target('linkedin', { status: 'uploading' })] }), 'owner')
    const card = view()
    expect(within(screen.getByTestId('socials-receipt-linkedin-text')).getByText('Publishing…')).toBeInTheDocument()
    expect(within(card).queryAllByRole('button')).toHaveLength(0)
  })

  it('a published post links to each live post and shows its history, with no Retry', () => {
    renderPage(post({
      status: 'published',
      targets: [target('linkedin', { status: 'published', permalink: 'https://www.linkedin.com/feed/update/urn:li:share:7/' })],
      review_log: [entry('approve'), entry('published')],
    }), 'owner')
    const card = view()
    const link = within(screen.getByTestId('socials-receipt-linkedin-text')).getByRole('link', { name: 'View the post' })
    expect(link.getAttribute('href')).toBe('https://www.linkedin.com/feed/update/urn:li:share:7/')
    expect(historyOf(card).getByText('Published')).toBeInTheDocument()
    expect(within(card).queryAllByRole('button')).toHaveLength(0)
  })

  it('a partly published post shows the failed channel and Retry, which retries it', async () => {
    renderPage(post({
      status: 'partially_published',
      copy: { base: 'Harvest Club opens Friday.', channels: { twitter: 'Friday. Harvest Club.' } },
      targets: [target('linkedin', { status: 'published' }), target('twitter', { status: 'failed', error: '429 Too Many Requests' })],
    }), 'owner')
    const card = view()
    expect(screen.getByTestId('socials-receipt-twitter-text')).toHaveTextContent('429 Too Many Requests')
    // Each channel's copy, as it publishes: X its own, LinkedIn the base.
    expect(within(within(card).getByRole('region', { name: 'Channels' })).getByText('Friday. Harvest Club.')).toBeInTheDocument()
    expect(within(card).getAllByRole('button').map((b) => b.textContent)).toEqual(['Retry the failed channels'])
    fireEvent.click(within(card).getByRole('button', { name: 'Retry the failed channels' }))
    await waitFor(() => expect(api.retrySocialPost).toHaveBeenCalledWith('p1'))
  })

  it('an archived post shows why it was rejected, and offers nothing', () => {
    renderPage(post({ status: 'archived', review_log: [entry('submit'), entry('reject', 'Not our voice')] }), 'owner')
    const card = view()
    expect(within(card).getByTestId('socials-post-status')).toHaveTextContent('Archived')
    expect(historyOf(card).getByText('Rejected')).toBeInTheDocument()
    expect(historyOf(card).getByText('Not our voice')).toBeInTheDocument()
    expect(within(card).queryAllByRole('button')).toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Back to calendar' })).toBeInTheDocument()
  })
})

describe("the Activity card's full render, the older pane's Render", () => {
  // The editor's header offers no full render for a post that failed or waits for approval,
  // and its Render preview of a video is the preview: the Activity card keeps the older pane's.
  const activity = (p: SocialPost, role: Workspace['role'] = 'owner') => withClient(<SocialsEditorActivity post={p} role={role} />)

  it('a post with a template that waits for approval renders with POST /render, as it is saved', async () => {
    activity(post({ status: 'needs_approval', template_id: 'tpl-video', format: 'video' }))
    expect(screen.getByText(RENDERS_AS_SAVED)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Render' }))
    await waitFor(() => expect(api.renderSocialPost).toHaveBeenCalledWith('p1'))
  })

  it('a failed one, or one with its file, says Render again', () => {
    activity(post({ status: 'failed', template_id: 'tpl-video', format: 'video' }))
    expect(screen.getByRole('button', { name: 'Render again' })).toBeInTheDocument()
    cleanup()
    activity(post({ status: 'needs_approval', template_id: 'tpl-video', format: 'video', media: { '9:16': ['d-1'] } }))
    expect(screen.getByRole('button', { name: 'Render again' })).toBeInTheDocument()
  })

  it('is not offered to a viewer, for a post with no template, or where Submit for approval renders', () => {
    const cases: Array<[SocialPost, Workspace['role']]> = [
      [post({ status: 'failed', template_id: 'tpl-video' }), 'viewer'],
      [post({ status: 'failed', template_id: null }), 'owner'],
      [post({ status: 'draft', template_id: 'tpl-video' }), 'owner'],
      [post({ status: 'changes_requested', template_id: 'tpl-video' }), 'owner'],
      [post({ status: 'approved', template_id: 'tpl-video' }), 'owner'],
      // Its file is older than its words: the approve controls say so and hold the one Render again.
      [post({ status: 'needs_approval', template_id: 'tpl-video', file_current: false }), 'owner'],
    ]
    for (const [p, role] of cases) {
      activity(p, role)
      expect(screen.queryByRole('button', { name: /^Render/ })).toBeNull()
      cleanup()
    }
  })
})
