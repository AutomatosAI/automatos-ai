/**
 * PRD-251 S2.3 (US-206) — the approval view, against a mocked apiClient.
 *
 * - Approve sends the content_hash of the version shown. A 409 says the post
 *   changed while it was reviewed and reloads it.
 * - An approval with unsourced claims asks a second confirmation naming them,
 *   and only that sends override_unsourced. A 422 naming claims asks the same.
 * - When the newest history entry voided an approval, a banner says so.
 * - The post's media renders through the shared FilePreview, from the presigned links.
 * - Each channel shows its copy, and each claim its source or a red Unsourced chip.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const api = vi.hoisted(() => ({
  approveSocialPost: vi.fn(),
  getSocialPostMedia: vi.fn(),
  listSocialPosts: vi.fn(async () => ({ posts: [], total: 0 })),
  submitSocialPost: vi.fn(),
  requestSocialPostChanges: vi.fn(),
  rejectSocialPost: vi.fn(),
}))
const previews = vi.hoisted(() => [] as Array<{ url?: string; previewType?: string; filename?: string }>)

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api, default: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/components/widgets/FileWidget/FilePreview')>()
  return {
    ...actual,
    FilePreview: (props: { url?: string; previewType?: string; filename?: string }) => {
      previews.push(props)
      return <div data-testid="file-preview">{props.url}</div>
    },
  }
})
vi.mock('../socials-voice-picker', () => ({ SocialsVoicePicker: () => null }))

import { toast } from 'sonner'
import { SocialsPostDetail } from '@/components/deliverables/socials/socials-post-detail'
import { SOCIAL_POST_REVIEW_STALE_MESSAGE } from '@/hooks/use-socials-api'
import type { SocialPost } from '@/lib/api-client'

const HASH = 'a'.repeat(64)

function post(overrides: Partial<SocialPost> = {}): SocialPost {
  return {
    id: 'p1', workspace_id: 'w1', created_by: 'u1', title: 'Harvest Club', brief: null,
    copy: { base: 'Harvest Club opens Friday.', channels: { x: 'Friday. Harvest Club.' } },
    format: 'image', template_id: null, variables: {}, sources: {}, media: {},
    status: 'needs_approval', content_hash: HASH, approved_hash: null, approved_by: null, approved_at: null,
    override_unsourced: false, review_log: [], scheduled_for: null, timezone: null,
    // F256: a post with no channel cannot be approved, so the post under review goes to X.
    targets: [target('x', 'text')], created_at: '2026-09-29T09:00:00Z', updated_at: '2026-09-29T09:00:00Z',
    ...overrides,
  }
}

function target(toolkit: string, post_kind: 'text' | 'image') {
  return { id: `t-${toolkit}`, toolkit, post_kind, options: {}, status: 'pending' as const, attempts: 0,
    remote_id: null, permalink: null, error: null, published_at: null }
}

let client: QueryClient
function show(p: SocialPost) {
  return render(
    <QueryClientProvider client={client}>
      <SocialsPostDetail post={p} role="owner" />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  previews.length = 0
  Object.values(api).forEach((fn) => fn.mockReset())
  api.listSocialPosts.mockResolvedValue({ posts: [], total: 0 })
  api.approveSocialPost.mockImplementation(async () => post({ status: 'approved', approved_hash: HASH }))
})
afterEach(() => { cleanup(); vi.clearAllMocks() })

describe('approving (D6)', () => {
  it('sends the content_hash of the version shown, and nothing else', async () => {
    show(post())
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('p1', HASH))
    expect(toast.success).toHaveBeenCalledWith('Approved')
  })

  it('a 409 says the post changed while it was reviewed and reloads the posts', async () => {
    api.approveSocialPost.mockRejectedValueOnce(Object.assign(new Error('changed'), { status: 409 }))
    show(post())
    const reload = vi.spyOn(client, 'invalidateQueries')
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'This post changed while you were reviewing it — review the new version',
    )
    expect(toast.error).toHaveBeenCalledWith(SOCIAL_POST_REVIEW_STALE_MESSAGE)
    await waitFor(() => expect(reload).toHaveBeenCalledWith({ queryKey: ['socials', 'w1'] }))
    expect(toast.success).not.toHaveBeenCalled()
  })
})

describe('unsourced claims (D7)', () => {
  const claimed = {
    variables: { members: { value: '1,200', claim: true }, growth: { value: '40%', claim: true }, headline: { value: 'Hi' } },
    sources: { growth: { kind: 'report', ref: 'r-9', as_of: '2026-09-01' } },
  }

  it('shows each claim with its source, and a red Unsourced chip on one without', () => {
    show(post(claimed as Partial<SocialPost>))
    expect(screen.getByTestId('socials-claim-growth')).toHaveTextContent('report r-9, as of 2026-09-01')
    expect(within(screen.getByTestId('socials-claim-members')).getByText('Unsourced')).toHaveClass('bg-destructive')
    expect(screen.queryByTestId('socials-claim-headline')).toBeNull()
  })

  it('approving needs a second confirmation naming them, which sends override_unsourced', async () => {
    show(post(claimed as Partial<SocialPost>))
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))
    const confirm = screen.getByRole('alertdialog', { name: 'Approve with unsourced claims' })
    expect(confirm).toHaveTextContent('These claims have no source: members')
    expect(api.approveSocialPost).not.toHaveBeenCalled()

    fireEvent.click(within(confirm).getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('alertdialog')).toBeNull()
    expect(api.approveSocialPost).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))
    fireEvent.click(screen.getByRole('button', { name: 'Approve with unsourced claims' }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenCalledWith('p1', HASH, { overrideUnsourced: true }))
  })

  it('a 422 naming claims (a source gone since) asks the same confirmation', async () => {
    api.approveSocialPost.mockRejectedValueOnce(
      Object.assign(new Error(JSON.stringify({ message: 'no source', claims: ['growth'], unresolved: { growth: 'gone' } })), { status: 422 }),
    )
    show(post())
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))
    const confirm = await screen.findByRole('alertdialog', { name: 'Approve with unsourced claims' })
    expect(confirm).toHaveTextContent('These claims have no source: growth')
    fireEvent.click(within(confirm).getByRole('button', { name: 'Approve with unsourced claims' }))
    await waitFor(() => expect(api.approveSocialPost).toHaveBeenLastCalledWith('p1', HASH, { overrideUnsourced: true }))
  })
})

describe('what the approver sees', () => {
  it('a banner reads "Approval reset: content changed" while the newest entry voided an approval', () => {
    const voided = post({ review_log: [
      { at: '2026-09-29T09:00:00Z', by: 'u1', action: 'approve', comment: null },
      { at: '2026-09-29T09:05:00Z', by: 'u1', action: 'approval_voided', comment: 'The content changed after approval.' },
    ] })
    show(voided)
    expect(screen.getByText('Approval reset: content changed')).toBeInTheDocument()
    cleanup()
    show(post({ review_log: [...voided.review_log, { at: '2026-09-29T09:06:00Z', by: 'u1', action: 'submit', comment: null }] }))
    expect(screen.queryByText('Approval reset: content changed')).toBeNull()
  })

  it('the media renders through FilePreview from the presigned links', async () => {
    api.getSocialPostMedia.mockResolvedValue([
      { aspect: '9:16', deliverable_id: 'd1', name: 'video-9x16.mp4', url: 'https://minio.test/v.mp4?sig=1',
        content_type: 'video/mp4', bytes: 10, error: null },
      { aspect: '1:1', deliverable_id: 'd2', name: 'image-1x1.png', url: 'https://minio.test/i.png?sig=2',
        content_type: 'image/png', bytes: 10, error: null },
    ])
    show(post({ media: { '9:16': [{}], '1:1': [{}] } }))
    await waitFor(() => expect(screen.getAllByTestId('file-preview')).toHaveLength(2))
    expect(api.getSocialPostMedia).toHaveBeenCalledWith('p1')
    expect(previews.map((p) => [p.url, p.previewType])).toEqual(
      expect.arrayContaining([
        ['https://minio.test/v.mp4?sig=1', 'video'],
        ['https://minio.test/i.png?sig=2', 'image'],
      ]),
    )
    expect(document.querySelector('img, video')).toBeNull()
  })

  it('a post with no media asks for none', () => {
    show(post())
    expect(api.getSocialPostMedia).not.toHaveBeenCalled()
  })

  it('each target channel shows the copy it publishes: its own, or the base', () => {
    show(post({ targets: [target('x', 'text'), target('linkedin', 'text'), target('linkedin', 'image')] }))
    const channels = screen.getByRole('region', { name: 'Channels' })
    expect(within(channels).getByText('Friday. Harvest Club.')).toBeInTheDocument()
    expect(within(channels).getByText('Harvest Club opens Friday.')).toBeInTheDocument()
    expect(within(channels).getByText(/text, image/)).toBeInTheDocument()
  })

  it('names each channel as the rest of Socials does: X, not Twitter', () => {
    show(post({ targets: [target('twitter', 'image'), target('linkedin', 'text')] }))
    const channels = screen.getByRole('region', { name: 'Channels' })
    expect(within(channels).getByText('X')).toBeInTheDocument()
    expect(within(channels).getByText('LinkedIn')).toBeInTheDocument()
    expect(within(channels).queryByText(/twitter/i)).toBeNull()
  })

  it('request changes needs a comment; reject sends its optional reason', async () => {
    api.rejectSocialPost.mockResolvedValue(post({ status: 'archived' }))
    show(post())
    fireEvent.click(screen.getByRole('button', { name: 'Request changes' }))
    expect(screen.getByRole('button', { name: 'Send request' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Reject' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reject post' }))
    await waitFor(() => expect(api.rejectSocialPost).toHaveBeenCalledWith('p1', undefined))
  })
})
