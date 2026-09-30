/**
 * PRD-251 Wave 3, US-308 (S3.3f) — the publish controls and receipts in the post
 * view, against a mocked apiClient.
 *
 * - An approved post schedules with the browser's timezone, and publishes now
 *   after a confirmation naming its channels.
 * - A scheduled post shows its slot in its own timezone, reschedules and
 *   unschedules.
 * - A partially published post shows each channel's permalink and error, and Retry
 *   calls the retry route; a missed post offers Reschedule and Publish now.
 * - A 409 says the post changed and reloads it, as approve does.
 * - apiClient calls publish-now and retry with POST.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { readFileSync } from 'fs'
import path from 'path'

const api = vi.hoisted(() => ({
  scheduleSocialPost: vi.fn(),
  unscheduleSocialPost: vi.fn(),
  publishSocialPostNow: vi.fn(),
  retrySocialPost: vi.fn(),
  getSocialPostMedia: vi.fn(async () => []),
  listSocialPosts: vi.fn(async () => ({ posts: [], total: 0 })),
}))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api, default: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'w1', role: 'owner', socials: { available: true, enabled: true } } }),
}))
vi.mock('../socials-voice-picker', () => ({ SocialsVoicePicker: () => null }))

import { toast } from 'sonner'
import { SocialsPostDetail } from '@/components/deliverables/socials/socials-post-detail'
import { SOCIAL_POST_CHANGED_MESSAGE } from '@/hooks/use-socials-api'
import type { SocialPost, SocialPostTarget } from '@/lib/api-client'
import { slotTimeLabel } from '@/lib/social-time'

const HASH = 'a'.repeat(64)

function target(toolkit: string, over: Partial<SocialPostTarget> = {}): SocialPostTarget {
  return {
    id: `t-${toolkit}`, toolkit, post_kind: 'text', options: {}, status: 'pending', attempts: 0,
    remote_id: null, permalink: null, error: null, published_at: null, notes: [], ...over,
  }
}

function post(overrides: Partial<SocialPost> = {}): SocialPost {
  return {
    id: 'p1', workspace_id: 'w1', created_by: 'u1', title: 'Harvest Club', brief: null,
    copy: { base: 'Harvest Club opens Friday.' }, format: 'image', template_id: null, variables: {}, sources: {}, media: {},
    status: 'approved', content_hash: HASH, approved_hash: HASH, approved_by: 'u2', approved_at: '2026-09-29T09:00:00Z',
    override_unsourced: false, review_log: [], scheduled_for: null, timezone: null,
    targets: [target('linkedin'), target('twitter')], created_at: '2026-09-29T09:00:00Z', updated_at: '2026-09-29T09:00:00Z',
    ...overrides,
  }
}

let client: QueryClient

function renderDetail(p: SocialPost) {
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SocialsPostDetail post={p} role="owner" />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  for (const fn of [api.scheduleSocialPost, api.unscheduleSocialPost, api.publishSocialPostNow, api.retrySocialPost]) {
    fn.mockReset().mockResolvedValue(post())
  }
  vi.mocked(toast.error).mockReset()
})
afterEach(() => cleanup())

describe('an approved post', () => {
  it('schedules with the browser\'s timezone', async () => {
    renderDetail(post())
    const publishing = screen.getByRole('region', { name: 'Publishing' })
    const browserTz = Intl.DateTimeFormat().resolvedOptions().timeZone
    fireEvent.change(within(publishing).getByLabelText(/Schedule: date and time/), { target: { value: '2026-11-09T09:30' } })
    fireEvent.click(within(publishing).getByRole('button', { name: 'Schedule' }))
    await waitFor(() => expect(api.scheduleSocialPost).toHaveBeenCalledTimes(1))
    expect(api.scheduleSocialPost).toHaveBeenCalledWith('p1', new Date('2026-11-09T09:30').toISOString(), browserTz)
  })

  it('publishes now only after a confirmation that names its channels', async () => {
    renderDetail(post())
    fireEvent.click(screen.getByRole('button', { name: 'Publish now' }))
    const confirm = screen.getByRole('alertdialog', { name: 'Publish now' })
    expect(confirm.textContent).toContain('LinkedIn, X')
    expect(api.publishSocialPostNow).not.toHaveBeenCalled()
    fireEvent.click(within(confirm).getByRole('button', { name: 'Publish' }))
    await waitFor(() => expect(api.publishSocialPostNow).toHaveBeenCalledWith('p1'))
  })

  it('shows the privacy level and AI label a TikTok target will publish with', () => {
    renderDetail(post({
      targets: [target('tiktok', { post_kind: 'video', options: { privacy_level: 'FOLLOWER_OF_CREATOR' } })],
      footage: { hook: { prompt: 'A harvest table', status: 'done' } } as SocialPost['footage'],
    }))
    const receipt = screen.getByTestId('socials-receipt-tiktok-video')
    // Publish uses the choice only when the account allows it (US-304's choice form).
    expect(receipt.textContent).toContain(
      'Privacy: FOLLOWER_OF_CREATOR if the account allows it, else the most private level the account allows.',
    )
    expect(receipt.textContent).toContain('AI-generated label: on')
  })
})

describe('a scheduled post', () => {
  const scheduled = post({ status: 'scheduled', scheduled_for: '2026-11-09T09:00:00Z', timezone: 'Asia/Tokyo' })

  it('shows its slot in its own timezone and unschedules', async () => {
    renderDetail(scheduled)
    expect(screen.getByTestId('socials-post-slot').textContent).toContain(slotTimeLabel(scheduled.scheduled_for, 'Asia/Tokyo'))
    expect(screen.getByTestId('socials-post-slot').textContent).toContain('18:00 GMT+9')
    fireEvent.click(screen.getByRole('button', { name: 'Unschedule' }))
    await waitFor(() => expect(api.unscheduleSocialPost).toHaveBeenCalledWith('p1'))
  })

  it('reschedules through the schedule route', async () => {
    renderDetail(scheduled)
    fireEvent.change(screen.getByLabelText(/Reschedule: date and time/), { target: { value: '2026-11-10T08:00' } })
    fireEvent.click(screen.getByRole('button', { name: 'Reschedule' }))
    await waitFor(() => expect(api.scheduleSocialPost).toHaveBeenCalledTimes(1))
    expect(api.scheduleSocialPost.mock.calls[0][1]).toBe(new Date('2026-11-10T08:00').toISOString())
  })
})

describe('receipts', () => {
  it('a partially published post shows each channel\'s permalink and error, and Retry calls the retry route', async () => {
    const partial = post({
      status: 'partially_published',
      targets: [
        target('linkedin', { status: 'published', remote_id: 'urn:li:share:7', permalink: 'https://www.linkedin.com/feed/update/urn:li:share:7/' }),
        target('twitter', { status: 'failed', attempts: 3, error: 'TWITTER_CREATION_OF_A_POST: 429 Too Many Requests' }),
      ],
    })
    renderDetail(partial)
    const link = within(screen.getByTestId('socials-receipt-linkedin-text')).getByRole('link', { name: 'View the post' })
    expect(link.getAttribute('href')).toBe('https://www.linkedin.com/feed/update/urn:li:share:7/')
    expect(link.getAttribute('target')).toBe('_blank')
    expect(screen.getByTestId('socials-receipt-twitter-text').textContent).toContain('429 Too Many Requests')
    fireEvent.click(screen.getByRole('button', { name: 'Retry the failed channels' }))
    await waitFor(() => expect(api.retrySocialPost).toHaveBeenCalledWith('p1'))
  })

  it('a published post offers no retry, and a skipped step is noted', () => {
    renderDetail(post({
      status: 'published',
      targets: [target('youtube', { post_kind: 'video', status: 'published', permalink: 'https://www.youtube.com/watch?v=x', notes: ['Skipped: YOUTUBE_UPDATE_THUMBNAIL: Needs public storage'] })],
    }))
    expect(screen.queryByRole('button', { name: 'Retry the failed channels' })).toBeNull()
    expect(screen.getByText(/Needs public storage/)).toBeTruthy()
  })
})

describe('a missed post', () => {
  it('says it missed its slot and offers Reschedule and Publish now', async () => {
    renderDetail(post({ status: 'missed', scheduled_for: '2026-11-09T09:00:00Z', timezone: 'Europe/Lisbon' }))
    expect(screen.getByRole('alert').textContent).toContain('Missed its slot')
    expect(screen.getByRole('button', { name: 'Reschedule' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Publish now' }))
    fireEvent.click(within(screen.getByRole('alertdialog', { name: 'Publish now' })).getByRole('button', { name: 'Publish' }))
    await waitFor(() => expect(api.publishSocialPostNow).toHaveBeenCalledWith('p1'))
  })
})

describe('a 409', () => {
  it('says the post changed and reloads it', async () => {
    api.publishSocialPostNow.mockRejectedValue(Object.assign(new Error('the post changed'), { status: 409 }))
    renderDetail(post())
    const invalidate = vi.spyOn(client, 'invalidateQueries')
    fireEvent.click(screen.getByRole('button', { name: 'Publish now' }))
    fireEvent.click(within(screen.getByRole('alertdialog', { name: 'Publish now' })).getByRole('button', { name: 'Publish' }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(SOCIAL_POST_CHANGED_MESSAGE))
    expect(invalidate).toHaveBeenCalled()
  })
})

describe('apiClient', () => {
  it('calls publish-now and retry with POST', () => {
    const src = readFileSync(path.resolve(__dirname, '../../../../lib/api-client.ts'), 'utf8')
    expect(src).toContain("`/api/socials/posts/${postId}/publish-now`, { method: 'POST' }")
    expect(src).toContain("`/api/socials/posts/${postId}/retry`, { method: 'POST' }")
    expect(src).toMatch(/`\/api\/socials\/posts\/\$\{postId\}\/schedule`, \{\s*method: 'POST'/)
  })
})
