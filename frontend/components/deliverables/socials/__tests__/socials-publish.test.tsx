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
 * - PRD-264 EUAI-05: every channel's receipt states its AI label decision, in the
 *   backend's words (GET /ai-labels): the receipt never decides it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { readFileSync } from 'fs'
import path from 'path'
import { format } from 'date-fns'

const api = vi.hoisted(() => ({
  scheduleSocialPost: vi.fn(),
  unscheduleSocialPost: vi.fn(),
  publishSocialPostNow: vi.fn(),
  retrySocialPost: vi.fn(),
  getSocialPostMedia: vi.fn(async () => []),
  listSocialPosts: vi.fn(async () => ({ posts: [], total: 0 })),
  get: vi.fn(),
}))
const NO_AI_MEDIA = { carries: false, footage: [], stills: [], voice: null, own: [], recorded: false }

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
  api.get.mockReset().mockResolvedValue({ ai_media: NO_AI_MEDIA, targets: [] })
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

  it('shows the privacy level and the AI label a TikTok target will publish with', async () => {
    api.get.mockResolvedValueOnce({
      ai_media: { ...NO_AI_MEDIA, carries: true, voice: 'kokoro', recorded: true },
      targets: [{ id: 't-tiktok', toolkit: 'tiktok', post_kind: 'video', at_publish: false,
        ai_label: "TikTok's AI-generated label is on: the post carries a synthetic voice (Kokoro)." }],
    })
    renderDetail(post({
      targets: [target('tiktok', { post_kind: 'video', options: { privacy_level: 'FOLLOWER_OF_CREATOR' } })],
    }))
    const receipt = screen.getByTestId('socials-receipt-tiktok-video')
    // Publish uses the choice only when the account allows it (US-304's choice form).
    expect(receipt.textContent).toContain(
      'Privacy: FOLLOWER_OF_CREATOR if the account allows it, else the most private level the account allows.',
    )
    expect(await screen.findByTestId('socials-ai-label-tiktok-video')).toHaveTextContent(
      "TikTok's AI-generated label is on: the post carries a synthetic voice (Kokoro).",
    )
    expect(api.get).toHaveBeenCalledWith('/api/socials/posts/p1/ai-labels')
  })

  it('states the AI label decision on every channel, not only TikTok', async () => {
    const youtube = "This video carries AI footage, and YouTube's upload takes no AI label. " +
      "Set 'Altered or synthetic content' in YouTube Studio for this video."
    const caption = (name: string) => `This post carries AI footage, and ${name} takes no AI label through its API, ` +
      'so the caption carries the disclosure: start it with "AI-generated."'
    api.get.mockResolvedValueOnce({
      ai_media: { ...NO_AI_MEDIA, carries: true, footage: ['hook'], recorded: true },
      targets: [
        { id: 't-youtube', toolkit: 'youtube', post_kind: 'video', ai_label: youtube, at_publish: false },
        { id: 't-linkedin', toolkit: 'linkedin', post_kind: 'text', ai_label: caption('LinkedIn'), at_publish: false },
        { id: 't-twitter', toolkit: 'twitter', post_kind: 'text', ai_label: caption('X'), at_publish: false },
      ],
    })
    renderDetail(post({ targets: [target('youtube', { post_kind: 'video' }), target('linkedin'), target('twitter')] }))
    expect(await screen.findByTestId('socials-ai-label-youtube-video')).toHaveTextContent(youtube)
    expect(screen.getByTestId('socials-ai-label-linkedin-text')).toHaveTextContent(caption('LinkedIn'))
    expect(screen.getByTestId('socials-ai-label-twitter-text')).toHaveTextContent(caption('X'))
  })

  it('never decides the AI label itself: a done footage slot alone says nothing', () => {
    renderDetail(post({
      targets: [target('tiktok', { post_kind: 'video' })],
      footage: { hook: { prompt: 'A harvest table', status: 'done' } } as SocialPost['footage'],
    }))
    expect(screen.getByTestId('socials-receipt-tiktok-video').textContent).not.toContain('AI-generated label')
    expect(screen.queryByTestId('socials-ai-label-tiktok-video')).toBeNull()
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

  it('a published target shows when it went out and its platform id next to View the post (P256-FIX-SOC-4)', () => {
    const at = '2026-10-14T18:05:00Z'
    renderDetail(post({
      status: 'published',
      targets: [target('instagram', {
        post_kind: 'carousel', status: 'published', remote_id: '17945112233', published_at: at,
        permalink: 'https://www.instagram.com/p/Cx7/',
      })],
    }))
    const receipt = within(screen.getByTestId('socials-receipt-instagram-carousel'))
    expect(receipt.getByRole('link', { name: 'View the post' }).getAttribute('href')).toBe('https://www.instagram.com/p/Cx7/')
    expect(receipt.getByText(`Published ${format(new Date(at), 'd MMM yyyy, HH:mm')}`)).toBeTruthy()
    expect(receipt.getByText(/id 17945112233$/)).toBeTruthy()
  })

  it('a target not yet out shows no time and no id', () => {
    renderDetail(post({ targets: [target('linkedin')] }))
    const receipt = screen.getByTestId('socials-receipt-linkedin-text')
    expect(receipt.textContent).not.toContain('Published ')
    expect(receipt.textContent).not.toContain(' id ')
  })

  it('a published target shows the AI label its publish used', async () => {
    const used = 'No AI media in this post: no AI label is needed.'
    api.get.mockResolvedValueOnce({
      ai_media: NO_AI_MEDIA,
      targets: [{ id: 't-linkedin', toolkit: 'linkedin', post_kind: 'text', ai_label: used, at_publish: true }],
    })
    renderDetail(post({ status: 'published', targets: [target('linkedin', { status: 'published' })] }))
    expect(await screen.findByTestId('socials-ai-label-linkedin-text')).toHaveTextContent(used)
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
