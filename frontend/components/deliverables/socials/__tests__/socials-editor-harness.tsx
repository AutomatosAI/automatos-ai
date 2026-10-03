/**
 * PRD-251B US-B109 — the post editor's test harness (not a test itself): the channels,
 * templates and footage a workspace offers, a stateful apiClient double, and the editor
 * rendered with its query client. The editor tests mock '@/lib/api-client' with
 * `apiModule` from here, so the double and the tests share one instance.
 */
import { vi } from 'vitest'
import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import type { SocialChannel, SocialPost, SocialTemplateSummary } from '@/lib/api-client'

const kind = (k: string, available = true) => ({ kind: k, available, reason: available ? null : 'no action', needs_public_storage: false })

export const CHANNELS: SocialChannel[] = [
  { toolkit: 'twitter', label: 'X', post_kinds: [kind('text'), kind('image'), kind('video')] as any, verified: true, setup_note: null, copy_limits: { text: 280 } },
  { toolkit: 'linkedin', label: 'LinkedIn', post_kinds: [kind('text'), kind('image'), kind('video')] as any, verified: true, setup_note: null, copy_limits: { text: 3000 } },
  { toolkit: 'instagram', label: 'Instagram', post_kinds: [kind('image'), kind('carousel'), kind('reel')] as any, verified: true, setup_note: null, copy_limits: { text: 2200 } },
  { toolkit: 'tiktok', label: 'TikTok', post_kinds: [kind('video')] as any, verified: true, setup_note: null, copy_limits: { text: 2200 } },
  { toolkit: 'reddit', label: 'Reddit (unverified channel)', post_kinds: [kind('image')] as any, verified: false, setup_note: null, copy_limits: { text: 300 } },
]

export const IMAGE_TEMPLATE: SocialTemplateSummary = {
  id: 'tpl-img', name: 'Fact card', description: null, format: 'social_image', kind: 'image',
  sizes: ['1080x1080', '1920x1080'], durations: [], thumbnail_url: 'https://cdn.test/fact-card.png', is_starter: true,
  updated_at: null, footage_slots: [],
  variables_schema: { headline: { type: 'text', label: 'Headline' } as any, members: { type: 'number', label: 'Members', claim: true } as any },
}

export const VIDEO_TEMPLATE: SocialTemplateSummary = {
  id: 'tpl-vid', name: 'Story promo', description: null, format: 'social_video', kind: 'video',
  sizes: ['1080x1920', '1920x1080'], durations: [15, 30, 40], thumbnail_url: null, is_starter: true,
  updated_at: null, footage_slots: ['broll', 'hook'], variables_schema: {},
}

/** PRD-251B: a template with a photo spot (the photo cards). */
export const PHOTO_TEMPLATE: SocialTemplateSummary = {
  ...IMAGE_TEMPLATE, id: 'tpl-photo', name: 'Photo + headline', thumbnail_url: null,
  footage_slots: ['photo'], image_slots: ['photo'], image_slot_labels: { photo: 'Photo' },
}

export const api = {
  listSocialChannels: vi.fn(),
  listSocialTemplates: vi.fn(),
  getSocialFootageSources: vi.fn(),
  listDeliverables: vi.fn(),
  createSocialPost: vi.fn(),
  updateSocialPost: vi.fn(),
  setSocialPostTargets: vi.fn(),
  setSocialPostSlot: vi.fn(),
  submitSocialPost: vi.fn(),
  renderSocialPost: vi.fn(),
  composeSocialPost: vi.fn(),
  uploadSocialPostMedia: vi.fn(),
  searchSocialSources: vi.fn(),
  getSocialPostMedia: vi.fn(),
  listSocialPosts: vi.fn(),
  get: vi.fn(),
  // PRD-251B US-B305: AI options for an image slot.
  makeSocialAiOptions: vi.fn(),
  pickSocialAiOption: vi.fn(),
  // PRD-251B: a Library picture in a template's photo spot.
  setSocialPostPhoto: vi.fn(),
}
export const apiModule = { apiClient: api, default: api }

export function post(over: Partial<SocialPost> = {}): SocialPost {
  return {
    id: 'post-1', workspace_id: 'w1', created_by: 'u1', title: 'Nothing posts without you', brief: 'The approval rule',
    copy: { base: 'Agents draft. A person approves.' }, format: 'image', template_id: null, variables: {}, sources: {},
    media: {}, status: 'draft', content_hash: 'h1', approved_hash: null, approved_by: null, approved_at: null,
    override_unsourced: false, review_log: [], scheduled_for: null, timezone: null, planned_for: null, length_seconds: null,
    targets: [], preview: null, created_at: '2026-10-14T07:02:00Z', updated_at: '2026-10-14T07:02:00Z',
    ...over,
  } as SocialPost
}

/** Every call answers as a server would: a save returns the post with what it was sent. */
export function resetApi() {
  Object.values(api).forEach((fn) => fn.mockReset())
  api.listSocialChannels.mockResolvedValue(CHANNELS)
  api.listSocialTemplates.mockImplementation(async (format?: string) =>
    format === 'video' ? [VIDEO_TEMPLATE] : format === 'text' ? [] : [IMAGE_TEMPLATE],
  )
  api.getSocialFootageSources.mockResolvedValue({ kinds: { video: { available: true, toolkit: 'fal_ai', label: 'fal.ai' } } })
  api.createSocialPost.mockImplementation(async (input: any) => post({ id: 'post-new', ...input }))
  api.updateSocialPost.mockImplementation(async (id: string, input: any) => post({ id, ...input }))
  api.setSocialPostTargets.mockImplementation(async (id: string) => post({ id }))
  api.setSocialPostSlot.mockImplementation(async (id: string, plannedFor: string | null, timezone: string) =>
    post({ id, planned_for: plannedFor, timezone }),
  )
  api.submitSocialPost.mockImplementation(async (id: string) => post({ id, status: 'needs_approval' }))
  api.renderSocialPost.mockImplementation(async (id: string) => post({ id, status: 'rendering' }))
  api.uploadSocialPostMedia.mockImplementation(async (id: string) => post({ id, media: { original: ['d-1'] } }))
  api.listDeliverables.mockImplementation(async ({ artifact_type }: { artifact_type: string }) => ({
    total: 1,
    deliverables: [{
      id: `d-${artifact_type}`, title: `A ${artifact_type}`, artifact_type, file_name: null,
      preview_url: artifact_type === 'image' ? 'https://cdn.test/a.png' : null,
      created_at: artifact_type === 'image' ? '2026-10-10T09:00:00Z' : '2026-10-11T09:00:00Z',
    }],
  }))
  api.searchSocialSources.mockResolvedValue({ candidates: [] })
  api.getSocialPostMedia.mockResolvedValue([])
  api.listSocialPosts.mockResolvedValue({ posts: [], total: 0 })
  api.get.mockResolvedValue({ name: 'Automatos' }) // the brand kit
  api.makeSocialAiOptions.mockImplementation(async (id: string, slot: string, prompt: string) =>
    post({ id, footage: { [slot]: { prompt, options: [], options_state: 'making' } } }),
  )
  api.pickSocialAiOption.mockImplementation(async (id: string) => post({ id }))
  api.setSocialPostPhoto.mockImplementation(async (id: string, slot: string) =>
    post({ id, footage: { [slot]: { prompt: 'Photo: your own picture (library-1.png)', status: 'done', toolkit: 'library', name: 'library-1.png' } } }),
  )
}

export function renderWith(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}
