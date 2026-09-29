/**
 * PRD-251 Wave 2, US-204 — a post's channels are approved content. The composer
 * writes them with apiClient.setSocialPostTargets: a PUT (the backend's own verb)
 * of `{ targets }` to /api/socials/posts/{id}/targets, answered with the post,
 * whose `targets` carry each one's options, status and receipt.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type SocialPost, type SocialPostTargetInput } from '@/lib/api-client'

afterEach(() => vi.restoreAllMocks())

const TARGETS: SocialPostTargetInput[] = [
  { toolkit: 'linkedin', post_kind: 'text', options: {} },
  { toolkit: 'tiktok', post_kind: 'video', options: { privacy_level: 'SELF_ONLY', is_aigc: true } },
]

const POST = {
  id: 'post-1',
  status: 'needs_approval',
  targets: [
    {
      id: 'target-1',
      toolkit: 'linkedin',
      post_kind: 'text',
      options: {},
      status: 'pending',
      attempts: 0,
      remote_id: null,
      permalink: null,
      error: null,
      published_at: null,
    },
  ],
} as unknown as SocialPost

describe('apiClient.setSocialPostTargets', () => {
  it('PUTs the targets to the post and returns the post the backend sent', async () => {
    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue(POST)

    await expect(apiClient.setSocialPostTargets('post-1', TARGETS)).resolves.toEqual(POST)
    expect(spy).toHaveBeenCalledTimes(1)
    expect(spy).toHaveBeenCalledWith('/api/socials/posts/post-1/targets', {
      method: 'PUT',
      body: JSON.stringify({ targets: TARGETS }),
    })
  })

  it('clears the channels with an empty list', async () => {
    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue({ ...POST, targets: [] })

    await apiClient.setSocialPostTargets('post-1', [])
    expect(spy).toHaveBeenCalledWith('/api/socials/posts/post-1/targets', {
      method: 'PUT',
      body: JSON.stringify({ targets: [] }),
    })
  })
})
