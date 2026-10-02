/**
 * PRD-251 Wave 2, US-202 (S3.4, D9) — the approval view reads a post's media as
 * presigned inline links: apiClient.getSocialPostMedia GETs
 * /api/socials/posts/{id}/media and hands back the backend's list as it is,
 * a file with no stored object included (its url null, its error saying why).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type SocialPostMediaLink } from '@/lib/api-client'

afterEach(() => vi.restoreAllMocks())

const LINKS: SocialPostMediaLink[] = [
  {
    aspect: '9:16',
    deliverable_id: '11111111-2222-4333-8444-555555555555',
    name: 'render-9x16.mp4',
    url: 'https://storage.test/automatos-ai/social-media/ws/post/render-9x16.mp4?X-Amz-Signature=abc',
    content_type: 'video/mp4',
    bytes: 3749,
    error: null,
  },
  {
    aspect: '1:1',
    deliverable_id: '66666666-7777-4888-9999-000000000000',
    name: null,
    url: null,
    content_type: null,
    bytes: null,
    error: 'This file is no longer in Deliverables.',
  },
]

describe('apiClient.getSocialPostMedia', () => {
  it('GETs the post media route and returns its links as the backend sent them', async () => {
    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue(LINKS)

    await expect(apiClient.getSocialPostMedia('post-1')).resolves.toEqual(LINKS)
    expect(spy).toHaveBeenCalledTimes(1)
    expect(spy).toHaveBeenCalledWith('/api/socials/posts/post-1/media')
  })
})
