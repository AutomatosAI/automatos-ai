/**
 * PRD-251 Wave 2, US-203 (D8, S3.2) — the composer reads the workspace's connected
 * channels: apiClient.listSocialChannels GETs /api/socials/channels and hands back
 * the backend's list as it is — an unavailable kind with its reason, a kind that
 * needs public storage, a generic channel still unverified, X's setup note.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type SocialChannel } from '@/lib/api-client'

afterEach(() => vi.restoreAllMocks())

const CHANNELS: SocialChannel[] = [
  {
    toolkit: 'twitter',
    label: 'X',
    post_kinds: [
      { kind: 'text', available: true, reason: null, needs_public_storage: false },
      {
        kind: 'video',
        available: false,
        reason: "Missing action TWITTER_UPLOAD_LARGE_MEDIA: Automatos's list of Composio actions does not hold it yet.",
        needs_public_storage: false,
      },
    ],
    verified: true,
    setup_note: 'Composio removed its managed X credentials in February 2026 — connect X with your own X API app in Composio',
  },
  {
    toolkit: 'youtube',
    label: 'YouTube',
    post_kinds: [{ kind: 'video', available: true, reason: null, needs_public_storage: true }],
    verified: true,
    setup_note: null,
  },
  {
    toolkit: 'reddit',
    label: 'Reddit (unverified channel)',
    post_kinds: [{ kind: 'image', available: true, reason: null, needs_public_storage: false }],
    verified: false,
    setup_note: null,
  },
]

describe('apiClient.listSocialChannels', () => {
  it('GETs the channels route and returns the channels as the backend sent them', async () => {
    const spy = vi.spyOn(apiClient, 'request').mockResolvedValue(CHANNELS)

    await expect(apiClient.listSocialChannels()).resolves.toEqual(CHANNELS)
    expect(spy).toHaveBeenCalledTimes(1)
    expect(spy).toHaveBeenCalledWith('/api/socials/channels')
  })
})
