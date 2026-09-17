/**
 * Channels in the local edition (2026-09-17): a polling-capable platform starts
 * on polling (no public URL), a webhook-only platform says inbound needs one.
 */
import { describe, it, expect } from 'vitest'

import { defaultModeFor, needsPublicUrlLocally, POLLING_CAPABLE_PLATFORMS } from '@/components/settings/ChannelsSettingsTab'

describe('channels in the local edition', () => {
  it('starts Telegram on polling locally and on the recommended webhook mode hosted', () => {
    expect(defaultModeFor('telegram', true)).toBe('polling')
    expect(defaultModeFor('telegram', false)).toBe('webhook')
  })

  it('leaves single-mode platforms without a picker default', () => {
    expect(defaultModeFor('slack', true)).toBeUndefined()
    expect(defaultModeFor('whatsapp', false)).toBeUndefined()
  })

  it('flags webhook-only platforms locally, never hosted, never Telegram', () => {
    expect(needsPublicUrlLocally('slack', true)).toBe(true)
    expect(needsPublicUrlLocally('whatsapp', true)).toBe(true)
    expect(needsPublicUrlLocally('telegram', true)).toBe(false)
    expect(needsPublicUrlLocally('slack', false)).toBe(false)
    expect(POLLING_CAPABLE_PLATFORMS.has('telegram')).toBe(true)
  })
})
