/**
 * PRD-263 PR-1E-a — the pure helpers moved out of SystemLLMSettingsTab.tsx keep what
 * the tab did inline: the heartbeat channel key parsed for delivery on save, one
 * connected channel per platform, and the fallback config when the endpoint fails.
 */
import { describe, it, expect } from 'vitest'
import {
  HeartbeatConfig,
  connectedChannelsFrom,
  fallbackOrchestratorConfig,
  heartbeatPayload,
} from '../orchestrator-settings-config'

const heartbeat = (notification_channel: string): HeartbeatConfig => ({
  ...fallbackOrchestratorConfig().heartbeat,
  notification_channel,
})

describe('heartbeatPayload', () => {
  const channels = [{ key: 'channel:42', platform: 'telegram' }]

  it('turns a channel key into its platform and channel_id', () => {
    const input = heartbeat('channel:42')
    const saved = heartbeatPayload(input, channels)

    expect(saved.notification_channel).toBe('telegram')
    expect(saved.channel_id).toBe('42')
    expect(input.notification_channel).toBe('channel:42')
    expect(input.channel_id).toBeUndefined()
  })

  it('keeps an unknown channel key as the channel, with its id', () => {
    const saved = heartbeatPayload(heartbeat('channel:9'), channels)

    expect(saved.notification_channel).toBe('channel:9')
    expect(saved.channel_id).toBe('9')
  })

  it('leaves built-in channels and platform names as they are', () => {
    for (const channel of ['in_app', 'webhook', 'slack']) {
      const saved = heartbeatPayload(heartbeat(channel), channels)
      expect(saved.notification_channel).toBe(channel)
      expect(saved.channel_id).toBeUndefined()
    }
  })
})

describe('connectedChannelsFrom', () => {
  it('lists configured integrations first, then one channel per platform not yet seen', () => {
    const integrations = { telegram_bot_token: { configured: true }, slack_bot_token: { configured: false } }
    const channels = [{ id: 5, platform: 'telegram' }, { id: 6, platform: 'slack' }, { id: 7, platform: 'slack' }]

    expect(connectedChannelsFrom(integrations, channels)).toEqual([
      { key: 'telegram_bot_token', platform: 'telegram' },
      { key: 'channel:6', platform: 'slack' },
    ])
    expect(connectedChannelsFrom({}, [])).toEqual([])
  })
})

describe('fallbackOrchestratorConfig', () => {
  it('keeps HARNESS off, weekly and full_auto, as the tab did', () => {
    const config = fallbackOrchestratorConfig()

    expect(config.harness).toEqual({ enabled: false, schedule: 'weekly', mode: 'full_auto' })
    expect(config.heartbeat.enabled).toBe(false)
    expect(config.heartbeat.interval_minutes).toBe(30)
    expect(config.personality_mode).toBe('friendly')
    expect(config.llm?.max_tokens).toBe(4000)
  })
})
