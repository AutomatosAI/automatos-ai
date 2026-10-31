/**
 * PRD-263 PR-1E-a — the pure helpers moved out of SystemLLMSettingsTab.tsx keep what
 * the tab did inline: the heartbeat channel key parsed for delivery on save, one
 * connected channel per platform, and the fallback config when the endpoint fails.
 * PR-1E-b (US-017) changed only HARNESS: the fallback is report_only with no Enable
 * switch, and a stored mode the list does not offer shows as Report only. US-054 adds
 * Propose to the list.
 */
import { describe, it, expect } from 'vitest'
import {
  HARNESS_MODE_OPTIONS,
  HeartbeatConfig,
  connectedChannelsFrom,
  fallbackOrchestratorConfig,
  harnessModeOption,
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
  it('falls back to HARNESS report_only and weekly, with no Enable switch', () => {
    const config = fallbackOrchestratorConfig()

    expect(config.harness).toEqual({ schedule: 'weekly', mode: 'report_only' })
    expect(config.heartbeat.enabled).toBe(false)
    expect(config.heartbeat.interval_minutes).toBe(30)
    expect(config.personality_mode).toBe('friendly')
    expect(config.llm?.max_tokens).toBe(4000)
  })
})

describe('harnessModeOption', () => {
  it('shows an offered mode as itself and anything else as Report only', () => {
    // PRD-263 US-054 offers Propose, so a stored propose now shows as itself.
    expect(HARNESS_MODE_OPTIONS.map((option) => option.value)).toEqual(['report_only', 'propose', 'disabled'])
    expect(harnessModeOption('disabled').label).toBe('Disabled')
    expect(harnessModeOption('propose').label).toBe('Propose')
    for (const mode of ['full_auto', 'manual', undefined]) {
      expect(harnessModeOption(mode).value).toBe('report_only')
    }
  })
})
