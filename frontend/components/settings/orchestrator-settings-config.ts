/**
 * Orchestrator settings (PRD-55): the shapes GET/PUT /api/workspaces/current/orchestrator
 * carry, the option lists Settings > Orchestrator offers, and the pure helpers its hooks
 * use. Moved out of SystemLLMSettingsTab.tsx unchanged (PRD-263 PR-1E-a) so each part of
 * the tab fits the component length rule.
 */

import { getDefaultModelConfig } from '@/lib/llm-defaults'

export interface LLMConfig {
  provider: string
  model_id: string
  temperature: number
  max_tokens: number
  top_p: number
  frequency_penalty: number
  presence_penalty: number
  stop: string[] | null
  timeout: number | null
  fallback_model_id: string | null
}

export interface HeartbeatConfig {
  enabled: boolean
  interval_minutes: number
  active_hours_start: string
  active_hours_end: string
  timezone: string
  checklist: string
  notification_channel: string
  channel_id?: string
}

/** HARNESS settings (PRD-263 US-017): the mode is the only control; the one schedule is weekly. */
export interface HarnessConfig {
  schedule?: string
  mode: string
}

export interface OrchestratorConfig {
  personality_mode: string
  custom_soul: string
  communication_style: string
  proactive_level: string
  thinking_level: string
  preferred_channel: string
  llm?: LLMConfig
  heartbeat: HeartbeatConfig
  harness: HarnessConfig
}

export interface ConnectedChannel {
  key: string
  platform: string
}

export const PERSONALITY_PRESETS: Record<string, { label: string; description: string }> = {
  friendly: { label: 'Friendly', description: 'Warm, approachable, celebrates wins with you' },
  professional: { label: 'Professional', description: 'Polished, structured, enterprise-appropriate' },
  technical: { label: 'Technical', description: 'Precise, detailed, developer-focused' },
  custom: { label: 'Custom', description: 'Write your own personality prompt' },
}

export const COMMUNICATION_STYLES: Record<string, { label: string; description: string }> = {
  concise: { label: 'Concise', description: 'Short, direct answers' },
  balanced: { label: 'Balanced', description: 'Clear answers with helpful context' },
  detailed: { label: 'Detailed', description: 'Thorough explanations with examples' },
}

export const PROACTIVE_LEVELS: Record<string, { label: string; description: string }> = {
  silent: { label: 'Silent', description: 'Heartbeat runs but never notifies' },
  notify: { label: 'Notify Only', description: 'Reports findings but takes no action' },
  act_notify: { label: 'Act & Notify', description: 'Takes action and tells you what it did' },
  autonomous: { label: 'Fully Autonomous', description: 'Acts independently, reports summary' },
}

export const THINKING_LEVELS: Record<string, string> = {
  off: 'Off',
  minimal: 'Minimal',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
}

export const DEFAULT_HARNESS_MODE = 'report_only'

/** The HARNESS modes (PRD-263 US-017, section 8 copy); US-054 added Propose. */
export const HARNESS_MODE_OPTIONS = [
  {
    value: DEFAULT_HARNESS_MODE,
    label: 'Report only',
    description: 'HARNESS writes a weekly report on your agents and changes nothing.',
  },
  {
    value: 'propose',
    label: 'Propose',
    description:
      'each change HARNESS suggests comes to you as an Approvals card, and nothing changes until you ' +
      'approve it. Cards appear only when proposals are switched on for this installation.',
  },
  {
    value: 'disabled',
    label: 'Disabled',
    description: 'HARNESS does not run for this workspace.',
  },
] as const

export const HARNESS_HELP_TEXT =
  'HARNESS runs once a week, on a weekday, and judges only agents with enough Tasks.'

/** The option a stored mode shows as: itself when offered, else Report only. */
export function harnessModeOption(mode?: string) {
  return HARNESS_MODE_OPTIONS.find((option) => option.value === mode) ?? HARNESS_MODE_OPTIONS[0]
}

/** The LLM block an edit starts from when the config has none yet. */
export const EMPTY_LLM_CONFIG: LLMConfig = {
  provider: '', model_id: '', temperature: 0.7, max_tokens: 4000, top_p: 1.0, frequency_penalty: 0.0,
  presence_penalty: 0.0, stop: null, timeout: null, fallback_model_id: null,
}

/** The config the tab shows when the orchestrator endpoint cannot be read. */
export function fallbackOrchestratorConfig(): OrchestratorConfig {
  return {
    personality_mode: 'friendly',
    custom_soul: '',
    communication_style: 'balanced',
    proactive_level: 'notify',
    thinking_level: 'medium',
    preferred_channel: 'in_app',
    heartbeat: {
      enabled: false,
      interval_minutes: 30,
      active_hours_start: '08:00',
      active_hours_end: '20:00',
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
      checklist: '- Check agent health status\n- Review pending webhook failures\n- Summarize today\'s activity',
      notification_channel: 'in_app',
    },
    llm: {
      ...getDefaultModelConfig(),
      max_tokens: 4000,
      stop: null,
      timeout: null,
    },
    harness: {
      schedule: 'weekly',
      mode: DEFAULT_HARNESS_MODE,
    },
  }
}

/** The heartbeat as saved: a channel key is parsed into platform + channel_id for delivery. */
export function heartbeatPayload(heartbeat: HeartbeatConfig, channels: ConnectedChannel[]): HeartbeatConfig {
  const hb = { ...heartbeat }
  const channelKey = hb.notification_channel
  if (channelKey?.startsWith('channel:')) {
    const channelId = channelKey.replace('channel:', '')
    const found = channels.find(ch => ch.key === channelKey)
    hb.notification_channel = found?.platform || channelKey
    hb.channel_id = channelId
  } else if (channelKey === 'in_app' || channelKey === 'webhook') {
    // Built-in channels, no channel_id needed
  } else {
    // Direct platform name (e.g. "telegram" from workspace integrations)
    hb.notification_channel = channelKey
  }
  return hb
}

/** The connected channels the notification dropdowns offer, one per platform. */
export function connectedChannelsFrom(integrations: any, channels: any): ConnectedChannel[] {
  const found: ConnectedChannel[] = []
  const seen = new Set<string>()
  const platformMap: Record<string, string> = { telegram_bot_token: 'telegram', slack_bot_token: 'slack' }
  if (integrations) {
    for (const [key, val] of Object.entries(integrations)) {
      const platform = platformMap[key]
      if (platform && (val as any)?.configured && !seen.has(platform)) {
        found.push({ key, platform })
        seen.add(platform)
      }
    }
  }
  if (Array.isArray(channels)) {
    for (const ch of channels) {
      if (!seen.has(ch.platform)) {
        found.push({ key: `channel:${ch.id}`, platform: ch.platform })
        seen.add(ch.platform)
      }
    }
  }
  return found
}
