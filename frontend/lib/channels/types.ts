// Shared ChannelConnection shape — mirrors orchestrator/api/channels.py
// list response. Source of truth: settings/ChannelsSettingsTab uses the
// same shape; we extracted it here so /lib/sites can reference it
// without importing UI code.

export type ChannelPlatform =
  | 'slack'
  | 'telegram'
  | 'whatsapp'
  | 'signal'
  | 'discord'
  | 'teams'
  | 'google_chat'
  | 'matrix'
  | 'line'
  | 'imessage'
  | 'irc'
  | 'webhook'

export interface ChannelConnection {
  id: string
  platform: ChannelPlatform | string
  status: 'active' | 'inactive' | 'error' | string
  message_count: number
  last_activity_at: string | null
  config?: Record<string, string>
  metadata?: Record<string, unknown>
  default_agent_id?: number | null
  created_at?: string
  updated_at?: string
}

// Platforms eligible as callback destinations. webhook + imessage +
// irc are excluded for v1 — we want bidirectional, real-time channels
// the merchant team monitors.
export const CALLBACK_PLATFORMS: readonly ChannelPlatform[] = [
  'slack',
  'telegram',
  'whatsapp',
  'signal',
  'discord',
  'teams',
  'google_chat',
  'matrix',
  'line',
] as const

export const PLATFORM_LABEL: Record<string, string> = {
  slack: 'Slack',
  telegram: 'Telegram',
  whatsapp: 'WhatsApp',
  signal: 'Signal',
  discord: 'Discord',
  teams: 'MS Teams',
  google_chat: 'Google Chat',
  matrix: 'Matrix',
  line: 'LINE',
  imessage: 'iMessage',
  irc: 'IRC',
  webhook: 'Webhook',
}

// Per-platform UX hint for the "target" sub-input (channel id, chat id,
// phone number, etc).
export const PLATFORM_TARGET_HINT: Record<string, { label: string; placeholder: string }> = {
  slack:       { label: 'Channel ID',   placeholder: 'C0123456789' },
  telegram:    { label: 'Chat ID',      placeholder: '@username or -100123456789' },
  whatsapp:    { label: 'Phone (E.164)', placeholder: '+447700900123' },
  signal:      { label: 'Phone (E.164)', placeholder: '+447700900123' },
  discord:     { label: 'Channel ID',   placeholder: '0123456789012345678' },
  teams:       { label: 'Channel URL',  placeholder: 'https://teams.microsoft.com/...' },
  google_chat: { label: 'Space ID',     placeholder: 'spaces/AAA0000' },
  matrix:      { label: 'Room ID',      placeholder: '!roomid:server' },
  line:        { label: 'User/Group ID', placeholder: 'U1234...' },
}
