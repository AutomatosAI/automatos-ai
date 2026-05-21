/**
 * Sites types — mirror of orchestrator/core/models/sites.py + the
 * settings shape from PRD-008-A.
 */

export type SiteType = 'shopify' | 'wix' | 'woocommerce' | 'custom'
export type SiteStatus = 'active' | 'paused' | 'disconnected' | 'error'

export interface SiteCapabilities {
  has_cart: boolean
  has_catalog: boolean
  has_volume_discounts: boolean
  has_customer_records: boolean
  has_working_hours_source: boolean
  supports_theme_override: boolean
}

// Platform-keyed destination — same shape heartbeat's "Report To"
// uses. The backend auto-resolves Telegram chat ids / WhatsApp targets
// from the workspace's connected channels (Settings → Channels). Slack
// optionally takes an explicit channel override; webhook needs a URL.
export type CallbackPlatform = 'telegram' | 'slack' | 'whatsapp' | 'webhook'

export interface CallbackDestination {
  platform: CallbackPlatform
  /** Slack-only: explicit channel id (e.g. "C01ABC..."). When omitted,
   *  uses ``workspace.settings.integrations.slack_default_channel``. */
  channel_id?: string
  /** Webhook-only: URL the orchestrator POSTs to. */
  webhook_url?: string
}

export interface CallbackSettings {
  enabled: boolean
  destinations: CallbackDestination[]
}

export interface CartIdleSettings {
  enabled: boolean
  idle_seconds: number
  greeting: string
  frequency_cap: {
    scope: 'session' | 'day' | 'product_session'
    max_pops: number
  }
}

export interface ProactiveSettings {
  enabled: boolean
  page_types: string[]
  triggers: Array<{ type: string; seconds?: number; percent?: number }>
  frequency_cap: {
    scope: 'session' | 'day' | 'product_session'
    max_pops: number
  }
  greeting_source: string
  canned_fallback: string
  agent_timeout_ms: number
  popup_style: string
  respect_consent: boolean
  dismissal_persistence: string
}

export interface SiteSettings {
  widget_proactive?: ProactiveSettings
  callback?: CallbackSettings
  cart_idle?: CartIdleSettings
}

export interface Site {
  id: string
  workspace_id: string
  type: SiteType
  external_id: string | null
  display_name: string
  status: SiteStatus
  settings: SiteSettings
  capabilities: SiteCapabilities
  created_at: string
  updated_at: string
}

export interface SitesListResponse {
  sites: Site[]
}
