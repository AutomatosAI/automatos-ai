/**
 * Sites types — mirror of orchestrator/core/models/sites.py + the
 * settings shape from PRD-008-A.
 */

export type SiteType = 'shopify' | 'wix' | 'woocommerce' | 'custom';
export type SiteStatus = 'active' | 'paused' | 'disconnected' | 'error';
export type CallbackDestinationType =
  | 'email'
  | 'slack_webhook'
  | 'crm_webhook'
  | 'shopify_customer_note';

export interface SiteCapabilities {
  has_cart: boolean;
  has_catalog: boolean;
  has_volume_discounts: boolean;
  has_customer_records: boolean;
  has_working_hours_source: boolean;
  supports_theme_override: boolean;
}

export interface CallbackDestination {
  type: CallbackDestinationType;
  address?: string;        // email
  url?: string;            // slack_webhook | crm_webhook
  channel_label?: string;  // slack_webhook
  auth_header?: string;    // crm_webhook
}

export interface WorkingHoursDay {
  start: string; // "09:00"
  end: string;   // "17:00"
}

export interface CallbackSettings {
  enabled: boolean;
  destinations: CallbackDestination[];
  fields?: {
    phone_required: boolean;
    name_required: boolean;
    urgency_optional: boolean;
    preferred_time_optional: boolean;
  };
  working_hours_only: boolean;
  working_hours: {
    tz: string;
    monday: WorkingHoursDay | 'closed';
    tuesday: WorkingHoursDay | 'closed';
    wednesday: WorkingHoursDay | 'closed';
    thursday: WorkingHoursDay | 'closed';
    friday: WorkingHoursDay | 'closed';
    saturday: WorkingHoursDay | 'closed';
    sunday: WorkingHoursDay | 'closed';
  };
  sla_hours: number;
  team_capacity: 'limited' | 'normal';
  intent_phrases: string[];
  rate_limit_per_hour: number;
}

export interface CartIdleSettings {
  enabled: boolean;
  idle_seconds: number;
  greeting: string;
  frequency_cap: {
    scope: 'session' | 'day' | 'product_session';
    max_pops: number;
  };
}

export interface ProactiveSettings {
  enabled: boolean;
  page_types: string[];
  triggers: Array<{ type: string; seconds?: number; percent?: number }>;
  frequency_cap: {
    scope: 'session' | 'day' | 'product_session';
    max_pops: number;
  };
  greeting_source: string;
  canned_fallback: string;
  agent_timeout_ms: number;
  popup_style: string;
  respect_consent: boolean;
  dismissal_persistence: string;
}

export interface SiteSettings {
  widget_proactive?: ProactiveSettings;
  callback?: CallbackSettings;
  cart_idle?: CartIdleSettings;
}

export interface Site {
  id: string;
  workspace_id: string;
  type: SiteType;
  external_id: string | null;
  display_name: string;
  status: SiteStatus;
  settings: SiteSettings;
  capabilities: SiteCapabilities;
  created_at: string;
  updated_at: string;
}

export interface SitesListResponse {
  sites: Site[];
}
