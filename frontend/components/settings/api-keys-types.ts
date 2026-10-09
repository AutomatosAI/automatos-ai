/**
 * Shared types for the BYOK API Keys settings surface (PRD-54).
 *
 * Mirrors the Pydantic response models in `orchestrator/api/user_api_keys.py`
 * so the frontend never guesses a field name (issue #830: the test mutation
 * used to be typed `{ success, message }` while the endpoint's
 * `ApiKeyTestResult` actually returns `valid`).
 */

/** Mirrors `ApiKeyValidation` — the live provider-test result on a key save. */
export interface ApiKeyValidation {
  valid: boolean
  message: string
  tested_at: string | null
}

/** Mirrors `ApiKeyOut`. `validation` is populated only on the save response. */
export interface ApiKeyOut {
  id: number
  provider: string
  display_name: string | null
  masked_key: string
  is_active: boolean
  last_used_at: string | null
  usage_count: number
  created_at: string
  validation: ApiKeyValidation | null
}

export interface AddKeyPayload {
  provider: string
  api_key: string
  display_name: string
  /** The key's own endpoint, for providers that take one (Azure, #873). */
  base_url?: string
  /** The workspace an organization-level Anthropic key bills (wrkspc_…). */
  workspace_id?: string
}

/** Mirrors `ApiKeyTestResult` returned by `POST /api/keys/{id}/test`. */
export interface ApiKeyTestResult {
  valid: boolean
  message: string
  provider: string
}

export interface PlatformKeyStatus {
  platform_keys: Record<string, { configured: boolean }>
}

export interface ByokPreferences {
  byok_overrides: Record<string, boolean>
}

export interface ProviderOption {
  value: string
  label: string
}
