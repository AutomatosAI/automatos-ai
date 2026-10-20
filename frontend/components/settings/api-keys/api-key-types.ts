/**
 * Types and option lists for the SDK API Keys surface (Settings › Widget SDK › API keys).
 * Mirrors the `/api/api-keys` request and response shapes in orchestrator/api/api_keys.py.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type KeyType = "public" | "server";

export type Permission =
  | "chat"
  | "documents:read"
  | "documents:write"
  | "data:query"
  | "data:execute"
  | "agents:read"
  | "agents:execute"
  | "missions:read"
  | "missions:execute"
  | "playbooks:read"
  | "playbooks:execute";

export type ExpiryOption = "none" | "30d" | "90d" | "1y";

export interface SdkApiKey {
  id: string;
  name: string;
  key_prefix: string;
  key_type: KeyType;
  permissions: Permission[];
  allowed_domains: string[] | null;
  rate_limit_requests: number | null;
  expires_at: string | null;
  created_at: string;
  last_used_at: string | null;
  is_active: boolean;
}

export interface CreateKeyPayload {
  name: string;
  key_type: KeyType;
  permissions: Permission[];
  allowed_domains?: string[];
  rate_limit_requests?: number | null;
  expires_at?: string | null;
}

export interface CreateKeyResponse {
  id: string;
  name: string;
  key: string;
  key_type: string;
  permissions: string[];
  created_at: string | null;
}

/** The create form's fields as the inputs hold them (rate limit and domains are raw text). */
export interface CreateKeyFormValues {
  formName: string;
  formKeyType: KeyType;
  formPermissions: Set<Permission>;
  formDomains: string;
  formRateLimit: string;
  formExpiry: ExpiryOption;
}

/** A key's status badge: its label, the Badge variant and the token colours. */
export interface KeyStatus {
  label: string;
  variant: "default" | "secondary" | "destructive" | "outline";
  className: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const ALL_PERMISSIONS: { value: Permission; label: string; group: string }[] = [
  { value: "chat", label: "Chat", group: "Chat" },
  { value: "documents:read", label: "Documents: Read", group: "Documents" },
  { value: "documents:write", label: "Documents: Write", group: "Documents" },
  { value: "data:query", label: "Data: Query", group: "Data" },
  { value: "data:execute", label: "Data: Execute", group: "Data" },
  { value: "agents:read", label: "Agents: Read", group: "Agents" },
  { value: "agents:execute", label: "Agents: Execute", group: "Agents" },
  { value: "missions:read", label: "Missions: Read", group: "Missions" },
  { value: "missions:execute", label: "Missions: Execute", group: "Missions" },
  { value: "playbooks:read", label: "Playbooks: Read", group: "Playbooks" },
  { value: "playbooks:execute", label: "Playbooks: Execute", group: "Playbooks" },
];

export const EXPIRY_OPTIONS: { value: ExpiryOption; label: string }[] = [
  { value: "none", label: "No expiry" },
  { value: "30d", label: "30 days" },
  { value: "90d", label: "90 days" },
  { value: "1y", label: "1 year" },
];
