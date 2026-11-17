/**
 * Types and option lists for the SDK API Keys surface (Settings › API keys).
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
  | "playbooks:execute"
  | "tasks:read"
  | "tasks:write"
  | "memory:read"
  | "memory:write"
  | "agents:chat"
  | "reports:write";

/** One scope the create form offers: its checkbox label and, where the label needs it, a hint. */
export interface PermissionOption {
  value: Permission;
  label: string;
  group: string;
  /** What the scope lets the key do, in a sentence under the checkbox. */
  hint?: string;
}

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

/** The scopes either kind of key is offered. "Select All" ticks these and no others. */
export const COMMON_PERMISSIONS: PermissionOption[] = [
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

/**
 * The scopes a product's own backend uses (PRD-274; orchestrator/api/api_keys.py
 * `VALID_PERMISSIONS`). The form offers them for a server key only, and each is ticked on its
 * own: "Select All" leaves them out, so one click never widens a key. The routes behind five of
 * them refuse a public key (orchestrator/core/auth/server_key_gate.py). The board's read routes
 * accept `tasks:read` from either kind; this form still offers it for a server key only, because
 * a public key can be read by anyone who opens the page it sits in.
 */
export const SERVER_KEY_PERMISSIONS: PermissionOption[] = [
  {
    value: "tasks:read",
    label: "Tasks: Read",
    group: "Tasks",
    hint: "Read the board's Tasks, what each one cost, and the fleet's live state.",
  },
  {
    value: "tasks:write",
    label: "Tasks: Write",
    group: "Tasks",
    hint: "File a Task, including one assigned to an agent with a prompt.",
  },
  {
    value: "memory:read",
    label: "Memory: Read",
    group: "Memory",
    hint: "Read and search memory: the workspace's, and any end user's.",
  },
  {
    value: "memory:write",
    label: "Memory: Write",
    group: "Memory",
    hint: "Store memory. Also erases an end user entirely: their memories, conversations and consent record.",
  },
  {
    value: "agents:chat",
    label: "Agents: Chat",
    group: "Agents",
    hint: "Chat with an agent as the key's own service user. This workspace pays for each turn.",
  },
  {
    value: "reports:write",
    label: "Reports: Write",
    group: "Reports",
    hint: "Report an agent's reply in the key's own chats, for the end user who read it.",
  },
];

export const EXPIRY_OPTIONS: { value: ExpiryOption; label: string }[] = [
  { value: "none", label: "No expiry" },
  { value: "30d", label: "30 days" },
  { value: "90d", label: "90 days" },
  { value: "1y", label: "1 year" },
];
