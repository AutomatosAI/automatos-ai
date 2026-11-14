/**
 * The create form's checks and the `POST /api/api-keys` body built from it. Pure functions,
 * so the create hook only sequences the request.
 */
import {
  SERVER_KEY_PERMISSIONS,
  type CreateKeyFormValues,
  type CreateKeyPayload,
  type ExpiryOption,
  type KeyType,
  type Permission,
} from "./api-key-types";

const EXPIRY_DAYS: Record<string, number> = {
  "30d": 30,
  "90d": 90,
  "1y": 365,
};

const MS_PER_DAY = 86400000;

const SERVER_KEY_ONLY: ReadonlySet<Permission> = new Set(SERVER_KEY_PERMISSIONS.map((p) => p.value));

/**
 * The ticked scopes the key will be given. A public key takes none of the server-key scopes:
 * the form hides them once the type is public, and one ticked before the switch is left behind.
 */
export function grantedPermissions(
  values: Pick<CreateKeyFormValues, "formKeyType" | "formPermissions">,
): Permission[] {
  const ticked = Array.from(values.formPermissions);
  if (values.formKeyType === "server") return ticked;
  return ticked.filter((permission) => !SERVER_KEY_ONLY.has(permission));
}

/** Why the form cannot be sent yet (the toast's text), or null when it can. */
export function createFormError(values: CreateKeyFormValues): string | null {
  if (!values.formName.trim()) return "Key name is required";
  if (grantedPermissions(values).length === 0) return "Select at least one permission";
  if (values.formKeyType === "public" && !values.formDomains.trim()) {
    return "Allowed domains are required for public keys";
  }
  return null;
}

/** One domain per non-blank line, for a public key; a server key sends none. */
function allowedDomains(formKeyType: KeyType, formDomains: string): Pick<CreateKeyPayload, "allowed_domains"> {
  if (formKeyType !== "public" || !formDomains.trim()) return {};
  return {
    allowed_domains: formDomains
      .split("\n")
      .map((d) => d.trim())
      .filter(Boolean),
  };
}

/** A positive whole number of requests per minute; anything else sends no limit. */
function rateLimit(formRateLimit: string): Pick<CreateKeyPayload, "rate_limit_requests"> {
  const rateLimitNum = parseInt(formRateLimit, 10);
  if (!isNaN(rateLimitNum) && rateLimitNum > 0) {
    return { rate_limit_requests: rateLimitNum };
  }
  return {};
}

/** The expiry as an ISO timestamp counted from now; "No expiry" sends none. */
function expiresAt(formExpiry: ExpiryOption): Pick<CreateKeyPayload, "expires_at"> {
  if (formExpiry === "none") return {};
  const now = new Date();
  const days = EXPIRY_DAYS[formExpiry] ?? 0;
  if (days <= 0) return {};
  const exp = new Date(now.getTime() + days * MS_PER_DAY);
  return { expires_at: exp.toISOString() };
}

/** The request body for a new key, from the form as it stands (fields in the API's order). */
export function buildCreatePayload(values: CreateKeyFormValues): CreateKeyPayload {
  return {
    name: values.formName.trim(),
    key_type: values.formKeyType,
    permissions: grantedPermissions(values),
    ...allowedDomains(values.formKeyType, values.formDomains),
    ...rateLimit(values.formRateLimit),
    ...expiresAt(values.formExpiry),
  };
}
