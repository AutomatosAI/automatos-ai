/**
 * Display helpers for the SDK API Keys table: dates, "last used" and the status badge.
 */
import type { KeyStatus, SdkApiKey } from "./api-key-types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** A key's date as "Jan 5, 2026" in the viewer's locale, or "--" when there is none. */
export function formatDate(iso: string | null): string {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** How long ago a key was used ("5m ago", "3d ago"), its date after 30 days, or "Never". */
export function formatRelative(iso: string | null): string {
  if (!iso) return "Never";
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1) return "Just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDays = Math.floor(diffHr / 24);
  if (diffDays < 30) return `${diffDays}d ago`;
  return formatDate(iso);
}

/** True when the key has an expiry date and it has passed. */
function isExpired(expiresAt: string | null): boolean {
  if (!expiresAt) return false;
  return new Date(expiresAt) < new Date();
}

// ---------------------------------------------------------------------------
// Key status helper
// ---------------------------------------------------------------------------

/** The status badge for a key: Revoked when inactive, Expired past its date, else Active. */
export function getKeyStatus(key: SdkApiKey): KeyStatus {
  if (!key.is_active) {
    return {
      label: "Revoked",
      variant: "destructive",
      className: "bg-destructive/15 text-destructive border-destructive/30",
    };
  }
  if (isExpired(key.expires_at)) {
    return {
      label: "Expired",
      variant: "secondary",
      className: "bg-warning/15 text-warning border-warning/30",
    };
  }
  return {
    label: "Active",
    variant: "default",
    className: "bg-success/15 text-success border-success/30",
  };
}
