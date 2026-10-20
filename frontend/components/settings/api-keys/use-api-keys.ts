"use client";

/**
 * The SDK key list (`GET /api/api-keys`) and revoking one (`DELETE /api/api-keys/{id}`), through
 * `apiClient`, which adds the Clerk JWT and the workspace ID.
 */
import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import type { SdkApiKey } from "./api-key-types";

// ---------------------------------------------------------------------------
// Fetch keys (uses apiClient which handles Clerk JWT + workspace ID)
// ---------------------------------------------------------------------------

/** Loads the workspace's SDK keys on mount; `fetchKeys` reloads them (Retry, after a change). */
export function useApiKeys() {
  const [keys, setKeys] = useState<SdkApiKey[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchKeys = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await apiClient.request<SdkApiKey[]>("/api/api-keys");
      setKeys(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load API keys";
      setError(msg);
      console.error("[ApiKeyManager] fetchKeys error:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  return { keys, isLoading, error, fetchKeys };
}

// ---------------------------------------------------------------------------
// Revoke key
// ---------------------------------------------------------------------------

/**
 * The key awaiting revoke confirmation, and `handleRevoke`, which revokes it, clears the
 * confirmation and reloads the list through `refresh`.
 */
export function useRevokeKey(refresh: () => Promise<void>) {
  const [revokeTarget, setRevokeTarget] = useState<SdkApiKey | null>(null);
  const [revokeLoading, setRevokeLoading] = useState(false);

  async function handleRevoke() {
    if (!revokeTarget) return;
    setRevokeLoading(true);
    try {
      await apiClient.request(`/api/api-keys/${revokeTarget.id}`, {
        method: "DELETE",
      });
      setRevokeTarget(null);
      await refresh();
      toast.success("API key revoked");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to revoke key";
      toast.error(msg);
      console.error("[ApiKeyManager] revoke error:", err);
    } finally {
      setRevokeLoading(false);
    }
  }

  return { revokeTarget, setRevokeTarget, revokeLoading, handleRevoke };
}
