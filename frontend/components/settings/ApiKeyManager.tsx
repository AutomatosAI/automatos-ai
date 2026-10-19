"use client";

/**
 * Settings › Widget SDK › API keys: lists the workspace's SDK keys, creates one (under step-up,
 * PRD-261 A-07f), shows its secret once and revokes keys. The pieces live in ./api-keys/.
 */
import { useState } from "react";
import { ApiKeysHeaderCard, ApiKeysListCard } from "./api-keys/api-keys-list";
import { CreateKeyDialog } from "./api-keys/create-key-dialog";
import { CreatedKeyDialog } from "./api-keys/created-key-dialog";
import { RevokeKeyDialog } from "./api-keys/revoke-key-dialog";
import { useApiKeys, useRevokeKey } from "./api-keys/use-api-keys";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** The SDK API Keys manager: the key list and its create, created-key and revoke dialogs. */
export function ApiKeyManager() {
  // Key list state
  const { keys, isLoading, error, fetchKeys } = useApiKeys();

  // Create dialog state
  const [createOpen, setCreateOpen] = useState(false);

  // Key created success dialog
  const [createdKey, setCreatedKey] = useState<string | null>(null);

  // Revoke confirmation
  const { revokeTarget, setRevokeTarget, revokeLoading, handleRevoke } = useRevokeKey(fetchKeys);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* Header */}
      <ApiKeysHeaderCard onCreate={() => setCreateOpen(true)} />

      {/* Key List */}
      <ApiKeysListCard
        keys={keys}
        isLoading={isLoading}
        error={error}
        onRetry={fetchKeys}
        onCreate={() => setCreateOpen(true)}
        onRevoke={setRevokeTarget}
      />

      {/* ------------------------------------------------------------------ */}
      {/* Create Key Dialog                                                   */}
      {/* ------------------------------------------------------------------ */}
      <CreateKeyDialog
        open={createOpen}
        setOpen={setCreateOpen}
        onCreated={setCreatedKey}
        refresh={fetchKeys}
      />

      {/* ------------------------------------------------------------------ */}
      {/* Key Created Success Dialog                                          */}
      {/* ------------------------------------------------------------------ */}
      <CreatedKeyDialog
        createdKey={createdKey}
        onClose={() => setCreatedKey(null)}
      />

      {/* ------------------------------------------------------------------ */}
      {/* Revoke Confirmation Dialog                                          */}
      {/* ------------------------------------------------------------------ */}
      <RevokeKeyDialog
        revokeTarget={revokeTarget}
        setRevokeTarget={setRevokeTarget}
        revokeLoading={revokeLoading}
        handleRevoke={handleRevoke}
      />
    </div>
  );
}
