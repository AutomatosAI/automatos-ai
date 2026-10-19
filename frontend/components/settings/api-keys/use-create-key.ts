"use client";

/**
 * The create-key form's state and minting a key from it (`POST /api/api-keys`, under step-up).
 */
import { useState } from "react";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import { themeErrorMessage, useStepUp } from "@/lib/theme-host/api";
import type {
  CreateKeyPayload,
  CreateKeyResponse,
  ExpiryOption,
  KeyType,
  Permission,
} from "./api-key-types";
import { buildCreatePayload, createFormError } from "./create-key-payload";

/**
 * PRD-261 A-07f (#1157): a new key is a standing credential, so in SaaS minting one needs a
 * sign-in from the last few minutes (orchestrator/api/api_keys.py `STEP_UP`). Called through
 * `useStepUp`, so a refusal opens Clerk's "confirm it's you" and the key is requested once more;
 * the local edition sends it as is.
 */
function postApiKey(payload: CreateKeyPayload) {
  return apiClient.post<CreateKeyResponse>("/api/api-keys", payload);
}

// ---------------------------------------------------------------------------
// Create dialog state
// ---------------------------------------------------------------------------

/** The create form's fields with their setters, `resetForm` and `togglePermission`. */
export function useCreateKeyForm() {
  const [formName, setFormName] = useState("");
  const [formKeyType, setFormKeyType] = useState<KeyType>("server");
  const [formPermissions, setFormPermissions] = useState<Set<Permission>>(new Set());
  const [formDomains, setFormDomains] = useState("");
  const [formRateLimit, setFormRateLimit] = useState("");
  const [formExpiry, setFormExpiry] = useState<ExpiryOption>("none");

  function resetForm() {
    setFormName("");
    setFormKeyType("server");
    setFormPermissions(new Set());
    setFormDomains("");
    setFormRateLimit("");
    setFormExpiry("none");
  }

  function togglePermission(perm: Permission) {
    setFormPermissions((prev) => {
      const next = new Set(prev);
      if (next.has(perm)) {
        next.delete(perm);
      } else {
        next.add(perm);
      }
      return next;
    });
  }

  return {
    formName, setFormName,
    formKeyType, setFormKeyType,
    formPermissions, setFormPermissions,
    formDomains, setFormDomains,
    formRateLimit, setFormRateLimit,
    formExpiry, setFormExpiry,
    resetForm, togglePermission,
  };
}

export type CreateKeyForm = ReturnType<typeof useCreateKeyForm>;

// ---------------------------------------------------------------------------
// Create key
// ---------------------------------------------------------------------------

interface CreateKeyOptions {
  /** The dialog's form (`useCreateKeyForm`). */
  form: CreateKeyForm;
  /** Shows the new key's secret, once. */
  onCreated: (key: string) => void;
  /** Closes the create dialog. */
  close: () => void;
  /** Reloads the key list. */
  refresh: () => Promise<void>;
}

/**
 * `handleCreate` checks the form, mints the key through `useStepUp(postApiKey)`, then shows the
 * secret, closes and resets the dialog and reloads the list; `createLoading` is true meanwhile.
 */
export function useCreateKey({ form, onCreated, close, refresh }: CreateKeyOptions) {
  const [createLoading, setCreateLoading] = useState(false);
  const createKey = useStepUp(postApiKey);

  async function handleCreate() {
    const invalid = createFormError(form);
    if (invalid) {
      toast.error(invalid);
      return;
    }

    setCreateLoading(true);
    try {
      const payload = buildCreatePayload(form);
      const data = await createKey(payload);
      onCreated(data.key);
      close();
      form.resetForm();
      await refresh();
      toast.success("API key created");
    } catch (err) {
      // The server's sentence, the closed confirmation, or "couldn't confirm it's you".
      toast.error(themeErrorMessage(err));
      console.error("[ApiKeyManager] create error:", err);
    } finally {
      setCreateLoading(false);
    }
  }

  return { createLoading, handleCreate };
}
