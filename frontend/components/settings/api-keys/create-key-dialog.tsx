"use client";

/**
 * The "Create API Key" dialog. It owns the form; closing it from the dialog itself (Escape, the
 * overlay, the close button) clears the form, while Cancel only closes it.
 */
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Loader2 } from "lucide-react";
import {
  AllowedDomainsField,
  ExpiryField,
  KeyNameField,
  KeyPermissionsField,
  KeyTypeField,
  RateLimitField,
} from "./create-key-fields";
import { useCreateKey, useCreateKeyForm } from "./use-create-key";

interface CreateKeyDialogProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  /** Shows the new key's secret, once. */
  onCreated: (key: string) => void;
  /** Reloads the key list after a key is created. */
  refresh: () => Promise<void>;
}

/** The create-key form in a dialog; Create Key mints the key under step-up. */
export function CreateKeyDialog({ open, setOpen, onCreated, refresh }: CreateKeyDialogProps) {
  const form = useCreateKeyForm();
  const { createLoading, handleCreate } = useCreateKey({
    form,
    onCreated,
    close: () => setOpen(false),
    refresh,
  });

  return (
    <Dialog
      open={open}
      onOpenChange={(open) => {
        setOpen(open);
        if (!open) form.resetForm();
      }}
    >
      <DialogContent className="sm:max-w-lg max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create API Key</DialogTitle>
          <DialogDescription>
            Generate a new API key for SDK access to this workspace.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 py-4">
          {/* Name */}
          <KeyNameField form={form} />

          {/* Key Type */}
          <KeyTypeField form={form} />

          {/* Permissions */}
          <KeyPermissionsField form={form} />

          {/* Allowed Domains (public keys only) */}
          {form.formKeyType === "public" && <AllowedDomainsField form={form} />}

          {/* Rate Limit */}
          <RateLimitField form={form} />

          {/* Expiry */}
          <ExpiryField form={form} />
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setOpen(false)}
            disabled={createLoading}
          >
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={createLoading}>
            {createLoading && (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            )}
            Create Key
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
