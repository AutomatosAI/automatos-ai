"use client";

/**
 * The "Revoke API Key" confirmation: names the key and warns that revoking cannot be undone.
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
import { AlertTriangle, Loader2 } from "lucide-react";
import type { SdkApiKey } from "./api-key-types";

interface RevokeKeyDialogProps {
  /** The key to revoke; the dialog is open while it is set. */
  revokeTarget: SdkApiKey | null;
  setRevokeTarget: (key: SdkApiKey | null) => void;
  revokeLoading: boolean;
  /** Revokes `revokeTarget` (`useRevokeKey`). */
  handleRevoke: () => void;
}

/** Asks to confirm revoking a key; Cancel or closing the dialog keeps it. */
export function RevokeKeyDialog({ revokeTarget, setRevokeTarget, revokeLoading, handleRevoke }: RevokeKeyDialogProps) {
  return (
    <Dialog
      open={!!revokeTarget}
      onOpenChange={(open) => {
        if (!open) setRevokeTarget(null);
      }}
    >
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            Revoke API Key
          </DialogTitle>
          <DialogDescription>
            Are you sure you want to revoke{" "}
            <span className="font-medium text-foreground">
              {revokeTarget?.name}
            </span>
            ? This action cannot be undone. Any applications using this key
            will immediately lose access.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setRevokeTarget(null)}
            disabled={revokeLoading}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleRevoke}
            disabled={revokeLoading}
          >
            {revokeLoading && (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            )}
            Revoke Key
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
