"use client";

/**
 * The one-time "API Key Created" dialog: the new secret with a copy button and the warning that
 * it will not be shown again.
 */
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Copy, AlertTriangle, Check } from "lucide-react";

/** How long the copy button shows its tick after a copy. */
const COPIED_FEEDBACK_MS = 2000;

interface CreatedKeyDialogProps {
  /** The new key's secret; the dialog is open while it is set. */
  createdKey: string | null;
  /** Forgets the secret (closes the dialog). */
  onClose: () => void;
}

/** Shows a just-created key's secret once, with copy-to-clipboard; Done forgets it. */
export function CreatedKeyDialog({ createdKey, onClose }: CreatedKeyDialogProps) {
  const [copied, setCopied] = useState(false);

  // ---------------------------------------------------------------------------
  // Copy to clipboard
  // ---------------------------------------------------------------------------

  function handleCopy(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), COPIED_FEEDBACK_MS);
    });
  }

  function close() {
    onClose();
    setCopied(false);
  }

  return (
    <Dialog
      open={!!createdKey}
      onOpenChange={(open) => {
        if (!open) close();
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Check className="h-5 w-5 text-success" />
            API Key Created
          </DialogTitle>
          <DialogDescription>
            Copy your API key now. It will not be shown again.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Warning */}
          <div className="rounded-lg border border-warning/30 bg-warning/5 p-3 flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
            <p className="text-xs text-warning/80">
              This key will only be shown once. Copy it now and store it
              securely. You will not be able to retrieve it later.
            </p>
          </div>

          {/* Key display */}
          <div className="flex items-center gap-2">
            <Input
              readOnly
              value={createdKey ?? ""}
              className="font-mono text-xs"
            />
            <Button
              variant="outline"
              size="icon"
              className={`shrink-0 transition-colors ${
                copied
                  ? "text-success border-green-500"
                  : ""
              }`}
              onClick={() => createdKey && handleCopy(createdKey)}
            >
              {copied ? (
                <Check className="h-4 w-4" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        <DialogFooter>
          <Button onClick={close}>
            Done
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
