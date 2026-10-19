"use client";

/**
 * The SDK API Keys header card and the key list card: loading, error with Retry, the empty
 * state, or the keys table.
 */
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Key, Plus, AlertTriangle, Loader2 } from "lucide-react";
import type { SdkApiKey } from "./api-key-types";
import { ApiKeysTable } from "./api-keys-table";

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/** The title card with the "Create API Key" button. */
export function ApiKeysHeaderCard({ onCreate }: { onCreate: () => void }) {
  return (
    <Card className="glass-card border-border/40">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-primary/10 p-2">
              <Key className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-xl">SDK API Keys</CardTitle>
              <CardDescription className="mt-1">
                Create and manage API keys for programmatic access to your
                workspace via the SDK.
              </CardDescription>
            </div>
          </div>
          <Button onClick={onCreate}>
            <Plus className="h-4 w-4 mr-2" />
            Create API Key
          </Button>
        </div>
      </CardHeader>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Key list states
// ---------------------------------------------------------------------------

function ApiKeysLoading() {
  return (
    <div className="flex items-center justify-center py-16">
      <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      <span className="ml-2 text-sm text-muted-foreground">
        Loading API keys...
      </span>
    </div>
  );
}

function ApiKeysError({ error, onRetry }: { error: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
      <AlertTriangle className="h-6 w-6 mb-2 text-destructive" />
      <p className="text-sm">{error}</p>
      <Button
        variant="outline"
        size="sm"
        className="mt-3"
        onClick={onRetry}
      >
        Retry
      </Button>
    </div>
  );
}

function ApiKeysEmpty({ onCreate }: { onCreate: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
      <Key className="h-10 w-10 mb-3 opacity-30" />
      <p className="text-sm font-medium">No API keys yet</p>
      <p className="text-xs mt-1">
        Create your first API key to start using the SDK.
      </p>
      <Button
        variant="outline"
        size="sm"
        className="mt-4"
        onClick={onCreate}
      >
        <Plus className="h-4 w-4 mr-2" />
        Create API Key
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Key list
// ---------------------------------------------------------------------------

interface ApiKeysListProps {
  keys: SdkApiKey[];
  isLoading: boolean;
  error: string | null;
  onRetry: () => void;
  onCreate: () => void;
  onRevoke: (key: SdkApiKey) => void;
}

function ApiKeysListBody({ keys, isLoading, error, onRetry, onCreate, onRevoke }: ApiKeysListProps) {
  if (isLoading) return <ApiKeysLoading />;
  if (error) return <ApiKeysError error={error} onRetry={onRetry} />;
  if (keys.length === 0) return <ApiKeysEmpty onCreate={onCreate} />;
  return <ApiKeysTable keys={keys} onRevoke={onRevoke} />;
}

/** The key list card: loading, the load error with Retry, the empty state, or the table. */
export function ApiKeysListCard(props: ApiKeysListProps) {
  return (
    <Card className="glass-card border-border/40">
      <CardContent className="pt-6">
        <ApiKeysListBody {...props} />
      </CardContent>
    </Card>
  );
}
