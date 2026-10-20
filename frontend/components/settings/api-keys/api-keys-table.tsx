"use client";

/**
 * The SDK keys table: one row per key with its prefix, type, permissions, dates, status and
 * the revoke button.
 */
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Trash2 } from "lucide-react";
import type { Permission, SdkApiKey } from "./api-key-types";
import { formatDate, formatRelative, getKeyStatus } from "./api-key-format";

// ---------------------------------------------------------------------------
// Permissions cell
// ---------------------------------------------------------------------------

/** Up to three permission badges; more than three shows the first two and "+N more". */
function PermissionBadges({ permissions }: { permissions: Permission[] }) {
  return (
    <div className="flex flex-wrap gap-1 max-w-[200px]">
      {permissions.length <= 3 ? (
        permissions.map((p) => (
          <Badge
            key={p}
            variant="secondary"
            className="text-[10px] px-1.5 py-0"
          >
            {p}
          </Badge>
        ))
      ) : (
        <>
          {permissions.slice(0, 2).map((p) => (
            <Badge
              key={p}
              variant="secondary"
              className="text-[10px] px-1.5 py-0"
            >
              {p}
            </Badge>
          ))}
          <Badge
            variant="secondary"
            className="text-[10px] px-1.5 py-0"
          >
            +{permissions.length - 2} more
          </Badge>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Row
// ---------------------------------------------------------------------------

interface ApiKeyRowProps {
  apiKey: SdkApiKey;
  onRevoke: (key: SdkApiKey) => void;
}

/** One key's row; the revoke button is disabled once the key is inactive. */
function ApiKeyRow({ apiKey: key, onRevoke }: ApiKeyRowProps) {
  const status = getKeyStatus(key);
  return (
    <TableRow>
      <TableCell className="font-medium">
        {key.name}
      </TableCell>
      <TableCell>
        <code className="rounded bg-muted/50 px-2 py-0.5 text-xs font-mono">
          {key.key_prefix}
        </code>
      </TableCell>
      <TableCell>
        <Badge
          variant="outline"
          className={
            key.key_type === "server"
              ? "bg-agent/15 text-agent border-agent/30"
              : "bg-info/15 text-info border-info/30"
          }
        >
          {key.key_type === "server" ? "Server" : "Public"}
        </Badge>
      </TableCell>
      <TableCell>
        <PermissionBadges permissions={key.permissions} />
      </TableCell>
      <TableCell className="text-xs text-muted-foreground">
        {formatDate(key.created_at)}
      </TableCell>
      <TableCell className="text-xs text-muted-foreground">
        {formatRelative(key.last_used_at)}
      </TableCell>
      <TableCell>
        <Badge
          variant={status.variant}
          className={status.className}
        >
          {status.label}
        </Badge>
      </TableCell>
      <TableCell className="text-right">
        <Button
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0 text-destructive hover:text-destructive"
          disabled={!key.is_active}
          onClick={() => onRevoke(key)}
          title="Revoke key"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </TableCell>
    </TableRow>
  );
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

/** The keys table; `onRevoke` asks to confirm revoking a key. */
export function ApiKeysTable({ keys, onRevoke }: { keys: SdkApiKey[]; onRevoke: (key: SdkApiKey) => void }) {
  return (
    <div className="rounded-md border border-border/40">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Key</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Permissions</TableHead>
            <TableHead>Created</TableHead>
            <TableHead>Last Used</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {keys.map((key) => (
            <ApiKeyRow key={key.id} apiKey={key} onRevoke={onRevoke} />
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
