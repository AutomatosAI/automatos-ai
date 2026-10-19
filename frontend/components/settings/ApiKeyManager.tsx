"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Copy, Key, Plus, Trash2, AlertTriangle, Check, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type KeyType = "public" | "server";

type Permission =
  | "chat"
  | "documents:read"
  | "documents:write"
  | "data:query"
  | "data:execute"
  | "agents:read"
  | "agents:execute"
  | "missions:read"
  | "missions:execute"
  | "playbooks:read"
  | "playbooks:execute";

type ExpiryOption = "none" | "30d" | "90d" | "1y";

interface SdkApiKey {
  id: string;
  name: string;
  key_prefix: string;
  key_type: KeyType;
  permissions: Permission[];
  allowed_domains: string[] | null;
  rate_limit_requests: number | null;
  expires_at: string | null;
  created_at: string;
  last_used_at: string | null;
  is_active: boolean;
}

interface CreateKeyPayload {
  name: string;
  key_type: KeyType;
  permissions: Permission[];
  allowed_domains?: string[];
  rate_limit_requests?: number | null;
  expires_at?: string | null;
}

interface CreateKeyResponse {
  id: string;
  name: string;
  key: string;
  key_type: string;
  permissions: string[];
  created_at: string | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ALL_PERMISSIONS: { value: Permission; label: string; group: string }[] = [
  { value: "chat", label: "Chat", group: "Chat" },
  { value: "documents:read", label: "Documents: Read", group: "Documents" },
  { value: "documents:write", label: "Documents: Write", group: "Documents" },
  { value: "data:query", label: "Data: Query", group: "Data" },
  { value: "data:execute", label: "Data: Execute", group: "Data" },
  { value: "agents:read", label: "Agents: Read", group: "Agents" },
  { value: "agents:execute", label: "Agents: Execute", group: "Agents" },
  { value: "missions:read", label: "Missions: Read", group: "Missions" },
  { value: "missions:execute", label: "Missions: Execute", group: "Missions" },
  { value: "playbooks:read", label: "Playbooks: Read", group: "Playbooks" },
  { value: "playbooks:execute", label: "Playbooks: Execute", group: "Playbooks" },
];

const EXPIRY_OPTIONS: { value: ExpiryOption; label: string }[] = [
  { value: "none", label: "No expiry" },
  { value: "30d", label: "30 days" },
  { value: "90d", label: "90 days" },
  { value: "1y", label: "1 year" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso: string | null): string {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatRelative(iso: string | null): string {
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

function isExpired(expiresAt: string | null): boolean {
  if (!expiresAt) return false;
  return new Date(expiresAt) < new Date();
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ApiKeyManager() {
  // Key list state
  const [keys, setKeys] = useState<SdkApiKey[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create dialog state
  const [createOpen, setCreateOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [formName, setFormName] = useState("");
  const [formKeyType, setFormKeyType] = useState<KeyType>("server");
  const [formPermissions, setFormPermissions] = useState<Set<Permission>>(new Set());
  const [formDomains, setFormDomains] = useState("");
  const [formRateLimit, setFormRateLimit] = useState("");
  const [formExpiry, setFormExpiry] = useState<ExpiryOption>("none");

  // Key created success dialog
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Revoke confirmation
  const [revokeTarget, setRevokeTarget] = useState<SdkApiKey | null>(null);
  const [revokeLoading, setRevokeLoading] = useState(false);

  // ---------------------------------------------------------------------------
  // Fetch keys (uses apiClient which handles Clerk JWT + workspace ID)
  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------
  // Create key
  // ---------------------------------------------------------------------------

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

  async function handleCreate() {
    if (!formName.trim()) {
      toast.error("Key name is required");
      return;
    }
    if (formPermissions.size === 0) {
      toast.error("Select at least one permission");
      return;
    }
    if (formKeyType === "public" && !formDomains.trim()) {
      toast.error("Allowed domains are required for public keys");
      return;
    }

    setCreateLoading(true);
    try {
      const payload: CreateKeyPayload = {
        name: formName.trim(),
        key_type: formKeyType,
        permissions: Array.from(formPermissions),
      };

      if (formKeyType === "public" && formDomains.trim()) {
        payload.allowed_domains = formDomains
          .split("\n")
          .map((d) => d.trim())
          .filter(Boolean);
      }

      const rateLimitNum = parseInt(formRateLimit, 10);
      if (!isNaN(rateLimitNum) && rateLimitNum > 0) {
        payload.rate_limit_requests = rateLimitNum;
      }

      if (formExpiry !== "none") {
        const now = new Date();
        const expiryMap: Record<string, number> = {
          "30d": 30,
          "90d": 90,
          "1y": 365,
        };
        const days = expiryMap[formExpiry] ?? 0;
        if (days > 0) {
          const exp = new Date(now.getTime() + days * 86400000);
          payload.expires_at = exp.toISOString();
        }
      }

      const data = await apiClient.request<CreateKeyResponse>("/api/api-keys", {
        method: "POST",
        body: payload as any,
      });
      setCreatedKey(data.key);
      setCreateOpen(false);
      resetForm();
      await fetchKeys();
      toast.success("API key created");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to create key";
      toast.error(msg);
      console.error("[ApiKeyManager] create error:", err);
    } finally {
      setCreateLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Revoke key
  // ---------------------------------------------------------------------------

  async function handleRevoke() {
    if (!revokeTarget) return;
    setRevokeLoading(true);
    try {
      await apiClient.request(`/api/api-keys/${revokeTarget.id}`, {
        method: "DELETE",
      });
      setRevokeTarget(null);
      await fetchKeys();
      toast.success("API key revoked");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to revoke key";
      toast.error(msg);
      console.error("[ApiKeyManager] revoke error:", err);
    } finally {
      setRevokeLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Copy to clipboard
  // ---------------------------------------------------------------------------

  function handleCopy(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  // ---------------------------------------------------------------------------
  // Key status helper
  // ---------------------------------------------------------------------------

  function getKeyStatus(key: SdkApiKey): {
    label: string;
    variant: "default" | "secondary" | "destructive" | "outline";
    className: string;
  } {
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

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* Header */}
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
            <Button onClick={() => setCreateOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create API Key
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* Key List */}
      <Card className="glass-card border-border/40">
        <CardContent className="pt-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">
                Loading API keys...
              </span>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <AlertTriangle className="h-6 w-6 mb-2 text-destructive" />
              <p className="text-sm">{error}</p>
              <Button
                variant="outline"
                size="sm"
                className="mt-3"
                onClick={fetchKeys}
              >
                Retry
              </Button>
            </div>
          ) : keys.length === 0 ? (
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
                onClick={() => setCreateOpen(true)}
              >
                <Plus className="h-4 w-4 mr-2" />
                Create API Key
              </Button>
            </div>
          ) : (
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
                  {keys.map((key) => {
                    const status = getKeyStatus(key);
                    return (
                      <TableRow key={key.id}>
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
                          <div className="flex flex-wrap gap-1 max-w-[200px]">
                            {key.permissions.length <= 3 ? (
                              key.permissions.map((p) => (
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
                                {key.permissions.slice(0, 2).map((p) => (
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
                                  +{key.permissions.length - 2} more
                                </Badge>
                              </>
                            )}
                          </div>
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
                            onClick={() => setRevokeTarget(key)}
                            title="Revoke key"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ------------------------------------------------------------------ */}
      {/* Create Key Dialog                                                   */}
      {/* ------------------------------------------------------------------ */}
      <Dialog
        open={createOpen}
        onOpenChange={(open) => {
          setCreateOpen(open);
          if (!open) resetForm();
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
            <div className="space-y-2">
              <Label htmlFor="key-name">
                Name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="key-name"
                placeholder="e.g. Production Backend"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                autoComplete="off"
              />
            </div>

            {/* Key Type */}
            <div className="space-y-2">
              <Label htmlFor="key-type">Key Type</Label>
              <Select
                value={formKeyType}
                onValueChange={(v) => setFormKeyType(v as KeyType)}
              >
                <SelectTrigger id="key-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="public">
                    Public — for client-side / browser use
                  </SelectItem>
                  <SelectItem value="server">
                    Server — for backend / server-side use
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {formKeyType === "public"
                  ? "Public keys are restricted by domain and have limited permissions."
                  : "Server keys should never be exposed in client-side code."}
              </p>
            </div>

            {/* Permissions */}
            <div className="space-y-3">
              <Label>
                Permissions <span className="text-destructive">*</span>
              </Label>
              <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                {ALL_PERMISSIONS.map((perm) => (
                  <div
                    key={perm.value}
                    className="flex items-center space-x-2"
                  >
                    <Checkbox
                      id={`perm-${perm.value}`}
                      checked={formPermissions.has(perm.value)}
                      onCheckedChange={() => togglePermission(perm.value)}
                    />
                    <Label
                      htmlFor={`perm-${perm.value}`}
                      className="text-sm font-normal cursor-pointer"
                    >
                      {perm.label}
                    </Label>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={() =>
                    setFormPermissions(
                      new Set(ALL_PERMISSIONS.map((p) => p.value))
                    )
                  }
                >
                  Select All
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={() => setFormPermissions(new Set())}
                >
                  Clear All
                </Button>
              </div>
            </div>

            {/* Allowed Domains (public keys only) */}
            {formKeyType === "public" && (
              <div className="space-y-2">
                <Label htmlFor="allowed-domains">
                  Allowed Domains <span className="text-destructive">*</span>
                </Label>
                <Textarea
                  id="allowed-domains"
                  placeholder={"example.com\napp.example.com\nlocalhost:3000"}
                  value={formDomains}
                  onChange={(e) => setFormDomains(e.target.value)}
                  rows={3}
                />
                <p className="text-xs text-muted-foreground">
                  One domain per line. Requests from other domains will be
                  rejected.
                </p>
              </div>
            )}

            {/* Rate Limit */}
            <div className="space-y-2">
              <Label htmlFor="rate-limit">Rate Limit (requests/minute)</Label>
              <Input
                id="rate-limit"
                type="number"
                min={0}
                placeholder="No limit"
                value={formRateLimit}
                onChange={(e) => setFormRateLimit(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Leave empty for no rate limit.
              </p>
            </div>

            {/* Expiry */}
            <div className="space-y-2">
              <Label htmlFor="expiry">Expiry</Label>
              <Select
                value={formExpiry}
                onValueChange={(v) => setFormExpiry(v as ExpiryOption)}
              >
                <SelectTrigger id="expiry">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {EXPIRY_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setCreateOpen(false)}
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

      {/* ------------------------------------------------------------------ */}
      {/* Key Created Success Dialog                                          */}
      {/* ------------------------------------------------------------------ */}
      <Dialog
        open={!!createdKey}
        onOpenChange={(open) => {
          if (!open) {
            setCreatedKey(null);
            setCopied(false);
          }
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
            <Button
              onClick={() => {
                setCreatedKey(null);
                setCopied(false);
              }}
            >
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ------------------------------------------------------------------ */}
      {/* Revoke Confirmation Dialog                                          */}
      {/* ------------------------------------------------------------------ */}
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
    </div>
  );
}
