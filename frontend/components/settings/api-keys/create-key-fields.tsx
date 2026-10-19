"use client";

/**
 * The create-key form's fields: name, key type, permissions, allowed domains (public keys),
 * rate limit and expiry. Each reads and writes the dialog's `useCreateKeyForm` state.
 */
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { ALL_PERMISSIONS, EXPIRY_OPTIONS, type ExpiryOption, type KeyType } from "./api-key-types";
import type { CreateKeyForm } from "./use-create-key";

interface FieldProps {
  form: CreateKeyForm;
}

/** The required key name. */
export function KeyNameField({ form }: FieldProps) {
  const { formName, setFormName } = form;
  return (
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
  );
}

/** Public (browser, domain-restricted) or server key, with the hint for the one chosen. */
export function KeyTypeField({ form }: FieldProps) {
  const { formKeyType, setFormKeyType } = form;
  return (
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
  );
}

/** One checkbox per permission, with Select All and Clear All. */
export function KeyPermissionsField({ form }: FieldProps) {
  const { formPermissions, setFormPermissions, togglePermission } = form;
  return (
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
  );
}

/** The domains a public key may be used from, one per line (required for public keys). */
export function AllowedDomainsField({ form }: FieldProps) {
  const { formDomains, setFormDomains } = form;
  return (
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
  );
}

/** Requests per minute; empty means no limit. */
export function RateLimitField({ form }: FieldProps) {
  const { formRateLimit, setFormRateLimit } = form;
  return (
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
  );
}

/** No expiry, 30 days, 90 days or 1 year. */
export function ExpiryField({ form }: FieldProps) {
  const { formExpiry, setFormExpiry } = form;
  return (
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
  );
}
