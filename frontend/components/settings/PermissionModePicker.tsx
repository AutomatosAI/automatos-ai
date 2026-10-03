'use client'

/**
 * Session permission modes: the four Claude Code users already know, applied by
 * the CLI host's gate to every ticket session on every CLI (orchestrator/core/session_permission_modes.py).
 * Plan included: a CLI without a plan mode of its own presents its plan as its final
 * message, and the plan reaches the operator as the Plan card (PRD-253 Wave P).
 *
 * The workspace picks a default on Settings → Session mode; an agent may override it.
 * Every mode keeps the gate's hard lines: no push or publish, the platform's
 * secrets out of reach, and the session sandbox around every command.
 */

import { Code, Hand, ScrollText, Zap, type LucideIcon } from 'lucide-react'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

export type PermissionMode = 'manual' | 'edits' | 'plan' | 'auto'

export interface PermissionModeOption {
  id: PermissionMode
  label: string
  description: string
  icon: LucideIcon
}

export const PERMISSION_MODES: PermissionModeOption[] = [
  { id: 'manual', label: 'Manual', icon: Hand, description: 'Asks for your approval before each edit and each command off the allowlist.' },
  { id: 'edits', label: 'Edit automatically', icon: Code, description: 'Edits files in its folders; asks before a command off the allowlist.' },
  { id: 'plan', label: 'Plan', icon: ScrollText, description: 'Explores and presents a plan; edits start once you approve it.' },
  { id: 'auto', label: 'Auto', icon: Zap, description: 'Runs what passes the safety checks and pauses for anything risky.' },
]

/** The value an agent stores for "use the workspace's default". */
export const WORKSPACE_DEFAULT = ''

export function permissionModeLabel(mode: string | null | undefined): string {
  return PERMISSION_MODES.find((m) => m.id === mode)?.label ?? 'Edit automatically'
}

export function isPermissionMode(value: unknown): value is PermissionMode {
  return PERMISSION_MODES.some((m) => m.id === value)
}

interface PermissionModePickerProps {
  value: PermissionMode | undefined
  disabled?: boolean
  onChange: (mode: PermissionMode) => void
}

/** Settings → Session mode: the workspace's default, one row per mode. */
export function PermissionModePicker({ value, disabled, onChange }: PermissionModePickerProps) {
  return (
    <div className="space-y-2" role="radiogroup" aria-label="Default permission mode for sessions">
      {PERMISSION_MODES.map(({ id, label, description, icon: Icon }) => (
        <label key={id} className="flex items-start gap-2" data-testid={`permission-mode-${id}`}>
          <input
            type="radio"
            name="permission-mode"
            className="mt-0.5"
            checked={value === id}
            disabled={disabled}
            onChange={() => onChange(id)}
          />
          <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-foreground" aria-hidden />
          <span>
            <span className="text-foreground">{label}</span> — {description}
          </span>
        </label>
      ))}
    </div>
  )
}

interface PermissionModeSelectProps {
  value: string
  onChange: (mode: string) => void
}

/** An agent's own mode, or the workspace's default (`WORKSPACE_DEFAULT`). */
export function PermissionModeSelect({ value, onChange }: PermissionModeSelectProps) {
  const current = isPermissionMode(value) ? value : 'default'
  return (
    <div className="space-y-1" data-testid="cli-permission-mode">
      <Label htmlFor="cli-permission-mode" className="text-xs">Permission mode</Label>
      <Select value={current} onValueChange={(next) => onChange(next === 'default' ? WORKSPACE_DEFAULT : next)}>
        <SelectTrigger id="cli-permission-mode"><SelectValue /></SelectTrigger>
        <SelectContent>
          <SelectItem value="default">Workspace default (Settings → Session mode)</SelectItem>
          {PERMISSION_MODES.map((m) => (
            <SelectItem key={m.id} value={m.id}>{m.label} — {m.description}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export default PermissionModePicker
