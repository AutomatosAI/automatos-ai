'use client'

/**
 * PRD-234 S4 — the Runtime group of an agent's Model step.
 *
 * Shared by the create wizard and the configuration modal so both offer the
 * same choice: an API model (the default — this workspace's keys or OpenRouter)
 * or the user's own Claude Code session on their machine. Local edition only:
 * in saas the group does not render and every agent stays `api`.
 *
 * The fields ride `Agent.configuration` (runtime / provider / model /
 * working_directory). The backend validates them (`core/cli_runtime.py`), so a
 * bad alias is refused at save, not discovered at claim.
 */

import { TerminalSquare } from 'lucide-react'
import { isLocal } from '@/lib/auth-edition'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

export type RuntimeKind = 'api' | 'cli'

export interface RuntimeFields {
  runtime: RuntimeKind
  cli_provider: string
  cli_model: string
  cli_working_directory: string
}

const DEFAULT_CLI_PROVIDER = 'claude'

export const DEFAULT_RUNTIME_FIELDS: RuntimeFields = {
  runtime: 'api',
  cli_provider: DEFAULT_CLI_PROVIDER,
  cli_model: '',
  cli_working_directory: '',
}

/** The aliases Claude Code itself resolves (`claude --model`); a full `claude-…` id also works. */
export const CLAUDE_MODEL_ALIASES = ['fable', 'opus', 'sonnet', 'haiku'] as const

function str(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/** Coerce a loosely-typed source (a form state, a decoded JSON) into RuntimeFields. */
export function normalizeRuntimeFields(source: object | null | undefined): RuntimeFields {
  const src = (source ?? {}) as Record<string, unknown>
  return {
    runtime: src.runtime === 'cli' ? 'cli' : 'api',
    cli_provider: str(src.cli_provider) || DEFAULT_CLI_PROVIDER,
    cli_model: str(src.cli_model),
    cli_working_directory: str(src.cli_working_directory),
  }
}

/** The fields as stored on an agent's `configuration` JSON. */
export function runtimeFieldsFromConfiguration(configuration: object | null | undefined): RuntimeFields {
  const cfg = (configuration ?? {}) as Record<string, unknown>
  return normalizeRuntimeFields({
    runtime: cfg.runtime,
    cli_provider: cfg.provider,
    cli_model: cfg.model,
    cli_working_directory: cfg.working_directory,
  })
}

/**
 * The configuration fragment to save. An api agent carries only `runtime: 'api'`;
 * a cli agent carries provider/model/working_directory, blanks as null so the
 * host falls back to the CLI's default model and its default directory.
 */
export function runtimeConfiguration(fields: RuntimeFields): Record<string, unknown> {
  if (fields.runtime !== 'cli') return { runtime: 'api' }
  return {
    runtime: 'cli',
    provider: fields.cli_provider || DEFAULT_CLI_PROVIDER,
    model: fields.cli_model.trim() || null,
    working_directory: fields.cli_working_directory.trim() || null,
  }
}

interface RuntimeSectionProps {
  value: RuntimeFields
  onChange: <K extends keyof RuntimeFields>(field: K, value: RuntimeFields[K]) => void
}

export function RuntimeSection({ value, onChange }: RuntimeSectionProps) {
  if (!isLocal) return null
  return (
    <div className="space-y-4 rounded-lg border border-border/40 p-4" data-testid="runtime-section">
      <div className="flex items-center gap-2">
        <TerminalSquare className="h-4 w-4 text-[hsl(var(--agent))]" />
        <Label className="text-sm font-medium">Runtime</Label>
      </div>
      <Select
        value={value.runtime}
        onValueChange={(next) => onChange('runtime', next === 'cli' ? 'cli' : 'api')}
      >
        <SelectTrigger aria-label="Runtime">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="api">API model (this workspace&apos;s keys or OpenRouter)</SelectItem>
          <SelectItem value="cli">Claude Code session (your own login, on your machine)</SelectItem>
        </SelectContent>
      </Select>
      {value.runtime === 'cli' && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            Tickets for this agent are run by your paired CLI host as interactive Claude Code sessions.
            The model settings below do not apply to sessions.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label htmlFor="cli-provider" className="text-xs">CLI</Label>
              <Select
                value={value.cli_provider || DEFAULT_CLI_PROVIDER}
                onValueChange={(next) => onChange('cli_provider', next)}
              >
                <SelectTrigger id="cli-provider"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="claude">Claude Code</SelectItem>
                  <SelectItem value="codex">Codex (S5 — not yet served by the host)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label htmlFor="cli-model" className="text-xs">Model (optional)</Label>
              <Input
                id="cli-model"
                placeholder={`${CLAUDE_MODEL_ALIASES.join(' · ')} · or a full id such as claude-opus-5`}
                value={value.cli_model}
                onChange={(e) => onChange('cli_model', e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Claude Code&apos;s own aliases, lowercase. Blank = the CLI&apos;s default. The model must be
                available to your login; it is not one of the API models below.
              </p>
            </div>
          </div>
          <div className="space-y-1">
            <Label htmlFor="cli-working-directory" className="text-xs">Working directory (absolute path, inside a directory the host registered)</Label>
            <Input
              id="cli-working-directory"
              placeholder="/Users/you/Development/your-repo"
              value={value.cli_working_directory}
              onChange={(e) => onChange('cli_working_directory', e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Blank = the host&apos;s default <span className="font-mono">./workspaces</span>. Git repositories get their own worktree per session; sessions never push.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default RuntimeSection
