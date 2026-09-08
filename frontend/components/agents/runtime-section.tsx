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

import { useEffect, useState } from 'react'
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
  /** PRD-239: tickets run in a git worktree of the workspace folder (a single repo), or in the folder itself (a workspace of many repos) */
  cli_worktree: boolean
}

const DEFAULT_CLI_PROVIDER = 'claude'

export const DEFAULT_RUNTIME_FIELDS: RuntimeFields = {
  runtime: 'api',
  cli_provider: DEFAULT_CLI_PROVIDER,
  cli_model: '',
  cli_working_directory: '',
  cli_worktree: true,
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
    cli_worktree: src.cli_worktree !== false,
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
    cli_worktree: cfg.worktree_per_ticket,
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
    worktree_per_ticket: fields.cli_worktree,
  }
}

/** PRD-239 S6: what the backend says a working directory would mean (`GET /api/v1/cli-hosts/workspace-check`). */
export interface WorkspaceCheck {
  path: string
  valid: boolean
  errors: string[]
  explorer_root: string | null
  browsable: boolean
  /** null = no paired host has announced its allowed directories yet */
  allowed: boolean | null
  allowed_roots: string[]
  projects_dir: string | null
}

export interface WorkspaceVerdict {
  tone: 'ok' | 'warn' | 'error'
  text: string
  /** The Canvas root to open when the folder is browsable. */
  canvasRoot: string | null
}

/** One line the operator can act on, from the check result. Pure. */
export function describeWorkspaceCheck(check: WorkspaceCheck): WorkspaceVerdict {
  if (!check.valid) {
    return { tone: 'error', text: check.errors[0] || 'This path cannot be used.', canvasRoot: null }
  }
  if (check.allowed === false) {
    const roots = check.allowed_roots.join(', ')
    return {
      tone: 'error',
      text: `Your CLI host may only run sessions inside: ${roots}. To allow this folder, set LOCAL_PROJECTS_DIR in the stack's .env to it (or a parent of it) and run \`make cli-host-install\` again; the Canvas explorer then shows it as projects/… too.`,
      canvasRoot: null,
    }
  }
  if (check.browsable && check.explorer_root) {
    const where = check.allowed === null ? ' (no host online to confirm it is allowed)' : ''
    return { tone: 'ok', text: `Browsable in the Canvas as ${check.explorer_root}${where}.`, canvasRoot: check.explorer_root }
  }
  const why = check.projects_dir
    ? `outside LOCAL_PROJECTS_DIR (${check.projects_dir}) and the workspace folder`
    : 'LOCAL_PROJECTS_DIR is not set on this instance'
  return {
    tone: 'warn',
    text: `Sessions can run here, but the folder is not browsable from the platform: ${why}. Deliverables stay references only.`,
    canvasRoot: null,
  }
}

const CHECK_DEBOUNCE_MS = 400

/** Ask the backend what a typed working directory means; debounced, never throws. */
function useWorkspaceCheck(path: string, enabled: boolean): { check: WorkspaceCheck | null; loading: boolean } {
  const [check, setCheck] = useState<WorkspaceCheck | null>(null)
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    const trimmed = path.trim()
    if (!enabled || !trimmed) {
      setCheck(null)
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const { apiClient } = await import('@/lib/api-client')
          const result = await apiClient.request<WorkspaceCheck>(
            `/api/v1/cli-hosts/workspace-check?path=${encodeURIComponent(trimmed)}`,
          )
          if (!cancelled) setCheck(result)
        } catch {
          if (!cancelled) setCheck(null)
        } finally {
          if (!cancelled) setLoading(false)
        }
      })()
    }, CHECK_DEBOUNCE_MS)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [path, enabled])
  return { check, loading }
}

const VERDICT_CLASS: Record<WorkspaceVerdict['tone'], string> = {
  ok: 'text-[hsl(var(--success))]',
  warn: 'text-[hsl(var(--warning))]',
  error: 'text-[hsl(var(--destructive))]',
}

interface RuntimeSectionProps {
  value: RuntimeFields
  onChange: <K extends keyof RuntimeFields>(field: K, value: RuntimeFields[K]) => void
}

export function RuntimeSection({ value, onChange }: RuntimeSectionProps) {
  // PRD-239 S6: the verdict on the typed working directory, live.
  const { check, loading } = useWorkspaceCheck(value.cli_working_directory, isLocal && value.runtime === 'cli')
  const verdict = check ? describeWorkspaceCheck(check) : null
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
            <Label htmlFor="cli-working-directory" className="text-xs">Workspace folder (absolute path on your machine, inside a folder your CLI host allows)</Label>
            <Input
              id="cli-working-directory"
              placeholder="/Users/you/Development/your-workspace"
              value={value.cli_working_directory}
              onChange={(e) => onChange('cli_working_directory', e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Claude Code starts here and loads this folder&apos;s CLAUDE.md files; the Canvas explorer opens here. One repo or a whole workspace of repos — your choice. Blank = the host&apos;s default <span className="font-mono">./workspaces</span>.
            </p>
            <label className="flex items-start gap-2 text-xs text-muted-foreground" data-testid="cli-worktree">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={value.cli_worktree}
                onChange={(e) => onChange('cli_worktree', e.target.checked)}
              />
              <span>
                Run each ticket in its own git worktree of this folder (your checkout stays untouched; sessions never push). Turn this off for a workspace of many repos — its own git tracks next to nothing, so a worktree would be empty. Your own sessions from the agent menu always run in the folder itself.
              </span>
            </label>
            {/* PRD-239 S6: what this folder means — valid, allowed by the host, browsable in the Canvas */}
            {value.cli_working_directory.trim() && (
              <p className="text-xs" data-testid="workspace-check">
                {loading && !verdict ? (
                  <span className="text-muted-foreground">Checking…</span>
                ) : verdict ? (
                  <span className={VERDICT_CLASS[verdict.tone]}>
                    {verdict.text}
                    {verdict.canvasRoot && (
                      <>
                        {' '}
                        <a
                          href={`/chat?repo=${encodeURIComponent(verdict.canvasRoot)}`}
                          className="underline underline-offset-2"
                          data-testid="workspace-open-canvas"
                        >
                          Open in the Canvas
                        </a>
                      </>
                    )}
                  </span>
                ) : (
                  <span className="text-muted-foreground">Could not check this folder right now (is session mode on?).</span>
                )}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default RuntimeSection
