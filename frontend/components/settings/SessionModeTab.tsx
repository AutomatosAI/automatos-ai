'use client'

/**
 * PRD-234 S4 — Settings → Session mode (local edition only).
 *
 * The operator's view of the CLI host lane: is session mode on, which hosts are
 * paired and online, and the one-time pairing code a new host needs. The code
 * is shown ONCE with the exact command to run; the token it is exchanged for
 * never reaches this UI.
 */

import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { TerminalSquare, Copy, Check, RefreshCw, Plug } from 'lucide-react'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

interface HostRow {
  id: string
  name: string
  status: 'pending' | 'paired' | 'revoked'
  online: boolean
  last_seen_at?: string | null
  paired_at?: string | null
  capabilities?: { claude?: { version?: string | null; path?: string | null; onboarded?: boolean } | null } | null
}

interface PairingCode {
  host_id: string
  code: string
  expires_at: string
  pair_command: string
}

/** PRD-239 S6c: Settings → Session mode — where tickets run, and the projects folder the stack was started with. */
export interface SessionModeSettings {
  default_folder: 'projects' | 'sessions'
  default_folder_explicit: boolean
  local_projects_dir: string | null
  projects_mount: string | null
  /** The deliverables root on the host (AUTOMATOS_WORKSPACE_DIR as `make up` exported it), null when unknown. */
  workspace_dir: string | null
  host_allowed_roots: string[]
}

export const PROJECTS_ENV_LINES = (folder: string, deliverables?: string) =>
  `LOCAL_PROJECTS_DIR=${folder}\nLOCAL_PROJECTS_MOUNT=rw` + (deliverables ? `\nAUTOMATOS_WORKSPACE_DIR=${deliverables}` : '')

/** The deliverables root we suggest for a projects folder: one place, beside the repos. */
export const suggestedDeliverablesRoot = (projectsFolder: string) => `${projectsFolder.replace(/\/$/, '')}/deliverables`

const inside = (path: string, roots: string[]) => roots.some((r) => path === r || path.startsWith(r.replace(/\/$/, '') + '/'))

/** The one-line state of the deliverables root — what API agents write and where a session with no folder runs. */
export function describeDeliverablesRoot(s: SessionModeSettings | undefined): { tone: 'ok' | 'warn' | 'muted'; text: string } {
  if (!s) return { tone: 'muted', text: '…' }
  if (!s.workspace_dir) {
    return {
      tone: 'muted',
      text: 'The compose default (./workspaces next to docker-compose.yml). Set AUTOMATOS_WORKSPACE_DIR in .env and start with make up to put it beside your projects.',
    }
  }
  if (!inside(s.workspace_dir, s.host_allowed_roots)) {
    return { tone: 'warn', text: `${s.workspace_dir} — mounted as the workspace root, but your CLI host does not allow it yet: run make cli-host-install again.` }
  }
  return { tone: 'ok', text: `${s.workspace_dir} — mounted as the workspace root; sessions without a folder run in sessions/<ticket> inside it.` }
}

/** The one-line state of the projects folder for the tab. */
export function describeProjectsFolder(s: SessionModeSettings | undefined): { tone: 'ok' | 'warn' | 'muted'; text: string } {
  if (!s) return { tone: 'muted', text: 'Checking…' }
  if (!s.local_projects_dir) return { tone: 'warn', text: 'No projects folder yet — agents can only work in the per-ticket sessions folders.' }
  const allowed = s.host_allowed_roots.some((r) => s.local_projects_dir === r || s.local_projects_dir!.startsWith(r.replace(/\/$/, '') + '/'))
  const mount = s.projects_mount === 'rw' ? 'the Canvas editor can save into it' : 'read-only in the Canvas editor (LOCAL_PROJECTS_MOUNT=rw to save)'
  if (!allowed) return { tone: 'warn', text: `${s.local_projects_dir} — mounted, ${mount}, but your CLI host does not allow it yet: run make cli-host-install again.` }
  return { tone: 'ok', text: `${s.local_projects_dir} — mounted, ${mount}, and your CLI host may run sessions anywhere inside it.` }
}

function useSessionModeHealth() {
  return useQuery({
    queryKey: ['health', 'session-mode'],
    queryFn: () => apiClient.request<{ edition?: string; cli_runtime_enabled?: boolean }>('/health'),
    staleTime: 10_000,
    refetchInterval: 30_000,
  })
}

function useSessionModeSettings(enabled: boolean) {
  return useQuery({
    queryKey: ['cli-hosts', 'settings'],
    queryFn: () => apiClient.request<SessionModeSettings>('/api/v1/cli-hosts/settings'),
    enabled,
    refetchInterval: 30_000,
  })
}

function useCliHosts(enabled: boolean) {
  return useQuery({
    queryKey: ['cli-hosts'],
    queryFn: () => apiClient.request<{ hosts: HostRow[] }>('/api/v1/cli-hosts'),
    enabled,
    refetchInterval: 15_000,
  })
}

function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value)
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        } catch {
          toast.error('Could not copy — select the text and copy it manually')
        }
      }}
      aria-label={label}
    >
      {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
    </Button>
  )
}

export function SessionModeTab() {
  const queryClient = useQueryClient()
  const health = useSessionModeHealth()
  const enabled = health.data?.cli_runtime_enabled === true
  const hosts = useCliHosts(enabled)
  const settings = useSessionModeSettings(enabled)
  const [hostName, setHostName] = useState('')
  const [saving, setSaving] = useState(false)
  const folderState = describeProjectsFolder(settings.data)
  const deliverablesState = describeDeliverablesRoot(settings.data)
  const projectsFolderForEnv = settings.data?.local_projects_dir || '/Users/you/Development'
  const deliverablesForEnv = settings.data?.workspace_dir || suggestedDeliverablesRoot(projectsFolderForEnv)
  const envLines = PROJECTS_ENV_LINES(projectsFolderForEnv, deliverablesForEnv)
  const saveDefaultFolder = async (choice: 'projects' | 'sessions') => {
    setSaving(true)
    try {
      await apiClient.request('/api/v1/cli-hosts/settings', { method: 'PUT', body: JSON.stringify({ default_folder: choice }) })
      queryClient.invalidateQueries({ queryKey: ['cli-hosts', 'settings'] })
      toast.success(choice === 'projects' ? 'New tickets run in your projects folder' : 'New tickets get their own sessions folder')
    } catch (err) {
      toast.error(`Could not save: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setSaving(false)
    }
  }
  const [pairing, setPairing] = useState<PairingCode | null>(null)
  const [minting, setMinting] = useState(false)

  const mintCode = async () => {
    setMinting(true)
    try {
      const code = await apiClient.post<PairingCode>('/api/v1/cli-hosts/pairing-codes', {
        name: hostName.trim() || undefined,
      })
      setPairing(code)
      queryClient.invalidateQueries({ queryKey: ['cli-hosts'] })
    } catch (err) {
      toast.error(`Could not issue a pairing code: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setMinting(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-secondary/30 border-border/30">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <TerminalSquare className="h-5 w-5 text-[hsl(var(--agent))]" />
            Session mode
            {health.isLoading ? null : enabled ? (
              <Badge variant="outline" className="ml-2">on</Badge>
            ) : (
              <Badge variant="secondary" className="ml-2">off</Badge>
            )}
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Agents with the runtime <span className="font-mono">Claude Code session</span> are your own Claude Code, on your
            machine, under your own login. Auto files tickets for them and they work in your folders; you can open any of
            their sessions in the Canvas and type alongside. Nothing runs through an API key.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {!health.isLoading && !enabled && (
            <div className="rounded-lg border border-border/40 p-4 text-sm space-y-2">
              <p className="font-medium">Session mode is off on this instance.</p>
              <p className="text-muted-foreground">
                Add <span className="font-mono">CLI_RUNTIME_ENABLED=true</span> to <span className="font-mono">.env</span>,
                run <span className="font-mono">make up</span>, then come back here to pair a host. Session mode exists in
                the local edition only.
              </p>
            </div>
          )}

          {enabled && (
            <>
              <div>
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">Hosts</Label>
                <div className="mt-2 space-y-2">
                  {(hosts.data?.hosts ?? []).length === 0 && (
                    <p className="text-sm text-muted-foreground">No host paired yet.</p>
                  )}
                  {(hosts.data?.hosts ?? []).map((h) => (
                    <div key={h.id} className="flex items-center justify-between rounded-lg border border-border/40 px-3 py-2 text-sm">
                      <div className="flex items-center gap-3">
                        <span
                          className={
                            'inline-block h-2 w-2 rounded-full ' +
                            (h.status !== 'paired' ? 'bg-muted-foreground' : h.online ? 'bg-[hsl(var(--success))]' : 'bg-[hsl(var(--warning))]')
                          }
                          aria-hidden
                        />
                        <span className="font-medium">{h.name}</span>
                        <span className="text-muted-foreground">
                          {h.status !== 'paired' ? h.status : h.online ? 'connected' : 'not running'}
                          {h.capabilities?.claude?.version ? ` · Claude Code ${h.capabilities.claude.version}` : ''}
                        </span>
                      </div>
                      <span className="font-mono text-xs text-muted-foreground">{h.id.slice(0, 8)}</span>
                    </div>
                  ))}
                </div>
                {hosts.data && hosts.data.hosts.some((h) => h.status === 'paired' && !h.online) && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    A paired host that is not running: start it with <span className="font-mono">make cli-host</span> from the repository.
                  </p>
                )}
              </div>

              <div className="rounded-lg border border-border/40 p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <Plug className="w-4 h-4" />
                  <p className="text-sm font-medium">Connect Claude Code</p>
                </div>
                <ol className="list-decimal pl-5 text-xs text-muted-foreground space-y-1">
                  <li>On the machine with your repositories, run <span className="font-mono">claude</span> once and log in — that is the login every session uses.</li>
                  <li>Get a pairing code here and run the command it gives you from the repository root. It installs a small host service that starts at login.</li>
                  <li>The host appears above as connected. Set your projects folder below, and each agent&apos;s workspace folder in its configuration.</li>
                </ol>
                <div className="flex flex-col sm:flex-row gap-2">
                  <Input
                    placeholder="Name for this machine (optional)"
                    value={hostName}
                    onChange={(e) => setHostName(e.target.value)}
                    className="sm:max-w-xs"
                  />
                  <Button type="button" onClick={mintCode} disabled={minting}>
                    {minting ? <RefreshCw className="w-4 h-4 animate-spin" /> : 'Get a pairing code'}
                  </Button>
                </div>
                {pairing && (
                  <div className="space-y-2 text-sm">
                    <p className="text-muted-foreground">
                      One-time code, valid for ten minutes. Run this on the machine that will run your sessions:
                    </p>
                    <div className="flex items-center gap-2">
                      <code className="flex-1 rounded bg-muted px-3 py-2 font-mono text-xs overflow-x-auto">{pairing.pair_command}</code>
                      <CopyButton value={pairing.pair_command} label="Copy the pair command" />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      From the repository root. The host may run sessions in your deliverables root and in
                      your projects folder (both below).
                    </p>
                  </div>
                )}
              </div>

              <div className="rounded-lg border border-border/40 p-4 space-y-3 text-xs text-muted-foreground" data-testid="projects-folder">
                <p className="text-sm font-medium text-foreground">Your projects folder</p>
                <p className={folderState.tone === 'ok' ? 'text-[hsl(var(--success))]' : folderState.tone === 'warn' ? 'text-[hsl(var(--warning))]' : ''} data-testid="projects-folder-state">
                  {folderState.text}
                </p>
                <p>
                  The folder Automatos may show and work in — one parent for everything: your repositories, a workspace of
                  many repos, anything Claude should be able to open. Each agent then gets its own workspace folder inside it
                  (Agent → Model → Workspace folder); the Canvas explorer and the agent&apos;s Claude Code session both open there.
                </p>
                <p>
                  This one lives in <span className="font-mono">.env</span>, not here: it is a Docker mount, fixed when the
                  containers start, so it cannot be changed from a running page. Set it once:
                </p>
                <div className="flex items-start gap-2">
                  <code className="flex-1 whitespace-pre rounded bg-muted px-3 py-2 font-mono text-xs overflow-x-auto">{envLines}</code>
                  <CopyButton value={envLines} label="Copy the .env lines" />
                </div>
                <p>
                  then <span className="font-mono">make up</span> (remounts it) and <span className="font-mono">make cli-host-install</span> (lets
                  the host run sessions there). Come back here: the line above turns green.
                </p>
              </div>
              <div className="rounded-lg border border-border/40 p-4 space-y-3 text-xs text-muted-foreground" data-testid="deliverables-root">
                <p className="text-sm font-medium text-foreground">Your deliverables root</p>
                <p className={deliverablesState.tone === 'ok' ? 'text-[hsl(var(--success))]' : deliverablesState.tone === 'warn' ? 'text-[hsl(var(--warning))]' : ''} data-testid="deliverables-root-state">
                  {deliverablesState.text}
                </p>
                <p>
                  Everything your agents write lands here — <span className="font-mono">artifacts/</span>, <span className="font-mono">reports/</span>,
                  <span className="font-mono">content/</span>, and <span className="font-mono">sessions/&lt;ticket&gt;</span> for a Claude Code session
                  with no folder of its own. It is mounted as the workspace root, so there is no workspace-id folder on your disk:
                  Deliverables → Explorer, the chat&apos;s Code mode and your sessions all show this one place. Put it beside your
                  projects folder (<span className="font-mono">AUTOMATOS_WORKSPACE_DIR</span> in the <span className="font-mono">.env</span> lines above).
                </p>
              </div>
              <div className="rounded-lg border border-border/40 p-4 space-y-3 text-xs text-muted-foreground" data-testid="default-folder">
                <p className="text-sm font-medium text-foreground">Where a ticket runs when its agent names no folder</p>
                <div className="space-y-2" role="radiogroup" aria-label="Default folder for tickets">
                  <label className="flex items-start gap-2">
                    <input
                      type="radio"
                      name="default-folder"
                      className="mt-0.5"
                      checked={settings.data?.default_folder === 'projects'}
                      disabled={saving || !settings.data?.local_projects_dir}
                      onChange={() => saveDefaultFolder('projects')}
                    />
                    <span>
                      <span className="text-foreground">Your projects folder</span>{settings.data?.local_projects_dir ? '' : ' (set it first)'} — the default. Most
                      tickets fix something in a repository or start a new one; the session runs at the top of your projects folder
                      and works from there.
                    </span>
                  </label>
                  <label className="flex items-start gap-2">
                    <input
                      type="radio"
                      name="default-folder"
                      className="mt-0.5"
                      checked={settings.data?.default_folder === 'sessions'}
                      disabled={saving}
                      onChange={() => saveDefaultFolder('sessions')}
                    />
                    <span>
                      <span className="text-foreground">A fresh folder per ticket</span> — <span className="font-mono">sessions/&lt;ticket&gt;</span> inside your deliverables root.
                      Keeps experiments apart; whatever the session writes there is registered as the ticket&apos;s deliverables.
                    </span>
                  </label>
                </div>
                <p>An agent with its own workspace folder always uses that folder; this only decides for agents without one.</p>
              </div>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>
                  A session uses the unmodified <span className="font-mono">claude</span> on your machine and your own login;
                  no credential is read, copied or set, and sessions never use <span className="font-mono">-p</span>.
                  Anthropic&apos;s terms permit signing in to the unmodified Claude Code binary with your own subscription and
                  assume &quot;ordinary, individual usage&quot; — how many sessions you run in parallel is your choice.
                </p>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default SessionModeTab
