'use client'

/**
 * The sections of Settings → Session mode (SessionModeTab.tsx): the paired hosts,
 * pairing a new one, the two folders the stack was started with, the default
 * permission mode and where a folder-less ticket runs.
 */

import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Copy, Check, RefreshCw, Plug } from 'lucide-react'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { PermissionModePicker, type PermissionMode } from './PermissionModePicker'

export interface HostRow {
  id: string
  name: string
  status: 'pending' | 'paired' | 'revoked'
  online: boolean
  last_seen_at?: string | null
  paired_at?: string | null
  /** Host 0.7.0 (CLI adapter design): every CLI the host knows, served or not, under `clis`; `providers` = the served ids. */
  capabilities?: {
    clis?: Record<string, { version?: string | null; path?: string | null; served?: boolean; reason?: string | null }> | null
    providers?: string[] | null
  } | null
}

export interface PairingCode {
  host_id: string
  code: string
  expires_at: string
  pair_command: string
}


/** A one-line state and its colour, as SessionModeTab's describe* helpers return it. */
export interface FolderState {
  tone: 'ok' | 'warn' | 'muted'
  text: string
}

const TONE_CLASS: Record<FolderState['tone'], string> = {
  ok: 'text-[hsl(var(--success))]',
  warn: 'text-[hsl(var(--warning))]',
  muted: '',
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

export function HostsList({ hosts }: { hosts: HostRow[] }) {
  return (
    <div>
      <Label className="text-xs uppercase tracking-wide text-muted-foreground">Hosts</Label>
      <div className="mt-2 space-y-2">
        {hosts.length === 0 && (
          <p className="text-sm text-muted-foreground">No host paired yet.</p>
        )}
        {hosts.map((h) => (
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
                {h.capabilities?.clis?.claude?.version ? ` · Claude Code ${h.capabilities.clis.claude.version}` : ''}
              </span>
            </div>
            <span className="font-mono text-xs text-muted-foreground">{h.id.slice(0, 8)}</span>
          </div>
        ))}
      </div>
      {hosts.some((h) => h.status === 'paired' && !h.online) && (
        <p className="mt-2 text-xs text-muted-foreground">
          A paired host that is not running: start it with <span className="font-mono">make cli-host</span> from the repository.
        </p>
      )}
    </div>
  )
}

export function ConnectClaudeCode() {
  const queryClient = useQueryClient()
  const [hostName, setHostName] = useState('')
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
  )
}

export function ProjectsFolderCard({ folderState, envLines }: { folderState: FolderState; envLines: string }) {
  return (
    <div className="rounded-lg border border-border/40 p-4 space-y-3 text-xs text-muted-foreground" data-testid="projects-folder">
      <p className="text-sm font-medium text-foreground">Your projects folder</p>
      <p className={TONE_CLASS[folderState.tone]} data-testid="projects-folder-state">
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
  )
}

export function DeliverablesRootCard({ deliverablesState }: { deliverablesState: FolderState }) {
  return (
    <div className="rounded-lg border border-border/40 p-4 space-y-3 text-xs text-muted-foreground" data-testid="deliverables-root">
      <p className="text-sm font-medium text-foreground">Your deliverables root</p>
      <p className={TONE_CLASS[deliverablesState.tone]} data-testid="deliverables-root-state">
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
  )
}

interface SettingCardProps<T> {
  settings: { default_folder?: string; local_projects_dir?: string | null; permission_mode?: PermissionMode } | undefined
  saving: boolean
  onSave: (choice: T) => void
}

export function PermissionModeCard({ settings, saving, onSave }: SettingCardProps<PermissionMode>) {
  return (
    <div className="rounded-lg border border-border/40 p-4 space-y-3 text-xs text-muted-foreground" data-testid="permission-mode">
      <p className="text-sm font-medium text-foreground">How much a session asks before it acts</p>
      <PermissionModePicker value={settings?.permission_mode} disabled={saving || !settings} onChange={onSave} />
      <p>
        The same modes as Claude Code. This is the default for every session agent; an agent can pick its own
        (Agent → Model → Permission mode). In every mode sessions never push or publish, the platform&apos;s secrets stay
        out of reach, and commands run inside the session sandbox. Questions reach you as cards on the ticket.
        Plan needs Claude Code: an agent on another CLI (Codex) runs Plan as Edit automatically.
      </p>
    </div>
  )
}

export function DefaultFolderCard({ settings, saving, onSave }: SettingCardProps<'projects' | 'sessions'>) {
  return (
    <div className="rounded-lg border border-border/40 p-4 space-y-3 text-xs text-muted-foreground" data-testid="default-folder">
      <p className="text-sm font-medium text-foreground">Where a ticket runs when its agent names no folder</p>
      <div className="space-y-2" role="radiogroup" aria-label="Default folder for tickets">
        <label className="flex items-start gap-2">
          <input
            type="radio"
            name="default-folder"
            className="mt-0.5"
            checked={settings?.default_folder === 'projects'}
            disabled={saving || !settings?.local_projects_dir}
            onChange={() => onSave('projects')}
          />
          <span>
            <span className="text-foreground">Your projects folder</span>{settings?.local_projects_dir ? '' : ' (set it first)'} — the
            session runs at the top of your projects folder and can reach every repository in it, and whatever a
            repository&apos;s own scripts load.
          </span>
        </label>
        <label className="flex items-start gap-2">
          <input
            type="radio"
            name="default-folder"
            className="mt-0.5"
            checked={settings?.default_folder === 'sessions'}
            disabled={saving}
            onChange={() => onSave('sessions')}
          />
          <span>
            <span className="text-foreground">A fresh folder per ticket</span> — the default. <span className="font-mono">sessions/&lt;ticket&gt;</span> inside
            your deliverables root: tickets stay apart and out of your repositories, and whatever the session writes there
            is registered as the ticket&apos;s deliverables. An agent that works in a repository names it as its folder.
          </span>
        </label>
      </div>
      <p>An agent with its own workspace folder always uses that folder; this only decides for agents without one.</p>
    </div>
  )
}
