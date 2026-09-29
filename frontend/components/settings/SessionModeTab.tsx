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
import { TerminalSquare } from 'lucide-react'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { permissionModeLabel, type PermissionMode } from './PermissionModePicker'
import {
  ConnectClaudeCode,
  DefaultFolderCard,
  DeliverablesRootCard,
  HostsList,
  PermissionModeCard,
  ProjectsFolderCard,
  type HostRow,
} from './SessionModeSections'

/** PRD-239 S6c: Settings → Session mode — where tickets run, and the projects folder the stack was started with. */
export interface SessionModeSettings {
  default_folder: 'projects' | 'sessions'
  default_folder_explicit: boolean
  local_projects_dir: string | null
  projects_mount: string | null
  /** The deliverables root on the host (AUTOMATOS_WORKSPACE_DIR as `make up` exported it), null when unknown. */
  workspace_dir: string | null
  host_allowed_roots: string[]
  /** The workspace's default permission mode for sessions; an agent may override it. */
  permission_mode?: PermissionMode
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


export function SessionModeTab() {
  const queryClient = useQueryClient()
  const health = useSessionModeHealth()
  const enabled = health.data?.cli_runtime_enabled === true
  const hosts = useCliHosts(enabled)
  const settings = useSessionModeSettings(enabled)
  const [saving, setSaving] = useState(false)
  const folderState = describeProjectsFolder(settings.data)
  const deliverablesState = describeDeliverablesRoot(settings.data)
  const projectsFolderForEnv = settings.data?.local_projects_dir || '/Users/you/Development'
  const deliverablesForEnv = settings.data?.workspace_dir || suggestedDeliverablesRoot(projectsFolderForEnv)
  const envLines = PROJECTS_ENV_LINES(projectsFolderForEnv, deliverablesForEnv)
  const saveSetting = async (body: Partial<Pick<SessionModeSettings, 'default_folder' | 'permission_mode'>>, done: string) => {
    setSaving(true)
    try {
      await apiClient.request('/api/v1/cli-hosts/settings', { method: 'PUT', body: JSON.stringify(body) })
      queryClient.invalidateQueries({ queryKey: ['cli-hosts', 'settings'] })
      toast.success(done)
    } catch (err) {
      toast.error(`Could not save: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setSaving(false)
    }
  }
  const saveDefaultFolder = (choice: 'projects' | 'sessions') =>
    saveSetting({ default_folder: choice }, choice === 'projects' ? 'New tickets run in your projects folder' : 'New tickets get their own sessions folder')
  const savePermissionMode = (mode: PermissionMode) =>
    saveSetting({ permission_mode: mode }, `New sessions run in ${permissionModeLabel(mode)} mode`)
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
              <HostsList hosts={hosts.data?.hosts ?? []} />

              <ConnectClaudeCode />

              <ProjectsFolderCard folderState={folderState} envLines={envLines} />
              <DeliverablesRootCard deliverablesState={deliverablesState} />
              <PermissionModeCard settings={settings.data} saving={saving} onSave={savePermissionMode} />
              <DefaultFolderCard settings={settings.data} saving={saving} onSave={saveDefaultFolder} />
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
