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

function useSessionModeHealth() {
  return useQuery({
    queryKey: ['health', 'session-mode'],
    queryFn: () => apiClient.request<{ edition?: string; cli_runtime_enabled?: boolean }>('/health'),
    staleTime: 10_000,
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
            Agents with the runtime <span className="font-mono">Claude Code session</span> run their tickets as
            your own Claude Code sessions on your machine, under your own login. Automatos assigns the ticket,
            tracks it on the board and records the result; nothing runs through an API key.
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
                  <p className="text-sm font-medium">Pair a host</p>
                </div>
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
                      From the repository root. The host registers <span className="font-mono">./workspaces</span>; add a real
                      repository with <span className="font-mono">CLI_HOST_ARGS=&quot;--allow /path/to/repo&quot;</span>.
                    </p>
                  </div>
                )}
              </div>

              <div className="rounded-lg border border-border/40 p-4 space-y-2 text-xs text-muted-foreground">
                <p className="text-sm font-medium text-foreground">Where sessions work</p>
                <p>
                  A ticket whose agent names no working directory runs in{' '}
                  <span className="font-mono">./workspaces/&lt;workspace id&gt;/sessions/&lt;ticket&gt;</span> — the folder
                  Deliverables → Explorer shows live. Files the session writes there are registered as the ticket&apos;s
                  deliverables when it finishes, and the task report carries the session log.
                </p>
                <p>
                  Your own repositories: set <span className="font-mono">LOCAL_PROJECTS_DIR=/path/to/your/projects</span> in{' '}
                  <span className="font-mono">.env</span> and run <span className="font-mono">make up</span> — the explorer shows it
                  under <span className="font-mono">projects/</span> and the host registers it. Point an agent&apos;s working
                  directory at a repository inside it. The folder is read-only inside the platform; sessions write to it
                  through the host on your machine. <span className="font-mono">LOCAL_PROJECTS_MOUNT=rw</span> lets the Code
                  Canvas editor save into it directly.
                </p>
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
