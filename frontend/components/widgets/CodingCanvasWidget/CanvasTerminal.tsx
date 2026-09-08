'use client'

/**
 * PRD-239 S7 — a real terminal in the Canvas: the operator's own shell on their
 * own machine, served by the CLI host on the loopback, in the ticket's directory
 * (or the agent's working directory). Type `claude` here yourself.
 *
 * The backend mints a short-lived grant for the paired host; the host learns it
 * on its next heartbeat, so the connect retries for a few seconds.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import '@xterm/xterm/css/xterm.css'
import { Loader2, TerminalSquare } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'
import { useCliHostHealth } from '@/hooks/use-cli-host-health'
import { connectWithRetry, describeLaunch, encodeInput, encodeResize, type TerminalLaunch } from './terminal-protocol'

export interface CanvasTerminalProps {
  taskId?: string | number | null
  cwd?: string | null
  /** PRD-239 S7 v2: open as soon as a host is online (the Runtime Canvas). */
  autoOpen?: boolean
  /** The terminal IS the session: the agent's Claude Code runs in it (no shell prompt first). */
  runtime?: boolean
  /** A plain shell in the ticket's folder, beside the session (an extra terminal tab). */
  shell?: boolean
}

interface TerminalGrant {
  ws_url: string
  cwd: string | null
  task_id: string | null
  expires_at: string
  launch?: TerminalLaunch | null
}

type TerminalStatus = 'idle' | 'opening' | 'connected' | 'closed' | 'error'

// The host learns the grant on its heartbeat (every 30 s): wait for a full one plus slack.
const CONNECT_BUDGET_MS = 50_000

export function CanvasTerminal({ taskId, cwd, autoOpen = false, runtime = false, shell = false }: CanvasTerminalProps) {
  const { data: health } = useCliHostHealth()
  const host = health?.online_hosts?.[0] ?? null
  const [status, setStatus] = useState<TerminalStatus>('idle')
  const [note, setNote] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const termRef = useRef<Terminal | null>(null)
  const fitRef = useRef<FitAddon | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const autoOpened = useRef(false)

  const closeSocket = useCallback(() => {
    const ws = wsRef.current
    wsRef.current = null
    if (ws && ws.readyState <= WebSocket.OPEN) ws.close()
  }, [])

  const open = useCallback(async () => {
    if (!host || !containerRef.current) return
    closeSocket()
    setStatus('opening')
    setNote(null)
    try {
      const grant = await apiClient.request<TerminalGrant>(`/api/v1/cli-hosts/${host.id}/terminal`, {
        method: 'POST',
        body: JSON.stringify({
          ...(taskId != null && taskId !== '' ? { task_id: Number(taskId) } : {}),
          ...(cwd ? { cwd } : {}),
          ...(shell ? { shell: true } : {}),
        }),
      })
      let term = termRef.current
      if (!term) {
        term = new Terminal({ cursorBlink: true, fontSize: 13, scrollback: 5000, theme: { background: '#0b0f14' } })
        const fit = new FitAddon()
        term.loadAddon(fit)
        term.open(containerRef.current)
        termRef.current = term
        fitRef.current = fit
      } else {
        term.reset()
      }
      fitRef.current?.fit()
      const ws = await connectWithRetry(grant.ws_url, CONNECT_BUDGET_MS)
      wsRef.current = ws
      ws.onmessage = (event: MessageEvent) => {
        if (event.data instanceof ArrayBuffer) term!.write(new Uint8Array(event.data))
        else if (typeof event.data === 'string') term!.write(event.data)
      }
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null
        setStatus('closed')
        setNote(grant.launch?.kind === 'claude' ? 'The session ended. Reopen to continue it.' : 'The shell ended. Open a new terminal to continue.')
      }
      ws.onerror = () => {
        setStatus('error')
        setNote('The connection to the host dropped.')
      }
      term.onData((data) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(encodeInput(data))
      })
      term.onResize(({ cols, rows }) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(encodeResize(cols, rows))
      })
      ws.send(encodeResize(term.cols, term.rows))
      term.focus()
      setStatus('connected')
      setNote(describeLaunch(grant.launch, grant.cwd, taskId != null ? `ticket #${taskId}'s session folder` : 'the host’s first allowed folder'))
    } catch (err) {
      setStatus('error')
      setNote(err instanceof Error ? err.message : 'Could not open a terminal')
    }
  }, [host, taskId, cwd, shell, closeSocket])

  // PRD-239 S7 v2: the Runtime Canvas opens the session itself, once, when a host is online.
  useEffect(() => {
    if (!autoOpen || autoOpened.current || !host || status !== 'idle') return
    autoOpened.current = true
    void open()
  }, [autoOpen, host, status, open])

  // Keep the terminal sized to its pane.
  useEffect(() => {
    const el = containerRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => fitRef.current?.fit())
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Close the shell with the pane.
  useEffect(() => () => {
    closeSocket()
    termRef.current?.dispose()
    termRef.current = null
  }, [closeSocket])

  const live = status === 'connected'
  return (
    <div className="flex h-full min-h-0 flex-col bg-[#0b0f14] text-foreground" data-testid="canvas-terminal">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex min-w-0 items-center gap-2 text-xs">
          <TerminalSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="font-medium">{shell ? 'Shell' : runtime ? 'Session' : 'Terminal'}</span>
          <span className="truncate text-muted-foreground" data-testid="canvas-terminal-note">
            {status === 'connected' && note ? `on your machine · ${note}` : status === 'opening' && runtime ? 'Starting the session on your machine…' : note}
          </span>
        </div>
        {!host ? (
          <span className="text-xs text-muted-foreground" data-testid="canvas-terminal-no-host">
            No CLI host online — start it with <code className="font-mono">make cli-host</code>.
          </span>
        ) : (
          <Button size="sm" variant={live ? 'outline' : 'default'} onClick={() => void (live ? closeSocket() : open())} disabled={status === 'opening'} data-testid="canvas-terminal-toggle">
            {status === 'opening' ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : null}
            {live ? 'Close' : status === 'opening' ? 'Opening…' : status === 'closed' || status === 'error' ? (runtime ? 'Reopen the session' : 'Open terminal') : runtime ? 'Open the session' : 'Open terminal'}
          </Button>
        )}
      </div>
      {status === 'idle' && host && !runtime && (
        <p className="px-3 py-2 text-xs text-muted-foreground">
          Opens your own shell here, in {taskId != null ? `ticket #${taskId}'s folder` : cwd ? cwd : 'the host’s folder'}. Run <code className="font-mono">claude</code> or <code className="font-mono">codex</code> yourself — your login, your subscription.
        </p>
      )}
      {status === 'idle' && host && runtime && !autoOpen && (
        <p className="px-3 py-2 text-xs text-muted-foreground">
          Opens the agent’s Claude Code session here, on your machine, under your own login — resumed where it left off.
        </p>
      )}
      <div ref={containerRef} className="min-h-0 flex-1 px-1 py-1" />
    </div>
  )
}
