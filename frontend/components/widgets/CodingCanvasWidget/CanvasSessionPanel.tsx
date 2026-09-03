'use client'

/**
 * CanvasSessionPanel — the streamed session surface beside the file tree
 * PRD-170 S3/S4
 *
 * Renders the headless SDK session: a start/stop control + live status, the
 * streamed agent turns (assistant text, tool calls, file edits), and the S4
 * approval cards (DiffCard) for any pending permission request. Auto-accept is a
 * session-scoped toggle (default OFF, file edits only) shown here so its state is
 * always visible.
 *
 * All state comes from `useCanvasSession` (which folds the tested pure modules);
 * this component is presentation only.
 */

import { useState } from 'react'
import { Loader2, Play, Send, Square, Sparkles } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { ScrollArea } from '@/components/ui/scroll-area'
import { DiffCard } from './DiffCard'
import { CanvasCommitControl } from './CanvasCommitControl'
import type { CanvasSessionController } from './useCanvasSession'
import type { CanvasSessionStatus, CanvasTurnItem } from './canvasSessionState'

interface CanvasSessionPanelProps {
  session: CanvasSessionController
  workspaceId: string | undefined
}

const STATUS_LABEL: Record<CanvasSessionStatus, string> = {
  idle: 'Idle',
  starting: 'Starting…',
  running: 'Running',
  stopped: 'Stopped',
  failed: 'Failed',
}

function statusVariant(
  status: CanvasSessionStatus
): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (status === 'running') return 'default'
  if (status === 'failed') return 'destructive'
  if (status === 'starting') return 'secondary'
  return 'outline'
}

export function CanvasSessionPanel({ session, workspaceId }: CanvasSessionPanelProps) {
  const { ui, approvals, starting, startError } = session
  const isLive = ui.status === 'running' || ui.status === 'starting'
  const pendingCards = approvals.cards.filter((c) => c.status === 'pending')
  const [prompt, setPrompt] = useState('')

  const submitPrompt = () => {
    const text = prompt.trim()
    if (!text) return
    void session.send(text)
    setPrompt('')
  }

  return (
    <div className="flex h-full flex-col border-l border-border bg-background" data-testid="canvas-session-panel">
      {/* Header: status + start/stop */}
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Auto Session</span>
          <Badge variant={statusVariant(ui.status)} data-testid="session-status">
            {STATUS_LABEL[ui.status]}
          </Badge>
        </div>
        {isLive ? (
          {session.external && (
            <span className="text-xs text-muted-foreground" data-testid="session-external-label">
              Live · Claude Code session for ticket #{String(session.taskId)} — approvals and takeover live on the ticket.
            </span>
          )}
          {!session.external && (
            <Button size="sm" variant="outline" onClick={() => void session.stop()} data-testid="session-stop">
              <Square className="mr-1 h-3.5 w-3.5" />
              Stop
            </Button>
          )}
        ) : (
          {!session.external && (
            <Button size="sm" onClick={() => void session.start()} disabled={starting} data-testid="session-start">
              {starting ? (
                <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="mr-1 h-3.5 w-3.5" />
              )}
              Start
            </Button>
          )}
        )}
      </div>

      {/* Auto-accept toggle (session-scoped, edits only, visibly indicated) */}
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <div className="flex flex-col">
          <span className="text-xs font-medium">Auto-accept edits</span>
          <span className="text-[11px] text-muted-foreground">
            Applies file edits without a prompt. Never bash.
          </span>
        </div>
        <Switch
          checked={approvals.autoAcceptEdits}
          onCheckedChange={(v: boolean) => void session.setAutoAccept(v)}
          data-testid="auto-accept-toggle"
        />
      </div>

      {(startError || ui.error) && (
        <div className="border-b border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive" data-testid="session-error">
          {startError || ui.error}
        </div>
      )}

      {/* Pending approval cards (S4) */}
      {pendingCards.length > 0 && (
        <div className="space-y-2 border-b border-border p-2" data-testid="approval-cards">
          {pendingCards.map((card) => (
            <DiffCard key={card.requestId} card={card} onDecide={(id, d) => void session.decide(id, d)} />
          ))}
        </div>
      )}

      {/* Streamed turns */}
      <ScrollArea className="flex-1">
        <div className="space-y-2 p-3" data-testid="session-turns">
          {ui.turns.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              {isLive
                ? 'Ask Auto to change this workspace — streamed turns and diffs appear here.'
                : 'Start a session to code with Auto in this workspace.'}
            </p>
          ) : (
            ui.turns.map((turn, i) => <TurnRow key={i} turn={turn} />)
          )}
        </div>
      </ScrollArea>

      {/* Prompt composer (PRD-203 C·S7) — the box to instruct Auto. */}
      {isLive && (
        <form
          className="flex items-end gap-2 border-t border-border p-2"
          onSubmit={(e) => {
            e.preventDefault()
            submitPrompt()
          }}
          data-testid="canvas-composer"
        >
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submitPrompt()
              }
            }}
            rows={2}
            placeholder="Ask Auto to change this workspace…"
            className="flex-1 resize-none rounded-md border border-border bg-background px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring"
            data-testid="canvas-composer-input"
          />
          <Button
            type="submit"
            size="sm"
            disabled={!prompt.trim()}
            data-testid="canvas-composer-send"
          >
            <Send className="h-3.5 w-3.5" />
          </Button>
        </form>
      )}

      {/* Commit + push the session's work (S5) — shown once a session exists. */}
      {(isLive || ui.status === 'stopped') && (
        <CanvasCommitControl workspaceId={workspaceId} changeSignal={session.treeRefreshTick} />
      )}
    </div>
  )
}

function TurnRow({ turn }: { turn: CanvasTurnItem }) {
  if (turn.kind === 'user') {
    return (
      <p
        className="whitespace-pre-wrap rounded-md bg-secondary/50 px-2 py-1 text-sm leading-relaxed"
        data-testid="user-turn"
      >
        {turn.text}
      </p>
    )
  }
  if (turn.kind === 'text') {
    return <p className="whitespace-pre-wrap text-sm leading-relaxed">{turn.text}</p>
  }
  if (turn.kind === 'file_edit') {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="secondary">edit</Badge>
        <span className="font-mono">{turn.path}</span>
      </div>
    )
  }
  // tool_call
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <Badge variant="outline">{turn.toolName || 'tool'}</Badge>
      {turn.path && <span className="font-mono">{turn.path}</span>}
    </div>
  )
}
