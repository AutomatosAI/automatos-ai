'use client'

/**
 * DiffCard — a single approval card in the canvas session
 * PRD-170 S4
 *
 * Renders one SDK permission prompt: a file edit shows an in-bundle Monaco DIFF
 * (old vs proposed content — no new heavy dep; @monaco-editor/react is already
 * bundled and exports DiffEditor); a bare permission (e.g. Bash) shows the tool.
 * Approve applies / Deny reverts + informs the session (the panel wires the
 * `onDecide` callback to the session). An auto-accepted card renders resolved
 * with a visible "auto-accepted" badge rather than action buttons.
 */

import dynamic from 'next/dynamic'
import { Check, X, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { ApprovalCard, ApprovalDecision } from './diffApproval'

// Monaco diff editor — SSR-unsafe, so dynamic like CodeEditor. Already bundled.
const DiffEditor = dynamic(
  () => import('@monaco-editor/react').then((mod) => mod.DiffEditor),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-24">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    ),
  }
)

interface DiffCardProps {
  card: ApprovalCard
  onDecide: (requestId: string, decision: ApprovalDecision) => void
}

export function DiffCard({ card, onDecide }: DiffCardProps) {
  const isPending = card.status === 'pending'

  return (
    <div
      className="rounded-md border border-border bg-card p-3 text-sm"
      data-testid="diff-card"
      data-status={card.status}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-medium">{card.toolName}</span>
          {card.diff && (
            <span className="text-xs text-muted-foreground">{card.diff.path}</span>
          )}
        </div>
        <StatusBadge card={card} />
      </div>

      {card.diff && (
        <div className="h-40 overflow-hidden rounded border border-border" data-testid="diff-view">
          <DiffEditor
            height="100%"
            language={card.diff.language}
            original={card.diff.oldContent}
            modified={card.diff.newContent}
            theme="vs-dark"
            options={{
              readOnly: true,
              renderSideBySide: false,
              minimap: { enabled: false },
              fontSize: 12,
              scrollBeyondLastLine: false,
              automaticLayout: true,
            }}
          />
        </div>
      )}

      {isPending && (
        <div className="mt-2 flex justify-end gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => onDecide(card.requestId, 'deny')}
            data-testid="diff-deny"
          >
            <X className="mr-1 h-3.5 w-3.5" />
            Deny
          </Button>
          <Button
            size="sm"
            onClick={() => onDecide(card.requestId, 'approve')}
            data-testid="diff-approve"
          >
            <Check className="mr-1 h-3.5 w-3.5" />
            Approve
          </Button>
        </div>
      )}
    </div>
  )
}

function StatusBadge({ card }: { card: ApprovalCard }) {
  if (card.autoAccepted) {
    return (
      <Badge variant="secondary" data-testid="auto-accepted-badge">
        auto-accepted
      </Badge>
    )
  }
  if (card.status === 'approved') {
    return <Badge variant="secondary">approved</Badge>
  }
  if (card.status === 'denied') {
    return <Badge variant="destructive">denied</Badge>
  }
  return <Badge variant="outline">pending</Badge>
}
