'use client'

/**
 * PRD-256 US-001 — the receipts block: what the turn's calls did, above the reply text.
 *
 * One line per call in ActivityTrail's chip layout, written by the platform (never the
 * model): "#0422: moved to Done", "Letter: tried, refused: <reason>". A turn that ran
 * nothing says "No actions in this turn."; a turn whose calls moved nothing on the board
 * says so. A message without receipts (every message before PRD-256) renders nothing here.
 *
 * US-002: live, the lines the platform puts above the reply (a refused write, work said done
 * that was not) follow the list, so they sit above the reply text as they do once saved.
 */
import { CheckCircle2, MinusCircle, XCircle } from 'lucide-react'
import type { Receipt } from '@/types'
import {
  NO_ACTIONS,
  NOTHING_ON_THE_BOARD,
  movedSomethingOnTheBoard,
  receiptLine,
  safeLink,
} from '@/lib/chat/receipts'

export interface ReceiptsBlockProps {
  /** undefined: a message from before receipts — nothing is rendered. */
  receipts?: Receipt[]
  /** The lines above the reply, live from the frame (a reloaded reply has them in its text). */
  above?: string[]
}

function statusIcon(r: Receipt) {
  if (r.status === 'refused') return <XCircle className="h-3.5 w-3.5 shrink-0 text-destructive/80" aria-label="Refused" />
  if (r.status === 'skipped') return <MinusCircle className="h-3.5 w-3.5 shrink-0 text-muted-foreground/70" aria-label="Not run" />
  return <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-success/80" aria-label="Done" />
}

function ReceiptRow({ receipt }: { receipt: Receipt }) {
  const line = receiptLine(receipt)
  const link = receipt.status === 'done' ? safeLink(receipt.link) : null
  return (
    <li
      className="flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-muted-foreground"
      data-testid="receipt-line"
      data-status={receipt.status}
    >
      {statusIcon(receipt)}
      {link ? (
        <a href={link} className="min-w-0 truncate hover:text-foreground hover:underline">{line}</a>
      ) : (
        <span className={`min-w-0 truncate ${receipt.status === 'refused' ? 'text-destructive/80' : ''}`}>{line}</span>
      )}
    </li>
  )
}

function NoteRow({ text }: { text: string }) {
  return (
    <li className="flex items-center gap-1.5 px-1.5 py-0.5 text-muted-foreground/80" data-testid="receipts-note">
      <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground/50" aria-hidden />
      <span>{text}</span>
    </li>
  )
}

function AboveLines({ lines }: { lines: string[] }) {
  return (
    <div className="space-y-1 text-sm text-foreground" data-testid="receipts-above">
      {lines.map((line, index) => (
        <p key={index}>{line}</p>
      ))}
    </div>
  )
}

export function ReceiptsBlock({ receipts, above }: ReceiptsBlockProps) {
  if (!receipts) return null
  return (
    <>
      <ol className="space-y-0.5 text-xs" aria-label="What I did" data-testid="receipts-block">
        {receipts.length === 0 && <NoteRow text={NO_ACTIONS} />}
        {receipts.map((receipt, index) => (
          <ReceiptRow key={`${receipt.action}-${index}`} receipt={receipt} />
        ))}
        {receipts.length > 0 && !movedSomethingOnTheBoard(receipts) && <NoteRow text={NOTHING_ON_THE_BOARD} />}
      </ol>
      {above && above.length > 0 && <AboveLines lines={above} />}
    </>
  )
}
