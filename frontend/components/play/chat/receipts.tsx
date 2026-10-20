'use client'

import { HostLink, type HostReceipt, movedSomethingOnTheBoard, receiptLine, receiptLink } from '@automatos/theme-sdk'

import { type LineTone, MARK_DONE, MARK_FAILED, MARK_MUTED, MARK_WAITING } from './labels'
import styles from './turn.module.css'

/**
 * PRD-260 §4 — "What Auto did" (mocks 01 and 02) in Play's design, on the
 * core's receipts and the core's rules (components/chatbot/receipts-block.tsx),
 * through the SDK (PRD-261 A-14a): the platform's own account of the turn, one
 * line per call in receiptLine's words, a done call's page linked only when
 * receiptLink says it is a path in this app; a turn that ran nothing says
 * NO_ACTIONS, one whose calls moved nothing on the board says
 * NOTHING_ON_THE_BOARD (the core's words, copied into Play). A reply from
 * before receipts (null) shows nothing. Beside a line, Play's tag
 * (receiptBadge): Done in the accent for a done write, Not done for a refused
 * call, Skipped for one not run; only the mark and the tag carry the tone,
 * never the whole line.
 */

export const RECEIPTS_TITLE = 'What Auto did'
export const BADGE_DONE = 'Done'
export const BADGE_NOT_DONE = 'Not done'
export const BADGE_SKIPPED = 'Skipped'
/** The core's closing notes (lib/chat/receipts.ts), word for word. */
export const NO_ACTIONS = 'No actions in this turn.'
export const NOTHING_ON_THE_BOARD = 'No actions on the board.'

const LOOK: Readonly<Record<HostReceipt['status'], { readonly mark: string; readonly tone: LineTone }>> = {
  done: { mark: MARK_DONE, tone: 'done' },
  refused: { mark: MARK_FAILED, tone: 'failed' },
  skipped: { mark: MARK_MUTED, tone: 'muted' },
  waiting: { mark: MARK_WAITING, tone: 'waiting' },
}

/**
 * The tag beside a receipt's line, as in the Play app: a done write says Done, a refused call
 * Not done, a skipped one Skipped; a done read and a waiting ask have none (the line itself says
 * "waiting for you"). Pure.
 */
export function receiptBadge(r: HostReceipt): string | null {
  if (r.status === 'refused') return BADGE_NOT_DONE
  if (r.status === 'skipped') return BADGE_SKIPPED
  return r.status === 'done' && r.kind === 'write' ? BADGE_DONE : null
}

function ReceiptItem({ receipt }: { readonly receipt: HostReceipt }) {
  const look = LOOK[receipt.status]
  const line = receiptLine(receipt)
  const link = receiptLink(receipt)
  const badge = receiptBadge(receipt)
  return (
    <li className={styles.receipt} data-tone={look.tone} data-status={receipt.status}>
      <span className={styles.mark} aria-hidden="true">
        {look.mark}
      </span>
      {link ? (
        <HostLink href={link} className={styles.receiptLink}>
          {line}
        </HostLink>
      ) : (
        <span>{line}</span>
      )}
      {badge && <span className={styles.badge}>{badge}</span>}
    </li>
  )
}

/** The core's closing note: nothing ran, or nothing moved on the board; none when a card moved. */
function noteFor(receipts: readonly HostReceipt[]): string | null {
  if (receipts.length === 0) return NO_ACTIONS
  return movedSomethingOnTheBoard(receipts) ? null : NOTHING_ON_THE_BOARD
}

export function Receipts({ receipts }: { readonly receipts: readonly HostReceipt[] | null }) {
  if (!receipts) return null
  const note = noteFor(receipts)
  return (
    <section className={styles.receipts} aria-label={RECEIPTS_TITLE}>
      <h3 className={styles.receiptsTitle}>{RECEIPTS_TITLE}</h3>
      {receipts.length > 0 && (
        <ul className={styles.receiptList}>
          {receipts.map((receipt, index) => (
            // Receipts carry no id, and a turn's list never reorders.
            <ReceiptItem key={`${index}-${receipt.action}`} receipt={receipt} />
          ))}
        </ul>
      )}
      {note && <p className={styles.receiptNote}>{note}</p>}
    </section>
  )
}
