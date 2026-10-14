'use client'

import Link from 'next/link'

import {
  movedSomethingOnTheBoard,
  NO_ACTIONS,
  NOTHING_ON_THE_BOARD,
  receiptLine,
  safeLink,
} from '@/lib/chat/receipts'
import type { Receipt } from '@/types'

import { type LineTone, MARK_DONE, MARK_FAILED, MARK_MUTED, MARK_WAITING } from './labels'
import styles from './turn.module.css'

/**
 * PRD-260 §4 — "What Auto did" (mocks 01 and 02) in Play's design, on the
 * core's receipts and the core's rules (components/chatbot/receipts-block.tsx):
 * the platform's own account of the turn, one line per call in receiptLine's
 * words, a done call's page linked only when safeLink says it is a path in
 * this app; a turn that ran nothing says NO_ACTIONS, one whose calls moved
 * nothing on the board says NOTHING_ON_THE_BOARD. A reply from before receipts
 * (undefined) shows nothing.
 */

export const RECEIPTS_TITLE = 'What Auto did'

const LOOK: Readonly<Record<Receipt['status'], { readonly mark: string; readonly tone: LineTone }>> = {
  done: { mark: MARK_DONE, tone: 'done' },
  refused: { mark: MARK_FAILED, tone: 'failed' },
  skipped: { mark: MARK_MUTED, tone: 'muted' },
  waiting: { mark: MARK_WAITING, tone: 'waiting' },
}

function ReceiptItem({ receipt }: { readonly receipt: Receipt }) {
  const look = LOOK[receipt.status]
  const line = receiptLine(receipt)
  const link = receipt.status === 'done' ? safeLink(receipt.link) : null
  return (
    <li className={styles.receipt} data-tone={look.tone} data-status={receipt.status}>
      <span className={styles.mark} aria-hidden="true">
        {look.mark}
      </span>
      {link ? (
        <Link href={link} className={styles.receiptLink}>
          {line}
        </Link>
      ) : (
        <span>{line}</span>
      )}
    </li>
  )
}

/** The core's closing note: nothing ran, or nothing moved on the board; none when a card moved. */
function noteFor(receipts: readonly Receipt[]): string | null {
  if (receipts.length === 0) return NO_ACTIONS
  return movedSomethingOnTheBoard([...receipts]) ? null : NOTHING_ON_THE_BOARD
}

export function Receipts({ receipts }: { readonly receipts: readonly Receipt[] | undefined }) {
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
