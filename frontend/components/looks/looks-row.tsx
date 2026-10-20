'use client'

import { useEffect, useId, useRef } from 'react'

import type { Look } from '../../lib/looks/types'

import styles from './looks.module.css'
import { cardStyle } from './swatch-style'

/**
 * PRD-260 — the Looks row (Ember looks-01), ported from Automatos Play's
 * components/looks/LooksRow.tsx: the presets, then the user's saved looks,
 * each a mini card drawn in its own ground, type, light and accent. Picking
 * one makes it the active look.
 */

export const LOOKS_ROW_LABEL = 'Looks'
export const CARD_TYPE_SAMPLE = 'Aa'

export interface LooksRowProps {
  readonly looks: readonly Look[]
  /** The active look's id: its card is pressed. */
  readonly activeId: string
  readonly onPick: (look: Look) => void
}

/**
 * Keeps the active card in view inside the row, e.g. a look just saved at its
 * end. Sideways only: the page itself never jumps.
 */
function useActiveInView(activeId: string) {
  const rowRef = useRef<HTMLUListElement>(null)
  useEffect(() => {
    const row = rowRef.current
    const active = row?.querySelector<HTMLElement>('[aria-pressed="true"]')
    if (!row || !active) return
    const rowBox = row.getBoundingClientRect()
    const box = active.getBoundingClientRect()
    if (box.left < rowBox.left) row.scrollLeft -= rowBox.left - box.left
    else if (box.right > rowBox.right) row.scrollLeft += box.right - rowBox.right
  }, [activeId])
  return rowRef
}

export function LooksRow({ looks, activeId, onPick }: LooksRowProps) {
  const headingId = useId()
  const rowRef = useActiveInView(activeId)
  return (
    <section className={styles.looks} aria-labelledby={headingId}>
      <h2 id={headingId} className={styles.sectionLabel}>
        {LOOKS_ROW_LABEL}
      </h2>
      <ul ref={rowRef} className={styles.row}>
        {looks.map((entry) => (
          <li key={entry.id} className={styles.rowItem}>
            <LookCard look={entry} active={entry.id === activeId} onPick={onPick} />
          </li>
        ))}
      </ul>
    </section>
  )
}

interface LookCardProps {
  readonly look: Look
  readonly active: boolean
  readonly onPick: (look: Look) => void
}

/** Its name is the look's name and line ('Ember Warm, editorial'); the swatch is decoration. */
function LookCard({ look, active, onPick }: LookCardProps) {
  return (
    <button type="button" className={styles.card} aria-pressed={active} onClick={() => onPick(look)}>
      <span className={styles.cardTop} style={cardStyle(look)} aria-hidden="true">
        <span className={styles.cardType}>{CARD_TYPE_SAMPLE}</span>
        <span className={styles.cardMarks}>
          <span className={styles.cardLight} />
          <span className={styles.cardAccent} />
        </span>
      </span>
      <span className={styles.cardText}>
        <span className={styles.cardName}>{look.name}</span>{' '}
        <span className={styles.cardSub}>{look.sub}</span>
      </span>
    </button>
  )
}
