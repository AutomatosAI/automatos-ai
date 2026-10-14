'use client'

import { AutoMascot } from '@/components/mascot/auto-mascot'

import styles from './looks.module.css'

/**
 * PRD-260 — the live preview on the Looks page (Ember looks-01), ported from
 * Automatos Play's components/looks/LookPreview.tsx: Auto with his caption, a
 * user bubble, tool chips, Auto's answer, a Deliverable card and the message
 * box, all drawn from the active look's variables on <html>, so it follows
 * every change at once. It is a picture of the chat: Open and Send are not
 * real buttons here.
 */

export const PREVIEW_LABEL = 'Preview'
export const PREVIEW_CAPTION = 'Ta-da.'

export function LookPreview() {
  return (
    <section className={styles.preview} aria-label={PREVIEW_LABEL}>
      <div className={styles.previewAuto}>
        <AutoMascot state="celebrate" caption={PREVIEW_CAPTION} />
      </div>
      <div className={styles.previewChat}>
        <p className={styles.previewBubble}>Can you make me a shortlist under £700?</p>
        <ul className={styles.previewChips}>
          <li className={styles.previewChip}>✓ Searched the web · 5 results</li>
          <li className={styles.previewChip}>✓ Made a report</li>
        </ul>
        <p className={styles.previewAnswer}>
          Three laptops make the cut. My pick is the{' '}
          <strong className={styles.previewHighlight}>ASUS Vivobook S 14</strong>.
        </p>
        <div className={styles.previewCard}>
          <span className={styles.previewTile} aria-hidden="true" />
          <span className={styles.previewCardText}>
            <span className={styles.previewCardTitle}>Laptop shortlist</span>{' '}
            <span className={styles.previewCardMeta}>Report · just now</span>
          </span>
          <span className={styles.previewButton}>Open</span>
        </div>
        <div className={styles.previewBox}>
          <span className={styles.previewPlaceholder}>Message Auto</span>
          <span className={styles.previewButton}>Send</span>
        </div>
      </div>
    </section>
  )
}
