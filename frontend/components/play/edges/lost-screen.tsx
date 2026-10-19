import Link from 'next/link'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { CHAT_HREF } from '@/lib/ticket-links'

import styles from './edges.module.css'
import { EdgeWordmark } from './wordmark'

/**
 * PRD-260 §4 — Play's 404 (Ember mock 09), ported from Automatos Play's
 * components/edges/LostScreen.tsx: the whole window, Auto lost in front of a
 * giant hollow "404", and one way back, to the chat. app/not-found.tsx renders
 * it under Play at >= 1024 px; every other style and width keeps the plain 404.
 */

export const LOST_TITLE = 'This page wandered off'
export const LOST_LINE = "Auto looked everywhere. It's not here."
export const BACK_TO_CHAT = 'Back to chat'

export function LostScreen() {
  return (
    <div className={styles.screen}>
      <header className={styles.top}>
        <EdgeWordmark />
      </header>
      <main className={styles.lost}>
        <div className={styles.stage404}>
          <span className={styles.big404} aria-hidden="true">
            404
          </span>
          <div className={styles.lostAuto}>
            <AutoMascot state="lost" showCaption={false} />
          </div>
        </div>
        <h1 className={styles.title}>{LOST_TITLE}</h1>
        <p className={styles.text}>{LOST_LINE}</p>
        <Link href={CHAT_HREF} className={styles.primary}>
          {BACK_TO_CHAT}
        </Link>
      </main>
    </div>
  )
}
