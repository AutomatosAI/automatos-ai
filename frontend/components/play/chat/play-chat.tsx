'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { FirstRunNudge } from '@/components/local/first-run-nudge'
import { AutoMascot } from '@/components/mascot/auto-mascot'
import { MascotColumn, PageColumns, PageHeader } from '@/components/play/shell/page-columns'
import { usePlayConversation } from '@/hooks/use-play-conversation'

import { PlayChatView } from './play-chat-view'
import styles from './opening.module.css'

/**
 * PRD-260 §4 — Play's chat at /chat, mounted by app/chat/layout.tsx through
 * PlayRouteSwitch under Play at >= 1024 px (K7), inside Play's frame
 * (MainLayout fullBleed). The conversation on screen is the core's
 * (usePlayConversation: the session store and lib/chat/api.ts); while it
 * settles or loads, Auto says so. The chat screen is keyed by the
 * conversation, so switching conversations starts it afresh.
 */

export const OPENING = 'Opening this conversation…'
export const OPENING_TITLE = 'Conversation'

export function PlayChat() {
  const { view, adopt } = usePlayConversation()
  return (
    <MainLayout fullBleed>
      {/* PRD-233 S3: the local edition's first-run nudge (nothing in saas, or once a key exists). */}
      <FirstRunNudge />
      {view ? <PlayChatView key={view.key} view={view} onChatIdAssigned={adopt} /> : <PlayChatOpening />}
    </MainLayout>
  )
}

/** A conversation on its way: Auto loading in his column, the words in the page. */
export function PlayChatOpening() {
  return (
    <PageColumns
      mascot={
        <MascotColumn>
          <AutoMascot state="loading" />
        </MascotColumn>
      }
    >
      <PageHeader title={OPENING_TITLE} />
      <div className={styles.opening}>
        <p className={styles.status} role="status">
          {OPENING}
        </p>
      </div>
    </PageColumns>
  )
}
