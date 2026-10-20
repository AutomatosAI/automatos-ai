'use client'

import { FirstRunNudge, useBoardEventStream } from '@automatos/theme-sdk'

import { usePlayConversation } from '../../../hooks/use-play-conversation'
import { AutoMascot } from '../../mascot/auto-mascot'
import { MascotColumn, PageColumns, PageHeader } from '../shell/page-columns'

import { PlayChatView } from './play-chat-view'
import styles from './opening.module.css'

/**
 * PRD-260 §4 — Play's chat at /chat, mounted by app/chat/layout.tsx through
 * ThemeSlot (PRD-261 A-04b) under Play at >= 1024 px (K7), inside Play's frame:
 * builtin Play's slot for the page (lib/theme-host/builtin/play-chat-lazy.tsx)
 * draws MainLayout full bleed around it, as ThemeSlot frames an installed
 * theme's page (MainLayout is host-only, docs/theme-host/sdk-surface-map.md).
 * The conversation on screen is the core's (usePlayConversation: the SDK's
 * useChatController, on the session store and lib/chat/api.ts); while it
 * settles or loads, Auto says so. The chat screen is keyed by the
 * conversation, so switching conversations starts it afresh; a `?draft=`
 * link's words go into the new chat's message box once. The page holds the
 * SSE lane open (the SDK's useBoardEventStream), as Studio's chat does, so
 * chat_changed reaches useChat's merge listener: an awaited (detached) reply
 * and a background post land live. It reaches the core only through the SDK
 * (PRD-261 A-14a).
 *
 * The conversation is held inside the frame (PlayChatScreen), never above it:
 * when the frame mounts again ("Automatos is asleep" clearing, AsleepGate), the
 * conversation mounts with it and loads the active one from the server, so
 * what was said meanwhile shows instead of the snapshot it had.
 */

export const OPENING = 'Opening this conversation…'
export const OPENING_TITLE = 'Conversation'

export function PlayChat() {
  // The chat_changed lane, as components/chatbot/chat.tsx:247-251 opens it; here, not in the keyed view, so one connection.
  useBoardEventStream(true)
  return (
    <>
      {/* PRD-233 S3: the local edition's first-run nudge (nothing in saas, or once a key exists). */}
      <FirstRunNudge />
      <PlayChatScreen />
    </>
  )
}

/** The conversation on screen, or Auto opening it: mounted (and so loaded afresh) with Play's frame. */
function PlayChatScreen() {
  const { conversation, draft, takeDraft } = usePlayConversation()
  if (!conversation) return <PlayChatOpening />
  return <PlayChatView key={conversation.key} conversation={conversation} draft={draft} onDraftTaken={takeDraft} />
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
