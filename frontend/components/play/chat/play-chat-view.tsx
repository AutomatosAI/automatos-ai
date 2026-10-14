'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { MascotColumn, PageColumns, PageHeader } from '@/components/play/shell/page-columns'
import { type AutoMascotDriver, type BackendStatus, useAutoMascot } from '@/hooks/use-auto-mascot'
import { type BackendConnection, useBackendHealth } from '@/hooks/use-backend-health'
import type { PlayConversationView } from '@/hooks/use-play-conversation'
import { useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { usePageContext } from '@/lib/page-context'
import { useChatSessionStore } from '@/stores/chat-session-store'

import { chatCaption } from './caption'
import { ExampleChips, FirstOpenHero } from './first-open'
import { MessageBox } from './message-box'
import { MessageList } from './message-list'
import { chatTitle, latestReply, runningToolInput } from './message-text'
import styles from './composer.module.css'

/**
 * PRD-260 §4, D5, D6 — the chat with Auto (mocks 01, 02 and 06), ported from
 * Automatos Play's components/chat/ChatView.tsx onto the core's data: sending
 * and streaming through useChatWithReceipts (the hook chat.tsx uses), a new
 * chat's id into the session store as chat.tsx does, and Auto driven by
 * useAutoMascot from the chat's data parts and status. The page lays out the
 * conversation and the message box, with Auto in the mascot column and his
 * caption saying what he is doing; an empty chat is the first open. Stop calls
 * the hook's stop(), which also cancels the turn on the server.
 */

export interface PlayChatViewProps {
  readonly view: PlayConversationView
  /** The backend named this new chat: the screen keeps it (usePlayConversation's adopt). */
  readonly onChatIdAssigned: (chatId: string) => void
  /** How long without input before Auto naps (tests shorten it). */
  readonly idleMs?: number
}

const BACKEND_OF: Readonly<Record<BackendConnection, BackendStatus | undefined>> = {
  connected: 'up',
  down: 'down',
  checking: undefined,
}

/** The chat and Auto, wired together: the chat's data parts reach Auto, its status moves him. */
function usePlayChat({ view, onChatIdAssigned, idleMs }: PlayChatViewProps) {
  const pageContext = usePageContext()
  const { connection } = useBackendHealth()
  // useChat takes its data callback before Auto exists; the ref hands the parts on.
  const toAuto = useRef<(part: unknown) => void>(() => undefined)
  const chat = useChatWithReceipts({
    id: view.chatId ?? '',
    initialMessages: view.messages,
    initialAwaitingReply: view.awaitingReply,
    pageContext,
    onData: (part: unknown) => toAuto.current(part),
    onChatIdUpdate: (chatId: string) => {
      onChatIdAssigned(chatId)
      // PRD-237: the session store is the channel to the sidebar and the widget, as in chat.tsx.
      useChatSessionStore.getState().chatIdAssigned(chatId)
    },
  })
  const auto = useAutoMascot({ status: chat.status, errorCode: chat.errorCode, backend: BACKEND_OF[connection], idleMs })
  useEffect(() => {
    toAuto.current = auto.onData
  }, [auto.onData])
  const { greet } = auto
  useEffect(() => {
    greet()
  }, [greet])
  return { chat, auto }
}

/** An example chip's words, put in a fresh message box (a new key) so it focuses with them. */
type Fill = { readonly text: string; readonly serial: number }

/** The chips' fill, and focus and blur as Auto's events (he listens while you type). */
function useComposer(auto: AutoMascotDriver) {
  const [fill, setFill] = useState<Fill | null>(null)
  const { dispatch } = auto
  const pick = useCallback((text: string) => setFill((prev) => ({ text, serial: (prev?.serial ?? 0) + 1 })), [])
  const onFocus = useCallback(() => dispatch({ type: 'input-focus' }), [dispatch])
  const onBlur = useCallback(() => dispatch({ type: 'input-blur' }), [dispatch])
  return { fill, pick, onFocus, onBlur }
}

export function PlayChatView(props: PlayChatViewProps) {
  const { chat, auto } = usePlayChat(props)
  const composer = useComposer(auto)
  const streaming = chat.status === 'streaming'
  const empty = chat.messages.length === 0
  const caption = chatCaption(auto.mascot, {
    runningInput: runningToolInput(latestReply(chat.messages)),
    backendDown: auto.state.backendDown,
  })
  const { isLoading, sendMessage, stop } = chat
  const send = useCallback(
    (text: string): boolean => {
      if (isLoading) return false
      void sendMessage(text)
      return true
    },
    [isLoading, sendMessage],
  )

  const mascot = empty ? undefined : (
    <MascotColumn>
      <AutoMascot state={auto.mascot} onClipEnded={auto.onClipEnded} {...caption} />
    </MascotColumn>
  )

  return (
    <PageColumns mascot={mascot}>
      <PageHeader title={chatTitle(chat.messages, props.view.title)} />
      <div className={styles.chat} data-empty={empty || undefined}>
        {empty ? (
          <FirstOpenHero state={auto.mascot} onClipEnded={auto.onClipEnded} {...caption} />
        ) : (
          <MessageList messages={chat.messages} streaming={streaming} />
        )}
        <div className={styles.footer}>
          {empty && <ExampleChips onPick={composer.pick} />}
          <MessageBox
            key={composer.fill ? `fill-${composer.fill.serial}` : 'box'}
            streaming={streaming}
            onSend={send}
            onStop={stop}
            onFocus={composer.onFocus}
            onBlur={composer.onBlur}
            initialDraft={composer.fill?.text}
            autoFocus={empty || composer.fill !== null}
          />
        </div>
      </div>
    </PageColumns>
  )
}
