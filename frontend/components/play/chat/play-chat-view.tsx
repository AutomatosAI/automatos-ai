'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { MascotColumn, PageColumns, PageHeader } from '@/components/play/shell/page-columns'
import { type AutoMascotDriver, type BackendStatus, useAutoMascot } from '@/hooks/use-auto-mascot'
import { type BackendConnection, useBackendHealth } from '@/hooks/use-backend-health'
import type { PlayConversationView } from '@/hooks/use-play-conversation'
import { type PlayTurn, usePlayTurn } from '@/hooks/use-play-turn'
import { useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { usePageContext } from '@/lib/page-context'
import type { ComposerFill } from '@/lib/play/draft'
import { useChatSessionStore } from '@/stores/chat-session-store'
import type { ChatMessage } from '@/types'

import { chatCaption } from './caption'
import { ExampleChips, FirstOpenHero } from './first-open'
import { MessageBox } from './message-box'
import { MessageList } from './message-list'
import { chatTitle, latestReply, runningToolInput } from './message-text'
import { isTrialBlocked, PlayTrialBanner, TurnEnd } from './turn-end'
import styles from './composer.module.css'

/**
 * PRD-260 §4, D5, D6 — the chat with Auto (mocks 01, 02 and 06), ported from
 * Automatos Play's components/chat/ChatView.tsx onto the core's data: sending
 * and streaming through useChatWithReceipts (the hook chat.tsx uses), a new
 * chat's id into the session store as chat.tsx does, and Auto driven by
 * useAutoMascot from the chat's data parts and status. The page lays out the
 * conversation and the message box, with Auto in the mascot column and his
 * caption saying what he is doing; an empty chat is the first open. Stop calls
 * the hook's stop(), which also cancels the turn on the server. A turn's
 * details (its Deliverables, a Mission suggestion, a failure) come from
 * usePlayTurn; Try again is the hook's reload(); a spent trial shows the
 * core's banner above the box.
 */

export interface PlayChatViewProps {
  readonly view: PlayConversationView
  /** The backend named this new chat: the screen keeps it (usePlayConversation's adopt). */
  readonly onChatIdAssigned: (chatId: string) => void
  /** How long without input before Auto naps (tests shorten it). */
  readonly idleMs?: number
  /** A link's words for the message box (a Deliverable's "Ask Auto about this"). */
  readonly draft?: ComposerFill | null
  /** The words are in the box: the link's draft is spent. */
  readonly onDraftTaken?: () => void
}

const BACKEND_OF: Readonly<Record<BackendConnection, BackendStatus | undefined>> = {
  connected: 'up',
  down: 'down',
  checking: undefined,
}

/** The chat, Auto and the turn's details, wired together: the chat's data parts reach both, its status moves Auto. */
function usePlayChat({ view, onChatIdAssigned, idleMs }: PlayChatViewProps) {
  const pageContext = usePageContext()
  const { connection } = useBackendHealth()
  // useChat takes its data callback before Auto and the turn exist; the ref hands the parts on.
  const onPart = useRef<(part: unknown) => void>(() => undefined)
  const chat = useChatWithReceipts({
    id: view.chatId ?? '',
    initialMessages: view.messages,
    initialAwaitingReply: view.awaitingReply,
    pageContext,
    onData: (part: unknown) => onPart.current(part),
    onChatIdUpdate: (chatId: string) => {
      onChatIdAssigned(chatId)
      // PRD-237: the session store is the channel to the sidebar and the widget, as in chat.tsx.
      useChatSessionStore.getState().chatIdAssigned(chatId)
    },
  })
  const auto = useAutoMascot({ status: chat.status, errorCode: chat.errorCode, backend: BACKEND_OF[connection], idleMs })
  const turn = usePlayTurn({ messages: chat.messages, streaming: chat.status === 'streaming', dispatch: auto.dispatch })
  const toAuto = auto.onData
  const toTurn = turn.onData
  useEffect(() => {
    onPart.current = (part: unknown) => {
      toAuto(part)
      toTurn(part)
    }
  }, [toAuto, toTurn])
  const { greet } = auto
  useEffect(() => {
    greet()
  }, [greet])
  return { chat, auto, turn }
}

/**
 * The box's fill (an example chip's words, or a link's draft, each in a fresh
 * box: a new key, so it focuses with them), and focus and blur as Auto's events
 * (he listens while you type). A link's draft is taken once.
 */
function useComposer(auto: AutoMascotDriver, linked: ComposerFill | null, onTaken?: () => void) {
  const [fill, setFill] = useState<ComposerFill | null>(linked)
  useEffect(() => {
    if (!linked) return
    setFill((prev) => (prev === linked ? prev : { text: linked.text, serial: (prev?.serial ?? 0) + 1 }))
    onTaken?.()
  }, [linked, onTaken])
  const { dispatch } = auto
  const pick = useCallback((text: string) => setFill((prev) => ({ text, serial: (prev?.serial ?? 0) + 1 })), [])
  const onFocus = useCallback(() => dispatch({ type: 'input-focus' }), [dispatch])
  const onBlur = useCallback(() => dispatch({ type: 'input-blur' }), [dispatch])
  return { fill, pick, onFocus, onBlur }
}

/** The newest reply's newest live Deliverable's kind, for the caption's "It's on your shelf too." */
function newestKind(turn: PlayTurn, reply: ChatMessage | undefined): string | undefined {
  const made = reply ? turn.deliverablesOf(reply.id) : []
  return made[made.length - 1]?.kind
}

/** Send and Try again: each clears the turn's suggestion first, as chat.tsx does on a new message. */
function useSending(chat: ReturnType<typeof useChatWithReceipts>, turn: PlayTurn) {
  const { isLoading, sendMessage, reload } = chat
  const { onSend } = turn
  const send = useCallback(
    (text: string): boolean => {
      if (isLoading) return false
      onSend()
      void sendMessage(text)
      return true
    },
    [isLoading, sendMessage, onSend],
  )
  const retry = useCallback(() => {
    if (isLoading) return
    onSend()
    reload()
  }, [isLoading, reload, onSend])
  return { send, retry }
}

export function PlayChatView(props: PlayChatViewProps) {
  const { chat, auto, turn } = usePlayChat(props)
  const composer = useComposer(auto, props.draft ?? null, props.onDraftTaken)
  const { send, retry } = useSending(chat, turn)
  const streaming = chat.status === 'streaming'
  const empty = chat.messages.length === 0
  const reply = latestReply(chat.messages)
  const caption = chatCaption(auto.mascot, {
    runningInput: runningToolInput(reply),
    deliverableKind: newestKind(turn, reply),
    backendDown: auto.state.backendDown,
  })

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
          <MessageList
            messages={chat.messages}
            streaming={streaming}
            deliverablesOf={turn.deliverablesOf}
            onRetry={retry}
            trialBlocked={isTrialBlocked(chat.errorCode)}
            after={
              <TurnEnd
                status={chat.status}
                errorCode={chat.errorCode}
                messages={chat.messages}
                onRetry={retry}
                suggestion={turn.suggestion}
                chatId={props.view.chatId ?? ''}
              />
            }
          />
        )}
        <div className={styles.footer}>
          {empty && <ExampleChips onPick={composer.pick} />}
          <PlayTrialBanner errorCode={chat.errorCode} />
          <MessageBox
            key={composer.fill ? `fill-${composer.fill.serial}` : 'box'}
            streaming={streaming}
            onSend={send}
            onStop={chat.stop}
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
