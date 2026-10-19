'use client'

import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { AutoMascot } from '@/components/mascot/auto-mascot'
import { MascotColumn, PageColumns, PageHeader } from '@/components/play/shell/page-columns'
import { type AutoMascotDriver, type BackendStatus, useAutoMascot } from '@/hooks/use-auto-mascot'
import { type BackendConnection, useBackendHealth } from '@/hooks/use-backend-health'
import { usePlayTurn } from '@/hooks/use-play-turn'
import { chatTitle, latestReply, replyProgress, runningToolInput } from '@/lib/chat/message-text'
import { useChatActions, useChatStream, useStreamHandlers } from '@/lib/chat-controller/controller'
import type { ChatConversationView, ChatStream, ChatTurn } from '@/lib/chat-controller/view-model'
import type { ComposerFill } from '@/lib/play/draft'
import type { ChatMessage } from '@/types'

import { ApprovalCards, LooseApprovalCards } from './approval-cards'
import { chatCaption } from './caption'
import { ExampleChips, FirstOpenHero } from './first-open'
import { MessageBox } from './message-box'
import { MessageList } from './message-list'
import { isTrialBlocked, PlayTrialBanner, TurnEnd } from './turn-end'
import styles from './composer.module.css'

/**
 * PRD-260 §4, D5, D6 — the chat with Auto (mocks 01, 02 and 06), ported from
 * Automatos Play's components/chat/ChatView.tsx onto the core's data, on the
 * chat controller (PRD-261 A-11a, lib/chat-controller): sending and streaming
 * through useChatStream (useChatWithReceipts, the hook chat.tsx uses; a new
 * chat's id into the session store as chat.tsx does), and Auto driven by
 * useAutoMascot from the chat's data parts, its status and the reply being
 * written (his thinking ends when the words start), between the stream and
 * the turn so the turn's events reach him. The page lays out the
 * conversation and the message box, with Auto in the mascot column and his
 * caption saying what he is doing; an empty chat is the first open. Stop tells
 * Auto first (no celebration for a stopped turn), then keeps the reply Auto was
 * writing, as stopped (the stream's stop, P260-FX-30), and calls the hook's
 * stop(), which also cancels the turn on the server: the conversation on screen,
 * and the one usePlayTurn reads, is useChat's with the stopped replies kept. A
 * turn's details (its Deliverables, a Mission suggestion, a failure, the
 * approvals it raised as the core's cards) come from usePlayTurn; Try again is
 * the hook's reload(); a spent trial shows the core's banner above the box. A
 * card whose reply left the conversation (Try again, a request failing
 * mid-turn, the conversation shown again) shows after it, in its column, until
 * answered or closed.
 */

export interface PlayChatViewProps {
  readonly view: ChatConversationView
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
  const { connection } = useBackendHealth()
  const shown = useChatStream({ view, onChatIdAssigned })
  const { chat } = shown
  const auto = useAutoMascot({
    status: chat.status,
    errorCode: chat.errorCode,
    reply: replyProgress(chat.messages),
    backend: BACKEND_OF[connection],
    idleMs,
  })
  const turn = usePlayTurn({
    chatId: view.chatId ?? '',
    messages: shown.messages,
    streaming: chat.status === 'streaming',
    dispatch: auto.dispatch,
  })
  const toAuto = auto.onData
  const { onData: toTurn, onChatNamed } = turn
  const onData = useCallback(
    (part: unknown) => {
      toAuto(part)
      toTurn(part)
    },
    [toAuto, toTurn],
  )
  useStreamHandlers(shown, onData, onChatNamed)
  const { greet } = auto
  useEffect(() => {
    greet()
  }, [greet])
  return { chat, auto, turn, shown }
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
function newestKind(turn: ChatTurn, reply: ChatMessage | undefined): string | undefined {
  const made = reply ? turn.deliverablesOf(reply.id) : []
  return made[made.length - 1]?.kind
}

/**
 * Send and Try again (the controller's: each clears the turn's suggestion
 * first, as chat.tsx does on a new message). Try again goes with the failed
 * turn's snag box, so focus moves to the message box, never to <body>
 * (P260-FX-22, M20). Stop tells Auto first (a done receipt no longer
 * celebrates the stopped turn), then runs the stream's stop, which keeps the
 * reply being written and stops the chat.
 */
function useSending(stream: ChatStream, turn: ChatTurn, auto: AutoMascotDriver, box: RefObject<HTMLTextAreaElement>) {
  const { send, retry: askAgain, stop } = useChatActions(stream.chat, turn.onSend, stream.stop)
  const { dispatch } = auto
  const retry = useCallback(() => {
    if (askAgain()) box.current?.focus()
  }, [askAgain, box])
  const onStop = useCallback(() => {
    dispatch({ type: 'stop' })
    stop()
  }, [dispatch, stop])
  return { send, retry, onStop }
}

/** The approval cards a reply raised, inside that reply; the loose ones (their reply is gone) after the conversation. */
function useApprovalSlots(turn: ChatTurn, chatId: string) {
  const { of, loose, answered, closed } = turn.approvals
  const approvalsOf = useCallback(
    (replyId: string) => <ApprovalCards cards={of(replyId)} chatId={chatId} onAnswered={answered} onClosed={closed} />,
    [of, answered, closed, chatId],
  )
  const looseCards = <LooseApprovalCards cards={loose} chatId={chatId} onAnswered={answered} onClosed={closed} />
  return { approvalsOf, looseCards }
}

export function PlayChatView(props: PlayChatViewProps) {
  const { chat, auto, turn, shown } = usePlayChat(props)
  const composer = useComposer(auto, props.draft ?? null, props.onDraftTaken)
  const box = useRef<HTMLTextAreaElement>(null)
  const { send, retry, onStop } = useSending(shown, turn, auto, box)
  const { approvalsOf, looseCards } = useApprovalSlots(turn, props.view.chatId ?? '')
  const streaming = chat.status === 'streaming'
  const empty = chat.messages.length === 0
  const reply = latestReply(chat.messages)
  const caption = chatCaption(auto.mascot, {
    runningInput: runningToolInput(reply),
    deliverableKind: newestKind(turn, reply),
    backendDown: auto.state.backendDown,
    approvalWaiting: auto.state.approvalWaiting,
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
            messages={shown.messages}
            streaming={streaming}
            deliverablesOf={turn.deliverablesOf}
            onRetry={retry}
            trialBlocked={isTrialBlocked(chat.errorCode)}
            approvalsOf={approvalsOf}
            isStopped={shown.isStopped}
            after={
              <TurnEnd
                status={chat.status}
                errorCode={chat.errorCode}
                messages={chat.messages}
                onRetry={retry}
                suggestion={turn.suggestion}
                chatId={props.view.chatId ?? ''}
                before={looseCards}
              />
            }
          />
        )}
        <div className={styles.footer}>
          {empty && <ExampleChips onPick={composer.pick} />}
          <PlayTrialBanner errorCode={chat.errorCode} />
          <MessageBox
            ref={box}
            key={composer.fill ? `fill-${composer.fill.serial}` : 'box'}
            streaming={streaming}
            onSend={send}
            onStop={onStop}
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
