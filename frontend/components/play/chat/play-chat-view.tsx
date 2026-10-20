'use client'

import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'
import {
  type HostChatController,
  type HostChatConversation,
  type HostChatDraft,
  type HostChatMessage,
  type HostHealth,
  useHealth,
} from '@automatos/theme-sdk'

import { type AutoMascotDriver, type BackendStatus, useAutoMascot } from '../../../hooks/use-auto-mascot'
import { type PlayTurn, usePlayTurn } from '../../../hooks/use-play-turn'
import { AutoMascot } from '../../mascot/auto-mascot'
import { MascotColumn, PageColumns, PageHeader } from '../shell/page-columns'

import { LooseApprovalCards } from './approval-cards'
import { chatCaption } from './caption'
import { ExampleChips, FirstOpenHero } from './first-open'
import { MessageBox } from './message-box'
import { MessageList } from './message-list'
import { latestReply, liveMessages, replyProgress, runningToolInput } from './reply'
import { PlayTrialBanner, TurnEnd } from './turn-end'
import styles from './composer.module.css'

/**
 * PRD-260 §4, D5, D6 — the chat with Auto (mocks 01, 02 and 06), ported from
 * Automatos Play's components/chat/ChatView.tsx onto the core's data, through
 * the theme SDK (PRD-261 A-14a): the conversation's chat is usePlayTurn's, the
 * SDK's useChatController(conversation) on the chat controller (sending and
 * streaming through useChatWithReceipts, the hook chat.tsx uses; a new chat's
 * id into the session store as chat.tsx does; Stop keeping the reply Auto was
 * writing, as stopped, P260-FX-30, and cancelling the turn on the server; Try
 * again; a turn's Deliverables, Mission suggestion, failure and approval cards
 * on its view model). Auto is driven by useAutoMascot from the chat's data
 * parts, its status and the reply being written (his thinking ends when the
 * words start), and plays the turn's events. The page lays out the
 * conversation and the message box, with Auto in the mascot column and his
 * caption saying what he is doing; an empty chat is the first open. Stop tells
 * Auto first (no celebration for a stopped turn), then stops the chat. A spent
 * trial shows the core's banner above the box. A card whose reply left the
 * conversation (Try again, a request failing mid-turn, the conversation shown
 * again) shows after it, in its column, until answered or closed.
 *
 * Two components, so Auto hears the chat in the PRD-260 order: the chat in
 * PlayChatView, Auto and the screen in its child, whose effects (Auto's
 * reading of the status and the reply) run before the turn's own (a failure,
 * an approval card waiting or settled) in every commit.
 */

export interface PlayChatViewProps {
  /** The conversation on screen, as usePlayConversation hands it out; key the view by `conversation.key`. */
  readonly conversation: HostChatConversation
  /** How long without input before Auto naps (tests shorten it). */
  readonly idleMs?: number
  /** A link's words for the message box (a Deliverable's "Ask Auto about this"). */
  readonly draft?: HostChatDraft | null
  /** The words are in the box: the link's draft is spent. */
  readonly onDraftTaken?: () => void
}

type ChatWithAutoProps = Omit<PlayChatViewProps, 'conversation'> & { readonly turn: PlayTurn }

const BACKEND_OF: Readonly<Record<HostHealth['connection'], BackendStatus | undefined>> = {
  connected: 'up',
  down: 'down',
  checking: undefined,
}

/** Auto, driven by the chat's status and the reply being written, and connected to the chat's parts and events. */
function usePlayAuto({ chat, connect }: PlayTurn, live: readonly HostChatMessage[], idleMs: number | undefined) {
  const { connection } = useHealth()
  const auto = useAutoMascot({
    status: chat.status,
    errorCode: chat.errorCode,
    reply: replyProgress(live),
    backend: BACKEND_OF[connection],
    idleMs,
  })
  // In the render that makes the driver, before any effect of this commit (use-play-turn.ts).
  connect(auto)
  const { greet } = auto
  useEffect(() => {
    greet()
  }, [greet])
  return auto
}

/**
 * The box's fill (an example chip's words, or a link's draft, each in a fresh
 * box: a new key, so it focuses with them), and focus and blur as Auto's events
 * (he listens while you type). A link's draft is taken once.
 */
function useComposer(auto: AutoMascotDriver, linked: HostChatDraft | null, onTaken?: () => void) {
  const [fill, setFill] = useState<HostChatDraft | null>(linked)
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
function newestKind(reply: HostChatMessage | undefined): string | undefined {
  const made = reply?.deliverables ?? []
  return made[made.length - 1]?.kind
}

/**
 * Try again (the chat's: it clears the turn's suggestion first, as chat.tsx
 * does on a new message) goes with the failed turn's snag box, so focus moves
 * to the message box, never to <body> (P260-FX-22, M20). Stop tells Auto first
 * (a done receipt no longer celebrates the stopped turn), then runs the chat's
 * stop, which keeps the reply being written and stops the turn.
 */
function useSending(chat: HostChatController, auto: AutoMascotDriver, box: RefObject<HTMLTextAreaElement>) {
  const { retry: askAgain, stop } = chat
  const { dispatch } = auto
  const retry = useCallback(() => {
    if (askAgain()) box.current?.focus()
  }, [askAgain, box])
  const onStop = useCallback(() => {
    dispatch({ type: 'stop' })
    stop()
  }, [dispatch, stop])
  return { retry, onStop }
}

/** The conversation's chat; Auto and the screen in the child (see above). */
export function PlayChatView({ conversation, ...screen }: PlayChatViewProps) {
  const turn = usePlayTurn(conversation)
  return <ChatWithAuto turn={turn} {...screen} />
}

function ChatWithAuto({ turn, idleMs, draft, onDraftTaken }: ChatWithAutoProps) {
  const { chat } = turn
  // The conversation as the chat hook holds it, as the PRD-260 chat read it for Auto, his caption and the turn's end.
  const live = liveMessages(chat.messages)
  const auto = usePlayAuto(turn, live, idleMs)
  const composer = useComposer(auto, draft ?? null, onDraftTaken)
  const box = useRef<HTMLTextAreaElement>(null)
  const { retry, onStop } = useSending(chat, auto, box)
  const { streaming, trialExhausted } = chat
  const empty = live.length === 0
  const reply = latestReply(live)
  const caption = chatCaption(auto.mascot, {
    runningInput: runningToolInput(reply),
    deliverableKind: newestKind(reply),
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
      <PageHeader title={chat.title} />
      <div className={styles.chat} data-empty={empty || undefined}>
        {empty ? (
          <FirstOpenHero state={auto.mascot} onClipEnded={auto.onClipEnded} {...caption} />
        ) : (
          <MessageList
            messages={chat.messages}
            streaming={streaming}
            onRetry={retry}
            trialBlocked={trialExhausted}
            after={
              <TurnEnd
                status={chat.status}
                trialExhausted={trialExhausted}
                messages={live}
                onRetry={retry}
                suggestion={chat.suggestion}
                planningMissionId={chat.planningMissionId}
                chatId={chat.conversation.chatId ?? ''}
                before={<LooseApprovalCards approvals={chat.looseApprovals} />}
              />
            }
          />
        )}
        <div className={styles.footer}>
          {empty && <ExampleChips onPick={composer.pick} />}
          <PlayTrialBanner errorCode={chat.errorCode} trialExhausted={trialExhausted} />
          <MessageBox
            ref={box}
            key={composer.fill ? `fill-${composer.fill.serial}` : 'box'}
            streaming={streaming}
            onSend={chat.send}
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
