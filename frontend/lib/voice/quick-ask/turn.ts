/**
 * Theme voice, web quick-ask: the turn and its spoken reply (automatos-go's quick-ask-reply.ts, PR #15, for a
 * browser).
 *
 *   send → a normal chat turn on the conversation with spokenMode (deps.sendTurn: lib/chat/turn-stream.ts); a new
 *   chat's id is adopted as it arrives. Each view of the answer → the reply feed (speakable sentences, as they
 *   complete) → the speaker, the AI notice first once per browser session (ai-notice.ts). The turn's end flushes the
 *   rest; a failed turn silences the speaker and says why in a plain sentence. Either way the open conversation is
 *   told the turn landed (deps.announceTurn), so it merges the saved messages.
 *   interruptReply → the speech stops and a turn still streaming is aborted; from Stop or a barge-in it is cancelled
 *   on the server too (the chat's Stop, PRD-237 S7), from a hidden tab it finishes into the chat unheard.
 */
import type { ChatTurnHandlers, ChatTurnOutcome } from '@/lib/chat/turn-stream'

import { queueWithNotice } from './ai-notice'
import { QUICK_ASK_ERRORS } from './copy'
import { createReplyFeed } from './reply-feed'
import type { SpeakerEvent } from './speaker'
import { dispatch, get, nextId, set, type Core, type Turn } from './store'

const LOG_PREFIX = '[voice]'
const UNREACHABLE: ChatTurnOutcome = Object.freeze({ kind: 'failed', reason: 'network', message: null })

function current(core: Core, turnId: number): Turn | null {
  const turn = get(core).turn
  return turn !== null && turn.id === turnId ? turn : null
}

/** New sentences to the speaker, the AI notice ahead of the session's first. Nothing is spoken once Auto stopped. */
function speak(core: Core, sentences: readonly string[]): void {
  if (sentences.length === 0) return
  dispatch(core, { type: 'queued' })
  const inner = get(core)
  if (!inner.state.speaking) return
  const queued = queueWithNotice(inner.noticed, core.deps.storage, sentences, inner.turn?.agentName)
  set(core, { noticed: queued.heard })
  core.speaker.speak(queued.speak)
}

function turnHandlers(core: Core, turnId: number): ChatTurnHandlers {
  return {
    onChatId: (chatId: string) => {
      if (current(core, turnId) !== null) dispatch(core, { type: 'chat_id', chatId })
    },
    onAgent: (name: string) => {
      const turn = current(core, turnId)
      if (turn !== null) set(core, { turn: Object.freeze({ ...turn, agentName: name }) })
    },
    onText: (answer: string) => {
      const turn = current(core, turnId)
      if (turn !== null) speak(core, turn.feed.next(answer))
    },
  }
}

function failureSentence(outcome: Extract<ChatTurnOutcome, { kind: 'failed' }>): string {
  if (outcome.reason === 'network') return QUICK_ASK_ERRORS.chatNetwork
  return outcome.message ?? QUICK_ASK_ERRORS.turn
}

/** The turn ended: the rest spoken, or the failure said; the conversation told either way. */
function settle(core: Core, turnId: number, outcome: ChatTurnOutcome): void {
  const turn = current(core, turnId)
  if (turn === null || outcome.kind === 'aborted') return
  const chatId = get(core).state.view.chatId
  if (outcome.kind === 'failed') {
    set(core, { turn: null })
    core.speaker.stop()
    dispatch(core, { type: 'failed', error: failureSentence(outcome) })
  } else {
    speak(core, turn.feed.finish())
    set(core, { turn: null })
    dispatch(core, { type: 'finished' })
  }
  if (chatId !== null) core.deps.announceTurn(chatId)
}

/** Sends the heard words to Auto as a chat turn on the conversation, the reply to be heard. */
export function sendTurn(core: Core, text: string): void {
  const turn: Turn = Object.freeze({
    id: nextId(core), controller: new AbortController(), feed: createReplyFeed(), agentName: null,
  })
  set(core, { turn })
  dispatch(core, { type: 'sent', text })
  if (current(core, turn.id) === null) return
  const locale = core.deps.locale
  const request = { chatId: get(core).state.view.chatId, text, spokenMode: true, ...(locale ? { locale } : {}) }
  void Promise.resolve()
    .then(() => core.deps.sendTurn(request, turnHandlers(core, turn.id), turn.controller.signal))
    .then((outcome) => settle(core, turn.id, outcome))
    .catch((error: unknown) => {
      console.warn(LOG_PREFIX, 'the spoken turn failed', error)
      settle(core, turn.id, UNREACHABLE)
    })
}

/** Auto stops: the speech, and the turn still streaming (cancelled on the server too when the member asked). */
export function interruptReply(core: Core, cancelOnServer: boolean): void {
  core.speaker.stop()
  const turn = get(core).turn
  if (turn === null) return
  set(core, { turn: null })
  turn.controller.abort()
  const chatId = get(core).state.view.chatId
  if (!cancelOnServer || chatId === null) return
  void Promise.resolve()
    .then(() => core.deps.cancelTurn(chatId))
    .catch((error: unknown) => console.warn(LOG_PREFIX, 'the turn could not be cancelled on the server', error))
}

/** What the speaker says: a sentence began (the reply grows), or the queue ran out. */
export function onSpeaker(core: Core, event: SpeakerEvent): void {
  dispatch(core, event.type === 'sentence' ? { type: 'sentence', text: event.text } : { type: 'drained' })
}
