import { captionFor } from '../../../lib/mascot/media'
import type { MascotName } from '../../../lib/mascot/types'

import { toolDetail } from './labels'

/**
 * PRD-260 §4 — what the caption under Auto says on the chat (mocks 01 and 02),
 * ported from Automatos Play's lib/chat/caption.ts. The state's own words come
 * from captionFor; the chat adds the detail it knows: the running search
 * query, that a Deliverable is on the shelf too, or that an approval card
 * waits for the owner's answer. Pure.
 */

export type ChatCaption = { readonly caption: string; readonly detail?: string }

export interface CaptionFacts {
  /** The input of the tool call running on the newest reply (its query, page or title). */
  readonly runningInput?: unknown
  /** The kind of the newest Deliverable the reply made ("report"), when it made one. */
  readonly deliverableKind?: string
  /** Automatos isn't answering: Auto sleeps for that reason, not for a nap. */
  readonly backendDown?: boolean
  /** An approval card waits for the owner's answer: Auto listens for it (T-005c). */
  readonly approvalWaiting?: boolean
}

export const ON_THE_SHELF = "It's on your shelf too."
/** Asleep after the idle timeout, not because Automatos is down. */
export const NAPPING = 'Having a little nap'
export const WAKE_HINT = 'Move or type to wake me.'
/** Listening for the owner's answer on an approval card. */
export const AWAITING_ANSWER = 'Waiting for your answer on the card.'

const BUSY: ReadonlySet<MascotName> = new Set<MascotName>(['searching', 'reading', 'working'])

export function chatCaption(state: MascotName, facts: CaptionFacts = {}): ChatCaption {
  if (state === 'sleeping' && !facts.backendDown) return { caption: NAPPING, detail: WAKE_HINT }
  if (state === 'listening' && facts.approvalWaiting) return { caption: captionFor(state), detail: AWAITING_ANSWER }
  if (state === 'celebrate' && facts.deliverableKind) {
    return { caption: captionFor(state, facts.deliverableKind), detail: ON_THE_SHELF }
  }
  if (BUSY.has(state)) {
    const detail = toolDetail(facts.runningInput)
    return detail ? { caption: captionFor(state), detail } : { caption: captionFor(state) }
  }
  return { caption: captionFor(state) }
}
