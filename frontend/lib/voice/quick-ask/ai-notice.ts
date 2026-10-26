/**
 * Theme voice, web quick-ask: the AI notice (PRD-264 EUAI-04). The greeting is spoken once per browser session, ahead
 * of the first reply's first sentence, and never again in that session: no reminder, whatever the time (Gerard's D73,
 * 23 Oct 2026). Ported from automatos-go src/voice/notice/ai-notice.ts (PR #15), with the session kept in this tab's
 * sessionStorage, so a new engine in the same tab (a provider remounted, another page) does not greet again; when
 * the storage can't be used (blocked site data) the engine's own memory stands in for it.
 *
 * The greeting is noted when it is queued, so a turn that fails, or ends with nothing to say, leaves it for the next
 * spoken reply. The words are the server's (copy.ts AI_VOICE_GREETING); only the name changes, to the answering
 * agent's when the turn named one that is not Auto.
 */
import { DEFAULT_AGENT_NAME } from '@/lib/ai-label/copy'

import { AI_NOTICE_HEARD, AI_NOTICE_STORAGE_KEY } from './config'
import { AI_VOICE_GREETING } from './copy'

export type NoticeStorage = Pick<Storage, 'getItem' | 'setItem'>

/** The sentences to hand the speaker, in order, and whether the session has now heard the notice. */
export interface NoticeQueue {
  readonly heard: boolean
  readonly speak: readonly string[]
}

const LOG_PREFIX = '[voice]'

/** Whether this tab's session has heard the notice; a storage that can't be read says no (the notice is said). */
export function noticeHeard(storage: NoticeStorage | null): boolean {
  if (storage === null) return false
  try {
    return storage.getItem(AI_NOTICE_STORAGE_KEY) === AI_NOTICE_HEARD
  } catch (error) {
    console.warn(LOG_PREFIX, 'the AI notice could not be read for this tab', error)
    return false
  }
}

function rememberHeard(storage: NoticeStorage | null): void {
  if (storage === null) return
  try {
    storage.setItem(AI_NOTICE_STORAGE_KEY, AI_NOTICE_HEARD)
  } catch (error) {
    console.warn(LOG_PREFIX, 'the AI notice could not be remembered for this tab', error)
  }
}

/** 'Hi, this is Auto, your AI assistant.', with the answering agent's name in place of Auto. */
export function noticeText(agentName?: string | null): string {
  const name = agentName?.replace(/\s+/g, ' ').trim() || DEFAULT_AGENT_NAME
  // A function replacement: a name holding '$&' or '$1' is said as written, not expanded.
  return AI_VOICE_GREETING.replace(DEFAULT_AGENT_NAME, () => name)
}

/**
 * The reply's new sentences for the speaker, with the greeting first when they are the session's first spoken
 * words. `heardHere` is the engine's own memory of it. No sentences yet: nothing to queue, and the greeting waits.
 */
export function queueWithNotice(
  heardHere: boolean,
  storage: NoticeStorage | null,
  sentences: readonly string[],
  agentName?: string | null,
): NoticeQueue {
  const heard = heardHere || noticeHeard(storage)
  if (sentences.length === 0 || heard) return Object.freeze({ heard, speak: Object.freeze([...sentences]) })
  rememberHeard(storage)
  return Object.freeze({ heard: true, speak: Object.freeze([noticeText(agentName), ...sentences]) })
}
