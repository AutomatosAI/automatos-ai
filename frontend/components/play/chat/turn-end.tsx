'use client'

import type { ReactNode } from 'react'
import {
  type HostChatMessage,
  type HostChatStatus,
  type HostMissionSuggestion,
  TrialExhaustedBanner,
  useTrial,
} from '@automatos/theme-sdk'

import { PlayMissionSuggestion } from './mission-suggestion'
import { SnagBox, TURN_FAILED } from './snag-box'
import styles from './turn.module.css'

/**
 * PRD-260 §4 — what follows the conversation in Play's chat, in the
 * conversation's column: the approval cards whose reply left it (M80), the
 * snag box for a turn that failed before Auto's reply began (useChat drops the
 * empty reply when the request itself fails, so there is no reply to hold it),
 * and the Mission Auto suggests once his answer is complete. PlayTrialBanner
 * is the core's TrialExhaustedBanner (the paused-credits banner, PRD-265) above
 * the message box, on two triggers: the workspace's credits say work on
 * Automatos's keys is paused (useTrial's `exhausted`), or the last send was
 * refused with the typed credits_exhausted code. All of it on the SDK (PRD-261
 * A-14a): the view model's status, suggestion and trialExhausted,
 * cards.TrialExhaustedBanner and context.useTrial.
 */

/** The name of what follows the conversation. */
export const TURN_END_LABEL = 'After the conversation'

const EXHAUSTED = 'exhausted'

/** The chat failed and no reply holds the failure: the newest message is not a reply with an error. */
export function failedBeforeReply(
  status: HostChatStatus,
  messages: readonly HostChatMessage[],
  trialExhausted: boolean,
): boolean {
  if (status !== 'error' || trialExhausted) return false
  const last = messages[messages.length - 1]
  return !(last?.role === 'assistant' && last.error)
}

export interface TurnEndProps {
  readonly status: HostChatStatus
  /** A spent trial refused the last send (the view model's trialExhausted). */
  readonly trialExhausted: boolean
  /** The conversation as the chat hook holds it (reply.ts liveMessages). */
  readonly messages: readonly HostChatMessage[]
  readonly onRetry: () => void
  readonly suggestion: HostMissionSuggestion | null
  /** The Mission the chat is planning, once one is. */
  readonly planningMissionId: string | null
  /** The conversation the Mission comes from ('' before the backend named a new chat). */
  readonly chatId: string
  /** Shown first: the approval cards whose reply left the conversation. */
  readonly before?: ReactNode
}

export function TurnEnd({ status, trialExhausted, messages, onRetry, suggestion, planningMissionId, chatId, before }: TurnEndProps) {
  return (
    <div className={styles.after} role="group" aria-label={TURN_END_LABEL}>
      {before}
      {failedBeforeReply(status, messages, trialExhausted) && <SnagBox lines={[TURN_FAILED]} onRetry={onRetry} />}
      {status !== 'streaming' && (
        <PlayMissionSuggestion suggestion={suggestion} planningMissionId={planningMissionId} chatId={chatId} messages={messages} />
      )}
    </div>
  )
}

export interface PlayTrialBannerProps {
  readonly errorCode: string | null
  /** A spent trial refused the last send (the view model's trialExhausted). */
  readonly trialExhausted: boolean
}

export function PlayTrialBanner({ errorCode, trialExhausted }: PlayTrialBannerProps) {
  const exhausted = useTrial()?.state === EXHAUSTED || trialExhausted
  if (!exhausted) return null
  return (
    <div className={styles.banner}>
      <TrialExhaustedBanner errorCode={errorCode} />
    </div>
  )
}
