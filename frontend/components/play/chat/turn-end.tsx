'use client'

import { TrialExhaustedBanner } from '@/components/onboarding/trial-exhausted-banner'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { TRIAL_EXHAUSTED_CODE } from '@/lib/trial'
import type { ChatMessage } from '@/types'

import { PlayMissionSuggestion } from './mission-suggestion'
import { SnagBox, TURN_FAILED } from './snag-box'
import type { MissionSuggestion } from './suggestion'
import styles from './turn.module.css'

/**
 * PRD-260 §4 — what follows the conversation in Play's chat: the snag box for
 * a turn that failed before Auto's reply began (useChat drops the empty reply
 * when the request itself fails, so there is no reply to hold it), and the
 * Mission Auto suggests once his answer is complete. PlayTrialBanner is the
 * core's TrialExhaustedBanner above the message box, on chat.tsx's two
 * triggers: the workspace's trial snapshot, or the typed trial_exhausted code
 * the last send returned.
 */

/** A spent trial blocked the last send (the core's errorCode). */
export function isTrialBlocked(errorCode: string | null | undefined): boolean {
  return errorCode === TRIAL_EXHAUSTED_CODE
}

/** The chat failed and no reply holds the failure: the newest message is not a reply with an error. */
export function failedBeforeReply(status: string, messages: readonly ChatMessage[], errorCode: string | null): boolean {
  if (status !== 'error' || isTrialBlocked(errorCode)) return false
  const last = messages[messages.length - 1]
  return !(last?.role === 'assistant' && last.error)
}

export interface TurnEndProps {
  readonly status: string
  readonly errorCode: string | null
  readonly messages: readonly ChatMessage[]
  readonly onRetry: () => void
  readonly suggestion: MissionSuggestion | null
  /** The conversation the Mission comes from ('' before the backend named a new chat). */
  readonly chatId: string
}

export function TurnEnd({ status, errorCode, messages, onRetry, suggestion, chatId }: TurnEndProps) {
  return (
    <div className={styles.after}>
      {failedBeforeReply(status, messages, errorCode) && <SnagBox lines={[TURN_FAILED]} onRetry={onRetry} />}
      {status !== 'streaming' && <PlayMissionSuggestion suggestion={suggestion} chatId={chatId} messages={messages} />}
    </div>
  )
}

export function PlayTrialBanner({ errorCode }: { readonly errorCode: string | null }) {
  const workspace = useWorkspaceOptional()?.workspace
  const exhausted = workspace?.onboarding?.trial?.state === 'exhausted' || isTrialBlocked(errorCode)
  if (!exhausted) return null
  return (
    <div className={styles.banner}>
      <TrialExhaustedBanner errorCode={errorCode} />
    </div>
  )
}
