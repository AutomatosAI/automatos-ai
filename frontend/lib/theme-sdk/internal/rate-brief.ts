/**
 * PRD-268 §6.4 "Feedback" (SDK 1.4.0) — the proactive area's useRateBrief (re-exported by proactive.ts): the thumbs up
 * or down that Auto's Read has on the web, for a theme that draws the brief card itself.
 *
 * It posts the existing digest feedback route (POST /api/activity/digest/feedback, api/activity.py: `{state_hash,
 * rating}`, rating 1 or -1, keyed by the state_hash the read described) through the host's own hook
 * (hooks/use-digest-api.ts useSubmitDigestFeedback), exactly as the host's brief card does
 * (components/proactive/brief-card.tsx). A theme is outside code: a state_hash that isn't some text, or a rating that
 * is neither `up` nor `down`, throws a TypeError before anything is sent. The write comes back as the proactive area's
 * other writes do (internal/query-view.ts).
 */
import { useSubmitDigestFeedback } from '@/hooks/use-digest-api'

import { refuse, useActionView, type ThemeAction } from './query-view'

/** Thumbs up or down. */
export type BriefRating = 'up' | 'down'

/** What the person thought of the brief: its `state_hash` (Brief@1), and their thumb. */
export interface RateBriefInput {
  readonly state_hash: string
  readonly rating: BriefRating
}

/** The route's answer. */
export interface BriefRated {
  readonly ok: boolean
}

/** The route's body: the digest feedback route takes 1 for up and -1 for down. */
interface DigestFeedbackBody {
  readonly state_hash: string
  readonly rating: 1 | -1
}

const RATINGS: Readonly<Record<BriefRating, DigestFeedbackBody['rating']>> = Object.freeze({ up: 1, down: -1 })
const NOT_A_RATING = "useRateBrief needs the brief's state_hash and a rating of up or down."

function checkedRating(input: RateBriefInput): DigestFeedbackBody {
  const stateHash = input?.state_hash
  const rating = input?.rating
  if (typeof stateHash !== 'string' || stateHash.trim() === '') return refuse(NOT_A_RATING)
  if (rating !== 'up' && rating !== 'down') return refuse(NOT_A_RATING)
  return { state_hash: stateHash, rating: RATINGS[rating] }
}

/** Rate the brief, thumbs up or down, through the host's digest feedback route. */
export function useRateBrief(): ThemeAction<RateBriefInput, BriefRated> {
  return useActionView<RateBriefInput, DigestFeedbackBody, BriefRated>(useSubmitDigestFeedback(), checkedRating)
}
