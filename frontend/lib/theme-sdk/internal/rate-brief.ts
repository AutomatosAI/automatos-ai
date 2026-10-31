/**
 * PRD-268 §6.4 "Feedback" (SDK 1.4.0) — the proactive area's useRateBrief (re-exported by proactive.ts): the thumbs up
 * or down that Auto's Read has on the web, for a theme that draws the brief card itself.
 *
 * It posts the existing digest feedback route (POST /api/activity/digest/feedback, api/activity.py: `{state_hash,
 * rating}`, rating 1 or -1, keyed by the state_hash the read described) through the host's own hook
 * (hooks/use-digest-api.ts useSubmitDigestFeedback), exactly as the host's brief card does
 * (components/proactive/brief-card.tsx). The server keeps one rating per person per state_hash: a second press
 * changes it. A theme is outside code: a state_hash that isn't some text, or a rating that is neither `up` nor `down`,
 * throws a TypeError before anything is sent. The write comes back as the proactive area's other writes do
 * (internal/query-view.ts).
 *
 * Given the brief's state_hash, `rating` is the person's own rating of it, read from GET
 * /api/activity/digest/feedback into the cache entry the host's brief card reads (useDigestRating): a press here
 * shows in the card's thumbs, and a press on the card shows here.
 */
import { useMemo } from 'react'

import { useDigestRating, useSubmitDigestFeedback, type DigestRatingValue } from '@/hooks/use-digest-api'

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

/** Rate the brief, and the person's own rating of the brief the hook was given. */
export interface RateBriefAction extends ThemeAction<RateBriefInput, BriefRated> {
  /** The person's rating of the brief whose state_hash the hook was given: up, down, or null (none yet, or no brief). */
  readonly rating: BriefRating | null
}

/** The route's body: the digest feedback route takes 1 for up and -1 for down. */
interface DigestFeedbackBody {
  readonly state_hash: string
  readonly rating: DigestRatingValue
}

const RATINGS: Readonly<Record<BriefRating, DigestRatingValue>> = Object.freeze({ up: 1, down: -1 })
const NOT_A_RATING = "useRateBrief needs the brief's state_hash and a rating of up or down."
const NOT_A_BRIEF = "useRateBrief needs the brief's state_hash to be some text, or nothing while there is no brief."

function checkedRating(input: RateBriefInput): DigestFeedbackBody {
  const stateHash = input?.state_hash
  const rating = input?.rating
  if (typeof stateHash !== 'string' || stateHash.trim() === '') return refuse(NOT_A_RATING)
  if (rating !== 'up' && rating !== 'down') return refuse(NOT_A_RATING)
  return { state_hash: stateHash, rating: RATINGS[rating] }
}

/** The brief the theme named: its state_hash, or null for none (nothing, or no text yet); anything else throws. */
function checkedBriefHash(stateHash: unknown): string | null {
  if (stateHash === undefined || stateHash === null) return null
  if (typeof stateHash !== 'string') return refuse(NOT_A_BRIEF)
  return stateHash.trim() === '' ? null : stateHash
}

function briefRatingOf(value: DigestRatingValue | null | undefined): BriefRating | null {
  if (value === RATINGS.up) return 'up'
  if (value === RATINGS.down) return 'down'
  return null
}

/**
 * Rate the brief, thumbs up or down, through the host's digest feedback route. Given the brief's state_hash, `rating`
 * is the person's own rating of it; without one (nothing, or an empty one while the brief loads), nothing is read and
 * `rating` is null.
 */
export function useRateBrief(stateHash?: string | null): RateBriefAction {
  const hash = checkedBriefHash(stateHash)
  const action = useActionView<RateBriefInput, DigestFeedbackBody, BriefRated>(useSubmitDigestFeedback(), checkedRating)
  const remembered = useDigestRating(hash).data?.rating
  return useMemo(() => Object.freeze({ ...action, rating: briefRatingOf(remembered) }), [action, remembered])
}
