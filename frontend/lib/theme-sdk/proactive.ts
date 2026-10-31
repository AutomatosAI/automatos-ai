/**
 * PRD-268 US-C21e (K12, SDK 1.3.0) — SDK area `proactive`: the day's brief, what Auto depends on, the open
 * suggestions with Start and Hide, and calibration, for a theme's own placements of the proactive surfaces. The
 * default components that draw them are the cards area's (cards.BriefCard, cards.SuggestionCard, …); Listen is
 * voice.BriefVoice.
 *
 * Each hook wraps the core hook the host's own surfaces read (hooks/use-brief.ts, use-readiness.ts,
 * use-suggestions.ts, use-calibration.ts), so a theme asks the same endpoints, through the same backend client, with
 * the same cache keys, polling and refresh events as the host. useSuggestions also reads who is signed in
 * (hooks/use-profile.ts, GET /api/profile): the one core hook added since, a host fix under the 1.3.0 contract. A read
 * comes back as `{ data, isLoading, isFetching, error, refetch }` and a write as `{ run, runAsync, isRunning, error,
 * result, reset }` (internal/query-view.ts): frozen copies, an error as its message, never react-query's own result.
 * The hooks need the host's providers above them, as every theme page the host draws has.
 *
 * - useBrief: the day's brief once the host has opened it (the Auto thread opens it); reading it never builds one.
 * - useReadiness, useCheckReadiness: Readiness@1, and Check again (the server limits how often a workspace checks).
 * - useSuggestions: the signed-in person's own open suggestions. The host's list (the one behind every surface) gives
 *   an owner or admin the whole workspace's, so only the rows addressed to the signed-in person (`user_id`) reach a
 *   theme, as the host's pinned line narrows to the thread's owner; until the person is known, none do.
 * - useStartSuggestion, useHideSuggestion: Start and Hide post the host's routes only. A theme is outside code, so an
 *   id that isn't a suggestion's, or a reason Hide doesn't know, throws a TypeError before anything is sent.
 * - useCalibration, useAnswerCalibration: inert until the server offers calibration (its routes are US-C22's):
 *   `available` is false, nothing is asked, and an answer is refused without a request.
 */
import { useMemo, useState } from 'react'

import { useBrief as useCoreBrief } from '@/hooks/use-brief'
import {
  useAnswerCalibration as useCoreAnswerCalibration,
  useCalibration as useCoreCalibration,
} from '@/hooks/use-calibration'
import { useProfile } from '@/hooks/use-profile'
import { useCheckReadiness as useCoreCheckReadiness, useReadiness as useCoreReadiness } from '@/hooks/use-readiness'
import {
  useHideSuggestion as useCoreHideSuggestion,
  useStartSuggestion as useCoreStartSuggestion,
  useSuggestions as useCoreSuggestions,
  type HideSuggestionVars,
  type StartSuggestionVars,
} from '@/hooks/use-suggestions'
import { HIDE_REASONS, type CalibrationAnswer as CoreCalibrationAnswer } from '@/lib/proactive/types'
import { isProactiveId } from '@/lib/proactive/wire'

import type {
  Brief,
  Calibration,
  CalibrationAnswer,
  HideAnswer,
  HideReason,
  Readiness,
  StartAnswer,
  Suggestion,
} from './internal/proactive-shapes'
import { frozenCopy, messageOf, refuse, useActionView, useQueryView } from './internal/query-view'

export type {
  Brief,
  BriefActionVerb,
  BriefCountedSection,
  BriefLabels,
  BriefOpensType,
  BriefRow,
  BriefSection,
  Calibration,
  CalibrationAnswer,
  CalibrationLine,
  CalibrationSource,
  DisplayState,
  EvidenceRef,
  HideAnswer,
  HideReason,
  ProposedAction,
  Readiness,
  ReadinessItem,
  ReadinessNeed,
  ReadinessState,
  Retry,
  StartAnswer,
  StartOpens,
  Suggestion,
  SuggestionSeverity,
  SuggestionStatus,
  SuggestionVia,
} from './internal/proactive-shapes'

/** A read: null data until it answers; `isLoading` is the first load only. */
export interface ProactiveQuery<T> {
  readonly data: T | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

/** A write: `run` reports failure in `error`; `runAsync` rejects with it. */
export interface ProactiveAction<V, R> {
  readonly run: (variables: V) => void
  readonly runAsync: (variables: V) => Promise<R>
  readonly isRunning: boolean
  readonly error: string | null
  readonly result: R | null
  readonly reset: () => void
}

/** The person's calibration: none while `available` is false. */
export interface CalibrationQuery extends ProactiveQuery<Calibration> {
  /** False until the server offers calibration. */
  readonly available: boolean
}

/** Sounds right or Reject one line: refused, with nothing sent, while `available` is false. */
export interface CalibrationAction extends ProactiveAction<CalibrationAnswer, void> {
  readonly available: boolean
}

/** Start: the suggestion's id, from useSuggestions. */
export interface StartSuggestionInput {
  readonly id: string
}

/** Hide: the suggestion's id, and the reason the person picked (left out: hidden with no reason). */
export interface HideSuggestionInput {
  readonly id: string
  readonly reason?: HideReason
}

const NOT_A_SUGGESTION_ID = 'needs the id of a suggestion from useSuggestions.'
const NOT_A_HIDE_REASON = `useHideSuggestion needs reason to be one of ${HIDE_REASONS.join(', ')}.`
const NOT_AN_ANSWER = 'useAnswerCalibration needs a calibration_id with an accept, or a reject and its line_id.'
/** Check again sends nothing; calibration has nothing to refetch. */
const NO_OP = (): void => undefined
/** What a person the server has no users row for is handed: no suggestion is addressed to them. */
const NONE_ADDRESSED: readonly Suggestion[] = Object.freeze([])

function checkedStart(input: StartSuggestionInput): StartSuggestionVars {
  return isProactiveId(input?.id) ? { id: input.id } : refuse(`useStartSuggestion ${NOT_A_SUGGESTION_ID}`)
}

function checkedReason(reason: unknown): HideReason | undefined {
  if (reason === undefined) return undefined
  return HIDE_REASONS.find((known) => known === reason) ?? refuse(NOT_A_HIDE_REASON)
}

function checkedHide(input: HideSuggestionInput): HideSuggestionVars {
  if (!isProactiveId(input?.id)) return refuse(`useHideSuggestion ${NOT_A_SUGGESTION_ID}`)
  const reason = checkedReason(input.reason)
  return reason === undefined ? { id: input.id } : { id: input.id, reason }
}

function someText(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0
}

/** The answer as the host sends it, or a TypeError when it is neither an accept nor a reject of one line. */
function checkedAnswer(answer: CalibrationAnswer): CoreCalibrationAnswer {
  const id = answer?.calibration_id
  if (!someText(id)) return refuse(NOT_AN_ANSWER)
  if (answer.answer === 'accept') return { calibration_id: id, answer: 'accept' }
  if (answer.answer === 'reject' && someText(answer.line_id)) return { calibration_id: id, answer: 'reject', line_id: answer.line_id }
  return refuse(NOT_AN_ANSWER)
}

/** The day's brief, once the host has opened it; null until then, and on a message without one. */
export function useBrief(): ProactiveQuery<Brief> {
  return useQueryView<Brief>(useCoreBrief())
}

/** Readiness@1 for the workspace: what Auto depends on, and what couldn't be checked. */
export function useReadiness(): ProactiveQuery<Readiness> {
  return useQueryView<Readiness>(useCoreReadiness())
}

/** Check again: every item now; its answer is the one every reader shows. Over the server's limit it fails. */
export function useCheckReadiness(): ProactiveAction<void, Readiness> {
  return useActionView<void, void, Readiness>(useCoreCheckReadiness(), NO_OP)
}

/** The signed-in person's users.id from GET /api/profile's answer; null when it names no one (or not as an integer). */
function viewerIdOf(profile: { readonly id?: unknown } | undefined): number | null {
  const id = profile?.id
  return typeof id === 'number' && Number.isInteger(id) ? id : null
}

/** The rows addressed to `viewer` (users.id), as a new frozen list of the already frozen rows. */
function addressedTo(rows: readonly Suggestion[], viewer: number | null): readonly Suggestion[] {
  if (viewer === null) return NONE_ADDRESSED
  return Object.freeze(rows.filter((row) => row.user_id === viewer))
}

/**
 * The signed-in person's own open suggestions, newest first, polled and refreshed as the host's list is. No data
 * until both the list and the person (GET /api/profile) have answered; a failed read of the person is the error.
 */
export function useSuggestions(): ProactiveQuery<readonly Suggestion[]> {
  const view = useQueryView<readonly Suggestion[]>(useCoreSuggestions())
  const person = useProfile()
  const known = person.data !== undefined
  const viewer = viewerIdOf(person.data)
  // isLoading (not isInitialLoading): no answer yet, also while the session loads and the read waits.
  const { isLoading: personLoading, isFetching, isError, error, refetch } = person
  return useMemo(
    () =>
      Object.freeze({
        data: view.data === null || !known ? null : addressedTo(view.data, viewer),
        isLoading: view.isLoading || personLoading,
        isFetching: view.isFetching || isFetching,
        error: view.error ?? (known ? null : messageOf(error)),
        refetch: () => {
          view.refetch()
          if (!known && isError) void refetch()
        },
      }),
    [view, known, viewer, personLoading, isFetching, isError, error, refetch],
  )
}

/** Start a suggestion: `{suggestion, opens}`, the ticket it filed or the chat to open. */
export function useStartSuggestion(): ProactiveAction<StartSuggestionInput, StartAnswer> {
  return useActionView<StartSuggestionInput, StartSuggestionVars, StartAnswer>(useCoreStartSuggestion(), checkedStart)
}

/** Hide a suggestion, with the reason the person picked. */
export function useHideSuggestion(): ProactiveAction<HideSuggestionInput, HideAnswer> {
  return useActionView<HideSuggestionInput, HideSuggestionVars, HideAnswer>(useCoreHideSuggestion(), checkedHide)
}

/** The person's calibration in this workspace: none offered on this server yet, and nothing asked. */
export function useCalibration(): CalibrationQuery {
  const read = useCoreCalibration()
  return useMemo(
    () =>
      Object.freeze({
        data: read.data === null ? null : (frozenCopy(read.data) as Calibration),
        available: read.available,
        isLoading: read.isLoading,
        isFetching: read.isLoading,
        error: null,
        refetch: NO_OP,
      }),
    [read],
  )
}

/** Answer the calibration through the host; refused, with nothing sent, while the server offers none. */
export function useAnswerCalibration(): CalibrationAction {
  const answering = useCoreAnswerCalibration()
  const [error, setError] = useState<string | null>(null)
  return useMemo(() => {
    const send = (answer: CalibrationAnswer): Promise<void> => answering.mutate(checkedAnswer(answer))
    return Object.freeze({
      available: answering.available,
      run: (answer: CalibrationAnswer) => {
        void send(answer).then(
          () => setError(null),
          (failure: unknown) => setError(messageOf(failure)),
        )
      },
      runAsync: (answer: CalibrationAnswer) =>
        new Promise<void>((resolve, reject) => {
          void send(answer).then(resolve, reject)
        }),
      isRunning: answering.isLoading,
      error,
      result: null,
      reset: () => setError(null),
    })
  }, [answering, error])
}
