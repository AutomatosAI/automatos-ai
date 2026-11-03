'use client'

/**
 * PRD-268 §6.11.5, K15, K17 (US-W06; the SDK's names from US-C21b) — calibration on day one, live.
 *
 * GET /api/auto/calibration (US-C22b) says whether the server offers calibration, whether to ask the person's
 * consent now, and where their calibration stands; every reader shares that one query. The SDK's proactive area
 * wraps `useCalibration` and `useAnswerCalibration` (lib/theme-sdk/proactive.ts), so their types stay as W2 left
 * them (K17): `useCalibration` gives the Calibration@1 and `available`, and `useAnswerCalibration` posts Reject
 * (…/{calibration_id}/lines/{line_id}/reject) and Sounds right (…/{calibration_id}/accept). What those types can't
 * carry is the host's own, beside them: `useCalibrationView` (the whole answer: the offer, the consent line, the
 * lines rejected and accepted, an answer's note) and `useCalibrationConsent` (the consent line's Go ahead and Not
 * now: POST /api/auto/calibration {consent, chat_id, locale}). An answer the server took goes into the shared view at
 * once, a read already in flight is dropped first, and the view is asked again however the answer ended (a 409
 * means it was answered elsewhere). A posted calibration announces its chat (`automatos:chat-changed`), so an open
 * chat shows the message now. While the server answers available false (CALIBRATION_ENABLED ships off) nothing is
 * offered, and every answer is refused with CALIBRATION_UNAVAILABLE without a request. An id that isn't a
 * calibration's or a line's never reaches a request path.
 */
import { useMemo } from 'react'
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import {
  answeredView,
  calibrationViewOf,
  consentedOf,
  isCalibrationLineId,
  NOT_AVAILABLE,
  pageLocale,
  type CalibrationConsented,
  type CalibrationView,
} from '@/lib/proactive/calibration'
import type { Calibration, CalibrationAnswer } from '@/lib/proactive/types'
import { isProactiveId, recordOf } from '@/lib/proactive/wire'

/** Why an answer is refused while the server offers no calibration. */
export const CALIBRATION_UNAVAILABLE = 'not available on this server'
export const CALIBRATION_PATH = '/api/auto/calibration'
/** The window event an open chat merges a background message on (lib/chat/hooks.ts). */
export const CHAT_CHANGED_EVENT = 'automatos:chat-changed'
/** The consent POST's 409, whose body carries the record as it is. */
const ALREADY_OFFERED = 'already_offered'
/** The view changes only on the person's own answers, which ask again themselves; a 409 covers one given elsewhere. */
export const CALIBRATION_STALE_MS = 5 * 60_000

export const calibrationQueryKeys = {
  mine: ['auto', 'calibration'] as const,
}

export interface CalibrationRead {
  readonly data: Calibration | null
  /** False until the server offers calibration. */
  readonly available: boolean
  readonly isLoading: boolean
}

export interface CalibrationAnswering {
  readonly available: boolean
  readonly isLoading: boolean
  /** Sounds right, or Reject one line; resolves once the server has the answer. */
  readonly mutate: (answer: CalibrationAnswer) => Promise<void>
}

/** Go ahead (true) or Not now (false), and the chat the person is in; an id that isn't a chat's is left out. */
export interface CalibrationConsentVars {
  readonly consent: boolean
  readonly chatId?: string
}

function viewPath(): string {
  const locale = pageLocale()
  return locale ? `${CALIBRATION_PATH}?locale=${encodeURIComponent(locale)}` : CALIBRATION_PATH
}

function answerPath(answer: CalibrationAnswer): string {
  if (!isProactiveId(answer.calibration_id)) throw new Error('not a calibration id')
  const base = `${CALIBRATION_PATH}/${answer.calibration_id}`
  if (answer.answer === 'accept') return `${base}/accept`
  if (!isCalibrationLineId(answer.line_id)) throw new Error('not a calibration line id')
  return `${base}/lines/${answer.line_id}/reject`
}

function consentBody({ consent, chatId }: CalibrationConsentVars) {
  const locale = pageLocale()
  return { consent, ...(isProactiveId(chatId) ? { chat_id: chatId } : {}), ...(locale ? { locale } : {}) }
}

function viewIn(queryClient: QueryClient): CalibrationView | undefined {
  return queryClient.getQueryData<CalibrationView>(calibrationQueryKeys.mine)
}

/** The read, keeping the last answer's note: GET carries none, and a read after an answer would drop it. */
async function readView(queryClient: QueryClient): Promise<CalibrationView> {
  const read = calibrationViewOf(await apiClient.request<unknown>(viewPath()))
  return { ...read, note: viewIn(queryClient)?.note ?? null }
}

/** Put what an answer came back with into the shared view, first dropping a read that may have started before it. */
async function holdAnswered(queryClient: QueryClient, body: unknown): Promise<void> {
  await queryClient.cancelQueries({ queryKey: calibrationQueryKeys.mine })
  queryClient.setQueryData<CalibrationView>(calibrationQueryKeys.mine, (before) => answeredView(before ?? NOT_AVAILABLE, body))
}

function askAgain(queryClient: QueryClient): void {
  void queryClient.invalidateQueries({ queryKey: calibrationQueryKeys.mine })
}

/** The record a refused consent carries when it was offered already (409 already_offered); undefined otherwise. */
function alreadyOffered(error: unknown): unknown {
  const body = recordOf((error as { body?: unknown } | null)?.body)
  return body?.code === ALREADY_OFFERED ? body.record ?? {} : undefined
}

function announcePosted(answer: CalibrationConsented): void {
  if (answer.chat_id === null || typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(CHAT_CHANGED_EVENT, { detail: { chat_id: answer.chat_id } }))
}

/** The whole of the server's answer, for the host's own surfaces: the offer, the consent line, what was answered. */
export function useCalibrationView() {
  const queryClient = useQueryClient()
  return useQuery<CalibrationView>({
    queryKey: calibrationQueryKeys.mine,
    queryFn: () => readView(queryClient),
    staleTime: CALIBRATION_STALE_MS,
    refetchOnWindowFocus: false,
  })
}

/** The person's calibration in this workspace: its Calibration@1 once built, and whether the server offers it. */
export function useCalibration(): CalibrationRead {
  const { data, isLoading } = useCalibrationView()
  const view = data ?? NOT_AVAILABLE
  return useMemo(() => Object.freeze({ data: view.calibration, available: view.available, isLoading }), [view, isLoading])
}

/** Reject one line, or Sounds right: refused with CALIBRATION_UNAVAILABLE, and nothing sent, while not available. */
export function useAnswerCalibration(): CalibrationAnswering {
  const queryClient = useQueryClient()
  const available = useCalibrationView().data?.available === true
  const { mutateAsync, isLoading } = useMutation<unknown, Error, CalibrationAnswer>({
    mutationFn: (answer) => apiClient.post<unknown>(answerPath(answer), {}),
    onSuccess: (body) => holdAnswered(queryClient, body),
    onSettled: () => askAgain(queryClient),
  })
  return useMemo(
    () =>
      Object.freeze({
        available,
        isLoading,
        mutate: (answer: CalibrationAnswer): Promise<void> =>
          available ? mutateAsync(answer).then(() => undefined) : Promise.reject(new Error(CALIBRATION_UNAVAILABLE)),
      }),
    [available, isLoading, mutateAsync],
  )
}

/** The consent line's answer: once per person per workspace; refused without a request while not available. */
export function useCalibrationConsent() {
  const queryClient = useQueryClient()
  return useMutation<CalibrationConsented, Error, CalibrationConsentVars>({
    mutationFn: async (vars) => {
      if (viewIn(queryClient)?.available !== true) throw new Error(CALIBRATION_UNAVAILABLE)
      return consentedOf(await apiClient.post<unknown>(CALIBRATION_PATH, consentBody(vars)))
    },
    onSuccess: async (answer) => {
      await holdAnswered(queryClient, answer)
      announcePosted(answer)
    },
    onError: async (error) => {
      const record = alreadyOffered(error)
      if (record !== undefined) await holdAnswered(queryClient, record)
    },
    onSettled: () => askAgain(queryClient),
  })
}
