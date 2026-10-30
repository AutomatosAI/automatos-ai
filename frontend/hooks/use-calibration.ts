'use client'

/**
 * PRD-268 §6.11.5, K12 (US-C21b) — calibration on day one: inert until its routes exist.
 *
 * The SDK's proactive area names `useCalibration` and `useAnswerCalibration` now, so a theme can
 * place the calibration card from 1.3.0 on; the routes (POST /api/auto/calibration and its accept
 * and reject) are US-C22's, in W3. Until then nothing is offered and nothing is asked: the
 * calibration is null and unavailable, and an answer is refused here, without a request.
 * CalibrationCard renders from the props it is given (K23).
 */
import type { Calibration, CalibrationAnswer } from '@/lib/proactive/types'

/** Why an answer is refused while the server has no calibration routes. */
export const CALIBRATION_UNAVAILABLE = 'not available on this server'

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

const NOTHING_OFFERED: CalibrationRead = Object.freeze({ data: null, available: false, isLoading: false })

const NOTHING_ASKED: CalibrationAnswering = Object.freeze({
  available: false,
  isLoading: false,
  mutate: (): Promise<void> => Promise.reject(new Error(CALIBRATION_UNAVAILABLE)),
})

/** The person's calibration in this workspace: none offered on this server yet. */
export function useCalibration(): CalibrationRead {
  return NOTHING_OFFERED
}

/** Answer the calibration: refused with CALIBRATION_UNAVAILABLE, and nothing is sent. */
export function useAnswerCalibration(): CalibrationAnswering {
  return NOTHING_ASKED
}
