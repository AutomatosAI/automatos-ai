/**
 * PRD-239 S4 — the error frames a turn can end with, parsed into one shape.
 */
import { describe, it, expect } from 'vitest'
import { errorFromDataPayload, parseErrorFrame, FALLBACK_TURN_ERROR } from '../errors'

describe('parseErrorFrame', () => {
  it('reads the {message, code} shape the backend emits', () => {
    expect(parseErrorFrame('{"message":"Researcher\'s model is not available","code":"model_unavailable"}')).toEqual({
      message: "Researcher's model is not available",
      code: 'model_unavailable',
    })
  })

  it('reads a bare JSON string (older frames) and raw text', () => {
    expect(parseErrorFrame('"turn died"')).toEqual({ message: 'turn died', code: null })
    expect(parseErrorFrame('not json at all')).toEqual({ message: 'not json at all', code: null })
  })

  it('never returns an empty sentence', () => {
    expect(parseErrorFrame('')).toEqual({ message: FALLBACK_TURN_ERROR, code: null })
    expect(parseErrorFrame('{}')).toEqual({ message: FALLBACK_TURN_ERROR, code: null })
    expect(parseErrorFrame('{"message":"","code":"x"}').message).toBe(FALLBACK_TURN_ERROR)
  })
})

describe('errorFromDataPayload', () => {
  it('prefers error over message and accepts error_code', () => {
    expect(errorFromDataPayload({ type: 'error', error: 'spent', error_code: 'trial_exhausted' })).toEqual({
      message: 'spent',
      code: 'trial_exhausted',
    })
    expect(errorFromDataPayload({ type: 'error', message: 'm' })).toEqual({ message: 'm', code: null })
    expect(errorFromDataPayload(null)).toEqual({ message: FALLBACK_TURN_ERROR, code: null })
  })
})
