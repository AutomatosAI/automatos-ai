/**
 * PRD-207 — mic-health: is the capture actually delivering signal?
 *
 * A live microphone always shows a noise floor comfortably above digital
 * silence; a capture bound to a dead or virtual device (a continuity iPhone
 * that never picked up, a recorder extension's loopback, an unplugged USB
 * mic) delivers flat zeros while the UI says "Mic live". First armed
 * morning three of five calls streamed exactly that — Retell's ASR received
 * no audio and the call sat "recording" with nothing to transcribe.
 *
 * Pure + immutable (explicit `now`, no Date.now inside) so the detector is
 * unit-testable, the orb-state discipline.
 */

export interface MicHealth {
  /** Start of the current observation window (ms). */
  windowStartAt: number
  /** Loudest RMS level seen inside the current window. */
  peak: number
  /** True once a full window stayed at digital silence. */
  silent: boolean
}

/** Peak RMS below this across a full window = the capture is delivering silence. */
export const MIC_SILENT_PEAK = 0.004
/** How long the capture must stay flat before we call it silent. */
export const MIC_SILENT_AFTER_MS = 3000

export function initialMicHealth(now: number): MicHealth {
  return { windowStartAt: now, peak: 0, silent: false }
}

/**
 * Fold one analyser reading in. Any real signal clears `silent` instantly
 * (a recovering mic must never keep wearing the banner); a full window of
 * near-zero peaks raises it and keeps it raised window over window.
 */
export function feedMicLevel(prev: MicHealth, level: number, now: number): MicHealth {
  if (level >= MIC_SILENT_PEAK) {
    return { windowStartAt: now, peak: level, silent: false }
  }
  const peak = Math.max(prev.peak, level)
  if (now - prev.windowStartAt >= MIC_SILENT_AFTER_MS) {
    return { windowStartAt: now, peak: 0, silent: true }
  }
  return { windowStartAt: prev.windowStartAt, peak, silent: prev.silent }
}
