/**
 * Theme voice, web quick-ask: levels() for a waveform or a mascot, read inside an animation frame. Neither level is a
 * measured audio signal: the browser's recognition keeps the microphone's audio to itself, and a second capture of
 * the same microphone stops recognition in some browsers (Safari), so the member's level is an envelope of
 * recognition activity; speech synthesis exposes no audio either, so Auto's is an envelope of speaking. A
 * microphone-silence check must not run on the member's level in quick-ask: it rests at 0 between words.
 */
import { AGENT_LEVEL_FLOOR, AGENT_LEVEL_PEAK, LEVEL_DECAY_MS, USER_LEVEL_PEAK } from './config'

/** A pulse at `at` falling from `peak` to 0 over LEVEL_DECAY_MS; 0 with no pulse yet. */
export function pulse(now: number, at: number | null, peak: number): number {
  if (at === null) return 0
  const since = now - at
  if (since < 0 || since >= LEVEL_DECAY_MS) return 0
  return peak * (1 - since / LEVEL_DECAY_MS)
}

/** The member's level: lifted by each recognised word while the microphone is open, else 0. */
export function userLevel(micOpen: boolean, now: number, lastWordsAt: number | null): number {
  return micOpen ? pulse(now, lastWordsAt, USER_LEVEL_PEAK) : 0
}

/** Auto's level: at least the floor while speaking, lifted by each word boundary; 0 when silent. */
export function agentLevel(speaking: boolean, now: number, lastBoundaryAt: number | null): number {
  return speaking ? Math.max(AGENT_LEVEL_FLOOR, pulse(now, lastBoundaryAt, AGENT_LEVEL_PEAK)) : 0
}
