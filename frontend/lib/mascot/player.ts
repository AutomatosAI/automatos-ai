import { type MascotName, ONE_SHOTS, URGENT } from './types'

/**
 * PRD-260 D6 — the mascot player's queue, ported from Automatos Play's
 * lib/mascot/player.ts: which clip is on screen and which waits for the clip
 * boundary. Pure; hooks/use-mascot-player.ts owns the <video> and the timers.
 *
 * Loops play without the `loop` attribute and restart on the boundary: a
 * looping clip never fires `ended`, and timeupdate can come as rarely as every
 * 250 ms, so the 80 ms window alone can miss a clip's end. `ended` is the
 * backstop.
 */

/** Every clip is 5 s long; also how long a one-shot lasts when no video plays. */
export const CLIP_MS = 5000

/** A clip this close to its end has reached the boundary. */
export const BOUNDARY_MS = 80

const MS_PER_SECOND = 1000

export type PlayerState = {
  /** The clip on screen. */
  readonly showing: MascotName
  /** The next clip, swapped in at the boundary. */
  readonly queued: MascotName | null
  /** A one-shot played to its end with nothing queued: it rests on its last frame. */
  readonly finished: boolean
}

/**
 * What the player does with the <video> at a boundary: play the queued clip,
 * play the same loop again, report that a one-shot ended, or nothing (already
 * finished).
 */
export type BoundaryAction = 'swap' | 'restart' | 'finish' | 'none'

export function startPlayer(state: MascotName): PlayerState {
  return { showing: state, queued: null, finished: false }
}

/** True once a clip is within BOUNDARY_MS of its end (or past it). */
export function atClipBoundary(currentTime: number, duration: number): boolean {
  if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(currentTime)) return false
  return (duration - currentTime) * MS_PER_SECOND <= BOUNDARY_MS
}

/**
 * The reducer's target changed. Urgent states (oops, hello) swap at once, and so
 * does anything after a finished one-shot; the rest wait for the boundary. Going
 * back to the clip on screen cancels the queue, so a 200 ms tool call never shows.
 * A finished one-shot re-targeted to itself stays finished and never reports its
 * end again: the player relies on the reducer never re-targeting the one-shot on
 * screen (lib/mascot/reducer.ts enqueue).
 */
export function retarget(player: PlayerState, target: MascotName): PlayerState {
  if (target === player.showing) {
    return player.queued === null ? player : { ...player, queued: null }
  }
  if (URGENT.has(target) || player.finished) return startPlayer(target)
  return player.queued === target ? player : { ...player, queued: target }
}

/** The clip on screen reached its end. */
export function clipBoundary(player: PlayerState): { player: PlayerState; action: BoundaryAction } {
  if (player.finished) return { player, action: 'none' }
  if (player.queued !== null) return { player: startPlayer(player.queued), action: 'swap' }
  if (ONE_SHOTS.has(player.showing)) return { player: { ...player, finished: true }, action: 'finish' }
  return { player, action: 'restart' }
}

/** Without video (reduced motion, a clip that won't play) the target shows at once. */
export function jumpTo(player: PlayerState, target: MascotName): PlayerState {
  const settled = player.showing === target && player.queued === null && !player.finished
  return settled ? player : startPlayer(target)
}
