import { INITIAL_MASCOT, mascotReducer } from '@/lib/mascot/reducer'
import type { MascotEvent, MascotName, MascotState } from '@/lib/mascot/types'
import type { Receipt } from '@/types'

/** PRD-260 — replays mascot events and records what Auto shows after each one. */

export function replay(events: readonly MascotEvent[], from: MascotState = INITIAL_MASCOT): MascotState {
  return events.reduce(mascotReducer, from)
}

/** The states shown after each event, with repeats (and the starting state) dropped. */
export function sequence(events: readonly MascotEvent[], from: MascotState = INITIAL_MASCOT): MascotName[] {
  const shown = events.reduce<{ state: MascotState; names: readonly MascotName[] }>(
    (acc, event) => {
      const state = mascotReducer(acc.state, event)
      const last = acc.names.length > 0 ? acc.names[acc.names.length - 1] : from.mascot
      return { state, names: state.mascot === last ? acc.names : [...acc.names, state.mascot] }
    },
    { state: from, names: [] },
  )
  return [...shown.names]
}

/** Freeze a value and everything in it, so a mutating reducer throws in strict mode. */
export function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value)
    Object.values(value).forEach(deepFreeze)
  }
  return value
}

/** A receipt in the core's shape (types/chat.ts). */
export function receipt(status: Receipt['status'], fields: Partial<Receipt> = {}): Receipt {
  return { action: 'a', kind: 'read', status, subject: 's', effect: 'e', link: null, reason: null, ...fields }
}
