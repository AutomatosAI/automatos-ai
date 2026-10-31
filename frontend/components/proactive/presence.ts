'use client'

/**
 * PRD-268 §6.12.4 (US-W01b) — a small store of what is on screen, read through useSyncExternalStore.
 *
 * A component marks a value while it shows something (a Mission suggestion waiting for an answer, the suggestion
 * pinned above the composer) and releases it when that ends or it unmounts; another component reads every marked
 * value. Each mark is its own token, so two marks, or one remounting under StrictMode, never release each other's,
 * and a release is safe to call twice. The store replaces its map and its listeners on each change and never
 * mutates them, so a snapshot stays the same object until something changes.
 */
import { useSyncExternalStore } from 'react'

type Listener = () => void
export type Marks<T> = ReadonlyMap<symbol, T>

const NO_MARKS: Marks<never> = new Map()

export interface Presence<T> {
  /** Mark `value` as shown; returns its release. */
  readonly mark: (value: T) => () => void
  /** Every value marked now, by token: the same object until the next change. */
  readonly snapshot: () => Marks<T>
  readonly subscribe: (listener: Listener) => () => void
}

export function createPresence<T>(name: string): Presence<T> {
  let marks: Marks<T> = NO_MARKS
  let listeners: ReadonlySet<Listener> = new Set()
  const emit = () => listeners.forEach((listener) => listener())

  const mark = (value: T) => {
    const token = Symbol(name)
    marks = new Map<symbol, T>([...Array.from(marks.entries()), [token, value]])
    emit()
    return () => {
      if (!marks.has(token)) return
      marks = new Map<symbol, T>(Array.from(marks.entries()).filter(([other]) => other !== token))
      emit()
    }
  }
  const subscribe = (listener: Listener) => {
    listeners = new Set([...Array.from(listeners), listener])
    return () => {
      listeners = new Set(Array.from(listeners).filter((other) => other !== listener))
    }
  }
  return { mark, snapshot: () => marks, subscribe }
}

/** Every value marked in `presence`, re-read on each change; none on the server. */
export function usePresence<T>(presence: Presence<T>): Marks<T> {
  return useSyncExternalStore(presence.subscribe, presence.snapshot, () => NO_MARKS)
}
