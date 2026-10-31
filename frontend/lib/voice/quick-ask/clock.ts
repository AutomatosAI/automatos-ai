/** Theme voice, web quick-ask: time, injectable so the engine's tests run on a clock they move by hand. */
export interface QuickAskClock {
  now(): number
  setTimeout(callback: () => void, ms: number): unknown
  clearTimeout(handle: unknown): void
}

/** The page's own clock and timers. */
export const SYSTEM_CLOCK: QuickAskClock = Object.freeze({
  now: () => Date.now(),
  setTimeout: (callback: () => void, ms: number): unknown => globalThis.setTimeout(callback, ms),
  clearTimeout: (handle: unknown) => globalThis.clearTimeout(handle as Parameters<typeof globalThis.clearTimeout>[0]),
})
