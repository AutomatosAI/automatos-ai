/**
 * PRD-261 A-10b — SDK area `events` (frontend/lib/theme-sdk/events.ts): the host's app-wide signals a theme may
 * follow, and the one it may send. A handler gets the event's detail as `unknown`: it came off the wire, so a
 * theme checks it before reading it.
 */

export type HostEventName = 'board-changed' | 'chat-changed'
export type HostEventHandler = (detail: unknown) => void

/** The host events a theme may follow. */
export declare const HOST_EVENT_NAMES: readonly HostEventName[]

/** Calls `handler` on every `name` event until the returned function is called. */
export declare function onHostEvent(name: HostEventName, handler: HostEventHandler): () => void

/** onHostEvent for a component's lifetime; the latest handler is called, without resubscribing. */
export declare function useHostEvent(name: HostEventName, handler: HostEventHandler): void

/** Opens the host's global search (the ⌘K dialog). */
export declare function openGlobalSearch(): void
