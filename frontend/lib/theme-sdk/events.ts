/**
 * PRD-261 A-10a — SDK area `events`: the host's app-wide signals a theme may follow, and the
 * one it may send. The host fans them out as window events today ('automatos:board-changed' and
 * 'automatos:chat-changed' from hooks/use-board-event-stream.ts, 'automatos:global-search-open'
 * from lib/global-search-event.ts); a theme names them by the SDK's short names and never by
 * the window event's, so the host may move them without breaking a theme.
 *
 * A handler gets the event's detail as `unknown`: it came off the wire (the board stream), so a
 * theme checks it before reading it.
 */
import { useEffect, useRef } from 'react'

import { openGlobalSearch as openHostGlobalSearch } from '@/lib/global-search-event'

export type HostEventName = 'board-changed' | 'chat-changed'
export type HostEventHandler = (detail: unknown) => void

const WINDOW_EVENTS: Readonly<Record<HostEventName, string>> = {
  'board-changed': 'automatos:board-changed',
  'chat-changed': 'automatos:chat-changed',
}

/** The host events a theme may follow. */
export const HOST_EVENT_NAMES: readonly HostEventName[] = Object.freeze(Object.keys(WINDOW_EVENTS) as HostEventName[])

function windowEventOf(name: unknown): string {
  if (typeof name !== 'string' || !Object.prototype.hasOwnProperty.call(WINDOW_EVENTS, name)) {
    throw new TypeError(`Automatos theme SDK: there is no host event "${String(name)}".`)
  }
  return WINDOW_EVENTS[name as HostEventName]
}

/** Calls `handler` on every `name` event until the returned function is called. */
export function onHostEvent(name: HostEventName, handler: HostEventHandler): () => void {
  const type = windowEventOf(name)
  if (typeof handler !== 'function') throw new TypeError('Automatos theme SDK: onHostEvent needs a handler function.')
  if (typeof window === 'undefined') return () => undefined
  const listener = (event: Event) => handler(event instanceof CustomEvent ? event.detail : undefined)
  window.addEventListener(type, listener)
  return () => window.removeEventListener(type, listener)
}

/** onHostEvent for a component's lifetime; the latest handler is called, without resubscribing. */
export function useHostEvent(name: HostEventName, handler: HostEventHandler): void {
  const latest = useRef(handler)
  useEffect(() => {
    latest.current = handler
  })
  useEffect(() => onHostEvent(name, (detail) => latest.current(detail)), [name])
}

/** Opens the host's global search (the ⌘K dialog). */
export function openGlobalSearch(): void {
  openHostGlobalSearch()
}
