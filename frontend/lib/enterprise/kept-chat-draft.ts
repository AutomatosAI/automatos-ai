'use client'

/**
 * PRD-277 US-502 — the chat box's unsent text survives a sign-in.
 *
 * When the session ends mid-work, or an admin action asks the person to sign in again, the browser
 * leaves the page for the organisation's sign-in and comes back. Just before it leaves,
 * `keepChatDraft()` puts the text the chat box holds in this tab's sessionStorage for 30 minutes
 * (the lib/marketplace/return-after-sign-in.ts pattern); the chat box takes it back once when it
 * mounts again. Only the text and its expiry are stored: no chat id, no address, no token.
 * Storage errors (blocked site data, a full store) mean nothing is kept, never a throw.
 *
 * The chat box registers what it holds (`useKeptChatDraft`). In `local` and `saas` the hook does
 * nothing at all (a build-time constant, so the rules of hooks hold).
 */
import { useEffect, useRef } from 'react'

import { isEnterprise } from '@/lib/edition-name'

export const DRAFT_STORAGE_KEY = 'automatos-chat-draft'
export const DRAFT_TTL_MS = 30 * 60 * 1000
/** The longest text kept: a long message still fits a tab's storage. */
export const DRAFT_MAX_LENGTH = 100_000
const LOG_PREFIX = '[chat-draft]'

export type DraftStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

let currentDraft: () => string = () => ''

function tabStorage(): DraftStorage | null {
  try {
    return window.sessionStorage
  } catch (error) {
    console.warn(LOG_PREFIX, 'sessionStorage is not available', error)
    return null
  }
}

/** Keep `text` (the registered chat box's, by default) for the next chat box this tab mounts. */
export function keepChatDraft(
  text: string = currentDraft(),
  storage: DraftStorage | null = tabStorage(),
  now: number = Date.now(),
): void {
  if (!text.trim() || storage === null) return
  try {
    storage.setItem(DRAFT_STORAGE_KEY, JSON.stringify({ text: text.slice(0, DRAFT_MAX_LENGTH), expiresAt: now + DRAFT_TTL_MS }))
  } catch (error) {
    console.warn(LOG_PREFIX, 'the chat draft could not be kept', error)
  }
}

function parseKept(raw: string, now: number): string | null {
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value !== 'object' || value === null) return null
    const { text, expiresAt } = value as Record<string, unknown>
    if (typeof text !== 'string' || typeof expiresAt !== 'number') return null
    return now < expiresAt && expiresAt <= now + DRAFT_TTL_MS ? text.slice(0, DRAFT_MAX_LENGTH) : null
  } catch (error) {
    console.warn(LOG_PREFIX, 'the kept chat draft was malformed', error)
    return null
  }
}

/** Once: the kept text, or null when none was kept or it expired. Taking it forgets it. */
export function takeChatDraft(storage: DraftStorage | null = tabStorage(), now: number = Date.now()): string | null {
  if (storage === null) return null
  try {
    const raw = storage.getItem(DRAFT_STORAGE_KEY)
    if (raw === null) return null
    storage.removeItem(DRAFT_STORAGE_KEY)
    return parseKept(raw, now)
  } catch (error) {
    console.warn(LOG_PREFIX, 'the kept chat draft could not be read', error)
    return null
  }
}

/** Enterprise: register the chat box's text, and give it back the text kept before a sign-in. */
function useEnterpriseKeptChatDraft(text: string, setText: (text: string) => void): void {
  const latest = useRef(text)
  latest.current = text
  useEffect(() => {
    const kept = takeChatDraft()
    if (kept !== null) setText(kept)
    const read = () => latest.current
    currentDraft = read
    return () => {
      if (currentDraft === read) currentDraft = () => ''
    }
    // Once per mounted chat box: the setter is React's own and stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}

function useNoKeptChatDraft(_text: string, _setText: (text: string) => void): void {}

export const useKeptChatDraft: (text: string, setText: (text: string) => void) => void = isEnterprise
  ? useEnterpriseKeptChatDraft
  : useNoKeptChatDraft
