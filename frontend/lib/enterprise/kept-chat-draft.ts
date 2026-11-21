'use client'

/**
 * PRD-277 US-502 — the chat box's unsent text survives a sign-in, for the person who typed it.
 *
 * When the session ends mid-work, or an admin action asks the person to sign in again, the browser
 * leaves the page for the organisation's sign-in and comes back. Just before it leaves,
 * `keepChatDraft()` puts the text the chat box holds in this tab's sessionStorage for 30 minutes
 * (the lib/marketplace/return-after-sign-in.ts pattern), with the id of the person who typed it;
 * the chat box takes it back once when it mounts again, and only when the same person is signed in.
 * Stored: the text, that person's id and the expiry. No chat id, no address, no token.
 *
 * The session provider names the signed-in person (`setDraftOwner`) as soon as the API answers
 * who-am-I, and checks what is kept then and when it mounts (`settleKeptDraft`): an expired or
 * malformed draft, or one another person typed, is removed at once, not only when the chat box
 * asks for it. Sign-out removes it (`forgetChatDraft`). Storage errors (blocked site data, a full
 * store) mean nothing is kept, never a throw.
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

interface KeptDraft {
  readonly text: string
  readonly owner: string
  readonly expiresAt: number
}

let currentDraft: () => string = () => ''
/** The signed-in person's id, as the session provider last named them; null before it knows. */
let draftOwner: string | null = null

function tabStorage(): DraftStorage | null {
  try {
    return window.sessionStorage
  } catch (error) {
    console.warn(LOG_PREFIX, 'sessionStorage is not available', error)
    return null
  }
}

function parseKept(raw: string, now: number): KeptDraft | null {
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value !== 'object' || value === null) return null
    const { text, owner, expiresAt } = value as Record<string, unknown>
    if (typeof text !== 'string' || typeof owner !== 'string' || typeof expiresAt !== 'number') return null
    const live = now < expiresAt && expiresAt <= now + DRAFT_TTL_MS
    return live ? { text: text.slice(0, DRAFT_MAX_LENGTH), owner, expiresAt } : null
  } catch (error) {
    console.warn(LOG_PREFIX, 'the kept chat draft was malformed', error)
    return null
  }
}

/** Remove the kept draft, whoever typed it. */
export function forgetChatDraft(storage: DraftStorage | null = tabStorage()): void {
  if (storage === null) return
  try {
    storage.removeItem(DRAFT_STORAGE_KEY)
  } catch (error) {
    console.warn(LOG_PREFIX, 'the kept chat draft could not be removed', error)
  }
}

/**
 * Remove the kept draft when it has expired or is malformed, or, once the signed-in person is
 * known (`owner`), when someone else typed it. A draft that may still be given back stays.
 */
export function settleKeptDraft(
  owner: string | null = draftOwner,
  storage: DraftStorage | null = tabStorage(),
  now: number = Date.now(),
): void {
  if (storage === null) return
  try {
    const raw = storage.getItem(DRAFT_STORAGE_KEY)
    if (raw === null) return
    const kept = parseKept(raw, now)
    if (kept === null || (owner !== null && kept.owner !== owner)) storage.removeItem(DRAFT_STORAGE_KEY)
  } catch (error) {
    console.warn(LOG_PREFIX, 'the kept chat draft could not be checked', error)
  }
}

/** The session provider names who is signed in (null: nobody yet), and the kept draft is checked against them. */
export function setDraftOwner(owner: string | null, storage: DraftStorage | null = tabStorage(), now: number = Date.now()): void {
  draftOwner = owner
  settleKeptDraft(owner, storage, now)
}

/** Keep `text` (the registered chat box's, by default) for `owner`'s next chat box in this tab; nothing without an owner. */
export function keepChatDraft(
  text: string = currentDraft(),
  storage: DraftStorage | null = tabStorage(),
  now: number = Date.now(),
  owner: string | null = draftOwner,
): void {
  if (!text.trim() || storage === null || owner === null) return
  const kept: KeptDraft = { text: text.slice(0, DRAFT_MAX_LENGTH), owner, expiresAt: now + DRAFT_TTL_MS }
  try {
    storage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(kept))
  } catch (error) {
    console.warn(LOG_PREFIX, 'the chat draft could not be kept', error)
  }
}

/** Once: the text `owner` kept, or null when none was kept, it expired, or someone else kept it. Taking it forgets it. */
export function takeChatDraft(
  storage: DraftStorage | null = tabStorage(),
  now: number = Date.now(),
  owner: string | null = draftOwner,
): string | null {
  if (storage === null || owner === null) return null
  try {
    const raw = storage.getItem(DRAFT_STORAGE_KEY)
    if (raw === null) return null
    storage.removeItem(DRAFT_STORAGE_KEY)
    const kept = parseKept(raw, now)
    return kept !== null && kept.owner === owner ? kept.text : null
  } catch (error) {
    console.warn(LOG_PREFIX, 'the kept chat draft could not be read', error)
    return null
  }
}

/** Enterprise: register the chat box's text, and give it back the text its person kept before a sign-in. */
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
