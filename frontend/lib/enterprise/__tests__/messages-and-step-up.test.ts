/**
 * PRD-277 Wave 5 — the plain messages, the kept chat draft and step-up by signing in again.
 *
 *   - Each sign-in refusal code has one sentence; an unknown code reads as the general one, so the
 *     address bar cannot make the page say anything the app did not write. None says whether a
 *     person is known.
 *   - `step_up_no_auth_time` tells the admin what to set at the identity provider (PRD-277 §5.5).
 *   - The chat draft keeps only its text and expiry, for 30 minutes, once.
 *   - A `step_up_required` refusal naming the API's start route sends the browser there with the
 *     page as `return_to`, after keeping the draft; any other `start` is not followed.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RETURN_TO_PARAM, SIGN_IN_START_PATH } from '@/lib/enterprise/auth-api'
import { DRAFT_STORAGE_KEY, DRAFT_TTL_MS, keepChatDraft, takeChatDraft, type DraftStorage } from '@/lib/enterprise/kept-chat-draft'
import {
  SIGN_IN_REFUSALS,
  SIGN_IN_REFUSAL_FALLBACK,
  isSessionEnded,
  signInRefusalMessage,
} from '@/lib/enterprise/messages'
import { signInAgainFor, signInAgainStartOf } from '@/lib/enterprise/step-up'

function memoryStorage(): DraftStorage & { readonly items: Map<string, string> } {
  const items = new Map<string, string>()
  return {
    items,
    getItem: (key) => items.get(key) ?? null,
    setItem: (key, value) => void items.set(key, value),
    removeItem: (key) => void items.delete(key),
  }
}

afterEach(() => {
  window.sessionStorage.clear()
  vi.restoreAllMocks()
})

describe('sign-in refusal messages', () => {
  it.each(Object.keys(SIGN_IN_REFUSALS))('has a sentence for %s', (code) => {
    expect(signInRefusalMessage(code)).toBe(SIGN_IN_REFUSALS[code])
  })

  it.each(['<script>', 'unknown_person', 'constructor', '__proto__'])('reads %j as the general sentence', (code) => {
    expect(signInRefusalMessage(code)).toBe(SIGN_IN_REFUSAL_FALLBACK)
  })

  it('says nothing when there is no code', () => {
    expect(signInRefusalMessage(null)).toBeNull()
    expect(signInRefusalMessage('')).toBeNull()
  })

  it('never says whether a person is known', () => {
    for (const sentence of Object.values(SIGN_IN_REFUSALS)) {
      expect(sentence).not.toMatch(/no such|not found|unknown|does not exist|deactivated|already (used|taken)/i)
    }
  })

  it("tells an admin to have the provider send auth_time when it does not", () => {
    expect(SIGN_IN_REFUSALS.step_up_no_auth_time).toMatch(/administrator/i)
    expect(SIGN_IN_REFUSALS.step_up_no_auth_time).toMatch(/auth_time/)
  })

  it("recognises the session lane's session_ended", () => {
    expect(isSessionEnded({ detail: { code: 'session_ended', reason: 'revoked' } })).toBe(true)
    expect(isSessionEnded({ detail: 'Not authenticated' })).toBe(false)
    expect(isSessionEnded(undefined)).toBe(false)
  })
})

describe('the kept chat draft', () => {
  it('keeps only the text and its expiry, and gives it back once', () => {
    const storage = memoryStorage()
    keepChatDraft('half a question', storage, 1_000)
    expect(JSON.parse(storage.items.get(DRAFT_STORAGE_KEY) ?? '')).toEqual({ text: 'half a question', expiresAt: 1_000 + DRAFT_TTL_MS })
    expect(takeChatDraft(storage, 2_000)).toBe('half a question')
    expect(takeChatDraft(storage, 2_000)).toBeNull()
  })

  it('keeps nothing for an empty box, and gives nothing back once expired or malformed', () => {
    const storage = memoryStorage()
    keepChatDraft('   ', storage, 0)
    expect(storage.items.size).toBe(0)
    keepChatDraft('late', storage, 0)
    expect(takeChatDraft(storage, DRAFT_TTL_MS + 1)).toBeNull()
    storage.setItem(DRAFT_STORAGE_KEY, '{"text": 5}')
    expect(takeChatDraft(storage, 0)).toBeNull()
  })

  it('never throws when the storage refuses', () => {
    const refusing: DraftStorage = {
      getItem: () => {
        throw new Error('blocked')
      },
      setItem: () => {
        throw new Error('blocked')
      },
      removeItem: () => {},
    }
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    expect(() => keepChatDraft('text', refusing, 0)).not.toThrow()
    expect(takeChatDraft(refusing, 0)).toBeNull()
  })
})

describe('step-up by signing in again', () => {
  const refusal = (start: unknown, status = 403) =>
    Object.assign(new Error('Confirm it is you'), {
      status,
      body: { code: 'step_up_required', detail: 'Confirm it is you', step_up: { start, max_age_minutes: 10 } },
    })

  it("follows only the API's own start route", () => {
    expect(signInAgainStartOf(refusal(`${SIGN_IN_START_PATH}?step_up=1`))).toBe(`${SIGN_IN_START_PATH}?step_up=1`)
    expect(signInAgainStartOf(refusal('https://elsewhere.example.test/api/v1/auth/oidc/start'))).toBeNull()
    expect(signInAgainStartOf(refusal('/api/v1/auth/other?step_up=1'))).toBeNull()
    expect(signInAgainStartOf(refusal(`${SIGN_IN_START_PATH}?step_up=1`, 401))).toBeNull()
    expect(signInAgainStartOf(new Error('plain'))).toBeNull()
  })

  it('keeps the draft and sends the browser to sign in again, back to this page', () => {
    window.history.replaceState(null, '', '/team?tab=people')
    const navigate = vi.fn()
    const pending = signInAgainFor(`${SIGN_IN_START_PATH}?step_up=1`, navigate)

    expect(pending).toBeInstanceOf(Promise)
    const target = new URL(navigate.mock.calls[0][0], 'https://automatos.example.test')
    expect(target.pathname).toBe(SIGN_IN_START_PATH)
    expect(target.searchParams.get('step_up')).toBe('1')
    expect(target.searchParams.get(RETURN_TO_PARAM)).toBe('/team?tab=people')
  })
})
