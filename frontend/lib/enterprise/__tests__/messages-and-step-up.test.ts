/**
 * PRD-277 Wave 5 — the plain messages, the kept chat draft and step-up by signing in again.
 *
 *   - Each sign-in refusal code has one sentence; an unknown code reads as the general one, so the
 *     address bar cannot make the page say anything the app did not write. None says whether a
 *     person is known.
 *   - `step_up_no_auth_time` tells the admin what to set at the identity provider (PRD-277 §5.5).
 *   - The chat draft keeps only its text, its person and its expiry, for 30 minutes, once, and
 *     only for that person; it is removed at sign-out, for another person, and once expired.
 *   - "Your session ended" says why when the API's `session_ended` gave a reason.
 *   - A `step_up_required` refusal naming the API's start route sends the browser there with the
 *     page as `return_to`, after keeping the draft; any other `start` is not followed.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RETURN_TO_PARAM, SIGN_IN_START_PATH } from '@/lib/enterprise/auth-api'
import {
  DRAFT_STORAGE_KEY,
  DRAFT_TTL_MS,
  forgetChatDraft,
  keepChatDraft,
  setDraftOwner,
  settleKeptDraft,
  takeChatDraft,
  type DraftStorage,
} from '@/lib/enterprise/kept-chat-draft'
import {
  SESSION_ENDED_BODY,
  SIGN_IN_REFUSALS,
  SIGN_IN_REFUSAL_FALLBACK,
  actionErrorMessage,
  isSessionEnded,
  sessionEndedBody,
  sessionEndedReason,
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

  it('says why the session ended when the API said, and the general words for any other 401', () => {
    expect(sessionEndedReason({ detail: { code: 'session_ended', reason: 'expired' } })).toBe('expired')
    expect(sessionEndedReason({ detail: { code: 'session_ended', reason: 'unknown' } })).toBeNull()
    expect(sessionEndedReason({ detail: { code: 'session_ended', reason: 'constructor' } })).toBeNull()
    expect(sessionEndedReason({ detail: 'Not authenticated' })).toBeNull()
    expect(sessionEndedReason(undefined)).toBeNull()
    expect(sessionEndedBody('revoked')).toMatch(/administrator/)
    expect(sessionEndedBody('revoked').endsWith(SESSION_ENDED_BODY)).toBe(true)
    expect(sessionEndedBody(null)).toBe(SESSION_ENDED_BODY)
  })

  it("an admin action's refusal reads as the session ending on any 401, else the API's sentence", () => {
    expect(actionErrorMessage(Object.assign(new Error('HTTP 401: Unauthorized'), { body: undefined }))).toContain(SESSION_ENDED_BODY)
    expect(actionErrorMessage(Object.assign(new Error('x'), { status: 409, body: { detail: 'No longer waiting.' } }))).toBe('No longer waiting.')
    expect(actionErrorMessage(Object.assign(new Error('x'), { body: { detail: { message: 'Not now.' } } }))).toBe('Not now.')
    expect(actionErrorMessage('odd')).toBe('Something went wrong. Try again.')
  })
})

const ADA = '7'
const GRACE = '8'

describe('the kept chat draft', () => {
  it('keeps only the text, its person and its expiry, and gives it back once, to that person', () => {
    const storage = memoryStorage()
    keepChatDraft('half a question', storage, 1_000, ADA)
    expect(JSON.parse(storage.items.get(DRAFT_STORAGE_KEY) ?? '')).toEqual({
      text: 'half a question', owner: ADA, expiresAt: 1_000 + DRAFT_TTL_MS,
    })
    expect(takeChatDraft(storage, 2_000, ADA)).toBe('half a question')
    expect(takeChatDraft(storage, 2_000, ADA)).toBeNull()
  })

  it('keeps nothing for an empty box or with nobody signed in, and gives nothing back once expired or malformed', () => {
    const storage = memoryStorage()
    keepChatDraft('   ', storage, 0, ADA)
    keepChatDraft('no one yet', storage, 0, null)
    expect(storage.items.size).toBe(0)
    keepChatDraft('late', storage, 0, ADA)
    expect(takeChatDraft(storage, DRAFT_TTL_MS + 1, ADA)).toBeNull()
    storage.setItem(DRAFT_STORAGE_KEY, '{"text": 5}')
    expect(takeChatDraft(storage, 0, ADA)).toBeNull()
  })

  it('never gives one person the draft another typed, and removes it', () => {
    const storage = memoryStorage()
    keepChatDraft('private words', storage, 0, ADA)
    expect(takeChatDraft(storage, 1, GRACE)).toBeNull()
    expect(storage.items.size).toBe(0)
  })

  it('is removed as soon as another person signs in, or once expired, not only when the chat box asks', () => {
    const storage = memoryStorage()
    keepChatDraft('mine', storage, 0, ADA)
    setDraftOwner(ADA, storage, 1)
    expect(storage.items.size).toBe(1)
    setDraftOwner(GRACE, storage, 1)
    expect(storage.items.size).toBe(0)

    keepChatDraft('mine again', storage, 0, ADA)
    settleKeptDraft(null, storage, DRAFT_TTL_MS + 1)
    expect(storage.items.size).toBe(0)
    setDraftOwner(null, storage, 0)
  })

  it('is removed at sign-out, whoever typed it', () => {
    const storage = memoryStorage()
    keepChatDraft('mine', storage, 0, ADA)
    forgetChatDraft(storage)
    expect(storage.items.size).toBe(0)
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
    expect(() => keepChatDraft('text', refusing, 0, ADA)).not.toThrow()
    expect(takeChatDraft(refusing, 0, ADA)).toBeNull()
    expect(() => settleKeptDraft(ADA, refusing, 0)).not.toThrow()
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
