/**
 * PRD-277 US-502 — the chat draft survives a session ending, in the enterprise build, for the
 * person who typed it.
 *
 * The chat box registers its text (useKeptChatDraft). When "Your session ended" sends the
 * browser to sign in, keepChatDraft() stores that text in this tab with the signed-in person's id;
 * the next chat box mounted gets it back once, and only while the same person is signed in.
 * Browser storage holds only the text, the person's id and its expiry.
 */
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))

import { DRAFT_STORAGE_KEY, keepChatDraft, setDraftOwner, useKeptChatDraft } from '@/lib/enterprise/kept-chat-draft'

const ADA = '7'

afterEach(() => {
  window.sessionStorage.clear()
  setDraftOwner(null)
})

describe('useKeptChatDraft (enterprise)', () => {
  it('keeps what the chat box holds and gives it back to the next one, once', () => {
    setDraftOwner(ADA)
    const first = renderHook(({ text }) => useKeptChatDraft(text, vi.fn()), { initialProps: { text: 'draft one' } })
    first.rerender({ text: 'draft one, longer' })
    keepChatDraft()
    first.unmount()

    const stored = JSON.parse(window.sessionStorage.getItem(DRAFT_STORAGE_KEY) ?? '{}')
    expect(Object.keys(stored).sort()).toEqual(['expiresAt', 'owner', 'text'])
    expect(stored.text).toBe('draft one, longer')
    expect(stored.owner).toBe(ADA)

    const setText = vi.fn()
    renderHook(() => useKeptChatDraft('', setText))
    expect(setText).toHaveBeenCalledWith('draft one, longer')
    expect(window.sessionStorage.getItem(DRAFT_STORAGE_KEY)).toBeNull()
  })

  it('gives nothing back to another person, and removes it when they sign in', () => {
    setDraftOwner(ADA)
    const box = renderHook(() => useKeptChatDraft('only for Ada', vi.fn()))
    keepChatDraft()
    box.unmount()

    setDraftOwner('8')
    expect(window.sessionStorage.getItem(DRAFT_STORAGE_KEY)).toBeNull()
    const setText = vi.fn()
    renderHook(() => useKeptChatDraft('', setText))
    expect(setText).not.toHaveBeenCalled()
  })

  it('keeps nothing once the chat box is gone', () => {
    setDraftOwner(ADA)
    const box = renderHook(() => useKeptChatDraft('gone', vi.fn()))
    box.unmount()
    keepChatDraft()
    expect(window.sessionStorage.getItem(DRAFT_STORAGE_KEY)).toBeNull()
  })
})
