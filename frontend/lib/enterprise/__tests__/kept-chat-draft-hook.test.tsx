/**
 * PRD-277 US-502 — the chat draft survives a session ending, in the enterprise build.
 *
 * The chat box registers its text (useKeptChatDraft). When "Your session ended" sends the
 * browser to sign in, keepChatDraft() stores that text in this tab; the next chat box mounted
 * gets it back once. Browser storage holds only the text and its expiry.
 */
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))

import { DRAFT_STORAGE_KEY, keepChatDraft, useKeptChatDraft } from '@/lib/enterprise/kept-chat-draft'

afterEach(() => window.sessionStorage.clear())

describe('useKeptChatDraft (enterprise)', () => {
  it('keeps what the chat box holds and gives it back to the next one, once', () => {
    const first = renderHook(({ text }) => useKeptChatDraft(text, vi.fn()), { initialProps: { text: 'draft one' } })
    first.rerender({ text: 'draft one, longer' })
    keepChatDraft()
    first.unmount()

    const stored = JSON.parse(window.sessionStorage.getItem(DRAFT_STORAGE_KEY) ?? '{}')
    expect(Object.keys(stored).sort()).toEqual(['expiresAt', 'text'])
    expect(stored.text).toBe('draft one, longer')

    const setText = vi.fn()
    renderHook(() => useKeptChatDraft('', setText))
    expect(setText).toHaveBeenCalledWith('draft one, longer')
    expect(window.sessionStorage.getItem(DRAFT_STORAGE_KEY)).toBeNull()
  })

  it('keeps nothing once the chat box is gone', () => {
    const box = renderHook(() => useKeptChatDraft('gone', vi.fn()))
    box.unmount()
    keepChatDraft()
    expect(window.sessionStorage.getItem(DRAFT_STORAGE_KEY)).toBeNull()
  })
})
