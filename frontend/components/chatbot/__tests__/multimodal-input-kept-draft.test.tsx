/**
 * PRD-277 US-502 — in enterprise the chat box takes back the text its person kept before signing
 * in again (lib/enterprise/kept-chat-draft.ts), once, and never another person's.
 */
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('@/lib/api-client', () => ({ apiClient: { uploadAttachment: vi.fn() } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('../agent-selector', () => ({ AgentSelector: () => null }))

import { DRAFT_STORAGE_KEY, keepChatDraft, setDraftOwner } from '@/lib/enterprise/kept-chat-draft'

import { MultimodalInput } from '../multimodal-input'

const PLACEHOLDER = 'Send a message...'

function drawChatBox() {
  render(<MultimodalInput chatId="kept-draft" status="ready" stop={vi.fn()} sendMessage={vi.fn()} selectedVisibilityType="private" />)
  return screen.getByPlaceholderText(PLACEHOLDER) as HTMLTextAreaElement
}

afterEach(() => {
  cleanup()
  window.sessionStorage.clear()
  setDraftOwner(null)
})

describe('the chat box after signing in again (enterprise)', () => {
  it('takes back the text its person kept, once', () => {
    keepChatDraft('half a question about the Q3 numbers', window.sessionStorage, Date.now(), '7')
    setDraftOwner('7')

    expect(drawChatBox().value).toBe('half a question about the Q3 numbers')
    expect(window.sessionStorage.getItem(DRAFT_STORAGE_KEY)).toBeNull()
    cleanup()
    expect(drawChatBox().value).toBe('')
  })

  it("never shows another person's kept text", () => {
    keepChatDraft('not for you', window.sessionStorage, Date.now(), '8')
    setDraftOwner('7')

    expect(drawChatBox().value).toBe('')
  })
})
