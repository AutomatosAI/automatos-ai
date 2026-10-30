// PRD-265 US-008 — Studio's chat shows the paused banner as soon as the page opens, not only after a refused
// send (lib/chat/use-studio-chat.ts). The chat page reads one `errorCode`; this hook reports the billing
// summary's paused state through it. useChatWithReceipts and the summary are stand-ins here; isWorkPaused is real.
import { readFileSync } from 'fs'
import path from 'path'

import { renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { BillingSummary } from '@/hooks/use-billing-summary'

const chat = vi.hoisted(() => ({ errorCode: null as string | null, messages: [] as unknown[] }))
const billing = vi.hoisted(() => ({ summary: undefined as Partial<BillingSummary> | undefined }))

vi.mock('@/lib/chat/use-chat-with-receipts', () => ({
  useChatWithReceipts: () => ({ errorCode: chat.errorCode, messages: chat.messages, reload: vi.fn() }),
}))
vi.mock('@/hooks/use-billing-summary', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/hooks/use-billing-summary')>()
  return { ...actual, useBillingSummary: () => ({ data: billing.summary }) }
})

import { errorCodeWithPause, useStudioChat } from '@/lib/chat/use-studio-chat'
import { CREDITS_EXHAUSTED_CODE } from '@/lib/credits'

const OPTIONS = {} as Parameters<typeof useStudioChat>[0]
const codeFor = () => renderHook(() => useStudioChat(OPTIONS)).result.current.errorCode

beforeEach(() => {
  chat.errorCode = null
  chat.messages = []
  billing.summary = undefined
})

describe('the chat page sees the paused state on load', () => {
  it('a workspace already out of credits reads credits_exhausted before anything is sent', () => {
    billing.summary = { paused: true, own_keys: false }
    expect(codeFor()).toBe(CREDITS_EXHAUSTED_CODE)
  })

  it('a workspace with keys of its own carries on, so it is not shown as paused', () => {
    billing.summary = { paused: true, own_keys: true }
    expect(codeFor()).toBeNull()
  })

  it('a refused send still shows the banner before the summary catches up', () => {
    chat.errorCode = CREDITS_EXHAUSTED_CODE
    billing.summary = { paused: false, own_keys: false }
    expect(codeFor()).toBe(CREDITS_EXHAUSTED_CODE)
  })

  it('with no summary (the local edition, or still loading) the last send decides alone', () => {
    expect(codeFor()).toBeNull()
    chat.errorCode = 'some_other_code'
    expect(codeFor()).toBe('some_other_code')
  })

  it('everything else the chat hook returns passes through untouched', () => {
    const messages = [{ id: 'm1' }]
    chat.messages = messages
    expect(renderHook(() => useStudioChat(OPTIONS)).result.current.messages).toBe(messages)
  })
})

describe('errorCodeWithPause', () => {
  it('paused wins; otherwise the last send keeps its own code', () => {
    expect(errorCodeWithPause(null, true)).toBe(CREDITS_EXHAUSTED_CODE)
    expect(errorCodeWithPause('some_other_code', true)).toBe(CREDITS_EXHAUSTED_CODE)
    expect(errorCodeWithPause('some_other_code', false)).toBe('some_other_code')
    expect(errorCodeWithPause(null, false)).toBeNull()
  })
})

describe('the wiring', () => {
  it("Studio's chat page streams through this hook", () => {
    const chatPage = readFileSync(path.resolve(__dirname, '../../../components/chatbot/chat.tsx'), 'utf8')
    expect(chatPage).toContain("import { useStudioChat as useChat } from '@/lib/chat/use-studio-chat'")
  })
})
