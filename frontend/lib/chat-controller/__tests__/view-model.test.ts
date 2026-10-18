/**
 * PRD-261 A-11a (M28) — the chat controller's pure view-model builders
 * (lib/chat-controller/view-model.ts): the chat URL less a link's parameters,
 * a new chat's view, and a stored conversation's view.
 */
import { describe, expect, it } from 'vitest'

import { CHAT_ID_PARAM, chatUrlWithout, draftView, storedView } from '@/lib/chat-controller/view-model'
import { CHAT_HREF } from '@/lib/ticket-links'
import type { Chat, ChatMessage } from '@/types'

describe('the chat view model', () => {
  it('drops only the named parameters from the chat URL, and never changes the params it read', () => {
    const params = new URLSearchParams(`${CHAT_ID_PARAM}=c1&draft=hi&tab=x`)
    expect(chatUrlWithout(params, CHAT_ID_PARAM, 'draft')).toBe(`${CHAT_HREF}?tab=x`)
    expect(chatUrlWithout(params, CHAT_ID_PARAM, 'draft', 'tab')).toBe(CHAT_HREF)
    expect(chatUrlWithout(null, CHAT_ID_PARAM)).toBe(CHAT_HREF)
    expect(params.toString()).toBe(`${CHAT_ID_PARAM}=c1&draft=hi&tab=x`)
  })

  it("keys a new chat by its serial, with nothing in it and no id until the backend names it", () => {
    expect(draftView(3)).toEqual({ key: 'draft-3', chatId: null, title: null, messages: [], awaitingReply: false })
    expect(draftView(4).key).not.toBe(draftView(3).key)
  })

  it("keys a stored conversation by its id, with its title and whether the server still writes its reply", () => {
    const messages = [{ id: 'u1', role: 'user', parts: [] }] as unknown as ChatMessage[]
    const chat = { id: 'c1', title: 'Laptops', turnInFlight: true } as unknown as Chat
    expect(storedView('c1', chat, messages)).toEqual({
      key: 'c1',
      chatId: 'c1',
      title: 'Laptops',
      messages,
      awaitingReply: true,
    })
    const untitled = { id: 'c2' } as unknown as Chat
    expect(storedView('c2', untitled, [])).toMatchObject({ title: null, awaitingReply: false })
  })
})
