/**
 * PRD-260 P260-FX-08 (M46) — the approval cards Play's chat filed this page
 * session, by conversation: each change is approvals.ts' pure fileApprovals or
 * withStatus on that conversation's cards, and a new chat's cards ('') go with
 * the name the backend gives it.
 */
import { beforeEach, describe, expect, it } from 'vitest'

import { approvalsInPart } from '@/components/play/chat/approvals'
import { cardsOfChat, usePlayApprovalsStore } from '@/stores/play-approvals-store'

const TOOL_PART = { type: 'tool-data', data: { tool_approval: { grant_id: 42, action: 'platform_delete_document' } } }
const MISSION_PART = { type: 'tool-data', data: { mission_approval: { mission_id: 'm-1', goal: 'Refund the customer', tasks: [] } } }
const AT = new Date(0)
const store = () => usePlayApprovalsStore.getState()
const keysOf = (chatId: string) => cardsOfChat(store().filed, chatId).map((card) => card.key)

beforeEach(() => {
  usePlayApprovalsStore.setState({ filed: [] })
})

describe('play-approvals-store', () => {
  it('files a card once, under the reply that raised it, waiting, in its own conversation', () => {
    expect(store().file('c1', approvalsInPart(TOOL_PART), 'r1', AT)).toBe(true)
    const filed = store().filed
    expect(store().file('c1', approvalsInPart(TOOL_PART), 'r2', AT)).toBe(false)
    expect(store().filed).toBe(filed)
    expect(cardsOfChat(store().filed, 'c1')).toEqual([
      expect.objectContaining({ key: 'tool-42', replyId: 'r1', status: 'waiting', filedAt: AT, chatId: 'c1' }),
    ])
    expect(keysOf('c2')).toEqual([])
  })

  it("changes a card's status in its own conversation only; a closed card stays closed", () => {
    store().file('c1', approvalsInPart(TOOL_PART), 'r1', AT)
    store().file('c2', approvalsInPart(MISSION_PART), 'r9', AT)
    const filed = store().filed
    store().setStatus('c2', 'tool-42', 'answered')
    expect(store().filed).toBe(filed)

    store().setStatus('c1', 'tool-42', 'closed')
    store().setStatus('c1', 'tool-42', 'answered')
    expect(cardsOfChat(store().filed, 'c1').map((card) => card.status)).toEqual(['closed'])
    expect(cardsOfChat(store().filed, 'c2').map((card) => card.status)).toEqual(['waiting'])
  })

  it("moves a new chat's cards ('') to the name the backend gives it, and no other conversation's", () => {
    store().file('', approvalsInPart(TOOL_PART), 'r1', AT)
    store().file('c2', approvalsInPart(MISSION_PART), 'r9', AT)
    store().rename('', 'c9')
    expect(keysOf('')).toEqual([])
    expect(keysOf('c9')).toEqual(['tool-42'])
    expect(keysOf('c2')).toEqual(['mission-m-1'])

    const named = store().filed
    store().rename('c9', 'c9')
    store().rename('', 'c10')
    expect(store().filed).toBe(named)
  })
})
