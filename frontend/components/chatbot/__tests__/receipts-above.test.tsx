/**
 * PRD-256 US-002 — the lines the platform puts above the reply: a write that was refused,
 * and work the reply says is done when no write went through. Live they ride the receipts
 * frame and follow the receipts list; reloaded they are the top of the reply's text.
 */
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ReceiptsBlock } from '../receipts-block'
import { aboveFromFrame, liveReceipts } from '@/lib/chat/receipts'
import type { Receipt } from '@/types'

afterEach(cleanup)

const TRIED = "I tried to approve the mission and it didn't go through: Mission m-1 is not waiting for approval."
const NOT_DONE =
  "Just to be clear: I haven't done that yet, and nothing has changed. Ask me again if you want it done."
const REFUSED: Receipt = {
  action: 'platform_approve_mission', kind: 'write', status: 'refused', subject: '', effect: 'mission approved',
  link: null, reason: 'Mission m-1 is not waiting for approval.',
}
const FRAME = `d:${JSON.stringify({ type: 'receipts', data: { receipts: [REFUSED], above: [TRIED, NOT_DONE] } })}`

describe('the lines above the reply', () => {
  it('come with the live frame and follow the receipts, in order', () => {
    const live = liveReceipts(JSON.parse(FRAME.slice(2)).data)
    expect(live).toEqual({ receipts: [REFUSED], receiptsAbove: [TRIED, NOT_DONE] })

    render(<ReceiptsBlock receipts={live.receipts} above={live.receiptsAbove} />)
    const above = screen.getByTestId('receipts-above')
    expect(above.querySelectorAll('p')).toHaveLength(2)
    expect(above.querySelectorAll('p')[0]).toHaveTextContent(TRIED)
    expect(above.querySelectorAll('p')[1]).toHaveTextContent(NOT_DONE)
    const block = screen.getByTestId('receipts-block')
    expect(block.compareDocumentPosition(above) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('are absent when a write went through, and on a reloaded message', () => {
    expect(liveReceipts({ receipts: [] })).toEqual({ receipts: [], receiptsAbove: undefined })
    render(<ReceiptsBlock receipts={[]} />)
    expect(screen.queryByTestId('receipts-above')).toBeNull()
    expect(screen.getByTestId('receipts-block')).toHaveTextContent('No actions in this turn.')
  })

  it('ignore a malformed list', () => {
    expect(aboveFromFrame({ above: 'not a list' })).toBeUndefined()
    expect(aboveFromFrame({ above: [42, '', '  ', NOT_DONE] })).toEqual([NOT_DONE])
    expect(aboveFromFrame(null)).toBeUndefined()
  })
})
