/**
 * PRD-256 FX-004 — an ask for the owner's click is a `waiting` receipt: an hourglass and
 * "waiting for you", never "tried, refused" and never a failure colour.
 */
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ReceiptsBlock } from '../receipts-block'
import { WAITING_SUFFIX, receiptLine, receiptsFromFrame } from '@/lib/chat/receipts'
import type { Receipt } from '@/types'

afterEach(cleanup)

const WAITING: Receipt = {
  action: 'platform_update_agent', kind: 'write', status: 'waiting', subject: 'Scout',
  effect: "card raised: change an agent 'Scout' (agent #12)", link: null, reason: null,
}

describe('a waiting receipt', () => {
  it('survives the frame filter', () => {
    expect(receiptsFromFrame({ receipts: [WAITING] })).toEqual([WAITING])
  })

  it('says what the card asks and that it waits for the owner, never refused', () => {
    expect(receiptLine(WAITING)).toBe(`card raised: change an agent 'Scout' (agent #12) — ${WAITING_SUFFIX}`)
    render(<ReceiptsBlock receipts={[WAITING]} />)
    const line = screen.getByTestId('receipt-line')
    expect(line).toHaveAttribute('data-status', 'waiting')
    expect(line).toHaveTextContent('waiting for you')
    expect(line).not.toHaveTextContent('refused')
    expect(screen.getByLabelText('Waiting for you')).toBeInTheDocument()
    expect(screen.queryByLabelText('Refused')).toBeNull()
    expect(screen.queryByRole('link')).toBeNull()
  })
})
