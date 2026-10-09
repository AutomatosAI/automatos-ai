/**
 * PRD-256 US-001 — the receipts block: what the turn's calls did, written by the
 * platform, above the reply. Live from the `receipts` frame, on reload from the
 * saved part; a message from before receipts renders as it always did.
 */
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ReceiptsBlock } from '../receipts-block'
import { receiptLine, receiptsFromFrame, receiptsOf } from '@/lib/chat/receipts'
import type { MessagePart, Receipt } from '@/types'

afterEach(cleanup)

const MOVED: Receipt = {
  action: 'platform_update_task_status', kind: 'write', status: 'done', subject: '#0422',
  effect: 'moved to Done', link: '/command-center?tab=board&task_id=422', reason: null,
}
const REFUSED: Receipt = {
  action: 'generate_document', kind: 'write', status: 'refused', subject: 'Letter to Maya Osei',
  effect: 'document made', link: null, reason: 'DATA_BAD_JSON: the document data is not valid JSON.',
}
const MEMORY: Receipt = {
  action: 'store_memory', kind: 'write', status: 'done', subject: '', effect: 'memory saved', link: null, reason: null,
}
const FRAME = `d:${JSON.stringify({ type: 'receipts', data: { receipts: [MOVED], model: 'google/gemini-2.5-flash' } })}`

const lines = () => screen.getAllByTestId('receipt-line')

describe('receipts', () => {
  it('renders the live frame: one line per call, a done card linked to the board', () => {
    const receipts = receiptsFromFrame(JSON.parse(FRAME.slice(2)).data)
    render(<ReceiptsBlock receipts={receiptsOf({ receipts, parts: [] })} />)
    expect(lines()).toHaveLength(1)
    expect(lines()[0]).toHaveTextContent('#0422: moved to Done')
    expect(screen.getByRole('link')).toHaveAttribute('href', '/command-center?tab=board&task_id=422')
    expect(screen.queryByText('No actions on the board.')).toBeNull()
  })

  it('renders from the saved part on reload', () => {
    const parts: MessagePart[] = [{ type: 'receipts', receipts: [MOVED] }, { type: 'text', text: 'Approved.' }]
    render(<ReceiptsBlock receipts={receiptsOf({ parts })} />)
    expect(lines()[0]).toHaveTextContent('#0422: moved to Done')
  })

  it('says so when the turn ran nothing', () => {
    render(<ReceiptsBlock receipts={receiptsOf({ parts: [{ type: 'receipts', receipts: [] }] })} />)
    expect(screen.getByTestId('receipts-block')).toHaveTextContent('No actions in this turn.')
    expect(screen.queryAllByTestId('receipt-line')).toHaveLength(0)
  })

  it('says a refused write was tried and refused, with its reason, and links nothing', () => {
    render(<ReceiptsBlock receipts={[REFUSED]} />)
    expect(lines()[0]).toHaveTextContent('Letter to Maya Osei: tried, refused: DATA_BAD_JSON')
    expect(lines()[0]).not.toHaveTextContent('document made')
    expect(screen.queryByRole('link')).toBeNull()
    expect(receiptLine(REFUSED)).toBe('Letter to Maya Osei: tried, refused: DATA_BAD_JSON: the document data is not valid JSON.')
  })

  it('says nothing moved on the board when the only write was to memory', () => {
    render(<ReceiptsBlock receipts={[MEMORY]} />)
    expect(lines()[0]).toHaveTextContent('memory saved')
    expect(screen.getByTestId('receipts-block')).toHaveTextContent('No actions on the board.')
  })

  it('renders nothing for a message from before receipts', () => {
    const parts: MessagePart[] = [{ type: 'text', text: 'Hello.' }]
    expect(receiptsOf({ parts })).toBeUndefined()
    const { container } = render(<ReceiptsBlock receipts={receiptsOf({ parts })} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('ignores a malformed frame and links only to pages in the app', () => {
    expect(receiptsFromFrame({})).toBeUndefined()
    expect(receiptsFromFrame({ receipts: [{ nonsense: true }, MOVED] })).toEqual([MOVED])
    render(<ReceiptsBlock receipts={[{ ...MOVED, link: 'https://evil.example/x' }]} />)
    expect(screen.queryByRole('link')).toBeNull()
  })
})
