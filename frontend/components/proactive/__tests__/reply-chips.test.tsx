/**
 * PRD-268 §6.12.3, K17, K23 (US-C21d) — ReplyChips, from its props: two or three chips of at most
 * 24 characters, under Auto's last message only and gone once a newer message exists; a tap sends
 * the chip's own text, once; each chip named "Reply: <chip>"; the chip's words marked as Auto's
 * (data-ai-generated); and the same parts in Studio light, Studio dark and Classic.
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { REPLY_CHIP_MAX_CHARS, ReplyChips, replyChipsOf } from '@/components/proactive/reply-chips'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const CHIPS = ['Window', 'Aisle', 'No preference']

const chipButtons = () => screen.queryAllByRole('button', { name: /^Reply: / })

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ReplyChips', () => {
  it('shows each chip, named "Reply: <chip>", under Auto\'s last message', () => {
    render(<ReplyChips chips={CHIPS} latest onReply={vi.fn()} />)
    expect(screen.getByRole('group', { name: 'Replies' })).toBeInTheDocument()
    expect(chipButtons().map((chip) => chip.getAttribute('aria-label'))).toEqual(['Reply: Window', 'Reply: Aisle', 'Reply: No preference'])
  })

  it('sends the chip’s own text as the person’s message, once', () => {
    const onReply = vi.fn()
    render(<ReplyChips chips={CHIPS} latest onReply={onReply} />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Window' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Aisle' }))
    expect(onReply).toHaveBeenCalledTimes(1)
    expect(onReply).toHaveBeenCalledWith('Window')
    chipButtons().forEach((chip) => expect(chip).toBeDisabled())
  })

  it('gives the chips back when the send fails', async () => {
    const onReply = vi.fn(() => Promise.reject(new Error('offline')))
    render(<ReplyChips chips={CHIPS} latest onReply={onReply} />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Window' }))
    expect(onReply).toHaveBeenCalledWith('Window')
    await act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)))
    chipButtons().forEach((chip) => expect(chip).toBeEnabled())
    expect(console.warn).toHaveBeenCalled()
  })

  it('starts a new set of chips open', () => {
    const onReply = vi.fn()
    const { rerender } = render(<ReplyChips chips={CHIPS} latest onReply={onReply} />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Window' }))
    rerender(<ReplyChips chips={['Yes', 'No']} latest onReply={onReply} />)
    chipButtons().forEach((chip) => expect(chip).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Yes' }))
    expect(onReply).toHaveBeenLastCalledWith('Yes')
  })

  it('is gone once a newer message exists', () => {
    const { rerender } = render(<ReplyChips chips={CHIPS} latest onReply={vi.fn()} />)
    expect(chipButtons()).toHaveLength(3)
    rerender(<ReplyChips chips={CHIPS} latest={false} onReply={vi.fn()} />)
    expect(chipButtons()).toHaveLength(0)
    expect(screen.queryByRole('group', { name: 'Replies' })).toBeNull()
  })

  it('shows at most three chips, and none unless at least two can be drawn', () => {
    const { rerender } = render(<ReplyChips chips={[...CHIPS, 'Later']} latest onReply={vi.fn()} />)
    expect(chipButtons()).toHaveLength(3)
    rerender(<ReplyChips chips={['Window']} latest onReply={vi.fn()} />)
    expect(chipButtons()).toHaveLength(0)
  })

  it(`leaves out a chip over ${REPLY_CHIP_MAX_CHARS} characters, an empty one and a repeat`, () => {
    const long = 'x'.repeat(REPLY_CHIP_MAX_CHARS + 1)
    const longest = 'y'.repeat(REPLY_CHIP_MAX_CHARS)
    expect(replyChipsOf(['Window', long, '  ', 'Window', ` ${longest} `, 7 as unknown as string])).toEqual(['Window', longest])
  })

  it("marks the chip's words as Auto's, inside the chip and not on the button", () => {
    render(<ReplyChips chips={CHIPS} latest onReply={vi.fn()} />)
    const chip = screen.getByRole('button', { name: 'Reply: Aisle' })
    expect(chip).not.toHaveAttribute('data-ai-generated')
    expect(chip.querySelector('[data-ai-generated="true"]')).toHaveTextContent('Aisle')
  })

  it("draws the model's words as text, never markup", () => {
    const { container } = render(<ReplyChips chips={['<img src=x>', 'Fine']} latest onReply={vi.fn()} />)
    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByRole('button', { name: 'Reply: <img src=x>' })).toBeInTheDocument()
  })
})

const LOOKS = [
  { name: 'Studio light', ground: 'studio' },
  { name: 'Studio dark', ground: 'studio dark' },
  { name: 'Classic', ground: '' },
] as const

describe.each(LOOKS)('ReplyChips in $name', ({ ground }) => {
  it("draws every chip from the same token classes (each look's token file paints them)", () => {
    render(
      <div className={ground}>
        <ReplyChips chips={CHIPS} latest onReply={vi.fn()} />
      </div>,
    )
    chipButtons().forEach((chip) => expect(chip).toHaveClass('rounded-xl', 'border-border', 'bg-card', 'text-foreground'))
  })
})

describe('reply chips source', () => {
  it('uses tokens only: no hex, no orange, no rounded-3xl', () => {
    const source = proactiveSource('reply-chips.tsx')
    expect(source).not.toMatch(HEX_COLOUR)
    expect(source).not.toMatch(ORANGE_CLASS)
    expect(source).not.toMatch(ROUNDED_3XL)
  })

  it(`keeps each component at most ${MAX_COMPONENT_LINES} lines`, () => {
    const spans = componentSpans(proactiveSource('reply-chips.tsx'))
    expect(spans.map((span) => span.name)).toEqual(['ReplyChips'])
    spans.forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
  })
})
