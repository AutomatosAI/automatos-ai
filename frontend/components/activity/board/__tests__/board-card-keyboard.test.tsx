/**
 * Issue #1090 — a board card is a keyboard-operable button, not a click-only div.
 * @hello-pangea/dnd is replaced by its render output so the card can be rendered in
 * isolation: the real Draggable needs a DragDropContext and the library's sensors.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import React from 'react'
import type { BoardTask } from '@/types/board'

vi.mock('@hello-pangea/dnd', () => ({
  Draggable: ({ children }: { children: (provided: unknown, snapshot: unknown) => React.ReactNode }) =>
    children(
      { innerRef: () => {}, draggableProps: { style: {} }, dragHandleProps: {} },
      { isDragging: false },
    ),
}))
vi.mock('../ticket-actions-menu', () => ({ TicketActionsMenu: () => null }))

import { BoardCard } from '../board-card'

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '1139', type: 'task', name: 'Price list for Tidewater', status: 'inbox', priority: 'medium',
    tags: [], review_mode: 'auto', source_id: '1139', ...over }
}

afterEach(() => cleanup())

describe('BoardCard keyboard', () => {
  it('opens with Enter and Space exactly as a click does', () => {
    const onOpen = vi.fn()
    render(<BoardCard task={ticket()} index={0} onOpen={onOpen} />)
    const card = screen.getByRole('button', { name: /Price list for Tidewater/ })
    expect(card).toHaveAttribute('tabindex', '0')

    fireEvent.click(card)
    expect(onOpen).toHaveBeenCalledTimes(1)

    onOpen.mockClear()
    fireEvent.keyDown(card, { key: 'Enter' })
    expect(onOpen).toHaveBeenCalledTimes(1)
    expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ id: '1139' }))

    onOpen.mockClear()
    expect(fireEvent.keyDown(card, { key: ' ' })).toBe(false) // the page does not scroll
    expect(onOpen).toHaveBeenCalledTimes(1)
  })
})
