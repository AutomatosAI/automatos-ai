/**
 * PRD-264 EUAI-09 (Art. 50(2)) — Auto's message in the floating AutoWidget is marked
 * machine-readable as AI-generated: its row carries `data-ai-generated="true"`, as the
 * core chat's MessageWithReceipts and Play's AutoReply do. A user message carries no mark.
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/types'

import { WidgetMessage } from '../widget-message'

afterEach(cleanup)

const user = { id: 'u1', role: 'user', content: 'Approve #0422', parts: [] } as unknown as ChatMessage
const reply = { id: 'a1', role: 'assistant', content: 'Done.', parts: [] } as unknown as ChatMessage

describe("Auto's message in the AutoWidget is marked AI-generated", () => {
  it('carries data-ai-generated="true" on the row that holds the reply', () => {
    render(<WidgetMessage message={reply} />)
    const row = screen.getByTestId('widget-message')
    expect(row).toHaveAttribute('data-ai-generated', 'true')
    expect(row).toHaveTextContent('Done.')
  })

  it('leaves a user message unmarked', () => {
    const { container } = render(<WidgetMessage message={user} />)
    expect(screen.getByTestId('widget-message')).toHaveTextContent('Approve #0422')
    expect(container.querySelector('[data-ai-generated]')).toBeNull()
  })
})
