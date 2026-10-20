/**
 * PRD-264 EUAI-09 (Art. 50(2)) — every assistant message in the chat is marked
 * machine-readable as AI-generated: the element that holds it carries
 * `data-ai-generated="true"`, with or without receipts. A user message carries no mark.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import type { ChatMessage, Receipt } from '@/types'

vi.mock('../message', () => ({
  Message: ({ message }: { message: ChatMessage }) => <div data-testid="the-message">{message.content}</div>,
}))

import { MessageWithReceipts } from '../message-with-receipts'

afterEach(cleanup)

const MOVED: Receipt = {
  action: 'platform_update_task_status', kind: 'write', status: 'done', subject: '#0422',
  effect: 'moved to Done', link: '/command-center?tab=board&task_id=422', reason: null,
}
const user = { id: 'u1', role: 'user', content: 'Approve #0422', parts: [] } as unknown as ChatMessage
const reply = { id: 'a1', role: 'assistant', content: 'Done.', parts: [] } as unknown as ChatMessage

function draw(message: ChatMessage) {
  return render(
    <MessageWithReceipts chatId="c1" setMessages={vi.fn() as any} regenerate={vi.fn()} isReadonly message={message} />,
  )
}

describe('an assistant message is marked AI-generated', () => {
  it('carries data-ai-generated="true" around the reply', () => {
    draw(reply)
    const holder = screen.getByTestId('ai-generated-message')
    expect(holder).toHaveAttribute('data-ai-generated', 'true')
    expect(holder).toContainElement(screen.getByTestId('the-message'))
  })

  it('carries it around the receipts and the reply alike', () => {
    draw({ ...reply, receipts: [MOVED] })
    const holder = screen.getByTestId('ai-generated-message')
    expect(holder).toHaveAttribute('data-ai-generated', 'true')
    expect(holder).toContainElement(screen.getByTestId('receipts-block'))
    expect(holder).toContainElement(screen.getByTestId('the-message'))
  })

  it('leaves a user message unmarked', () => {
    const { container } = draw(user)
    expect(screen.getByTestId('the-message')).toHaveTextContent('Approve #0422')
    expect(container.querySelector('[data-ai-generated]')).toBeNull()
  })
})
