/**
 * PRD-264 EUAI-09 (Art. 50(2)) — Auto's reply in Play's chat is marked machine-readable
 * as AI-generated: its root carries `data-ai-generated="true"`, while it streams and once
 * it is done, as the core chat's MessageWithReceipts does. The core's cards and the
 * markdown renderer are stand-ins; the reply itself is real.
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/types'

vi.mock('../play-markdown', () => ({
  PlayMarkdown: ({ text }: { text: string }) => <div data-testid="the-answer">{text}</div>,
}))
vi.mock('../reply-cards', () => ({ ReplyCards: () => null }))

import { AutoReply, SPEAKER } from '../auto-reply'

afterEach(cleanup)

const reply = { id: 'a1', role: 'assistant', content: 'Done.', parts: [] } as unknown as ChatMessage

describe("Auto's reply in Play is marked AI-generated", () => {
  it('carries data-ai-generated="true" on the reply that holds the answer', () => {
    render(<AutoReply message={reply} streaming={false} />)
    const holder = screen.getByRole('article', { name: `${SPEAKER} said` })
    expect(holder).toHaveAttribute('data-ai-generated', 'true')
    expect(holder).toContainElement(screen.getByTestId('the-answer'))
  })

  it('carries it while the reply still streams', () => {
    render(<AutoReply message={{ ...reply, content: '' }} streaming />)
    expect(screen.getByRole('article', { name: `${SPEAKER} said` })).toHaveAttribute('data-ai-generated', 'true')
  })
})
