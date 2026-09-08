/**
 * PRD-239 — a reply renders a persisted ticket card (S2) and a failed turn
 * shows its sentence in the bubble (S4).
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: vi.fn() } }))
vi.mock('framer-motion', () => ({ motion: { div: (p: any) => <div data-testid={p['data-testid']}>{p.children}</div> } }))
vi.mock('../message-actions', () => ({ MessageActions: () => null }))

import { Message } from '../message'
import type { ChatMessage } from '@/types'

const base = {
  chatId: 'c1',
  setMessages: vi.fn() as any,
  regenerate: vi.fn(),
  isReadonly: true,
}

afterEach(cleanup)

describe('Message (PRD-239)', () => {
  it('renders a task_card part persisted with a session agent reply', () => {
    const message = {
      id: 'm1',
      role: 'assistant',
      content: '',
      parts: [
        { type: 'text', text: "Bob runs as a Claude Code session on your CLI host, so I've filed ticket #118 for this message." },
        { type: 'task_card', card: { id: 118, title: 'Chat with Bob: Hey Bob', status: 'assigned', assigned_agent: 'Bob' } },
      ],
    } as unknown as ChatMessage
    render(<Message {...base} message={message} />)
    expect(screen.getByTestId('task-card')).toBeInTheDocument()
    expect(screen.getByText('Ticket #118')).toBeInTheDocument()
    expect(screen.getByText(/filed ticket #118/)).toBeInTheDocument()
  })

  it('shows the failure sentence in the bubble instead of silence', () => {
    const message = {
      id: 'm2',
      role: 'assistant',
      content: '',
      parts: [],
      error: { message: "Researcher's model is not available: OpenRouter does not offer the model.", code: 'model_unavailable' },
    } as unknown as ChatMessage
    render(<Message {...base} message={message} />)
    expect(screen.getByTestId('turn-error')).toHaveTextContent("Researcher's model is not available")
  })
})
