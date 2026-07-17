/**
 * PRD-205 S7 -- the "Auto badge" on background-authored messages.
 *
 * The message badge slot historically rendered only the string source union
 * ('rag' | 'codegraph' | ...) via toUpperCase(). Background messages carry a
 * persisted OBJECT source ({origin, label, link_type, link_id}) lifted into
 * metadata.source by getChatMessages; the slot must render its label AS-IS
 * (never uppercased -- the server's exact label) while string sources keep
 * their existing rendering.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'

// Keep the test hermetic: actions and voice pull in api-client / audio.
vi.mock('@/components/chatbot/message-actions', () => ({
  MessageActions: () => null,
}))
vi.mock('@/components/voice/VoiceMessage', () => ({
  VoiceMessage: () => null,
}))

import { Message } from '@/components/chatbot/message'
import type { ChatMessage } from '@/types'

const AUTO_LABEL = 'Auto \u00b7 background'

function renderMessage(metadata: Record<string, unknown> | undefined) {
  const message = {
    id: 'm-1',
    role: 'assistant',
    parts: [{ type: 'text', text: 'Watched it. 8.4/10, passed.' }],
    metadata,
  } as unknown as ChatMessage
  return render(
    <Message
      chatId="chat-1"
      message={message}
      setMessages={vi.fn()}
      regenerate={vi.fn()}
      isReadonly={true}
    />
  )
}

describe('background message badge', () => {
  it('renders the persisted source label as-is', () => {
    renderMessage({
      source: {
        origin: 'watcher',
        label: AUTO_LABEL,
        link_type: 'watch',
        link_id: 'w-1',
      },
    })
    expect(screen.getByText(AUTO_LABEL)).toBeInTheDocument()
  })

  it('falls back to the Auto label when the object has none', () => {
    renderMessage({ source: { origin: 'scheduled_task' } })
    expect(screen.getByText(AUTO_LABEL)).toBeInTheDocument()
  })

  it('keeps the legacy uppercase rendering for string sources', () => {
    renderMessage({ source: 'rag' })
    expect(screen.getByText('RAG')).toBeInTheDocument()
  })

  it('renders no badge without a source', () => {
    renderMessage({ processing_time: 0.5 })
    expect(screen.queryByText(AUTO_LABEL)).toBeNull()
  })
})
