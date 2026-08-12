/**
 * PRD-180 S4 (F037) — composio_execute is visible in the running/error chips
 * (vitest proxy for the browser AC; the loop is headless).
 *
 * Before: composio_execute was name-filtered OUT of the tool indicators, so
 * every external-app action (Gmail send, Slack post, …) ran invisibly. Now it
 * shows a running chip and an error chip, surfacing the resolved action name
 * where the executor logged it.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

// The tool-chip AC doesn't depend on the actions bar / voice player — stub the
// heavy children so the test stays focused and dep-light.
vi.mock('../message-actions', () => ({
  MessageActions: () => null,
}))
import { Message } from '../message'

function assistantWithTool(tool: Record<string, unknown>) {
  return {
    id: 'm-1',
    role: 'assistant' as const,
    content: 'Working on it.',
    parts: [{ type: 'text' as const, text: 'Working on it.' }],
    toolCalls: [tool],
  }
}

const noop = () => {}

function renderMessage(message: any) {
  return render(
    <Message
      chatId="c-1"
      message={message}
      isLoading={false}
      setMessages={noop as any}
      regenerate={noop}
      isReadonly={false}
    />,
  )
}

describe('composio_execute indicator (F037)', () => {
  it('renders a running chip for a running composio_execute call', () => {
    renderMessage(
      assistantWithTool({
        toolCallId: 't1',
        toolName: 'composio_execute',
        state: 'running',
        input: { action: 'GMAIL_SEND_EMAIL' },
      }),
    )
    // No longer filtered out — the chip shows, and surfaces the resolved action.
    expect(screen.getByText(/Composio · GMAIL_SEND_EMAIL/)).toBeInTheDocument()
  })

  it('renders an error chip when a composio_execute call fails', () => {
    renderMessage(
      assistantWithTool({
        toolCallId: 't2',
        toolName: 'composio_execute',
        state: 'error',
        error: 'Gmail auth expired',
        input: { action: 'GMAIL_SEND_EMAIL' },
      }),
    )
    expect(screen.getByText(/Composio · GMAIL_SEND_EMAIL failed/)).toBeInTheDocument()
  })

  it('falls back to a plain Composio label when no action is resolvable', () => {
    renderMessage(
      assistantWithTool({
        toolCallId: 't3',
        toolName: 'composio_execute',
        state: 'running',
        input: {},
      }),
    )
    expect(screen.getByText('Composio')).toBeInTheDocument()
  })
})
