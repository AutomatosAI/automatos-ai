import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

vi.mock('../agent-selector', () => ({ AgentSelector: () => null }))
vi.mock('@/lib/api-client', () => ({ apiClient: { uploadAttachment: vi.fn() } }))

import { MultimodalInput } from '../multimodal-input'

afterEach(cleanup)

const props = {
  chatId: 'accessibility-test',
  selectedVisibilityType: 'private' as const,
}

describe('chat composer accessible actions', () => {
  it('names Send, hides its icon, and preserves empty-input and submit behavior', () => {
    const sendMessage = vi.fn()
    render(<MultimodalInput {...props} status="ready" stop={vi.fn()} sendMessage={sendMessage} />)

    const send = screen.getByRole('button', { name: /send message/i })
    expect(send.querySelector('svg')).toHaveAttribute('aria-hidden', 'true')
    expect(send).toBeDisabled()
    fireEvent.click(send)
    expect(sendMessage).not.toHaveBeenCalled()

    const input = screen.getByRole('textbox')
    fireEvent.change(input, { target: { value: '   ' } })
    expect(send).toBeDisabled()
    fireEvent.change(input, { target: { value: '  Hello  ' } })
    expect(send).toBeEnabled()
    fireEvent.click(send)
    expect(sendMessage).toHaveBeenCalledOnce()
    expect(sendMessage).toHaveBeenCalledWith({
      role: 'user', content: 'Hello', attachment_ids: [],
      parts: [{ type: 'text', text: 'Hello' }],
    })
    expect(input).toHaveValue('')
    expect(send).toBeDisabled()
  })

  it('names Stop, hides its icon, and stops generation without sending', () => {
    const stop = vi.fn()
    const sendMessage = vi.fn()
    render(<MultimodalInput {...props} status="streaming" stop={stop} sendMessage={sendMessage} />)

    const button = screen.getByRole('button', { name: /stop generating/i })
    expect(button.querySelector('svg')).toHaveAttribute('aria-hidden', 'true')
    expect(button).toBeEnabled()
    expect(screen.queryByRole('button', { name: /send message/i })).not.toBeInTheDocument()
    const input = screen.getByRole('textbox')
    fireEvent.change(input, { target: { value: 'Hello' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    fireEvent.click(button)
    expect(stop).toHaveBeenCalledOnce()
    expect(sendMessage).not.toHaveBeenCalled()
  })

  it('keeps Enter submission and Shift+Enter behavior with the named Send action', () => {
    const sendMessage = vi.fn()
    render(<MultimodalInput {...props} status="ready" stop={vi.fn()} sendMessage={sendMessage} />)

    expect(screen.getByRole('button', { name: /send message/i })).toBeDisabled()
    const input = screen.getByRole('textbox')
    fireEvent.change(input, { target: { value: 'Hello' } })
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
    expect(sendMessage).not.toHaveBeenCalled()
    expect(input).toHaveValue('Hello')
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(sendMessage).toHaveBeenCalledOnce()
    expect(input).toHaveValue('')
  })
})
