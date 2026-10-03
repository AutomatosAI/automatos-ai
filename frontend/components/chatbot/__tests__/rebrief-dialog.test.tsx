/**
 * PRD-252 R2 — "Update ticket and re-queue" opens on the brief Auto proposed,
 * and sends the brief the owner keeps back onto the ticket.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

const rebrief = vi.hoisted(() => ({ mutate: vi.fn(), isLoading: false }))
const messages = vi.hoisted(() => ({ load: vi.fn() }))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))

vi.mock('@/hooks/use-board-tasks', () => ({ useRebriefTask: () => rebrief }))
vi.mock('@/lib/chat/api', () => ({ getChatMessages: (id: string) => messages.load(id) }))
vi.mock('@/stores/chat-session-store', () => ({
  useChatSessionStore: (pick: (s: unknown) => unknown) => pick({ session: { activeChatId: 'chat-1' } }),
}))
vi.mock('sonner', () => ({ toast }))

import { RebriefDialog } from '../rebrief-dialog'

const discussion = { ticketId: '612', number: '#0042', title: 'Welcome email', agentName: 'Words', missionId: null }
const AGREED = 'Write to Priya at Gull & Anchor: two short paragraphs, no prices.'

beforeEach(() => {
  rebrief.mutate.mockReset()
  toast.success.mockReset()
  messages.load.mockResolvedValue([
    { id: 'u', role: 'user', parts: [{ type: 'text', text: 'Shorter please.' }] },
    { id: 'a', role: 'assistant', parts: [{ type: 'text', text: `Agreed:\n\`\`\`\n${AGREED}\n\`\`\`` }] },
  ])
})
afterEach(cleanup)

describe('RebriefDialog', () => {
  it('opens on the brief Auto proposed, and sends the one the owner keeps', async () => {
    const onDone = vi.fn()
    render(<RebriefDialog discussion={discussion} open onOpenChange={vi.fn()} onDone={onDone} />)
    const box = screen.getByRole('textbox', { name: /The agreed brief/ })
    await waitFor(() => expect(box).toHaveValue(AGREED))
    expect(messages.load).toHaveBeenCalledWith('chat-1')

    fireEvent.change(box, { target: { value: `  ${AGREED} Sign it Sam.  ` } })
    rebrief.mutate.mockImplementation((_vars, opts) => opts.onSuccess())
    fireEvent.click(screen.getByRole('button', { name: 'Update and re-queue' }))

    expect(rebrief.mutate).toHaveBeenCalledWith({ taskId: '612', brief: `${AGREED} Sign it Sam.` }, expect.anything())
    expect(toast.success).toHaveBeenCalledWith('Ticket #0042 is back with Words, working from the brief you agreed.')
    expect(onDone).toHaveBeenCalled()
  })

  it('sends nothing empty, and says when the conversation could not be read', async () => {
    messages.load.mockRejectedValue(new Error('offline'))
    render(<RebriefDialog discussion={discussion} open onOpenChange={vi.fn()} onDone={vi.fn()} />)
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveAttribute('placeholder', expect.stringContaining('Could not read the conversation')))
    expect(screen.getByRole('button', { name: 'Update and re-queue' })).toBeDisabled()
  })
})
