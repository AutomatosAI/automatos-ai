// The chat composer (components/chatbot/multimodal-input.tsx, with composer-actions.tsx and composer-attachments.tsx)
// keeps what the browser specs, screen readers and the chat read from it: the box's placeholder (e2e/browser's
// CORE_COMPOSER), the hidden file picker, the attach button that opens it and waits on uploads, the attachments strip
// (an upload under way named, an uploaded file removable), the picked agent's tools (four named logos, then a count)
// and Send carrying the attachments' ids. The composer has no data-testid of its own: these names are its hooks.
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Agent } from '../agent-selector'

const fixtures = vi.hoisted(() => ({
  upload: vi.fn(),
  agent: {
    id: 7,
    name: 'Research Desk',
    agent_type: 'custom',
    description: '',
    status: 'active',
    is_default: false,
    tags: [],
    tools: ['Gmail', 'Slack', 'GitHub', 'Notion', 'Linear'].map((name, index) => ({ id: index + 1, name })),
  },
}))
vi.mock('@/lib/api-client', () => ({ apiClient: { uploadAttachment: fixtures.upload } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
// The picker hands the composer the picked agent, as the real one does once its agents load.
vi.mock('../agent-selector', async () => {
  const { useEffect } = await import('react')
  return {
    AgentSelector: ({ onAgentData }: { onAgentData?: (agent: Agent | null) => void }) => {
      useEffect(() => onAgentData?.(fixtures.agent), [onAgentData])
      return null
    },
  }
})

import { MultimodalInput } from '../multimodal-input'

const PLACEHOLDER = 'Send a message...'
const ACCEPTED = 'image/*,.pdf,.doc,.docx,.xls,.xlsx,.txt,.csv,.md,.json,.py,.js,.ts,.tsx'

interface UploadAnswer {
  readonly attachment_id: string
  readonly filename: string
  readonly mime: string
  readonly media_type: 'image' | 'document'
  readonly size_bytes: number
}

const NOTES: UploadAnswer = { attachment_id: 'att-1', filename: 'notes.pdf', mime: 'application/pdf', media_type: 'document', size_bytes: 4 }

afterEach(() => {
  cleanup()
  fixtures.upload.mockReset()
})

function draw(props: Partial<React.ComponentProps<typeof MultimodalInput>> = {}) {
  const sendMessage = vi.fn()
  const view = render(
    <MultimodalInput chatId="structure-test" status="ready" stop={vi.fn()} sendMessage={sendMessage} selectedVisibilityType="private" {...props} />,
  )
  const picker = view.container.querySelector<HTMLInputElement>('input[type="file"]')
  if (!picker) throw new Error('no file picker drawn')
  return { sendMessage, picker }
}

function composerForm(): HTMLFormElement {
  const form = screen.getByPlaceholderText(PLACEHOLDER).closest('form')
  if (!form) throw new Error('no composer form')
  return form
}

/** The attach button: the form's first button while no file is in the strip (an icon, with no name of its own). */
function attachButton(): HTMLElement {
  return within(composerForm()).getAllByRole('button')[0]
}

function pick(picker: HTMLInputElement, name: string): void {
  fireEvent.change(picker, { target: { files: [new File(['note'], name, { type: 'application/pdf' })] } })
}

describe('the composer keeps its hooks', () => {
  it('draws the box, the hidden picker the attach button opens, and Send', () => {
    const { picker } = draw()
    expect(screen.getByPlaceholderText(PLACEHOLDER).tagName).toBe('TEXTAREA')
    expect(picker).toHaveClass('hidden')
    expect(picker.multiple).toBe(true)
    expect(picker.getAttribute('accept')).toBe(ACCEPTED)
    const opened = vi.spyOn(picker, 'click').mockImplementation(() => undefined)
    fireEvent.click(attachButton())
    expect(opened).toHaveBeenCalledOnce()
    expect(screen.getByRole('button', { name: 'Send message' })).toHaveAttribute('type', 'submit')
  })

  it("shows the picked agent's first four tools as named buttons, then how many more; a click asks for suggestions", async () => {
    const onToolIconClick = vi.fn()
    draw({ onAgentChange: vi.fn(), onToolIconClick })
    for (const name of ['Gmail', 'Slack', 'GitHub', 'Notion']) {
      expect(await screen.findByRole('button', { name: `Show suggestions for ${name}` })).toBeInTheDocument()
    }
    expect(screen.queryByRole('button', { name: 'Show suggestions for Linear' })).toBeNull()
    expect(screen.getByText('+1')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Show suggestions for Slack' }))
    expect(onToolIconClick).toHaveBeenCalledWith('Slack')
  })

  it('names an upload under way, then shows the file, removable, and Send carries its id', async () => {
    let finish: (answer: UploadAnswer) => void = () => undefined
    fixtures.upload.mockImplementation(() => new Promise<UploadAnswer>((resolve) => { finish = resolve }))
    const { picker, sendMessage } = draw()
    pick(picker, NOTES.filename)
    expect(screen.getByText('(uploading…)')).toBeInTheDocument()
    expect(attachButton()).toBeDisabled()
    act(() => finish(NOTES))
    expect(await screen.findByRole('button', { name: 'Remove attachment' })).toBeInTheDocument()
    expect(screen.queryByText('(uploading…)')).toBeNull()
    expect(screen.getByText(NOTES.filename)).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText(PLACEHOLDER), { target: { value: 'Summarise it' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    expect(sendMessage).toHaveBeenCalledWith({
      role: 'user',
      content: 'Summarise it',
      attachment_ids: [NOTES.attachment_id],
      parts: [
        { type: 'file', filename: NOTES.filename, mediaType: 'application/octet-stream', attachment_id: NOTES.attachment_id },
        { type: 'text', text: 'Summarise it' },
      ],
    })
    expect(screen.queryByText(NOTES.filename)).toBeNull()
  })

  it('takes an uploaded file away on Remove attachment', async () => {
    fixtures.upload.mockResolvedValue(NOTES)
    const { picker } = draw()
    pick(picker, NOTES.filename)
    fireEvent.click(await screen.findByRole('button', { name: 'Remove attachment' }))
    expect(screen.queryByText(NOTES.filename)).toBeNull()
  })
})
