// Theme voice — Studio's quick ask in the chat composer (components/chatbot/composer-voice.tsx, in the action row of
// composer-actions.tsx), over a HostVoice stub: the host's voice button sits right before Send (and Stop), at Send's
// size; a hold starts the quick ask on the conversation on screen and letting go sends it; a click no pointer made
// toggles; there is no hotkey (Space is the page's and the box's); its own controls never submit the composer's form;
// it is never fenced on the AI label, while the box and Send are; outside a voice provider it is drawn disabled and
// says why; a browser that can't recognise speech says so, and typing and Send still work.
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const voice = vi.hoisted(() => ({ current: null as unknown }))
vi.mock('@/components/voice/voice-provider', () => ({ useOptionalHostVoice: () => voice.current }))
vi.mock('../agent-selector', () => ({ AgentSelector: () => null }))
vi.mock('@/lib/api-client', () => ({ apiClient: { uploadAttachment: vi.fn() } }))

import { AI_LABEL_FENCE_ATTR } from '@/components/ai-label/ai-label-fence'
import { type HostVoiceStubPatch, hostVoiceStub } from '@/components/voice/__tests__/support/host-voice-stub'
import { AI_VOICE_LABEL } from '@/components/voice/AiVoiceLabel'
import { VOICE_BUTTON_WORDS } from '@/components/voice/voice-button-view'
import type { HostVoice } from '@/components/voice/voice-provider'
import { PAGE_AI_LABEL, useAiLabelStore } from '@/lib/ai-label/store'
import { QUICK_ASK_ERRORS } from '@/lib/voice/quick-ask/copy'

import { MultimodalInput } from '../multimodal-input'

const IDLE: HostVoiceStubPatch = { snapshot: { mode: null, phase: 'idle', micOpen: false, chatId: null } }
const QUICK_LISTENING: HostVoiceStubPatch = { snapshot: { mode: 'quick', phase: 'listening', micOpen: true, chatId: null } }
const UNSUPPORTED: HostVoiceStubPatch = {
  snapshot: { mode: 'quick', phase: 'error', micOpen: false, chatId: null, error: QUICK_ASK_ERRORS.unsupported },
}
const SEND = /send message/i
const FENCE = `[${AI_LABEL_FENCE_ATTR}]`

type Status = 'ready' | 'streaming'

afterEach(() => {
  cleanup()
  voice.current = null
  useAiLabelStore.getState().releaseLabel(PAGE_AI_LABEL)
})

function drawComposer(stub: HostVoice | null, status: Status) {
  voice.current = stub
  const sendMessage = vi.fn()
  const view = render(
    <MultimodalInput chatId="chat-on-screen" status={status} stop={vi.fn()} sendMessage={sendMessage} selectedVisibilityType="private" />,
  )
  const root = view.container.querySelector<HTMLElement>('[data-voice-button="root"]')
  const press = view.container.querySelector<HTMLButtonElement>('[data-voice-button="press"]')
  if (!root || !press) throw new Error('the composer drew no voice button')
  return { sendMessage, root, press }
}

/** The composer inside a voice provider whose session is `patch`. */
function draw(patch: HostVoiceStubPatch, status: Status = 'ready') {
  const stub = hostVoiceStub(patch)
  return { stub, ...drawComposer(stub, status) }
}

function typeAndSend(text: string): void {
  fireEvent.change(screen.getByRole('textbox'), { target: { value: text } })
  fireEvent.click(screen.getByRole('button', { name: SEND }))
}

describe("the composer's voice button", () => {
  it('sits right before Send, in a slot of its size', () => {
    const { root, press } = draw(IDLE)
    const send = screen.getByRole('button', { name: SEND })
    const slot = root.parentElement
    expect(slot?.nextElementSibling?.contains(send)).toBe(true)
    expect(send.closest('div')?.contains(press)).toBe(true)
    expect(slot?.className).toContain('h-10 w-10')
    expect(send.className).toContain('h-10 w-10')
    expect(root.dataset.voiceSize).toBe('md')
    expect(press).toHaveAccessibleName(VOICE_BUTTON_WORDS.hold)
  })

  it('sits right before Stop while a reply streams', () => {
    const { root } = draw(IDLE, 'streaming')
    const stop = screen.getByRole('button', { name: /stop generating/i })
    expect(root.parentElement?.nextElementSibling?.contains(stop)).toBe(true)
  })

  it('holds to talk in quick mode, into the conversation on screen: down starts, letting go sends', () => {
    const { stub, root, press } = draw(IDLE)
    expect({ ...root.dataset }).toMatchObject({ voicePress: 'hold', voiceMode: 'quick', voiceState: 'idle' })
    fireEvent.pointerDown(press)
    expect(vi.mocked(stub.start).mock.calls).toStrictEqual([[{ mode: 'quick' }]])
    fireEvent.pointerUp(press)
    expect(stub.release).toHaveBeenCalledTimes(1)
  })

  it('takes a click no pointer made (a key, a screen reader) as a toggle', () => {
    const idle = draw(IDLE)
    fireEvent.click(idle.press)
    expect(idle.stub.start).toHaveBeenCalledWith({ mode: 'quick' })
    cleanup()
    const hearing = draw(QUICK_LISTENING)
    fireEvent.click(hearing.press)
    expect(hearing.stub.release).toHaveBeenCalledTimes(1)
  })

  it('has no hotkey: Space on the page and in the box stays theirs', () => {
    const { stub } = draw(IDLE)
    fireEvent.keyDown(document.body, { code: 'Space' })
    fireEvent.keyUp(document.body, { code: 'Space' })
    fireEvent.keyDown(screen.getByRole('textbox'), { code: 'Space' })
    expect(stub.start).not.toHaveBeenCalled()
  })

  it('shows the AI voice caption and Stop while the quick ask is its own; Stop never sends the draft', () => {
    const { stub, root, sendMessage } = draw(QUICK_LISTENING)
    expect(within(root).getByTestId('ai-voice-label')).toHaveTextContent(AI_VOICE_LABEL)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'A draft' } })
    fireEvent.click(within(root).getByRole('button', { name: VOICE_BUTTON_WORDS.stop }))
    expect(stub.stop).toHaveBeenCalledTimes(1)
    expect(sendMessage).not.toHaveBeenCalled()
    expect(screen.getByRole('textbox')).toHaveValue('A draft')
    expect(root.parentElement?.nextElementSibling?.contains(screen.getByRole('button', { name: SEND }))).toBe(true)
  })
})

describe('the composer around it', () => {
  it('still types and sends, by Send and by Enter', () => {
    const { sendMessage } = draw(IDLE)
    typeAndSend('Hello')
    expect(sendMessage).toHaveBeenCalledWith({ role: 'user', content: 'Hello', attachment_ids: [], parts: [{ type: 'text', text: 'Hello' }] })
    const box = screen.getByRole('textbox')
    fireEvent.change(box, { target: { value: 'Again' } })
    fireEvent.keyDown(box, { key: 'Enter' })
    expect(sendMessage).toHaveBeenCalledTimes(2)
    expect(box).toHaveValue('')
  })

  it('fences the box and Send on the AI label, never the voice button', () => {
    useAiLabelStore.getState().setHidden(PAGE_AI_LABEL, true)
    const { stub, press } = draw(IDLE)
    const box = screen.getByRole('textbox')
    expect(box).toBeDisabled()
    expect(box.closest(FENCE)).not.toBeNull()
    expect(screen.getByRole('button', { name: SEND }).closest(FENCE)).toHaveProperty('disabled', true)
    expect(press).toBeEnabled()
    expect(press.closest('fieldset')).toBeNull()
    fireEvent.pointerDown(press)
    expect(stub.start).toHaveBeenCalledWith({ mode: 'quick' })
  })

  it("says the browser can't recognise speech, and the box still types and sends", () => {
    const { stub, sendMessage } = draw(UNSUPPORTED)
    expect(screen.getAllByText(QUICK_ASK_ERRORS.unsupported).length).toBeGreaterThan(0)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Typed instead' } })
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(stub.start).toHaveBeenCalledWith({ mode: 'quick' })
    expect(sendMessage).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: SEND }))
    expect(sendMessage).toHaveBeenCalledWith(expect.objectContaining({ content: 'Typed instead' }))
  })

  it("outside a voice provider draws the button disabled, saying voice isn't available here, and still sends", () => {
    const { press, sendMessage } = drawComposer(null, 'ready')
    expect(press).toBeDisabled()
    expect(press).toHaveAttribute('title', VOICE_BUTTON_WORDS.unavailable)
    typeAndSend('Hello')
    expect(sendMessage).toHaveBeenCalledOnce()
  })
})
